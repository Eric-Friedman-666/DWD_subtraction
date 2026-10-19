"""Rolling single-use block processes with bounded, checkpoint-safe recovery.

Only JSON and committed filenames cross the supervision boundary. A worker owns
its sampler backend, never returns a large posterior over an IPC queue, and must
exit before the controller changes the round snapshot or performs subtraction.
"""
from collections import deque
from datetime import datetime, timezone
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
import json
import os
import pickle
import shutil
import time
import traceback
from uuid import uuid4

from script.dwd_run_status import write_run_status

_worker_context = None


class BlockWorkerError(RuntimeError):
    """A block exhausted its recovery budget; the round must not be committed."""


def worker_progress(phase, *, force=False, **fields):
    """Mark actual forward progress, not a background 'still alive' heartbeat."""
    if _worker_context is None:
        return
    now = time.monotonic()
    if not force and phase == _worker_context['phase'] and now - _worker_context['last_write'] < 5.:
        return
    _worker_context.update(phase=phase, last_write=now,
        sequence=_worker_context['sequence'] + 1)
    if phase == _worker_context.get('startup_ready_phase'):
        _worker_context['startup_ready'] = True
    write_run_status(_worker_context['path'], controller_pid=_worker_context['controller_pid'],
        worker_pid=os.getpid(), block_index=_worker_context['index'],
        round_label=_worker_context['label'], attempt=_worker_context['attempt'],
        attempt_token=_worker_context['token'], sequence=_worker_context['sequence'],
        phase=phase, startup_ready=_worker_context.get('startup_ready', False),
        **({key: _worker_context[key] for key in ('gpu_device', 'gpu_slot_index', 'worker_slot_index')}
           if 'gpu_device' in _worker_context else {}), **fields)


def _store_payload(value):
    encoded = pickle.dumps(value, protocol=5)
    memory = SharedMemory(create=True, size=len(encoded))
    try:
        memory.buf[:len(encoded)] = encoded
    except BaseException:
        memory.close()
        memory.unlink()
        raise
    return memory, len(encoded)


def _load_payload(description):
    name, size = description
    memory = SharedMemory(name=name)
    try:
        return pickle.loads(bytes(memory.buf[:size]))
    finally:
        memory.close()


def _release_payload(description):
    memory, _ = description
    memory.close()
    memory.unlink()


def _run_one(job_payload, initialization_payload, progress):
    global _worker_context
    # The environment is also inherited at Process.start(), so even imports in
    # the spawned __main__ see this device. Reassert before payload unpickling.
    if progress.get('gpu_device') is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = progress['gpu_device']
        os.environ['DWD_ASSIGNED_GPU'] = progress['gpu_device']
        os.environ['DWD_GPU_SLOT_INDEX'] = str(progress['gpu_slot_index'])
        os.environ['DWD_WORKER_SLOT_INDEX'] = str(progress['worker_slot_index'])
        if progress.get('gpu_memory_fraction') is not None:
            os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(progress['gpu_memory_fraction'])
    _worker_context = dict(progress, sequence=0, phase=None, last_write=0.)
    try:
        worker_progress('initializing', force=True)
        initializer, initargs = _load_payload(initialization_payload)
        function, arguments = _load_payload(job_payload)
        if initializer is not None:
            initializer(*initargs)
        worker_progress('running', force=True)
        function(*arguments)
        worker_progress('saved', force=True)
    except BaseException as error:
        worker_progress('failed', force=True, error=repr(error),
            traceback=traceback.format_exc()[-8000:])
        raise


def _stop(processes, grace):
    """Stop only owned children; never wait indefinitely in pool shutdown."""
    processes = list(processes)
    for process in processes:
        if process.is_alive():
            process.terminate()
    deadline = time.monotonic() + grace
    while any(process.is_alive() for process in processes) and time.monotonic() < deadline:
        time.sleep(.02)
    for process in processes:
        if process.is_alive():
            process.kill()
    deadline = time.monotonic() + grace
    while any(process.is_alive() for process in processes) and time.monotonic() < deadline:
        time.sleep(.02)
    for process in processes:
        process.join(timeout=0)
    if any(process.is_alive() for process in processes):
        raise BlockWorkerError('An owned worker could not be reaped; refusing to change the shared round snapshot')


def _read_progress(path, token, pid):
    try:
        fields = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return fields if fields.get('attempt_token') == token and fields.get('worker_pid') == pid else {}


def _archive_attempt(work, label, index, attempt, token, progress):
    """Preserve partial chains/results offline instead of reusing or deleting them."""
    checkpoint = work / f'round_{label}_block_{index:05d}'
    archive = work / 'failed_attempts' / f'round_{label}_block_{index:05d}' / f'attempt_{attempt}_{token}'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if checkpoint.exists():
        checkpoint.rename(archive)
    else:
        archive.mkdir()
    if progress.is_file():
        shutil.copy2(progress, archive / 'worker.status.json')


def run_block_jobs(function, jobs, *, max_workers, initializer=None, initargs=(),
        work, label, progress_timeout=600., max_attempts=3, poll_interval=.1,
        stop_grace=3., startup_ready_phase=None, status_file=None, run_status=None,
        gpu_devices=None, workers_per_gpu=None, worker_gpu_memory_fraction=None):
    """Fill free slots immediately, retry failed jobs, and enforce the round barrier.

    A progress timeout kills the affected process, archives its incomplete bank,
    and retries in a fresh process/CUDA context. Healthy long fits remain alive
    through real sampler progress. A repeatedly failing block fails the round;
    it is never converted into an empty result or silently skipped.
    """
    if max_workers < 1 or progress_timeout <= 0 or max_attempts < 1 or poll_interval <= 0 or stop_grace <= 0:
        raise ValueError('Worker limits, timeouts and attempt counts must be positive')
    if (gpu_devices is None) != (workers_per_gpu is None):
        raise ValueError('GPU devices and workers_per_gpu must be provided together')
    if worker_gpu_memory_fraction is not None:
        if gpu_devices is None or not 0. < float(worker_gpu_memory_fraction) <= 1.:
            raise ValueError('Worker GPU memory fraction requires GPU admission and must be in (0, 1]')
        worker_gpu_memory_fraction = float(worker_gpu_memory_fraction)
    if gpu_devices is not None:
        gpu_devices = tuple(str(device).strip() for device in gpu_devices)
        if (not gpu_devices or any(not device or ',' in device for device in gpu_devices)
                or len(set(gpu_devices)) != len(gpu_devices)
                or not isinstance(workers_per_gpu, int) or workers_per_gpu < 1
                or max_workers > workers_per_gpu * len(gpu_devices)):
            raise ValueError('Unique GPU devices, positive integer capacity and sufficient GPU slots required')
    work, label = Path(work), str(label)
    progress_directory = work / 'worker_status'
    progress_directory.mkdir(parents=True, exist_ok=True)
    context = get_context('spawn')
    incoming, retries, active, seen = iter(jobs), deque(), {}, set()
    exhausted = False
    summary = dict(configured_slots=int(max_workers), spawned_processes=0,
        completed_blocks=0, retries=0, forced_cleanup_after_commit=0, maximum_active=0)
    if gpu_devices is not None:
        summary.update(configured_workers_per_gpu=workers_per_gpu, gpu_devices=list(gpu_devices),
            maximum_active_by_gpu={device: 0 for device in gpu_devices},
            worker_gpu_memory_fraction=worker_gpu_memory_fraction)
    # Passing large NumPy initialization arguments directly to Process.start()
    # can block its spawn pipe while the child imports JAX. Only tiny shared-
    # memory descriptors cross that pipe; decoding stays inside the watchdog.
    initialization_payload = _store_payload((initializer, initargs))

    def publish(event, **details):
        if status_file is not None and run_status is not None:
            run_status.update(last_worker_event=dict(event=event, **details),
                active_worker_jobs=[dict(block_index=index, worker_pid=entry['process'].pid,
                    attempt=entry['attempt'], phase=entry.get('phase', 'spawning'),
                    **({key: entry[key] for key in ('gpu_device', 'gpu_slot_index', 'worker_slot_index')}
                       if 'gpu_device' in entry else {}))
                    for index, entry in active.items()], last_round_worker_summary=dict(summary))
            write_run_status(status_file, **run_status)

    try:
        while active or retries or not exhausted:
            while len(active) < max_workers and (retries or not exhausted):
                # Stagger GPU context/kernel initialization, not the fits. Once
                # this worker reports readiness, fill the next rolling slot.
                assignment = {}
                if gpu_devices is None:
                    if any(not entry['startup_ready'] for entry in active.values()):
                        break
                else:
                    # Each physical GPU has an independent readiness gate and
                    # stable slots. Unequal job lengths cannot overload one GPU
                    # via an ever-increasing spawn-identity modulo assignment.
                    available = []
                    for gpu_number, device in enumerate(gpu_devices):
                        occupants = [entry for entry in active.values() if entry['gpu_device'] == device]
                        if (len(occupants) < workers_per_gpu
                                and all(entry['startup_ready'] for entry in occupants)):
                            occupied = {entry['gpu_slot_index'] for entry in occupants}
                            slot = next(slot for slot in range(workers_per_gpu) if slot not in occupied)
                            available.append((len(occupants), gpu_number, slot, device))
                    if not available:
                        break
                    _, gpu_number, slot, device = min(available)
                    assignment = dict(gpu_device=device, gpu_slot_index=slot,
                        worker_slot_index=gpu_number * workers_per_gpu + slot,
                        gpu_memory_fraction=worker_gpu_memory_fraction)
                if retries:
                    index, table, attempt = retries.popleft()
                else:
                    job = next(incoming, None)
                    if job is None:
                        exhausted = True
                        break
                    index, table = job
                    index, attempt = int(index), 1
                    if index in seen:
                        raise ValueError(f'Duplicate block job: {index}')
                    seen.add(index)
                result = work / f'round_{label}_block_{index:05d}' / 'result.pkl'
                if result.is_file():
                    summary['completed_blocks'] += 1
                    publish('checkpoint_available', block_index=index)
                    continue
                token = uuid4().hex
                progress = progress_directory / f'round_{label}_block_{index:05d}.status.json'
                metadata = dict(path=progress, controller_pid=os.getpid(), index=index,
                    label=label, attempt=attempt, token=token,
                    startup_ready_phase=startup_ready_phase, **assignment)
                payload = _store_payload((function, (index, table)))
                process = context.Process(target=_run_one,
                    args=((payload[0].name, payload[1]),
                        (initialization_payload[0].name, initialization_payload[1]), metadata))
                try:
                    # spawn imports __main__ before entering _run_one. Bind the
                    # device in the child's inherited environment, not merely
                    # in an initializer that may run after JAX imports.
                    environment = ({'CUDA_VISIBLE_DEVICES': assignment['gpu_device'],
                        'DWD_ASSIGNED_GPU': assignment['gpu_device'],
                        'DWD_GPU_SLOT_INDEX': str(assignment['gpu_slot_index']),
                        'DWD_WORKER_SLOT_INDEX': str(assignment['worker_slot_index'])}
                        if assignment else {})
                    if assignment.get('gpu_memory_fraction') is not None:
                        environment['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(assignment['gpu_memory_fraction'])
                    previous = {key: os.environ.get(key) for key in environment}
                    try:
                        os.environ.update(environment)
                        process.start()
                    finally:
                        for key, value in previous.items():
                            if value is None:
                                os.environ.pop(key, None)
                            else:
                                os.environ[key] = value
                except BaseException:
                    try:
                        if process.pid is not None:
                            _stop([process], stop_grace)
                            process.close()
                    finally:
                        _release_payload(payload)
                    raise
                active[index] = dict(process=process, table=table, attempt=attempt,
                    payload=payload,
                    token=token, progress=progress, last_progress=time.monotonic(),
                    sequence=-1, result=result, phase='spawning',
                    startup_ready=startup_ready_phase is None, **assignment)
                summary['spawned_processes'] += 1
                summary['maximum_active'] = max(summary['maximum_active'], len(active))
                if assignment:
                    device = assignment['gpu_device']
                    occupancy = sum(entry['gpu_device'] == device for entry in active.values())
                    summary['maximum_active_by_gpu'][device] = max(
                        summary['maximum_active_by_gpu'][device], occupancy)
                publish('started', block_index=index, attempt=attempt, worker_pid=process.pid)
            advanced = False
            for index, entry in list(active.items()):
                process = entry['process']
                worker_pid = process.pid
                fields = _read_progress(entry['progress'], entry['token'], process.pid)
                if fields.get('sequence', -1) > entry['sequence']:
                    entry.update(sequence=fields['sequence'], phase=fields['phase'],
                        last_progress=time.monotonic())
                    entry['startup_ready'] |= bool(fields.get('startup_ready'))
                committed = entry['result'].is_file()
                # A saved result is not a reaped process. Let normal GPU/Python
                # teardown finish, retaining this slot and the round barrier.
                # The same no-progress watchdog bounds a genuinely stuck exit;
                # stop_grace is only the TERM/KILL grace, not a job-exit deadline.
                if not process.is_alive():
                    process.join(timeout=0)
                    exitcode = process.exitcode
                    process.close()
                    _release_payload(entry['payload'])
                    del active[index]
                    advanced = True
                    if committed:
                        summary['completed_blocks'] += 1
                        publish('completed', block_index=index, attempt=entry['attempt'], exitcode=exitcode)
                        continue
                    reason = fields.get('error', f'Worker exited {exitcode} without a committed result')
                elif time.monotonic() - entry['last_progress'] >= progress_timeout:
                    reason = f'No forward progress for {progress_timeout:g}s in {entry["phase"]}'
                    _stop([process], stop_grace)
                    if entry['result'].is_file():
                        summary['forced_cleanup_after_commit'] += 1
                    process.close()
                    _release_payload(entry['payload'])
                    del active[index]
                    advanced = True
                else:
                    continue
                # An atomic commit may win the race with timeout cancellation.
                if entry['result'].is_file():
                    summary['completed_blocks'] += 1
                    publish('completed_after_cleanup', block_index=index, attempt=entry['attempt'])
                    continue
                _archive_attempt(work, label, index, entry['attempt'], entry['token'], entry['progress'])
                if entry['attempt'] >= max_attempts:
                    raise BlockWorkerError(f'Round {label} block {index:05d} failed after '
                        f'{max_attempts} attempts: {reason}')
                summary['retries'] += 1
                retries.appendleft((index, entry['table'], entry['attempt'] + 1))
                if run_status is not None:
                    recovery = dict(block_index=index, attempt=entry['attempt'], worker_pid=worker_pid,
                        reason=reason, updated_utc=datetime.now(timezone.utc).isoformat())
                    run_status['worker_recovery_events'] = [
                        *run_status.get('worker_recovery_events', []), recovery][-8:]
                print(f'Round {label} block {index:05d}: {reason}; retrying in a fresh worker', flush=True)
                publish('retrying', block_index=index, attempt=entry['attempt'], reason=reason)
            if active and not advanced:
                time.sleep(poll_interval)
        publish('round_workers_reaped')
        return summary
    except BaseException as error:
        if run_status is not None:
            run_status.update(phase='failed', complete=False, error=repr(error),
                failed_utc=datetime.now(timezone.utc).isoformat())
        publish('failure_cleanup', error=repr(error))
        try:
            _stop((entry['process'] for entry in active.values()), stop_grace)
        except BlockWorkerError as cleanup_error:
            if run_status is not None:
                run_status['cleanup_error'] = str(cleanup_error)
            publish('failed_unreaped_workers', error=repr(error))
            raise cleanup_error from error
        else:
            for entry in active.values():
                entry['process'].close()
                _release_payload(entry['payload'])
            active.clear()
            publish('failed', error=repr(error))
        raise
    finally:
        for entry in active.values():
            _release_payload(entry['payload'])
        _release_payload(initialization_payload)

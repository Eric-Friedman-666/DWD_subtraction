"""One read-only snapshot: JSON, logs, /proc, filenames and nvidia-smi; NO HDF5."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        return dict(unavailable=str(error))


def kernel_faults(text, *, pci_bus_id, tracked_pids, started_epoch, boot_epoch):
    """Correlate bounded kernel diagnostics without touching a CUDA context."""
    bus = pci_bus_id.lower().split(':')[-2:]
    bus = ':'.join(bus).removesuffix('.0')
    faults = []
    for line in text.splitlines():
        match = re.search(r'^\[\s*([\d.]+)\].*Xid \(PCI:([^)]*)\):\s*(\d+),\s*pid=(\d+)', line)
        if not match or match[2].lower().split(':', 1)[-1] != bus:
            continue
        when = boot_epoch + float(match[1])
        if when < started_epoch:
            continue
        faults.append(dict(xid=int(match[3]), pid=int(match[4]),
            pipeline_pid=int(match[4]) in tracked_pids,
            utc=datetime.fromtimestamp(when, timezone.utc).isoformat(), text=line))
    return faults[-8:]


def inspect_run(run, *, query_gpu=True, query_kernel=True):
    """Never acquire a lock on any science, cache or sampler HDF5 file."""
    run = Path(run).resolve()
    launch = _read_json(run / 'launch.json')
    pid = int(launch['pid'])
    proc = Path('/proc') / str(pid)
    result = dict(run=str(run), controller_pid=pid, launch=launch)
    alive = False
    try:
        command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        status = (proc / 'status').read_text()
        state = re.search(r'^State:\s+(\S+)', status, re.M).group(1)
        alive = (state not in ('Z', 'X') and 'eric_subtraction_all.py' in command
            and os.readlink(proc / 'cwd') == launch['cwd'])
        if launch.get('controller_start_ticks') is not None:
            start_ticks = (proc / 'stat').read_text().rsplit(') ', 1)[1].split()[19]
            alive = alive and start_ticks == str(launch['controller_start_ticks'])
        result.update(controller_state=state, controller_command=command,
            controller_identity_matches=alive)
    except (OSError, AttributeError):
        pass
    result['controller_alive'] = alive
    workers = []
    for child in Path('/proc').iterdir():
        if not child.name.isdigit():
            continue
        try:
            text = (child / 'status').read_text()
            if int(re.search(r'^PPid:\s+(\d+)', text, re.M).group(1)) != pid:
                continue
            command = (child / 'cmdline').read_bytes().decode(errors='replace')
            state = re.search(r'^State:\s+(\S+)', text, re.M).group(1)
            if 'spawn_main' in command and 'resource_tracker' not in command and state not in ('Z', 'X'):
                workers.append(dict(pid=int(child.name), state=state))
        except (OSError, AttributeError):
            continue
    result.update(live_worker_count=len(workers), worker_states=dict(Counter(w['state'] for w in workers)),
        live_worker_pids=[w['pid'] for w in workers])
    status = _read_json(run / 'subtraction.status.json')
    cache_status = _read_json(run / 'subtraction.checkpoints/reference_waveforms.status.json')
    result['status'] = status if status.get('controller_pid') == pid else dict(unavailable='No current-controller snapshot')
    result['reference_cache_status'] = (cache_status if cache_status.get('controller_pid') == pid
        else dict(unavailable='No current-controller reference-cache snapshot'))
    log = run / 'controller.log'
    try:
        with log.open('rb') as saved:
            saved.seek(0, 2)
            saved.seek(max(0, saved.tell() - 65536))
            lines = [s for s in re.split(r'[\r\n]+', saved.read().decode(errors='replace')) if s.strip()]
        result['recent_log_lines'] = lines[-18:]
        result['recent_errors'] = [s for s in lines if re.search(
            r'Traceback|Error|Exception|out of memory|CUDA_ERROR', s, re.I)]
    except OSError as error:
        result['log_unavailable'] = str(error)
    status = result['status']
    current_round = status.get('current_round')
    result['current_round'] = current_round
    nonempty = status.get('nonempty_input_block_indices')
    total = status.get('selected_block_count')
    files = {}
    if current_round is not None:
        for entry in (run / 'subtraction.checkpoints').glob('round_*_block_*'):
            match = re.fullmatch(r'round_(\d+)_block_(\d+)', entry.name)
            if match and int(match[1]) == current_round and entry.is_dir():
                files[int(match[2])] = dict(result=(entry / 'result.pkl').is_file(),
                    backend=(entry / 'eryn_chain_00.h5').is_file())
    result['current_round_worker_results'] = sum(v['result'] for v in files.values())
    if nonempty is not None and total is not None and alive:
        nonempty = set(nonempty)
        empty = status['empty_input_block_count']
        completed = sum(v['result'] for i, v in files.items() if i in nonempty)
        fits = sum(v['result'] and v['backend'] for i, v in files.items() if i in nonempty)
        unfinished = [i for i, v in files.items() if i in nonempty and not v['result']]
        result['progress'] = dict(completed=completed, total=len(nonempty),
            percent=100 * completed / len(nonempty) if nonempty else None,
            accounting_definition='current_round_nonempty_input_banks_only',
            selected_block_count=total, selected_blocks_accounted_for=empty + completed,
            zero_work_skips=empty,
            nonempty_worker_results=completed, completed_rjmcmc_fits=fits,
            unfinished_nonempty_directories=unfinished,
            active_block_jobs=len(unfinished) if workers and status.get('phase') != 'failed' else 0)
        # A newly spawning worker may not have created its bank directory yet.
        if 'active_worker_jobs' in status:
            live = {worker['pid'] for worker in workers}
            active = []
            for job in status['active_worker_jobs']:
                index, worker_pid = job['block_index'], job['worker_pid']
                if (not alive or status.get('phase') == 'failed' or worker_pid not in live
                        or index not in nonempty or files.get(index, {}).get('result')):
                    continue
                job = dict(job)
                progress = _read_json(run / 'subtraction.checkpoints/worker_status' /
                    f'round_{current_round:02d}_block_{index:05d}.status.json')
                if (progress.get('controller_pid') == pid and progress.get('worker_pid') == worker_pid
                        and progress.get('attempt') == job['attempt']):
                    job['progress'] = progress
                    job['phase'] = progress['phase']
                active.append(job)
            result['active_worker_jobs'] = active
            result['progress'].update(active_block_jobs=len(active),
                waiting_nonempty_blocks=max(0, len(nonempty) - completed - len(active)))
    else:
        result['progress'] = dict(unavailable=('Controller identity unavailable or mismatched'
            if not alive else 'Round occupancy not published yet'),
            verified_worker_results=result['current_round_worker_results'])
    if query_gpu:
        snapshot = subprocess.run(['nvidia-smi', '-q', '-x'], capture_output=True, text=True, timeout=20)
        if snapshot.returncode:
            result['gpu_unavailable'] = snapshot.stderr.strip()
        else:
            tracked = {pid, *(w['pid'] for w in workers)}
            contexts = []
            for gpu in ET.fromstring(snapshot.stdout).findall('gpu'):
                uuid = gpu.findtext('uuid')
                processes = [{v.tag: v.text for v in p} for p in gpu.findall('processes/process_info')]
                contexts.extend(dict(gpu_uuid=uuid, **p) for p in processes if int(p['pid']) in tracked)
                if uuid == launch['gpu_uuid']:
                    result['gpu'] = dict(uuid=uuid,
                        pci_bus_id=gpu.findtext('pci/pci_bus_id', gpu.attrib.get('id', '')),
                        utilization={v.tag: v.text for v in gpu.find('utilization')},
                        memory={v.tag: v.text for v in gpu.find('fb_memory_usage')},
                        compute_context_count=len(processes))
            result['pipeline_cuda_context_count'] = len(contexts)
            result['wrong_gpu_contexts'] = [p for p in contexts if p['gpu_uuid'] != launch['gpu_uuid']]
    if query_kernel and query_gpu and result.get('gpu', {}).get('pci_bus_id'):
        try:
            kernel = subprocess.run(['dmesg'], capture_output=True, text=True, timeout=5)
            if kernel.returncode:
                result['kernel_unavailable'] = kernel.stderr.strip()
            else:
                boot = float(next(line.split()[1] for line in Path('/proc/stat').read_text().splitlines()
                    if line.startswith('btime ')))
                tracked = {pid, *result['live_worker_pids'],
                    *(event['worker_pid'] for event in status.get('worker_recovery_events', []))}
                result['gpu_kernel_faults'] = kernel_faults(kernel.stdout,
                    pci_bus_id=result['gpu']['pci_bus_id'], tracked_pids=tracked,
                    started_epoch=datetime.fromisoformat(launch['started_utc']).timestamp(), boot_epoch=boot)
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            result['kernel_unavailable'] = str(error)
    now = datetime.now(timezone.utc)
    result.update(as_of_utc=now.isoformat(), as_of_nz=now.astimezone(ZoneInfo('Pacific/Auckland')).isoformat())
    end = now if alive else None
    if status.get('failed_utc'):
        end = datetime.fromisoformat(status['failed_utc'])
    elif status.get('complete'):
        end = datetime.fromisoformat(status['updated_utc'])
    elif not alive and result.get('recent_errors'):
        end = datetime.fromtimestamp(log.stat().st_mtime, timezone.utc)
        result['failure_time_approximate'] = True
    result['elapsed_seconds'] = ((end - datetime.fromisoformat(launch['started_utc'])).total_seconds()
        if end is not None else None)
    result['terminal'] = not alive
    return result


if __name__ == '__main__':
    print(json.dumps(inspect_run(Path(sys.argv[1])), indent=2))

"""Compute-host-only CPU tests; fake GPU names never initialize CUDA."""
from functools import partial
from multiprocessing.util import Finalize
from pathlib import Path
import json
import os
import tempfile
import time
import unittest

from script.dwd_worker_supervisor import BlockWorkerError, run_block_jobs, worker_progress

IMPORTED_CUDA = os.environ.get('CUDA_VISIBLE_DEVICES')


def exit_marker(path, seconds):
    time.sleep(seconds)
    Path(path).write_text(str(time.monotonic()))


def assert_assigned_initializer(directory):
    # This module is imported by multiprocessing before its target starts.
    assert IMPORTED_CUDA == os.environ['DWD_ASSIGNED_GPU']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == os.environ['DWD_ASSIGNED_GPU']
    assert os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] == '0.045'
    gpu = os.environ['DWD_ASSIGNED_GPU']
    lock = Path(directory) / f'initializing_{gpu}'
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    try:
        time.sleep(.02)
    finally:
        os.close(descriptor)
        lock.unlink()
    worker_progress('f_screening', force=True)


def assigned_job(index, options, *, work, label):
    work = Path(work)
    gpu = os.environ['DWD_ASSIGNED_GPU']
    slot = int(os.environ['DWD_GPU_SLOT_INDEX'])
    global_slot = int(os.environ['DWD_WORKER_SLOT_INDEX'])
    # Slot lock persists through Python finalization; a replacement cannot use
    # it just because result.pkl exists.
    lock = work / f'slot_{gpu}_{slot}'
    if lock.exists():
        try:
            os.kill(int(lock.read_text()), 0)
        except ProcessLookupError:
            lock.unlink()  # A watchdog-reaped fixture, never a live occupant.
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.close(descriptor)
    lock.write_text(str(os.getpid()))
    Finalize(None, lock.unlink, exitpriority=-1)
    started = time.monotonic()
    marker = work / f'start_{label}_{index}_{os.getpid()}.json'
    marker.write_text(json.dumps(dict(index=index, gpu=gpu, slot=slot,
        global_slot=global_slot, pid=os.getpid(), started=started)))
    if options.get('gate'):
        deadline = time.monotonic() + 90.
        while len(list(work.glob(f'start_{label}_*.json'))) < options['gate']:
            if time.monotonic() > deadline:
                raise TimeoutError('All requested GPU slots did not become concurrently resident')
            worker_progress('sampling', force=True)
            time.sleep(.05)
    first = work / f'first_{index}'
    first_attempt = not first.exists()
    first.touch()
    if options.get('fail_once') and first_attempt:
        raise RuntimeError('Injected first-attempt GPU failure')
    if options.get('always_fail'):
        raise RuntimeError('Injected persistent GPU failure')
    until = time.monotonic() + options.get('duration', .05)
    while time.monotonic() < until:
        worker_progress('sampling', force=True)
        time.sleep(.02)
    bank = work / f'round_{label}_block_{index:05d}'
    bank.mkdir(exist_ok=True)
    temporary = bank / 'result.partial'
    temporary.write_text(json.dumps(dict(index=index, gpu=gpu, slot=slot,
        global_slot=global_slot, pid=os.getpid(), started=started, saved=time.monotonic())))
    temporary.replace(bank / 'result.pkl')
    if options.get('exit_delay'):
        Finalize(None, exit_marker, args=(work/f'exited_{index}_{os.getpid()}',
            options['exit_delay']), exitpriority=0)
    if options.get('exit_hang'):
        Finalize(None, time.sleep, args=(60.,), exitpriority=0)


class GpuAssignmentTests(unittest.TestCase):
    def run_jobs(self, directory, jobs, *, devices=('fake0', 'fake1'), capacity=2, **settings):
        return run_block_jobs(partial(assigned_job, work=Path(directory), label='01'), jobs,
            initializer=assert_assigned_initializer, initargs=(directory,),
            max_workers=len(devices)*capacity, gpu_devices=devices, workers_per_gpu=capacity,
            worker_gpu_memory_fraction=.045,
            work=directory, label='01', startup_ready_phase='f_screening',
            progress_timeout=settings.pop('progress_timeout', 30.),
            poll_interval=.02, stop_grace=.2, **settings)

    def rows(self, directory):
        return [json.loads(path.read_text()) for path in sorted(
            Path(directory).glob('round_01_block_*/result.pkl'))]

    def assert_reaped(self, rows):
        for row in rows:
            with self.assertRaises(ProcessLookupError):
                os.kill(row['pid'], 0)

    def test_four_gpus_sixteen_slots_unequal_rolling_refill_and_reap_barrier(self):
        with tempfile.TemporaryDirectory() as directory:
            original = {key: os.environ.get(key) for key in
                ('CUDA_VISIBLE_DEVICES', 'DWD_ASSIGNED_GPU', 'DWD_GPU_SLOT_INDEX', 'DWD_WORKER_SLOT_INDEX',
                 'XLA_PYTHON_CLIENT_MEM_FRACTION')}
            jobs = [(i, dict(gate=64 if i < 64 else None,
                duration=.35 if i % 7 == 0 else .02, exit_delay=.1)) for i in range(96)]
            summary = self.run_jobs(directory, jobs,
                devices=('fake0', 'fake1', 'fake2', 'fake3'), capacity=16, progress_timeout=100.)
            self.assertEqual(summary['maximum_active'], 64)
            self.assertEqual(summary['maximum_active_by_gpu'],
                {device: 16 for device in ('fake0', 'fake1', 'fake2', 'fake3')})
            self.assertEqual(summary['spawned_processes'], 96)
            self.assertEqual(summary['completed_blocks'], 96)
            self.assertEqual(summary['forced_cleanup_after_commit'], 0)
            rows = self.rows(directory)
            self.assertEqual(len({row['pid'] for row in rows}), 96)
            self.assertEqual(len(list(Path(directory).glob('exited_*'))), 96)
            self.assertEqual({key: os.environ.get(key) for key in original}, original)
            self.assert_reaped(rows)
            self.assertFalse(list(Path(directory).glob('slot_*')))

    def test_gpu_gates_are_independent_and_per_gpu_initialization_is_serial(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = self.run_jobs(directory, [(i, dict(gate=4)) for i in range(4)])
            self.assertEqual(summary['maximum_active_by_gpu'], {'fake0': 2, 'fake1': 2})
            self.assert_reaped(self.rows(directory))

    def test_fresh_retry_and_postcommit_cleanup_preserve_per_gpu_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = self.run_jobs(directory, [(0, dict(fail_once=True)),
                (1, dict(exit_hang=True)), (2, {}), (3, {})],
                devices=('fake0', 'fake1'), capacity=1, progress_timeout=1.5)
            self.assertEqual(summary['retries'], 1)
            self.assertEqual(summary['forced_cleanup_after_commit'], 1)
            self.assertEqual(summary['completed_blocks'], 4)
            self.assertEqual(summary['maximum_active_by_gpu'], {'fake0': 1, 'fake1': 1})
            self.assertEqual(len(list(Path(directory).glob('failed_attempts/round_01_block_00000/attempt_*'))), 1)
            self.assert_reaped(self.rows(directory))

    def test_retry_exhaustion_fails_closed_without_round_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            status = dict(phase='fitting_blocks', last_completed_round=0)
            with self.assertRaisesRegex(BlockWorkerError, 'failed after 3 attempts'):
                self.run_jobs(directory, [(0, dict(always_fail=True))],
                    devices=('fake0',), capacity=1, max_attempts=3,
                    status_file=Path(directory)/'status.json', run_status=status)
            self.assertEqual(status['phase'], 'failed')
            self.assertEqual(status['last_completed_round'], 0)
            self.assertEqual(status['active_worker_jobs'], [])
            self.assertEqual(len(list(Path(directory).glob('failed_attempts/round_01_block_00000/attempt_*'))), 3)

    def test_gpu_capacity_arguments_fail_closed(self):
        bad = [dict(gpu_devices=('x',)), dict(workers_per_gpu=1),
               dict(gpu_devices=('x', 'x'), workers_per_gpu=1),
               dict(gpu_devices=('x',), workers_per_gpu=0),
               dict(gpu_devices=('x',), workers_per_gpu=1)]
        with tempfile.TemporaryDirectory() as directory:
            for arguments in bad:
                with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                    run_block_jobs(assigned_job, [], max_workers=2,
                        work=directory, label='01', **arguments)


if __name__ == '__main__':
    if os.uname().sysname == 'Darwin':
        raise SystemExit('Run these tests on a server/OzSTAR compute host, not macOS')
    unittest.main()

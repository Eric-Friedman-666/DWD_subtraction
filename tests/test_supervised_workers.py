"""SERVER-ONLY fault injection for rolling, single-use worker supervision."""
from functools import partial
from multiprocessing import get_context
from multiprocessing.util import Finalize
from pathlib import Path
import json
import os
import signal
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from script.dwd_worker_supervisor import BlockWorkerError, run_block_jobs, worker_progress


def initializer_fails_once(directory):
    marker = Path(directory) / 'initializer_failed_once'
    if not marker.exists():
        marker.touch()
        raise RuntimeError('injected initializer failure')


def slow_deserialization(seconds):
    time.sleep(seconds)
    return 'decoded'


class SlowPayload:
    def __reduce__(self):
        return slow_deserialization, (2.,)


def payload_initializer(value, data):
    assert value == 'decoded' and len(data) == 2 * 1024 * 1024


def delayed_exit_marker(path, delay):
    time.sleep(delay)
    Path(path).touch()


def checkpoint_job(index, options, *, work, label):
    work = Path(work)
    bank = work / f'round_{label}_block_{index:05d}'
    bank.mkdir(parents=True, exist_ok=True)
    (work / f'start_{label}_{index}_{os.getpid()}').touch()
    marker = work / f'first_attempt_{label}_{index}'
    first = not marker.exists()
    marker.touch()
    mode = options.get('mode', 'normal')
    if options.get('startup_delay'):
        lock = work / f'initializing_{label}'
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            time.sleep(options['startup_delay'])
        finally:
            os.close(descriptor)
            lock.unlink()
        worker_progress('f_screening', force=True)
        # Parent polling may miss the readiness phase itself; the flag must stick.
        worker_progress('sampling', force=True)
    if mode == 'rolling_gate' and index == 0:
        deadline = time.monotonic() + 20
        while not list(work.glob(f'start_{label}_2_*')):
            if time.monotonic() >= deadline:
                raise TimeoutError('The next job did not start in a free rolling slot')
            time.sleep(.02)
    if mode == 'gate64':
        deadline = time.monotonic() + 30
        while len(list(work.glob(f'start_{label}_*'))) < 64:
            if time.monotonic() >= deadline:
                raise TimeoutError('All 64 configured slots were not launched')
            time.sleep(.02)
    if mode in {'cuda_once', 'exit_once', 'hang_once', 'ignore_term_once', 'stale_once'} and first:
        (bank / 'result.partial').write_text('incomplete result')
        (bank / 'eryn_chain_00.h5').write_text('dummy backend filename fixture, not an HDF5 file')
        if mode == 'cuda_once':
            raise RuntimeError('injected CUDA_ERROR_ILLEGAL_ADDRESS')
        if mode == 'exit_once':
            os._exit(17)
        if mode == 'ignore_term_once':
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if mode == 'stale_once':
            path = work / 'worker_status' / f'round_{label}_block_{index:05d}.status.json'
            until = time.monotonic() + 20
            while time.monotonic() < until:
                path.write_text(json.dumps(dict(attempt_token='wrong-attempt',
                    worker_pid=os.getpid(), sequence=999999, phase='sampling')))
                time.sleep(.03)
        else:
            time.sleep(20)
    if mode == 'healthy_progress':
        until = time.monotonic() + 2.5
        cycle = 0
        while time.monotonic() < until:
            worker_progress('sampling', force=True, cycle=cycle)
            cycle += 1
            time.sleep(.05)
    if mode == 'always_fail':
        raise RuntimeError('injected persistent failure')
    payload = dict(index=index, pid=os.getpid(), value=options.get('value', index * index),
        completed=time.monotonic())
    temporary = bank / 'result.partial'
    temporary.write_text(json.dumps(payload))
    temporary.replace(bank / 'result.pkl')
    if options.get('exit_delay'):
        Finalize(None, delayed_exit_marker,
            args=(work / f'exited_{label}_{index}_{os.getpid()}', options['exit_delay']),
            exitpriority=0)
    if mode == 'cleanup_hang':
        time.sleep(20)
    if mode == 'exit_hang':
        Finalize(None, time.sleep, args=(20.,), exitpriority=0)


class SupervisedWorkerTests(unittest.TestCase):
    def test_large_startup_payload_cannot_block_the_parent_watchdog(self):
        with tempfile.TemporaryDirectory() as directory:
            started = time.monotonic()
            with self.assertRaisesRegex(BlockWorkerError, 'No forward progress'):
                run_block_jobs(partial(checkpoint_job, work=Path(directory), label='01'),
                    [(i, {}) for i in range(4)], max_workers=4, work=directory, label='01',
                    initializer=payload_initializer, initargs=(SlowPayload(), bytes(2 * 1024 * 1024)),
                    progress_timeout=.6, max_attempts=1, poll_interval=.02, stop_grace=.1)
            self.assertLess(time.monotonic() - started, 4., 'Large spawn arguments blocked timeout enforcement')

    def test_initializer_failure_retries_in_a_new_process(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = self.run_jobs(directory, [(0, {})], workers=1,
                initializer=initializer_fails_once, initargs=(directory,), max_attempts=2)
            self.assertEqual(summary['completed_blocks'], 1)
            self.assertEqual(summary['retries'], 1)
            self.assert_reaped(self.results(directory))

    def test_unreapable_process_fails_closed_and_publishes_cleanup_error(self):
        class Unreapable:
            pid = 123456789
            terminations = 0
            kills = 0
            def start(self): pass
            def is_alive(self): return True
            def terminate(self): self.terminations += 1
            def kill(self): self.kills += 1
            def join(self, **kwargs): pass
        process = Unreapable()
        with tempfile.TemporaryDirectory() as directory, patch(
                'script.dwd_worker_supervisor.get_context',
                return_value=SimpleNamespace(Process=lambda **kwargs: process)):
            status = dict(phase='fitting_blocks', last_completed_round=0)
            with self.assertRaisesRegex(BlockWorkerError, 'could not be reaped'):
                run_block_jobs(checkpoint_job, [(0, {})], max_workers=1, work=directory,
                    label='01', progress_timeout=.03, poll_interval=.01, stop_grace=.01,
                    status_file=Path(directory) / 'status.json', run_status=status)
            self.assertGreater(process.terminations, 0)
            self.assertGreater(process.kills, 0)
            self.assertEqual(status['phase'], 'failed')
            self.assertIn('cleanup_error', status)
            self.assertEqual(status['last_completed_round'], 0)

    def run_jobs(self, directory, jobs, *, workers=2, label='01', **settings):
        return run_block_jobs(partial(checkpoint_job, work=Path(directory), label=label), jobs,
            max_workers=workers, work=directory, label=label,
            progress_timeout=settings.pop('progress_timeout', 10.),
            poll_interval=.02, stop_grace=.2, **settings)

    def results(self, directory, label='01'):
        return [json.loads(path.read_text()) for path in sorted(
            Path(directory).glob(f'round_{label}_block_*/result.pkl'))]

    def assert_reaped(self, results):
        for row in results:
            with self.assertRaises(ProcessLookupError):
                os.kill(row['pid'], 0)

    def test_rolling_refill_single_use_pids_and_reaped_round_barrier(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = self.run_jobs(directory, [(i, {'mode': 'rolling_gate'}) for i in range(6)])
            rows = self.results(directory)
            self.assertEqual([row['index'] for row in rows], list(range(6)))
            self.assertEqual(len({row['pid'] for row in rows}), 6)
            self.assertEqual(summary['spawned_processes'], 6)
            self.assertEqual(summary['retries'], 0)
            self.assertGreater(rows[0]['completed'], rows[2]['completed'])
            self.assert_reaped(rows)

    def test_64_slots_recycle_768_jobs_in_three_rounds_with_slow_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            all_pids = set()
            for number in range(1, 4):
                label = f'{number:02d}'
                summary = self.run_jobs(directory, [(i, dict(mode='gate64', value=number,
                    exit_delay=.5, startup_delay=.01)) for i in range(256)],
                    workers=64, label=label, startup_ready_phase='f_screening', progress_timeout=45.)
                rows = self.results(directory, label)
                self.assertEqual(summary['maximum_active'], 64)
                self.assertEqual(summary['spawned_processes'], 256)
                self.assertEqual(summary['completed_blocks'], 256)
                self.assertEqual(summary['forced_cleanup_after_commit'], 0)
                self.assertEqual(summary['retries'], 0)
                self.assertEqual(len({row['pid'] for row in rows}), 256)
                self.assertTrue(all(row['value'] == number for row in rows))
                self.assertEqual(len(list(Path(directory).glob(f'exited_{label}_*'))), 256,
                    'A saved worker was killed before normal teardown finished')
                self.assert_reaped(rows)
                all_pids.update(row['pid'] for row in rows)
            self.assertEqual(len(all_pids), 768)

    def test_cuda_exception_and_abrupt_exit_retry_and_preserve_failed_files(self):
        for mode in ['cuda_once', 'exit_once']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                summary = self.run_jobs(directory, [(0, {'mode': mode}), (1, {})], max_attempts=2)
                self.assertEqual(summary['completed_blocks'], 2)
                self.assertEqual(summary['retries'], 1)
                archives = list((Path(directory) / 'failed_attempts').glob('round_01_block_00000/attempt_*'))
                self.assertEqual(len(archives), 1)
                self.assertEqual((archives[0] / 'result.partial').read_text(), 'incomplete result')
                self.assertTrue((archives[0] / 'eryn_chain_00.h5').is_file())
                self.assertFalse((Path(directory) / 'round_01_block_00000/eryn_chain_00.h5').exists())
                self.assert_reaped(self.results(directory))

    def test_silent_hang_is_killed_and_retried_without_blocking_other_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            status = {}
            summary = run_block_jobs(partial(checkpoint_job, work=Path(directory), label='01'),
                [(0, {'mode': 'hang_once'}), (1, {})], max_workers=2, work=directory, label='01',
                progress_timeout=1.5, max_attempts=2, poll_interval=.02, stop_grace=.2,
                status_file=Path(directory) / 'controller.status.json', run_status=status)
            self.assertEqual(summary['retries'], 1)
            self.assertIn('No forward progress', status['worker_recovery_events'][0]['reason'])
            self.assertEqual(status['active_worker_jobs'], [])
            self.assert_reaped(self.results(directory))

    def test_sigterm_ignoring_worker_is_force_killed_then_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = run_block_jobs(partial(checkpoint_job, work=Path(directory), label='01'),
                [(0, {'mode': 'ignore_term_once'})], max_workers=1, work=directory, label='01',
                progress_timeout=1.5, max_attempts=2, poll_interval=.02, stop_grace=.15)
            self.assertEqual(summary['retries'], 1)
            self.assert_reaped(self.results(directory))

    def test_real_progress_allows_work_longer_than_the_stall_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = run_block_jobs(partial(checkpoint_job, work=Path(directory), label='01'),
                [(0, {'mode': 'healthy_progress'})], max_workers=1, work=directory, label='01',
                progress_timeout=1.5, max_attempts=1, poll_interval=.02, stop_grace=.2)
            self.assertEqual(summary['retries'], 0)
            self.assert_reaped(self.results(directory))

    def test_fresh_json_from_the_wrong_attempt_cannot_hide_a_hang(self):
        with tempfile.TemporaryDirectory() as directory:
            summary = run_block_jobs(partial(checkpoint_job, work=Path(directory), label='01'),
                [(0, {'mode': 'stale_once'})], max_workers=1, work=directory, label='01',
                progress_timeout=1.5, max_attempts=2, poll_interval=.02, stop_grace=.2)
            self.assertEqual(summary['retries'], 1)

    def test_committed_result_survives_a_hang_during_worker_cleanup(self):
        for mode in ('cleanup_hang', 'exit_hang'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                summary = run_block_jobs(partial(checkpoint_job, work=Path(directory), label='01'),
                    [(0, {'mode': mode})], max_workers=1, work=directory, label='01',
                    progress_timeout=1.5, max_attempts=1, poll_interval=.02, stop_grace=.2)
                self.assertEqual(summary['completed_blocks'], 1)
                self.assertEqual(summary['retries'], 0)
                self.assertEqual(summary['forced_cleanup_after_commit'], 1)
                self.assert_reaped(self.results(directory))

    def test_retry_exhaustion_fails_round_and_leaves_unrelated_process_alive(self):
        unrelated = get_context('spawn').Process(target=time.sleep, args=(20,))
        unrelated.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                status = dict(phase='fitting_blocks', last_completed_round=0)
                with self.assertRaisesRegex(BlockWorkerError, 'failed after 2 attempts'):
                    self.run_jobs(directory, [(0, {'mode': 'always_fail'})], max_attempts=2,
                        status_file=Path(directory) / 'controller.status.json', run_status=status)
                self.assertTrue(unrelated.is_alive())
                self.assertEqual(status['phase'], 'failed')
                self.assertEqual(status['last_completed_round'], 0)
                self.assertEqual(status['active_worker_jobs'], [])
                self.assertIn('failed_utc', status)
                self.assertEqual(self.results(directory), [])
        finally:
            unrelated.terminate()
            unrelated.join(5)
            unrelated.close()

    def test_empty_round_launches_nothing_and_checkpoint_resume_does_not_refit(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.run_jobs(directory, [])['spawned_processes'], 0)
            self.run_jobs(directory, [(0, {})])
            first = self.results(directory)
            summary = self.run_jobs(directory, [(0, {'mode': 'always_fail'})])
            self.assertEqual(summary['spawned_processes'], 0)
            self.assertEqual(self.results(directory), first)


if __name__ == '__main__':
    unittest.main()

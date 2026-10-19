"""Server-only lock-safe reporting fixtures; never inspect a running science HDF5."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from script.dwd_run_status import write_run_status
from script.monitor_dwd_run import inspect_run, kernel_faults


class ReadOnlyRunMonitorTests(unittest.TestCase):
    def test_worker_snapshot_can_name_its_owning_controller(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'worker.json'
            write_run_status(path, controller_pid=123, worker_pid=456)
            self.assertEqual(json.loads(path.read_text())['controller_pid'], 123)

    @unittest.skipUnless(Path('/proc').is_dir(), 'Linux /proc compute-host fixture required')
    def test_failure_clock_is_frozen_and_partial_round_not_finalized(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'launch.json').write_text(json.dumps(dict(pid=os.getpid(), cwd=os.getcwd(),
                started_utc='2026-10-16T11:00:00+00:00', gpu_uuid='test')))
            write_run_status(run / 'subtraction.status.json', current_round=2,
                phase='failed', failed_utc='2026-10-16T13:00:00+00:00', complete=False,
                last_completed_round=1, finalized_identified_dwd_count=4,
                finalized_round_counts={'01': 4})
            result = inspect_run(run, query_gpu=False)
            self.assertEqual(result['elapsed_seconds'], 7200.)
            self.assertEqual(result['status']['last_completed_round'], 1)
            self.assertEqual(result['status']['finalized_identified_dwd_count'], 4)

    def test_kernel_fault_requires_assigned_gpu_and_current_run_time(self):
        text = '\n'.join([
            '[10.0] NVRM: Xid (PCI:0000:e3:00): 31, pid=123, MMU Fault ACCESS_TYPE_VIRT_READ',
            '[40.0] NVRM: Xid (PCI:0000:e3:00): 31, pid=456, MMU Fault ACCESS_TYPE_VIRT_READ',
            '[50.0] NVRM: Xid (PCI:0000:a2:00): 31, pid=456, MMU Fault',
            '[60.0] NVRM: Xid (PCI:0000:e3:00): 43, pid=789, channel stopped'])
        faults = kernel_faults(text, pci_bus_id='00000000:E3:00.0', tracked_pids={456},
            started_epoch=1020., boot_epoch=1000.)
        self.assertEqual([event['xid'] for event in faults], [31, 43])
        self.assertEqual([event['pipeline_pid'] for event in faults], [True, False])
        self.assertNotIn('OOM', faults[0]['text'])

    @unittest.skipUnless(Path('/proc').is_dir(), 'Linux /proc compute-host fixture required')
    def test_spawning_jobs_are_counted_without_a_checkpoint_or_cuda_context(self):
        with tempfile.TemporaryDirectory() as directory:
            run, pid = Path(directory), os.getpid()
            (run / 'launch.json').write_text(json.dumps(dict(pid=pid, cwd=os.getcwd(),
                started_utc=datetime.now(timezone.utc).isoformat(), gpu_uuid='test')))
            write_run_status(run / 'subtraction.status.json', current_round=1, complete=False,
                phase='fitting_blocks', selected_block_count=4, nonempty_input_block_indices=[1, 2],
                empty_input_block_count=2, active_worker_jobs=[dict(block_index=1, worker_pid=88888,
                    attempt=1, phase='spawning')])
            child = run / '88888'
            child.mkdir()
            (child / 'status').write_text(f'State:\tS\nPPid:\t{pid}\n')
            (child / 'cmdline').write_bytes(b'python3\0spawn_main\0')
            original_read, original_iterdir = Path.read_bytes, Path.iterdir
            def command(path):
                return b'python3\0eric_subtraction_all.py\0' if path == Path(f'/proc/{pid}/cmdline') else original_read(path)
            def listing(path):
                return iter([child]) if path == Path('/proc') else original_iterdir(path)
            with patch.object(Path, 'read_bytes', command), patch.object(Path, 'iterdir', listing), \
                    patch('os.readlink', return_value=os.getcwd()):
                result = inspect_run(run, query_gpu=False)
            self.assertTrue(result['controller_alive'])
            self.assertEqual(result['progress']['active_block_jobs'], 1)
            self.assertEqual(result['progress']['waiting_nonempty_blocks'], 1)
            self.assertEqual(result['progress']['unfinished_nonempty_directories'], [])

    @unittest.skipUnless(Path('/proc').is_dir(), 'Linux /proc compute-host fixture required')
    def test_live_hdf5_is_never_opened_and_skips_fits_and_active_jobs_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            work = run / 'subtraction.checkpoints'
            work.mkdir()
            launch = dict(pid=os.getpid(), cwd=os.getcwd(),
                started_utc=datetime.now(timezone.utc).isoformat(), gpu_uuid='test-gpu',
                block_workers=64, relative_snr_threshold=7.)
            (run / 'launch.json').write_text(json.dumps(launch))
            (run / 'controller.log').write_text('Round 01\n')
            write_run_status(run / 'subtraction.status.json', current_round=1, complete=False,
                phase='fitting_blocks', selected_block_count=5, nonempty_input_block_indices=[0, 1, 2],
                empty_input_block_count=2, finalized_identified_dwd_count=0)
            for index in (0, 1, 2):
                block = work / f'round_01_block_{index:05d}'
                block.mkdir()
                if index < 2:
                    (block / 'result.pkl').touch()
                if index in (0, 2):
                    (block / 'eryn_chain_00.h5').touch()
            # Even a stray empty-bank checkpoint must never inflate the bar.
            empty = work / 'round_01_block_00004'
            empty.mkdir()
            (empty / 'result.pkl').touch()
            (empty / 'eryn_chain_00.h5').touch()
            original = Path.open
            original_read = Path.read_bytes
            def guarded(path, *args, **kwargs):
                if path.suffix in ('.h5', '.pkl'):
                    raise AssertionError(f'Monitor opened scientific file: {path}')
                return original(path, *args, **kwargs)
            def command(path):
                return b'python3\0eric_subtraction_all.py\0' if path == Path(f'/proc/{os.getpid()}/cmdline') else original_read(path)
            with patch.object(Path, 'open', guarded), patch.object(Path, 'read_bytes', command):
                result = inspect_run(run, query_gpu=False)
            self.assertEqual(result['progress']['completed'], 2)
            self.assertEqual(result['progress']['total'], 3)
            self.assertAlmostEqual(result['progress']['percent'], 200 / 3)
            self.assertEqual(result['progress']['selected_blocks_accounted_for'], 4)
            self.assertEqual(result['progress']['zero_work_skips'], 2)
            self.assertEqual(result['progress']['nonempty_worker_results'], 2)
            self.assertEqual(result['progress']['completed_rjmcmc_fits'], 1)
            self.assertEqual(result['progress']['active_block_jobs'], 0)
            self.assertEqual(result['progress']['unfinished_nonempty_directories'], [2])

    @unittest.skipUnless(Path('/proc').is_dir(), 'Linux /proc compute-host fixture required')
    def test_terminal_empty_round_has_no_fabricated_worker_percentage(self):
        with tempfile.TemporaryDirectory() as directory:
            run, pid = Path(directory), os.getpid()
            (run / 'launch.json').write_text(json.dumps(dict(pid=pid, cwd=os.getcwd(),
                started_utc=datetime.now(timezone.utc).isoformat(), gpu_uuid='test')))
            write_run_status(run / 'subtraction.status.json', current_round=2, complete=False,
                phase='finalizing_round', selected_block_count=4990,
                nonempty_input_block_indices=[], empty_input_block_count=4990)
            original_read = Path.read_bytes
            def command(path):
                return b'python3\0eric_subtraction_all.py\0' if path == Path(f'/proc/{pid}/cmdline') else original_read(path)
            with patch.object(Path, 'read_bytes', command):
                result = inspect_run(run, query_gpu=False)
            self.assertEqual(result['progress']['completed'], 0)
            self.assertEqual(result['progress']['total'], 0)
            self.assertIsNone(result['progress']['percent'])
            self.assertEqual(result['progress']['zero_work_skips'], 4990)

    @unittest.skipUnless(Path('/proc').is_dir(), 'Linux /proc compute-host fixture required')
    def test_wrong_controller_cannot_publish_overall_percentage(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'launch.json').write_text(json.dumps(dict(pid=os.getpid(), cwd=directory,
                started_utc=datetime.now(timezone.utc).isoformat(), gpu_uuid='test')))
            write_run_status(run / 'subtraction.status.json', current_round=1, complete=False,
                phase='fitting_blocks', selected_block_count=4990,
                nonempty_input_block_indices=[0], empty_input_block_count=4989)
            result = inspect_run(run, query_gpu=False)
            self.assertIn('unavailable', result['progress'])
            self.assertNotIn('percent', result['progress'])

    @unittest.skipUnless(Path('/proc').is_dir(), 'Linux /proc compute-host fixture required')
    def test_initialization_has_no_fabricated_round_percentage(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / 'launch.json').write_text(json.dumps(dict(pid=999999999, cwd=directory,
                started_utc=datetime.now(timezone.utc).isoformat(), gpu_uuid='test')))
            (run / 'controller.log').write_text('Reference cache preparing\n')
            result = inspect_run(run, query_gpu=False)
            self.assertIsNone(result['current_round'])
            self.assertIn('unavailable', result['progress'])
            self.assertIsNone(result['elapsed_seconds'])


if __name__ == '__main__':
    unittest.main()

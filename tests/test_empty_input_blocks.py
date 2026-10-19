"""Controller-side empty-input skipping, rolling spawn and checkpoint resume."""
import ast
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from copy import deepcopy
from itertools import islice
from multiprocessing import get_context
from pathlib import Path
import os
import pickle
import tempfile
import time
import unittest
from unittest.mock import patch

import h5py
import numpy as np

import test_subtraction_all as fixtures

ROOT = Path(__file__).resolve().parents[1]


def checkpoint_nonempty_block(index, table, directory):
    directory = Path(directory)
    assert len(table)
    (directory / f'start_{index}').touch()
    if index == 0:
        deadline = time.monotonic() + 20
        while not (directory / 'start_4').exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Rolling refill did not start block four')
            time.sleep(.01)
    result = dict(block=index, pid=os.getpid(), value=table.copy())
    with (directory / f'block_{index}.pkl').open('wb') as saved:
        pickle.dump(result, saved)
    return result


class EmptyInputBlockTests(unittest.TestCase):
    run_case = fixtures.GlobalRoundTests.run_case

    def test_production_job_filter_skips_empty_banks_with_real_rolling_spawn(self):
        tree = ast.parse((ROOT / 'eric_subtraction_all.py').read_text())
        assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'block_jobs' for target in node.targets))
        indices = np.arange(8)
        tables = [np.full((int(index % 2 == 0), 11), index) for index in indices]
        jobs = eval(compile(ast.Expression(deepcopy(assignment.value)), '<block_jobs>', 'eval'),
            dict(selected_block_indices=indices, block_template_tables=tables))
        with tempfile.TemporaryDirectory() as directory:
            results = []
            with ProcessPoolExecutor(max_workers=2, mp_context=get_context('spawn'),
                                     max_tasks_per_child=1) as executor:
                pending = {executor.submit(checkpoint_nonempty_block, index, table, directory)
                           for index, table in islice(jobs, 2)}
                while pending:
                    done, pending = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                    self.assertTrue(done, 'Rolling spawn stalled')
                    for future in done:
                        results.append(future.result())
                        job = next(jobs, None)
                        if job is not None:
                            pending.add(executor.submit(checkpoint_nonempty_block, *job, directory))
            self.assertEqual(sorted(result['block'] for result in results), [0, 2, 4, 6])
            self.assertEqual(len({result['pid'] for result in results}), 4)
            self.assertEqual(sorted(path.name for path in Path(directory).glob('start_*')),
                             ['start_0', 'start_2', 'start_4', 'start_6'])
            for result in results:
                with (Path(directory) / f"block_{result['block']}.pkl").open('rb') as saved:
                    np.testing.assert_array_equal(pickle.load(saved)['value'], tables[result['block']])
                with self.assertRaises(ProcessLookupError):
                    os.kill(result['pid'], 0)

    def test_all_empty_inputs_submit_no_jobs_and_preserve_observation(self):
        submissions = []

        class RecordingExecutor(fixtures.IsolatedWorkerExecutor):
            def submit(self, function, index, table):
                submissions.append((index, table.copy()))
                return super().submit(function, index, table)

        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory,
                catalogue_relative_snrs=(1., .9), executor_factory=RecordingExecutor)
            self.assertEqual(submissions, [])
            self.assertEqual(state['search'], [])
            self.assertEqual(state['sampler'], [])
            self.assertEqual(result['rounds'], 1)
            self.assertFalse(state['waveform'] or state['tdi'])
            self.assertFalse(list(output.with_suffix('.checkpoints').glob('round_*_block_*')))
            with h5py.File(output) as saved:
                np.testing.assert_array_equal(saved['config/selected_block_indices'][:], [0, 1, 2, 3])
                for name in ('selected_count', 'recovered_count', 'identified_count', 'ghost_count', 'accepted_count'):
                    np.testing.assert_array_equal(saved[f'rounds/01/{name}'][:], np.zeros(4))
                np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_td'][:], original)
                self.assertTrue(saved.attrs['complete'])

    def test_sparse_inputs_keep_zero_counts_and_do_not_read_missing_checkpoints(self):
        submissions = []

        class RecordingExecutor(fixtures.IsolatedWorkerExecutor):
            def submit(self, function, index, table):
                self_test.assertGreater(len(table), 0)
                submissions.append((function.keywords['label'], int(index)))
                return super().submit(function, index, table)

        self_test = self
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory, bins=(10, 36),
                executor_factory=RecordingExecutor)
            self.assertEqual(submissions, [('01', 0), ('01', 2)])
            self.assertEqual(result['rounds'], 2)
            self.assertEqual(len(state['tdi']), 1)
            self.assertEqual(len(state['sampler']), 2)
            directories = sorted(path.name for path in output.with_suffix('.checkpoints').glob('round_*_block_*'))
            self.assertEqual(directories, ['round_01_block_00000', 'round_01_block_00002'])
            with h5py.File(output) as saved:
                for name in ('selected_count', 'recovered_count', 'identified_count', 'accepted_count'):
                    np.testing.assert_array_equal(saved[f'rounds/01/{name}'][:], [1, 0, 1, 0])
                    np.testing.assert_array_equal(saved[f'rounds/02/{name}'][:], np.zeros(4))
                np.testing.assert_array_equal(saved['final_subtraction/dwd_id'][:], [1, 2])
                np.testing.assert_allclose(saved['final_subtraction/signal_xyz_td'][:], original, atol=1.e-12)

    def test_nonempty_input_that_fails_f_still_runs_screening(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory, bins=(), catalogue_bins=(10,))
            self.assertEqual(len(state['search']), 1)
            self.assertEqual(len(state['search'][0][0]), 1)
            self.assertEqual(state['sampler'], [])
            self.assertFalse(state['tdi'])
            self.assertEqual(result['rounds'], 1)
            result_path = output.with_suffix('.checkpoints') / 'round_01_block_00000/result.pkl'
            with result_path.open('rb') as saved:
                recovered, filtered, bins, posterior = pickle.load(saved)
            self.assertEqual(recovered.shape, (0, 8))
            self.assertEqual(filtered.shape, (0, 11))
            self.assertGreater(len(bins), 0)
            self.assertIsNone(posterior)
            with h5py.File(output) as saved:
                np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_td'][:], original)

    def test_completed_nonempty_checkpoint_resumes_without_refitting(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'result.h5'

            def interrupt(templates, state):
                if int(templates[0, 0]) == 1:
                    completed = output.with_suffix('.checkpoints') / 'round_01_block_00002/result.pkl'
                    deadline = time.monotonic() + 10
                    while not completed.exists() and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertTrue(completed.exists())
                    raise RuntimeError('deliberate unfinished sparse round')

            with self.assertRaisesRegex(RuntimeError, 'deliberate unfinished sparse round'):
                self.run_case(directory, bins=(10, 36), worker_hook=interrupt,
                    executor_factory=fixtures.IsolatedWorkerExecutor)
            _, state, _, output = self.run_case(directory, bins=(10, 36))
            self.assertEqual(len(state['sampler']), 1)
            self.assertEqual(len(state['tdi']), 1)
            with h5py.File(output) as saved:
                np.testing.assert_array_equal(saved['final_subtraction/dwd_id'][:], [1, 2])

    def test_scientific_outputs_match_pre_skip_controller(self):
        reference = os.environ.get('DWD_EMPTY_SKIP_REFERENCE')
        if not reference:
            self.skipTest('Set DWD_EMPTY_SKIP_REFERENCE to the saved pre-change source directory')
        for settings in (dict(bins=(10, 36)), dict(catalogue_relative_snrs=(1., .9)),
                         dict(bins=(), catalogue_bins=(10,)), dict(wait_for_round=(23,))):
            with self.subTest(settings=settings), tempfile.TemporaryDirectory() as before, \
                 tempfile.TemporaryDirectory() as after:
                with patch.object(fixtures, 'ROOT', Path(reference)):
                    _, _, _, expected = self.run_case(before, **settings)
                _, _, _, actual = self.run_case(after, **settings)
                with h5py.File(expected) as old, h5py.File(actual) as new:
                    old_paths, new_paths = [], []
                    old.visititems(lambda name, obj: old_paths.append(name) if isinstance(obj, h5py.Dataset) else None)
                    new.visititems(lambda name, obj: new_paths.append(name) if isinstance(obj, h5py.Dataset) else None)
                    self.assertEqual(old_paths, new_paths)
                    for name in old_paths:
                        if name.endswith('/elapsed_seconds'):
                            continue
                        np.testing.assert_array_equal(new[name][()], old[name][()], err_msg=name)
                    self.assertEqual(dict(old.attrs), dict(new.attrs))


if __name__ == '__main__':
    unittest.main()

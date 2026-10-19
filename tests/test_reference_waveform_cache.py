"""Persistent reference caching; mock coordination and real GPU replay are separate."""
import ast
from contextlib import contextmanager
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from time import monotonic
import unittest
from unittest.mock import patch

import h5py
import numpy as np

import test_subtraction_all as fixtures
from script.dwd_reference_cache import _open_cache, precompute_reference_waveforms
from script.dwd_run_status import write_run_status


class Response:
    calls = []
    files_open = 0

    def __init__(self, covariance, data, times, geometry, frequency, duration, rate):
        if self.files_open:
            raise AssertionError('Waveform calculation inside an HDF5 context')
        np.testing.assert_array_equal(covariance,
            np.broadcast_to(np.eye(3), (len(frequency), 3, 3)))
        np.testing.assert_array_equal(data, 0.)
        self.frequency = frequency

    def build(self, points):
        if self.files_open:
            raise AssertionError('Waveform calculation inside an HDF5 context')
        self.calls.append(points.copy())
        return np.broadcast_to(points[:, :, None, None]
            + (1 + 2j) * self.frequency[None, None, None, :],
            (len(points), 4, 3, len(self.frequency))).copy()


def coefficients(plus, cross, psi, phi):
    return np.stack((plus, cross, psi, phi), axis=-1)


def reconstruct(amplitudes, bases):
    return np.einsum('sm,smck->sck', amplitudes, bases)


class ReferenceWaveformCacheTests(unittest.TestCase):
    def setUp(self):
        Response.calls = []
        Response.files_open = 0
        self.blocks = np.array([[.1, .2], [.2, .3], [.3, .4]])
        self.frequency = np.arange(0., .51, .01)
        self.tables = [np.array([[91., 1.e-22, .15, 1.e-17, .4, .7, .2, -.1, .3, 20., 8.],
                                [17., 2.e-22, .16, 2.e-17, .2, .9, .5, -.2, .4, 30., 9.]]),
                       np.empty((0, 11)),
                       np.array([[82., 3.e-22, .35, 3.e-17, .3, 1.1, .8, -.3, .5, 40., 10.]])]

    def prepare(self, path, **overrides):
        settings = dict(cache_file=path, selected_blocks=np.arange(3),
            template_tables=self.tables, frequency_blocks=self.blocks, block_width=.1,
            frequency=self.frequency, duration=100., rate=1., response_times=np.array([0., 1.]),
            response_geometry=np.array([2., 3.]), response_type=Response,
            amplitude_coordinates=coefficients, reconstruct_waveform=reconstruct)
        settings.update(overrides)
        return precompute_reference_waveforms(**settings)

    def test_cold_generation_and_warm_reload_are_exact_without_gpu_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'references.h5'
            cold = self.prepare(path)
            self.assertEqual(len(Response.calls), 2)
            self.assertEqual(set(cold), {0, 2})
            np.testing.assert_array_equal(cold[0]['dwd_id'], [91, 17])
            manifest = path.read_bytes()
            with patch.object(Response, '__init__', side_effect=AssertionError('GPU initialized on cache hit')):
                warm = self.prepare(path)
            self.assertEqual(len(Response.calls), 2)
            self.assertEqual(path.read_bytes(), manifest)
            for block in cold:
                for name in ('dwd_id', 'retained_bins', 'waveforms'):
                    np.testing.assert_array_equal(cold[block][name], warm[block][name])
                self.assertEqual(cold[block]['waveforms'].dtype, np.dtype('complex128'))

    def test_cached_values_equal_direct_physical_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = self.prepare(Path(directory) / 'references.h5')
            for block, table in enumerate(self.tables):
                if not len(table):
                    continue
                bins = cache[block]['retained_bins']
                response = Response(np.broadcast_to(np.eye(3), (len(bins), 3, 3)),
                    np.zeros((3, len(bins))), None, None, self.frequency[bins], 100., 1.)
                cosine = np.cos(table[:, 5])
                amplitude = coefficients(table[:, 1] * (1 + cosine**2),
                    2 * table[:, 1] * cosine, table[:, 8], table[:, 4]) / 1.e-22
                direct = reconstruct(amplitude, response.build(table[:, [2, 3, 6, 7]]))
                np.testing.assert_array_equal(cache[block]['waveforms'], direct)

    def test_empty_input_banks_create_no_cache_or_gpu_response(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'references.h5'
            self.assertEqual(self.prepare(path, template_tables=[np.empty((0, 11))] * 3), {})
            self.assertFalse(path.exists())
            self.assertEqual(Response.calls, [])

    def test_progress_bar_counts_only_nonempty_input_banks(self):
        bars = []
        def progress(iterable, **kwargs):
            values = list(iterable)
            bars.append((values, kwargs))
            return iter(values)
        with tempfile.TemporaryDirectory() as directory, \
             patch('script.dwd_reference_cache.tqdm', progress):
            self.prepare(Path(directory) / 'references.h5')
        self.assertEqual([int(index) for index, table in bars[0][0]], [0, 2])
        self.assertEqual(bars[0][1]['total'], 2)

    def test_changed_physical_parameters_or_grid_invalidate_persistent_entries(self):
        for change in ('amplitude', 'frequency_grid', 'duration', 'response_geometry'):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'references.h5'
                self.prepare(path)
                original_calls = len(Response.calls)
                tables = [table.copy() for table in self.tables]
                overrides = {}
                if change == 'amplitude':
                    tables[0][0, 1] *= 2
                    overrides['template_tables'] = tables
                elif change == 'frequency_grid':
                    overrides['frequency'] = self.frequency + .0001
                elif change == 'duration':
                    overrides['duration'] = 200.
                else:
                    overrides['response_geometry'] = np.array([2., 4.])
                self.prepare(path, **overrides)
                self.assertEqual(len(Response.calls), original_calls + 2)
                with h5py.File(path, 'r') as saved:
                    self.assertEqual(len(saved), 2, 'Old keyed cache must not be overwritten')

    def test_incomplete_block_is_rebuilt_without_replacing_complete_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'references.h5'
            expected = self.prepare(path)
            with h5py.File(path, 'a') as saved:
                key = next(iter(saved))
                saved[f'{key}/block_00000'].attrs['complete'] = False
                saved[f'{key}/block_00000/waveforms'][:] = np.nan
            previous_calls = len(Response.calls)
            actual = self.prepare(path)
            self.assertEqual(len(Response.calls), previous_calls + 1)
            for block in expected:
                np.testing.assert_array_equal(actual[block]['waveforms'], expected[block]['waveforms'])

    def test_changed_waveform_implementation_uses_a_different_cache_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'references.h5'
            self.prepare(path)
            previous_calls = len(Response.calls)
            original_read = Path.read_bytes
            def changed_read(source_path):
                value = original_read(source_path)
                return value + b'\n# simulated implementation revision\n' if source_path.name == 'dwd_reference_cache.py' else value
            with patch.object(Path, 'read_bytes', changed_read):
                self.prepare(path)
            self.assertEqual(len(Response.calls), previous_calls + 2)
            with h5py.File(path, 'r') as saved:
                self.assertEqual(len(saved), 2)

    def test_corrupt_complete_cache_is_rejected_not_silently_used(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'references.h5'
            self.prepare(path)
            with h5py.File(path, 'a') as saved:
                key = next(iter(saved))
                saved[f'{key}/block_00000/dwd_id'][0] = 999
            with self.assertRaisesRegex(ValueError, 'Invalid reference cache: block 0'):
                self.prepare(path)

    def test_hdf5_lock_errors_propagate_without_lock_bypass(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('script.dwd_reference_cache.monotonic', side_effect=[0., 31.]), \
             patch('script.dwd_reference_cache.h5py.File', side_effect=BlockingIOError(errno.EAGAIN, 'writer lock')):
            with self.assertRaisesRegex(BlockingIOError, 'writer lock'):
                self.prepare(Path(directory) / 'references.h5')

    def test_equal_scalar_types_strides_and_buffer_aliases_reuse_the_same_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'references.h5'
            expected = self.prepare(path)
            storage = np.zeros((4, len(self.frequency)), dtype=complex)
            storage[0] = self.frequency
            alias = np.array([0., 1.])
            with patch.object(Response, '__init__', side_effect=AssertionError('Equivalent inputs missed cache')):
                actual = self.prepare(path, frequency=storage[0].real,
                    duration=np.float64(100.), rate=np.float64(1.), block_width=np.float64(.1),
                    response_times=alias, response_geometry=np.array([2., 3.]))
            for block in expected:
                np.testing.assert_array_equal(actual[block]['waveforms'], expected[block]['waveforms'])
            with h5py.File(path, 'r') as saved:
                self.assertEqual(len(saved), 1)

    def test_buffer_aliasing_and_array_copying_do_not_change_the_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'references.h5'
            shared = np.array([2., 3.])
            self.prepare(path, response_times=shared, response_geometry={'a': shared, 'b': shared})
            with patch.object(Response, '__init__', side_effect=AssertionError('Array alias changed cache key')):
                self.prepare(path, response_times=shared.copy(),
                    response_geometry={'b': shared.copy(), 'a': shared.copy()})

    def test_transient_lock_is_retried_but_other_errors_are_not_hidden(self):
        original = h5py.File
        calls = []
        def locked_twice(*args, **kwargs):
            calls.append(1)
            if len(calls) <= 2:
                raise BlockingIOError(errno.EAGAIN, 'reader owns shared lock')
            return original(*args, **kwargs)
        with tempfile.TemporaryDirectory() as directory, \
             patch('script.dwd_reference_cache.h5py.File', side_effect=locked_twice), \
             patch('script.dwd_reference_cache.sleep') as wait:
            self.prepare(Path(directory) / 'references.h5')
            self.assertEqual(wait.call_count, 2)
        with patch('script.dwd_reference_cache.h5py.File', side_effect=PermissionError('permission denied')), \
             patch('script.dwd_reference_cache.sleep') as wait:
            with self.assertRaises(PermissionError):
                with _open_cache('unused', 'a'):
                    pass
            wait.assert_not_called()

    def test_real_cross_process_shared_hdf5_lock_is_waited_out(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'references.h5'
            with h5py.File(path, 'w') as saved:
                saved.create_dataset('fixture', data=[1])
            code = ("import h5py,sys,time\n"
                    "with h5py.File(sys.argv[1], 'r') as h:\n"
                    " print('locked', flush=True)\n time.sleep(.5)\n")
            reader = subprocess.Popen([sys.executable, '-c', code, str(path)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                self.assertEqual(reader.stdout.readline().strip(), 'locked')
                start = monotonic()
                with _open_cache(path, 'a', timeout=3.) as saved:
                    saved.create_dataset('after_reader', data=[2])
                self.assertGreater(monotonic() - start, .25)
                self.assertEqual(reader.wait(timeout=5), 0, reader.stderr.read())
            finally:
                if reader.poll() is None:
                    reader.terminate()
                    reader.wait(timeout=5)
                reader.stdout.close()
                reader.stderr.close()

    def test_calculations_run_only_outside_hdf5_contexts(self):
        original = h5py.File
        @contextmanager
        def tracked(*args, **kwargs):
            with original(*args, **kwargs) as saved:
                Response.files_open += 1
                try:
                    yield saved
                finally:
                    Response.files_open -= 1
        with tempfile.TemporaryDirectory() as directory, \
             patch('script.dwd_reference_cache._open_cache', tracked):
            self.prepare(Path(directory) / 'references.h5')
            self.assertEqual(Response.files_open, 0)

    def test_cache_status_is_complete_and_does_not_require_opening_hdf5_to_monitor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'references.h5'
            self.prepare(path)
            with patch('h5py.File', side_effect=AssertionError('Monitor opened HDF5')):
                status = json.loads(path.with_suffix('.status.json').read_text())
            self.assertTrue(status['complete'])
            self.assertEqual(status['generated_blocks'], 2)
            self.assertEqual(status['loaded_blocks'], 0)
            self.assertEqual(status['reference_sources'], 3)


class AtomicRunStatusTests(unittest.TestCase):
    def test_concurrent_reads_never_see_a_partial_json_document(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run.status.json'
            write_run_status(path, sequence=0, payload=[0] * 100)
            errors = []
            def writer():
                try:
                    for n in range(1, 201):
                        write_run_status(path, sequence=n, payload=[n] * 100)
                except Exception as error:
                    errors.append(error)
            thread = threading.Thread(target=writer)
            thread.start()
            while thread.is_alive():
                value = json.loads(path.read_text())
                self.assertEqual(value['payload'], [value['sequence']] * 100)
            thread.join()
            self.assertEqual(errors, [])
            self.assertEqual(json.loads(path.read_text())['sequence'], 200)

    def test_reporting_io_error_does_not_abort_scientific_work(self):
        with patch.object(Path, 'write_text', side_effect=OSError('status disk unavailable')), \
             patch('builtins.print') as log:
            write_run_status('/unused/status.json', phase='fitting_blocks')
            self.assertIn('Status snapshot unavailable', log.call_args.args[0])


class CachedControllerTests(unittest.TestCase):
    run_case = fixtures.GlobalRoundTests.run_case

    def test_one_precompute_before_rounds_and_only_remaining_ids_are_matched(self):
        seen = []
        with tempfile.TemporaryDirectory() as directory, \
             patch('script.dwd_reference_cache.precompute_reference_waveforms',
                   wraps=precompute_reference_waveforms) as prepare:
            result, _, _, output = self.run_case(directory, wait_for_round=(23,),
                association_hook=lambda r, b, ids: seen.append((r, b, ids.tolist())))
            self.assertEqual(prepare.call_count, 1)
            self.assertEqual(result['rounds'], 3)
            self.assertEqual(seen, [(1, 0, [1]), (2, 1, [2])])
            with h5py.File(output.with_suffix('.checkpoints') / 'reference_waveforms.h5', 'r') as saved:
                group = saved[next(iter(saved))]
                self.assertEqual(sorted(group), ['block_00000', 'block_00001'])
                self.assertTrue(all(block.attrs['complete'] for block in group.values()))

    def test_full_scientific_output_matches_uncached_same_block_controller(self):
        baseline = os.environ.get('DWD_REFERENCE_CACHE_BASELINE')
        if not baseline:
            self.skipTest('Server runner supplies the preserved uncached SNR>7 controller')
        cases = (dict(), dict(wait_for_round=(23,)), dict(empty_fit=True),
                 dict(bins=(), catalogue_bins=(10,)), dict(catalogue_relative_snrs=(7., 6.)),
                 dict(bins=(10, 23, 36, 50), catalogue_ids=(27, 27, 99, 27)),
                 dict(bins=(10, 23), subtraction_frequency_range=(.00015, .00025)),
                 dict(bins=(10,), catalogue_bins=(10, 11)))
        for settings in cases:
            with self.subTest(settings=settings), tempfile.TemporaryDirectory() as old, \
                 tempfile.TemporaryDirectory() as new:
                with patch.object(fixtures, 'ROOT', Path(baseline)):
                    _, _, _, expected = self.run_case(old, **settings)
                _, _, _, actual = self.run_case(new, **settings)
                with h5py.File(expected, 'r') as before, h5py.File(actual, 'r') as after:
                    paths = []
                    before.visititems(lambda name, obj: paths.append(name) if isinstance(obj, h5py.Dataset) else None)
                    other_paths = []
                    after.visititems(lambda name, obj: other_paths.append(name) if isinstance(obj, h5py.Dataset) else None)
                    self.assertEqual(paths, other_paths)
                    for name in paths:
                        if not name.endswith('/elapsed_seconds'):
                            np.testing.assert_array_equal(after[name][()], before[name][()], err_msg=name)
                    self.assertEqual(dict(before.attrs), dict(after.attrs))

    def test_cached_frequency_grid_mismatch_stops_before_subtraction(self):
        def bad_cache(*args, **kwargs):
            values = precompute_reference_waveforms(*args, **kwargs)
            values[0]['retained_bins'][0] += 1
            return values
        with tempfile.TemporaryDirectory() as directory, \
             patch('script.dwd_reference_cache.precompute_reference_waveforms', side_effect=bad_cache):
            with self.assertRaisesRegex(ValueError, 'Reference/checkpoint mismatch: block 0'):
                self.run_case(directory)

    def test_status_snapshot_counts_only_committed_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, output = self.run_case(directory, wait_for_round=(23,))
            status = json.loads(output.with_suffix('.status.json').read_text())
            self.assertTrue(status['complete'])
            self.assertEqual(status['phase'], 'complete')
            self.assertEqual(status['relative_snr_threshold'], 7.)
            with h5py.File(output, 'r') as saved:
                self.assertEqual(status['last_completed_round'], saved.attrs['last_completed_round'])
                self.assertEqual(status['finalized_identified_dwd_count'],
                    len(saved['final_subtraction/recovered_dwd_id']))
                self.assertEqual(status['finalized_round_counts'],
                    {label: int(group['identified_dwd_count'][()]) for label, group in saved['rounds'].items()})


if __name__ == '__main__':
    unittest.main()

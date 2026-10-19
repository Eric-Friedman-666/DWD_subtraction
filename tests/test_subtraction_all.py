"""Round coordination and catalogue independence; numerical screening has separate tests."""
import ast
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import islice
import json
import os
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import FunctionType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import h5py
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_pipeline(orbit_file, snr_file, observation_file, output_file, **settings):
    """Execute the guarded controller with fixture parameter assignments."""
    round_limit = settings.pop('_round_limit', None)
    source = ROOT / 'eric_subtraction_all.py'
    tree = ast.parse(source.read_text())
    module = ModuleType('eric_subtraction_all')
    scope = module.__dict__
    scope.update(orbit_file=orbit_file, snr_file=snr_file,
                 observation_file=observation_file, output_file=output_file, **settings)
    for main in (node for node in tree.body if isinstance(node, ast.If)):
        main.test = ast.copy_location(ast.Constant(True), main.test)
        main.body = [node for node in main.body if not (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in scope for target in node.targets))]
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.FunctionDef) and node.name in settings)]
    if round_limit is not None:
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.While))
        loop.body.extend(ast.parse(f"if round_number >= {round_limit}:\n    stopping_reason = 'bounded_test_round_limit'\n    break\n").body)
        ast.fix_missing_locations(tree)
    previous_module = sys.modules.get('eric_subtraction_all')
    sys.modules['eric_subtraction_all'] = module
    try:
        exec(compile(tree, str(source), 'exec'), scope)
    finally:
        # Restore only this temporary controller module. patch.dict(sys.modules)
        # also unloads newly imported native JAX modules, which cannot safely be
        # initialized a second time in the same interpreter.
        if previous_module is None:
            sys.modules.pop('eric_subtraction_all', None)
        else:
            sys.modules['eric_subtraction_all'] = previous_module
    return {**scope['pipeline_result'], 'temporary_directory': str(scope['work'])}


def save_tree(group, values):
    for name, value in values.items():
        if isinstance(value, dict):
            save_tree(group.create_group(name), value)
        else:
            group.create_dataset(name, data=value)


def threaded_block_jobs(function, jobs, *, max_workers, initializer, initargs, **settings):
    """Keep mocked science in threads; real supervision has separate spawn tests."""
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    from multiprocessing import get_context
    original_worker = initializer.__globals__['single_block_process']
    try:
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=get_context('spawn'),
                initializer=initializer, initargs=initargs, max_tasks_per_child=1) as executor:
            # The fixture executor installs a per-thread worker namespace at construction.
            function = partial(initializer.__globals__['single_block_process'], **function.keywords)
            jobs = iter(jobs)
            pending = {executor.submit(function, *job) for job in islice(jobs, max_workers)}
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    job = next(jobs, None)
                    if job is not None:
                        pending.add(executor.submit(function, *job))
    finally:
        initializer.__globals__['single_block_process'] = original_worker


class IsolatedWorkerExecutor(ThreadPoolExecutor):
    """Mock science in threads while preserving separate worker namespaces."""
    def __init__(self, max_workers, mp_context, initializer, initargs, max_tasks_per_child=1):
        if max_tasks_per_child != 1:
            raise AssertionError('Block workers must exit after one task')
        self.worker = threading.local()
        self.finalizers = []
        worker_numbers = iter(range(1, max_workers + 1))
        scope = initializer.__globals__
        original_worker = scope['single_block_process']
        finalize = scope['Finalize']

        def start():
            worker_scope = dict(scope)
            worker_scope['current_process'] = lambda: SimpleNamespace(_identity=(next(worker_numbers),))
            worker_scope['os'] = SimpleNamespace(environ={}, getpid=os.getpid)
            def register(*args, **kwargs):
                result = finalize(*args, **kwargs)
                self.finalizers.append(result)
                return result
            worker_scope['Finalize'] = register
            FunctionType(initializer.__code__, worker_scope, argdefs=initializer.__defaults__)(
                deepcopy(initargs[0]), initargs[1], *initargs[2:])
            self.worker.function = FunctionType(original_worker.__code__, worker_scope)

        def dispatch(*args, **kwargs):
            return self.worker.function(*args, **kwargs)

        scope['single_block_process'] = dispatch
        super().__init__(max_workers=max_workers, initializer=start)

    def shutdown(self, wait=True, **kwargs):
        super().shutdown(wait=wait, **kwargs)
        for finalizer in self.finalizers:
            finalizer()


class GlobalRoundTests(unittest.TestCase):
    def run_case(self, directory, bins=(10, 23), *, catalogue_bins=None, rejected=(),
                 wait_for_round=(), neighbor_alias=False, duplicate_alias=False, empty_fit=False,
                 subtraction_frequency_range=None, catalogue_relative_snrs=None, catalogue_ids=None,
                 executor_factory=None, worker_hook=None, discover_without_candidates=False,
                 association_confirmed_counts=None, fitted_amplitude=1., screening_scale=None,
                 association_hook=None):
        directory = Path(directory)
        duration, rate, N = 128000., .001, 128
        catalogue_bins = bins if catalogue_bins is None else catalogue_bins
        fields = ('Amplitude', 'Frequency', 'FrequencyDerivative', 'InitialPhase', 'Inclination',
            'EclipticLongitude', 'EclipticLatitude', 'Polarization', 'SNR_absolute', 'SNR_relative')
        rows = np.zeros((len(catalogue_bins), 1), dtype=[(name, 'f8') for name in fields])
        for i, carrier in enumerate(catalogue_bins):
            rows[i, 0]['Amplitude'], rows[i, 0]['Frequency'] = 1., carrier / duration
            rows[i, 0]['Inclination'], rows[i, 0]['SNR_absolute'] = np.pi / 2, 20.
            rows[i, 0]['SNR_relative'] = 20.
        if catalogue_relative_snrs is not None:
            rows['SNR_relative'][:, 0] = catalogue_relative_snrs
        if catalogue_ids is None:
            catalogue_ids = np.arange(1, len(rows) + 1, dtype=np.int64)
        snr, observation, output = [directory / name for name in ('snr.h5', 'observation.h5', 'result.h5')]
        with h5py.File(snr, 'w') as saved:
            saved.create_dataset('catalogue', data=np.column_stack((catalogue_ids,
                *[rows[field][:, 0] for field in fields])))
            saved.create_dataset('settings/observation_duration_s', data=duration)
            saved.create_dataset('settings/detector_sampling_rate_hz', data=rate)
            saved.create_dataset('status/complete', data=True)
        def wave(carrier):
            if duplicate_alias and carrier == 23:
                carrier = 10
            if neighbor_alias and carrier in (23, 24):
                original_wave = np.tile(np.sin(2 * np.pi * 10 * np.arange(N) / N), (3, 1))
                original_wave[[1, 2] if carrier == 23 else [0]] = 0
                return original_wave
            return np.tile(np.sin(2 * np.pi * carrier * np.arange(N) / N), (3, 1))
        original = sum((wave(carrier) for carrier in bins), np.zeros((3, N)))
        with h5py.File(observation, 'w') as saved:
            saved.create_dataset('observation/total_xyz_td', data=original)
            saved['observation'].attrs.update(duration=duration, rate=rate)
            saved['observation'].attrs['detector_config'] = json.dumps(dict(
                T_obs=duration, dt=1 / rate, orbits='stored_orbit_location.h5', orbit_dataset='tcb/ltt',
                gpu_batch_size=3, minimum_envelope_samples=1024, include_padding=True,
                aafilter=('kaiser', 222., .2 * rate, .7 * rate),
                clock_offsets=.02, clock_freqoffsets=.03, clock_freqlindrifts=.04,
                clock_freqquaddrifts=.05))
        state = dict(sampler=[], waveform=[], tdi=[], geometry=0, screen=[], search=[], new_signal=None, waveform_config=[], instrument_config=[], events=[])
        thread_data = threading.local()
        class Builder:
            def __init__(self, covariance, data, times, geometry, frequency, T_obs, fs):
                self.frequency = frequency
                thread_data.frequency = frequency
            def build(self, doppler):
                basis = np.zeros((len(doppler), 4, 3, len(self.frequency)), complex)
                indices = np.rint(self.frequency * duration).astype(int)
                for i, point in enumerate(doppler):
                    carrier = int(round(point[0] * duration))
                    basis[i, 0] = np.fft.rfft(wave(carrier), axis=-1)[:, indices]
                    basis[i, 3, 0] = carrier
                return basis
        def geometry(work, frequency, T_obs, fs, **settings):
            state['geometry'] += 1
            return Builder(None, None, None, None, frequency, T_obs, fs), np.array([0.]), None
        def reconstruct(coefficients, bases):
            return np.einsum('sm,smck->sck', coefficients, bases)
        def inner(a, b, *args):
            return np.sum(np.asarray(a).conj() * b, axis=(-2, -1)).real
        def periodogram(data, T_obs, fs):
            state['events'].append(('covariance', time.monotonic()))
            state['screen'].append(data.copy())
            # Keep the coordination fixtures independent of the numerical estimator.
            return np.broadcast_to(np.eye(3), (data.shape[-1], 3, 3)).copy()
        def statistics(data, bases, covariance, T_obs, fs):
            if sys._getframe(1).f_code.co_name == 'single_block_process':
                templates = sys._getframe(1).f_locals['dwd_template_table']
                if worker_hook is not None:
                    worker_hook(templates, state)
                state['search'].append((templates.copy(), data.copy()))
                frequency = thread_data.frequency
                state.setdefault('search_frequency', []).append(frequency.copy())
                self.assertEqual(bases.shape, (len(templates), 4, 3, len(frequency)))
                self.assertEqual(data.shape, (3, len(frequency)))
                self.assertEqual(covariance.shape, (len(frequency), 3, 3))
                self.assertEqual(templates.shape[1], 11)
                indices = np.rint(frequency * duration).astype(int)
                active = indices[np.sum(abs(data)**2, axis=0) > 1.e-6]
                template_bins = np.rint(templates[:, 2] * duration).astype(int)
                if duplicate_alias:
                    template_bins = np.where(template_bins == 23, 10, template_bins)
                elif neighbor_alias:
                    template_bins = np.where(np.isin(template_bins, [23, 24]), 10, template_bins)
                retained = np.isin(template_bins, active)
                if any(int(round(f * duration)) in wait_for_round for f in templates[retained, 2]) and not state['waveform']:
                    retained[:] = False
                if screening_scale is None:
                    scores = np.where(retained, 50., 0.)
                else:
                    original_power = np.sum(abs(np.fft.rfft(original, axis=-1))**2, axis=0)[template_bins]
                    current_power = np.sum(abs(data)**2, axis=0)[np.searchsorted(indices, template_bins)]
                    scores = np.where(retained, screening_scale * current_power / original_power, 0.)
                state.setdefault('f_values', []).append((templates[:, 0].copy(), scores.copy()))
                return scores
            scores = []
            for basis in bases:
                carrier = int(round(basis[3, 0, 0].real))
                scores.append(0. if carrier in rejected else (100. if np.vdot(basis[0], data).real > 1.e-6 else 0.))
            return np.asarray(scores) / 2
        def sampler(data, covariance, table, block_range, bounds, T_obs, fs, checkpoints, **settings):
            state['sampler'].append((table.copy(), data.copy(), settings, bounds.copy(), block_range.copy()))
            self.assertEqual(table.shape[1], 11)
            doppler = table[:, [2, 3, 6, 7]]
            if discover_without_candidates and not len(table):
                frequency = settings['physical_frequency']
                found = (np.sum(abs(data)**2, axis=0) > 1.e-6) & (frequency >= block_range[0]) & (frequency < block_range[1])
                doppler = np.column_stack((frequency[found][:1], np.zeros((min(np.count_nonzero(found), 1), 3))))
            self.assertNotIn('chirp_coordinate', settings)
            self.assertNotIn('chain_count', settings)
            self.assertNotIn('inverse_temperatures', settings)
            if empty_fit:
                doppler = doppler[:0]
            bases = Builder(covariance, data, None, None, settings['physical_frequency'], T_obs, fs).build(doppler)
            return dict(doppler=doppler.copy(), bases=bases,
                fit_result=dict(coefficients=np.tile([fitted_amplitude, 0., 0., 0.], (len(doppler), 1))), config={})
        def associate(recovered, templates, ids, inverse, T_obs, fs, **settings):
            if association_hook is not None:
                caller = sys._getframe(1).f_locals
                association_hook(caller['round_number'], caller['block_index'], ids.copy())
            matched = np.full(len(recovered), -1, dtype=np.int64)
            used = set()
            for i, source in enumerate(recovered):
                if len(templates):
                    overlap = np.array([np.vdot(candidate, source).real / max(np.linalg.norm(candidate) * np.linalg.norm(source), 1.e-300) for candidate in templates])
                    for candidate in np.argsort(-overlap, kind='stable'):
                        if overlap[candidate] > .9 and int(ids[candidate]) not in used:
                            matched[i] = ids[candidate]
                            used.add(int(ids[candidate]))
                            break
            if association_confirmed_counts is not None:
                block = sys._getframe(1).f_locals['block_index']
                count = association_confirmed_counts.get(int(block), len(recovered))
                if isinstance(count, (tuple, list)):
                    count = count[min(len(state['tdi']), len(count) - 1)]
                matched[count:] = -1
            return dict(matched_dwd_id=matched, best_overlap=np.ones(len(matched)))
        def waveform(parameters, config):
            state['events'].append(('subtraction', time.monotonic()))
            state['waveform'].append(parameters.copy())
            state['waveform_config'].append(config.copy())
            state['new_signal'] = sum((row[0] * wave(int(round(row[1] * duration))) for row in parameters), np.zeros_like(original))
            return np.zeros((N + 4 * round(86400 * rate), 6))
        class Response:
            def __init__(self, **settings):
                pass
            def init_response_file(self, saved, dt, size, t0, frame):
                saved.create_dataset('tcb/y', shape=(size, 6))
        class Instrument:
            central_freq = 1.
            def __init__(self, **settings):
                state['instrument_config'].append(settings.copy())
            def disable_all_noises(self):
                pass
            def disable_dopplers(self):
                pass
            def export_numpy_full(self):
                return SimpleNamespace(measurements={}, args={})
        def tdi(measurements, config, **settings):
            state['events'].append(('tdi', time.monotonic()))
            state['tdi'].append(settings)
            return state['new_signal']
        replacements = {
            'lisagwresponse': dict(GalacticBinary=Response), 'lisainstrument': dict(Instrument=Instrument),
            'pytdi': dict(Data=SimpleNamespace(from_instrument=lambda value: value)),
            'eric_generate_waveform': dict(DWD_simulation_gpu=waveform, tdi_gpu=tdi),
            'script.eric_generate_waveform': dict(GPUResponse=Builder),
            'eric_rjmcmc': dict(run_eryn_rjmcmc=sampler),
            'script.eric_Fstats': dict(build_response=geometry,
                physical_amplitude_coordinates=lambda plus, cross, psi, phi: np.column_stack((plus, np.zeros((len(plus), 3)))),
                physical_amplitudes=lambda coeff: dict(A=coeff[:, 0], phi0=np.zeros(len(coeff)), iota=np.zeros(len(coeff)), psi_bme=np.zeros(len(coeff))), reconstruct_waveform=reconstruct),
            'script.dwd_blind_search': dict(F_statistic=statistics),
            'script.eric_preprocess': dict(periodogram=periodogram, inner_product=inner,
                effective_noise_cov=lambda psd, **settings: psd.copy()),
            'script.dwd_bank_association': dict(assign_bank_ids=associate),
            # These coordination fixtures mock screening, sampling, matching,
            # and synthesis. Numerical winner rescaling is checked separately
            # by test_amplitude_only_refit and the real crowded-source smoke.
            'script.dwd_amplitude_refit': dict(joint_amplitude_only_refit=
                lambda data, sources, *args, **settings: (np.ones(len(sources)), {})),
            'script.dwd_worker_supervisor': dict(run_block_jobs=threaded_block_jobs,
                worker_progress=lambda *args, **settings: None),
            'script.oneblock_observation': dict(read_checkpoint=lambda saved: {}, write_checkpoint=lambda saved, values: None),
            'script.tool': dict(_save_tree=save_tree, single_cpu=nullcontext)}
        modules = {}
        for name, values in replacements.items():
            module = ModuleType(name)
            module.__dict__.update(values)
            modules[name] = module
        # Baseline fixtures use one thread; separate scheduling tests use three.
        if executor_factory is None:
            executor_factory = lambda **settings: IsolatedWorkerExecutor(**{**settings, 'max_workers': 1})
        with patch.dict(sys.modules, modules), patch('concurrent.futures.ProcessPoolExecutor', executor_factory):
            result = run_pipeline(directory / 'orbit.h5', snr, observation, output,
                frequency_range=(.00005, .00045), freq_block_width=.0001, sample_number=(1, 2),
                subtraction_frequency_range=subtraction_frequency_range)
        return result, state, original, output

    def test_rolling_four_blocks_ordered_results_and_round_barrier(self):
        starts, finishes, active = [], [], set()
        lock = threading.Lock()
        fourth_started = threading.Event()
        first_three = threading.Barrier(3)
        maximum_active = 0
        case = self

        class RollingExecutor(IsolatedWorkerExecutor):
            def __init__(self, max_workers, mp_context, initializer, initargs, max_tasks_per_child):
                case.assertEqual(max_workers, 3)
                case.assertEqual(mp_context.get_start_method(), 'spawn')
                case.assertEqual(max_tasks_per_child, 1)
                super().__init__(max_workers, mp_context, initializer, initargs, max_tasks_per_child)

            def submit(self, function, block_index, *args, **settings):
                def measured():
                    nonlocal maximum_active
                    label = function.keywords['label']
                    with lock:
                        active.add((label, int(block_index)))
                        maximum_active = max(maximum_active, len(active))
                        starts.append((label, int(block_index), time.monotonic()))
                    try:
                        if label == '01':
                            if block_index < 3:
                                first_three.wait(timeout=5)
                            if block_index in (0, 2):
                                case.assertTrue(fourth_started.wait(timeout=5), 'batch barrier prevented block four')
                            if block_index == 3:
                                fourth_started.set()
                        return function(block_index, *args)
                    finally:
                        with lock:
                            finishes.append((label, int(block_index), time.monotonic()))
                            active.remove((label, int(block_index)))
                return super().submit(measured)

        with tempfile.TemporaryDirectory() as sequential, tempfile.TemporaryDirectory() as rolling:
            baseline, _, _, baseline_output = self.run_case(sequential, bins=(10, 23, 36, 50), catalogue_ids=(27, 27, 99, 27))
            result, state, _, output = self.run_case(rolling, bins=(10, 23, 36, 50),
                catalogue_ids=(27, 27, 99, 27), executor_factory=RollingExecutor)
            self.assertEqual(maximum_active, 3)
            start_four = next(t for label, block, t in starts if label == '01' and block == 3)
            self.assertTrue(all(start_four < t for label, block, t in finishes if label == '01' and block in (0, 2)))
            self.assertEqual(finishes[0][:2], ('01', 1))
            self.assertEqual(result['source_count'], baseline['source_count'])
            self.assertEqual(result['rounds'], 2)
            subtraction = next(t for event, t in state['events'] if event == 'subtraction')
            tdi = next(t for event, t in state['events'] if event == 'tdi')
            self.assertGreater(subtraction, max(t for label, _, t in finishes if label == '01'))
            second_covariance = [t for event, t in state['events'] if event == 'covariance'][1]
            self.assertGreater(second_covariance, tdi)
            self.assertFalse(any(label == '02' for label, _, _ in starts))
            self.assertEqual(sum(event == 'tdi' for event, _ in state['events']), 1)
            with h5py.File(output) as actual, h5py.File(baseline_output) as expected:
                paths = []
                expected.visititems(lambda name, obj: paths.append(name) if isinstance(obj, h5py.Dataset) else None)
                for path in paths:
                    if path.endswith('/elapsed_seconds'):
                        continue
                    np.testing.assert_array_equal(actual[path][()], expected[path][()], err_msg=path)
                np.testing.assert_array_equal(actual['final_subtraction/dwd_id'][:], [27, 27, 99, 27])

    def test_out_of_order_neighbor_results_preserve_block_identification_decisions(self):
        for duplicate in (False, True):
            other_started = threading.Event()
            def wait_for_neighbor(templates, state):
                if len(templates):
                    if templates[0, 0] == 1:
                        self.assertTrue(other_started.wait(timeout=5))
                    else:
                        other_started.set()
            with self.subTest(duplicate=duplicate), tempfile.TemporaryDirectory() as sequential, tempfile.TemporaryDirectory() as rolling:
                settings = dict(bins=(10,), catalogue_bins=(10, 23) if duplicate else (10, 23, 24),
                                neighbor_alias=not duplicate, duplicate_alias=duplicate,
                                association_confirmed_counts={0: 1, 1: 0})
                _, _, _, reference = self.run_case(sequential, **settings)
                result, _, _, output = self.run_case(rolling, **settings, worker_hook=wait_for_neighbor,
                    executor_factory=IsolatedWorkerExecutor)
                self.assertEqual(result['source_count'], 1)
                with h5py.File(output) as actual, h5py.File(reference) as expected:
                    for key in actual['rounds/01/blocks']:
                        a, e = actual[f'rounds/01/blocks/{key}'], expected[f'rounds/01/blocks/{key}']
                        if 'acceptance' in a:
                            for field in a['acceptance']:
                                np.testing.assert_array_equal(a['acceptance'][field][:] if a['acceptance'][field].shape else a['acceptance'][field][()],
                                    e['acceptance'][field][:] if e['acceptance'][field].shape else e['acceptance'][field][()])
                    np.testing.assert_array_equal(actual['rounds/01/accepted_count'][:], [1, 0, 0, 0])
                    self.assertFalse(actual['rounds/01/blocks/00001/acceptance/passed'][()])
                    self.assertNotIn('duplicate_previous_or_this_round', actual['rounds/01/blocks/00001/acceptance'])
                    np.testing.assert_array_equal(actual['final_subtraction/dwd_id'][:], expected['final_subtraction/dwd_id'][:])
                    np.testing.assert_array_equal(actual['final_subtraction/residual_xyz_td'][:], expected['final_subtraction/residual_xyz_td'][:])

    def test_import_guard_and_worker_ownership(self):
        tree = ast.parse((ROOT / 'eric_subtraction_all.py').read_text())
        worker = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'single_block_process')
        guard = next(node for node in tree.body if isinstance(node, ast.If))
        self.assertEqual(ast.unparse(guard.test), "__name__ == '__main__'")
        self.assertEqual(worker.name, 'single_block_process')
        returned = next(node for node in worker.body if isinstance(node, ast.Return))
        self.assertEqual([node.id for node in returned.value.elts],
            ['dwd_recover_table', 'filter_result', 'retained_bins', 'posterior'])
        self.assertFalse(any(isinstance(node, ast.Subscript) and isinstance(node.ctx, ast.Store)
            and isinstance(node.value, ast.Name) and node.value.id == 'posterior' for node in ast.walk(worker)))
        worker_names = {node.id for node in ast.walk(worker) if isinstance(node, ast.Name)}
        self.assertFalse(worker_names & {'output_file', 'proposal_doppler', 'proposal_ids', 'proposals', 'accepted_ids'})
        self.assertEqual([node.name for node in tree.body if isinstance(node, ast.FunctionDef)], ['initialize_worker', 'single_block_process'])
        loads = [node for node in ast.walk(worker) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'load'
                 and isinstance(node.func.value, ast.Name) and node.func.value.id == 'np']
        self.assertFalse(loads)
        source = (ROOT / 'eric_subtraction_all.py').read_text()
        self.assertFalse(any(name in source for name in ('shared.pkl', 'dwd_template_table.npy',
            'lisa_data_fd.npy', 'effective_noise_covariance.npy', 'np.save(', 'np.load(')))
        self.assertIn("checkpoints / 'result.pkl'", source)
        supervision = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                           and isinstance(node.func, ast.Name) and node.func.id == 'run_block_jobs')
        self.assertEqual(next(ast.unparse(item.value) for item in supervision.keywords
                              if item.arg == 'max_workers'), 'worker_count')
        self.assertNotIn('ProcessPoolExecutor', source)
        supervisor = (ROOT / 'script/dwd_worker_supervisor.py').read_text()
        self.assertIn("get_context('spawn')", supervisor)
        self.assertIn('function(*arguments)', supervisor)

    def test_uncommitted_round_resumes_completed_blocks_without_refitting(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as baseline:
            result_file = Path(directory) / 'result.h5'
            def interrupt(templates, state):
                if len(templates) and int(templates[0, 0]) == 1:
                    completed = result_file.with_suffix('.checkpoints') / 'round_01_block_00001' / 'result.pkl'
                    deadline = time.monotonic() + 5
                    while not completed.exists() and time.monotonic() < deadline:
                        time.sleep(.01)
                    self.assertTrue(completed.exists())
                    raise RuntimeError('deliberate unfinished block')
            with self.assertRaisesRegex(RuntimeError, 'deliberate unfinished block'):
                self.run_case(directory, bins=(10, 23, 36, 50), worker_hook=interrupt,
                    executor_factory=IsolatedWorkerExecutor)
            with h5py.File(result_file) as saved:
                self.assertEqual(saved.attrs.get('last_completed_round', 0), 0)
            _, state, _, output = self.run_case(directory, bins=(10, 23, 36, 50))
            _, _, _, expected = self.run_case(baseline, bins=(10, 23, 36, 50))
            self.assertLess(len(state['sampler']), 4, 'completed block checkpoints were refitted')
            self.assertEqual(len(state['tdi']), 1)
            with h5py.File(output) as actual, h5py.File(expected) as reference:
                for name in reference['final_subtraction']:
                    np.testing.assert_array_equal(actual['final_subtraction/' + name][:],
                        reference['final_subtraction/' + name][:])

    def test_committed_round_resumes_without_subtracting_its_signal_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, original, output = self.run_case(directory, bins=(10, 23))
            with h5py.File(output, 'a') as saved:
                signal = saved['final_subtraction/signal_xyz_td'][:]
                residual = saved['final_subtraction/residual_xyz_td'][:]
                del saved['final_subtraction']
                del saved['rounds/02']
                saved.attrs['last_completed_round'] = 1
                saved.attrs['complete'] = False
            _, state, _, output = self.run_case(directory, bins=(10, 23))
            self.assertFalse(state['sampler'])
            self.assertFalse(state['tdi'])
            with h5py.File(output) as saved:
                np.testing.assert_array_equal(saved['final_subtraction/signal_xyz_td'][:], signal)
                np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_td'][:], residual)
                np.testing.assert_array_equal(signal + residual, original)

    def test_recovery_reporting_is_unique_without_changing_component_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, output = self.run_case(directory, bins=(10, 23, 36, 50), catalogue_ids=(27, 27, 99, 27))
            with h5py.File(output) as saved:
                np.testing.assert_array_equal(saved['final_subtraction/dwd_id'][:], [27, 27, 99, 27])
                np.testing.assert_array_equal(saved['final_subtraction/recovered_dwd_id'][:], [27, 99])
                np.testing.assert_array_equal(saved['rounds/01/recovered_dwd_id'][:], [27, 99])

    def test_shared_memory_and_work_directory_cleanup_on_failure_or_interrupt(self):
        from multiprocessing.shared_memory import SharedMemory
        for error in (RuntimeError, KeyboardInterrupt):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                names = []
                def allocate(*args, **kwargs):
                    memory = SharedMemory(*args, **kwargs)
                    if kwargs.get('create'):
                        names.append(memory.name)
                    return memory
                def fail(templates, state):
                    if len(templates):
                        raise error('bounded cleanup test')
                with patch('multiprocessing.shared_memory.SharedMemory', side_effect=allocate):
                    with self.assertRaises(error):
                        self.run_case(directory, worker_hook=fail)
                self.assertEqual(len(names), 2)
                self.assertTrue((Path(directory) / 'result.checkpoints').is_dir())
                for name in names:
                    with self.assertRaises(FileNotFoundError):
                        SharedMemory(name=name)

    def test_seven_steps_and_observation_table_contract(self):
        import re
        source = (ROOT / 'eric_subtraction_all.py').read_text()
        self.assertEqual(re.findall(r'# Step (\d+)\.', source), list('1234567'))
        tree = ast.parse(source)
        guards = [node for node in tree.body if isinstance(node, ast.If)]
        self.assertEqual(len(guards), 2)
        self.assertTrue(all(ast.unparse(node.test) == "__name__ == '__main__'" for node in guards))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertFalse(names & {'total_xyz_td', 'residual_xyz_td', 'residual_xyz_fd',
                                 'accumulated_xyz_td', 'template_dwd', 'search', 'maximum_rounds'})
        self.assertIn("print(f'Round {label}'", source)
        with tempfile.TemporaryDirectory() as directory:
            result, _, original, output = self.run_case(directory, wait_for_round=(23,))
            self.assertEqual(result['rounds'], 3)
            with h5py.File(output) as saved:
                td, fd = saved['final_subtraction/lisa_data_td'][:], saved['final_subtraction/lisa_data_fd'][:]
                self.assertEqual(td.shape, (4, original.shape[-1]))
                self.assertEqual(fd.shape, (4, original.shape[-1] // 2 + 1))
                np.testing.assert_array_equal(td[0], np.arange(original.shape[-1]) * 1000.)
                np.testing.assert_array_equal(fd[0], np.fft.rfftfreq(original.shape[-1], 1000.))
                np.testing.assert_array_equal(fd[1:], np.fft.rfft(td[1:], axis=-1))
                for group in saved['rounds'].values():
                    self.assertEqual(group['parameters'].shape[1], 8)
                    self.assertEqual(len(group['parameters']), sum(group['accepted_count'][:]))
                    self.assertEqual(len(group['dwd_id']), len(group['block_index']))

    def test_candidate_bank_can_exceed_fixed_source_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, _, output = self.run_case(directory, bins=(10,), catalogue_bins=(10,) * 70, empty_fit=True)
            self.assertEqual(len(state['sampler'][0][0]), 70)
            self.assertEqual(state['sampler'][0][2]['maximum_sources'], 64)
            self.assertEqual(result['stopping_reason'], 'no_accepted_blocks')
            self.assertFalse(state['tdi'])
            with h5py.File(output) as saved:
                self.assertEqual(saved['config/maximum_sources'][()], 64)
                self.assertEqual(saved['rounds/01/selected_count'][0], 70)
                filtered = saved['rounds/01/blocks/00000/screening/search']
                self.assertIsInstance(filtered, h5py.Dataset)
                self.assertEqual(filtered.shape, (70, 11))
                np.testing.assert_array_equal(filtered[:], state['sampler'][0][0])
                np.testing.assert_array_equal(filtered[:, 0], np.arange(1, 71))
                self.assertEqual(saved['rounds/01/accepted_count'][0], 0)

    def test_saved_relative_snr_filters_templates_and_preserves_original_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory,
                bins=(10, 12, 23, 24), catalogue_relative_snrs=(.9, 1., 1.1, 20.))
            self.assertEqual(result['source_count'], 2)
            first_round_templates = np.concatenate([templates for templates, _ in state['search'][:4]])
            np.testing.assert_array_equal(np.rint(first_round_templates[:, 2] * 128000.), [23, 24])
            tree = ast.parse((ROOT / 'eric_subtraction_all.py').read_text())
            main = next(node for node in tree.body if isinstance(node, ast.If))
            start = next(i for i, node in enumerate(main.body) if isinstance(node, ast.With)
                and isinstance(node.items[0].context_expr.args[0], ast.Name)
                and node.items[0].context_expr.args[0].id == 'snr_file')
            field_definition = next(node for node in main.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'fields' for t in node.targets))
            scope = dict(np=np, h5py=h5py, snr_file=Path(directory) / 'snr.h5', SNR_r_threshold=1.)
            exec(compile(ast.Module(body=[field_definition, main.body[start]], type_ignores=[]),
                str(ROOT / 'eric_subtraction_all.py'), 'exec'), scope)
            table = scope['dwd_template_table']
            self.assertEqual(table.shape, (2, 11))
            np.testing.assert_array_equal(table[:, 0], [3, 4])
            with h5py.File(Path(directory) / 'snr.h5') as saved:
                np.testing.assert_array_equal(table, saved['catalogue'][2:])
            with h5py.File(output) as saved:
                self.assertEqual(saved['config/relative_snr_threshold'][()], 1.)
                self.assertEqual(saved['config/relative_snr_source'][()].decode(), 'fixed_saved_catalogue')
                self.assertEqual(saved['config/subtraction_policy'][()].decode(), 'identified_only')
                np.testing.assert_array_equal(saved['final_subtraction/dwd_id'][:], [3, 4])
                filtered = saved['rounds/01/blocks/00001/screening/search']
                self.assertIsInstance(filtered, h5py.Dataset)
                np.testing.assert_array_equal(filtered[:], table)
                np.testing.assert_array_equal(filtered[:, 0], [3, 4])
                np.testing.assert_allclose(saved['final_subtraction/residual_xyz_td'][:],
                    original - saved['final_subtraction/signal_xyz_td'][:], atol=1.e-14)

    def test_no_saved_relative_snr_above_threshold_keeps_original_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory,
                catalogue_relative_snrs=(1., .9))
            self.assertEqual(result['source_count'], 0)
            self.assertTrue(all(len(templates) == 0 for templates, _ in state['search']))
            self.assertEqual(len(state['sampler']), 0)
            self.assertFalse(state['waveform'] or state['tdi'])
            with h5py.File(output) as saved:
                np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_td'][:], original)
                for block in saved['rounds/01/blocks'].values():
                    self.assertIsInstance(block['screening/search'], h5py.Dataset)
                    self.assertEqual(block['screening/search'].shape, (0, 11))
                    self.assertNotIn('rjmcmc', block)
                    self.assertNotIn('checkpoints', block)

    def test_remaining_templates_exclude_identified_ids_and_keep_saved_snrs(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, _, output = self.run_case(directory, bins=(10, 23),
                catalogue_relative_snrs=(1.1, 6.), wait_for_round=(23,))
            self.assertEqual(result['rounds'], 3)
            for first, ids, snrs in ((0, [1, 2], [1.1, 6.]), (4, [2], [6.]), (8, [], [])):
                templates = np.concatenate([table for table, _ in state['search'][first:first + 4]])
                np.testing.assert_array_equal(templates[:, 0], ids)
                np.testing.assert_array_equal(templates[:, 10], snrs)
            with h5py.File(output) as saved:
                self.assertTrue(saved['config/exclude_recovered_templates'][()])
                for label, excluded in (('01', []), ('02', [1]), ('03', [1, 2])):
                    np.testing.assert_array_equal(saved[f'rounds/{label}/excluded_dwd_id'][:], excluded)
                for label, remaining in (('01', [1, 2]), ('02', [2]), ('03', [])):
                    np.testing.assert_array_equal(saved[f'rounds/{label}/remaining_template_dwd_id'][:], remaining)
            with h5py.File(Path(directory) / 'snr.h5') as saved:
                np.testing.assert_array_equal(saved['catalogue'][:, 10], [1.1, 6.])

    def test_identified_template_is_excluded_even_when_its_residual_still_passes_f(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory, bins=(10,), fitted_amplitude=.25)
            self.assertEqual(result['rounds'], 2)
            self.assertEqual(result['source_count'], 1)
            self.assertEqual(len(state['sampler']), 1)
            np.testing.assert_array_equal(state['search'][0][0][:, 0], [1])
            self.assertTrue(all(len(table) == 0 for table, _ in state['search'][4:]))
            with h5py.File(output) as saved:
                np.testing.assert_array_equal(saved['rounds/02/excluded_dwd_id'][:], [1])
                np.testing.assert_allclose(saved['final_subtraction/residual_xyz_td'][:], .75 * original, atol=1.e-12)

    def test_empty_filtered_bank_skips_prior_discovery_and_preserves_data(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory, bins=(10,),
                catalogue_relative_snrs=(.9,), discover_without_candidates=True)
            self.assertEqual(result['source_count'], 0)
            self.assertEqual(result['rounds'], 1)
            self.assertEqual(len(state['sampler']), 0)
            with h5py.File(output) as saved:
                np.testing.assert_array_equal(saved['rounds/01/selected_count'][:], 0)
                np.testing.assert_array_equal(saved['rounds/01/accepted_count'][:], 0)
                self.assertEqual(saved['rounds/01/blocks/00000/screening/search'].shape, (0, 11))
                self.assertEqual(saved['final_subtraction/dwd_id'].shape, (0,))
                np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_td'][:], original)

    def test_ids_are_read_from_the_single_table_without_renumbering(self):
        with tempfile.TemporaryDirectory() as directory:
            _, state, _, output = self.run_case(directory, catalogue_ids=(2111, 29000000))
            np.testing.assert_array_equal(state['search'][0][0][:, 0], [2111])
            with h5py.File(output) as saved:
                np.testing.assert_array_equal(saved['final_subtraction/dwd_id'][:], [2111, 29000000])

    def test_snr_writer_saves_the_numerical_eleven_column_table(self):
        tree = ast.parse((ROOT / 'eric_snr.py').read_text())
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in ('catalogue_columns', 'merged_rows') for t in node.targets)]
        create = next(node for node in ast.walk(tree) if isinstance(node, ast.With)
            and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == 'create_dataset' and call.args
                and isinstance(call.args[0], ast.Constant) and call.args[0].value == 'catalogue'
                for call in ast.walk(node)))
        write = next(node for node in ast.walk(tree) if isinstance(node, ast.With)
            and isinstance(node.items[0].context_expr, ast.Call)
            and isinstance(node.items[0].context_expr.func, ast.Attribute)
            and node.items[0].context_expr.func.attr == 'File'
            and any(isinstance(n, ast.Name) and n.id == 'merged_rows' for n in ast.walk(node)))
        physical_fields = ('Amplitude', 'Frequency', 'FrequencyDerivative', 'InitialPhase',
            'Inclination', 'EclipticLongitude', 'EclipticLatitude', 'Polarization')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'snr.h5'
            rows = np.zeros(3, dtype=[('Name', 'S7')] + [(name, 'f8') for name in
                (*physical_fields, 'SNR_absolute', 'SNR_relative')])
            for i, field in enumerate(rows.dtype.names[1:]):
                rows[field] = np.arange(3) + i / 10.
            with h5py.File(path, 'w') as saved:
                saved.create_group('settings')
                saved.create_dataset('sky/igb/cat', data=rows[:, None])
            scope = dict(np=np, h5py=h5py, output_file=path, source_count=3,
                PARAMETER_FIELDS=physical_fields, dwd_ids=np.arange(3), rows=rows,
                dataset_path='sky/igb/cat', catalogue_offsets={'igb': 0}, group='igb', first=0, last=3)
            nodes = [next(node for node in assignments if node.targets[0].id == 'catalogue_columns'),
                create, next(node for node in assignments if node.targets[0].id == 'merged_rows'), write]
            exec(compile(ast.Module(body=nodes, type_ignores=[]), 'snr_export', 'exec'), scope)
            with h5py.File(path) as saved:
                self.assertEqual(saved['catalogue'].shape, (3, 11))
                np.testing.assert_array_equal(saved['catalogue'][:, 0], [1, 2, 3])
                expected = np.column_stack([rows[name] for name in rows.dtype.names[1:]])
                np.testing.assert_array_equal(saved['catalogue'][:, 1:], expected)
                self.assertEqual(tuple(saved['settings/catalogue_columns'].asstr()[:]),
                    ('ID', *physical_fields, 'SNR_absolute', 'SNR_relative'))

    def test_grid_is_saved_as_contiguous_start_end_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, output = self.run_case(directory, bins=(), catalogue_bins=())
            with h5py.File(output) as saved:
                blocks = saved['config/frequency_blocks_hz'][:]
                self.assertEqual(blocks.shape, (4, 2))
                np.testing.assert_allclose(blocks, [[.00005, .00015], [.00015, .00025],
                    [.00025, .00035], [.00035, .00045]], rtol=0, atol=1e-18)
                np.testing.assert_allclose(blocks[:-1, 1], blocks[1:, 0], rtol=0, atol=1e-18)

    def test_subtraction_reuses_saved_detector_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            _, state, _, _ = self.run_case(directory)
            config = state['waveform_config'][0]
            self.assertEqual(config['gpu_batch_size'], 3)
            self.assertEqual(config['minimum_envelope_samples'], 1024)
            self.assertEqual(config['T_obs'], 128000.)
            self.assertEqual(config['dt'], 1000.)
            self.assertEqual(config['orbits'], str(Path(directory) / 'orbit.h5'))
            instrument = state['instrument_config'][0]
            self.assertEqual(instrument['aafilter'], ['kaiser', 222., .0002, .0007])
            self.assertEqual(instrument['clock_offsets'], .02)
            self.assertEqual(instrument['clock_freqquaddrifts'], .05)

    def test_one_snapshot_and_one_tdi_sum_for_multiple_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory)
            self.assertEqual(result['rounds'], 2)
            self.assertEqual(len(state['sampler']), 2)
            self.assertEqual(len(state['waveform']), 1)
            self.assertEqual(len(state['waveform'][0]), 2)
            self.assertEqual(len(state['tdi']), 1)
            self.assertEqual(state['geometry'], 1)
            with h5py.File(output) as saved:
                self.assertTrue(saved.attrs['complete'])
                self.assertTrue(saved.attrs['single_file_complete'])
                for round_group in saved['rounds'].values():
                    self.assertNotIn('complete', round_group.attrs)
                    self.assertNotIn('committed', round_group.attrs)
                    self.assertEqual(set(round_group.attrs), {'visited_block_count',
                        'selected_block_count', 'accepted_block_count', 'residual_output_sha256'})
                first = saved['rounds/01']
                self.assertEqual(first.attrs['visited_block_count'], 4)
                self.assertEqual(len(state['search']), 8)
                self.assertNotIn('residual_input_sha256', first.attrs)
                original_fd = np.fft.rfft(original, axis=-1)
                for key, group in first['blocks'].items():
                    self.assertNotIn('residual_input_sha256', group.attrs)
                    bins = saved['blocks'][key]['retained_bins'][:]
                    np.testing.assert_array_equal(group['screening/residual_input_xyz_fd'][:], original_fd[:, bins])
                np.testing.assert_allclose(saved['final_subtraction/signal_xyz_td'][:], original, atol=1.e-14)
                np.testing.assert_array_equal(saved['final_subtraction/dwd_id'][:], [1, 2])
                np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_fd'][:], np.fft.rfft(saved['final_subtraction/residual_xyz_td'][:], axis=-1))
    def test_one_block_padding_is_fixed_for_every_selected_block_and_round(self):
        for bounds in (None, (.00005, .00025), (.00015, .00025)):
            with self.subTest(bounds=bounds), tempfile.TemporaryDirectory() as directory:
                result, state, original, output = self.run_case(directory,
                    subtraction_frequency_range=bounds,
                    wait_for_round=(23,) if bounds != (.00015, .00025) else ())
                self.assertGreaterEqual(result['rounds'], 2)
                frequency = np.fft.rfftfreq(original.shape[-1], d=1000.)
                with h5py.File(output) as saved:
                    width = saved['config/frequency_block_width'][()]
                    blocks = saved['config/frequency_blocks_hz'][:]
                    selected = saved['config/selected_block_indices'][:]
                    self.assertEqual(len(state['search']), result['rounds'] * len(selected))
                    for round_index, round_group in enumerate(saved['rounds'].values()):
                        for position, block_index in enumerate(selected):
                            key = f'{block_index:05d}'
                            lower, upper = blocks[block_index]
                            expected = np.flatnonzero((frequency >= lower - width)
                                & (frequency <= upper + width) & (frequency > 0))
                            np.testing.assert_array_equal(saved[f'blocks/{key}/retained_bins'][:], expected)
                            self.assertEqual(set(saved[f'blocks/{key}']), {'retained_bins',
                                'block_index', 'frequency_range', 'physical_frequency_hz'})
                            np.testing.assert_array_equal(saved[f'blocks/{key}/physical_frequency_hz'][:], frequency[expected])
                            call_index = round_index * len(selected) + position
                            np.testing.assert_array_equal(state['search_frequency'][call_index], frequency[expected])
                            templates, data = state['search'][call_index]
                            self.assertEqual(data.shape, (3, len(expected)))
                            np.testing.assert_array_equal(round_group[f'blocks/{key}/screening/residual_input_xyz_fd'][:], data)
                            self.assertTrue(np.all((templates[:, 2] >= lower)
                                & ((templates[:, 2] < upper)
                                   | ((block_index == len(blocks) - 1) & (templates[:, 2] == upper)))))

    def test_direct_grid_has_no_crop_or_identity_indexing(self):
        source = (ROOT / 'eric_subtraction_all.py').read_text()
        tree = ast.parse(source)
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        self.assertFalse(names & {'center_bin', 'wide_bins', 'wide_frequency', 'core_indices',
                                 'round_residual_fd', 'total_xyz_fd', 'new_bases', 'ids'})
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                self.assertNotIn('core_indices', [keyword.arg for keyword in node.keywords])
        sampler = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name) and node.func.id == 'run_eryn_rjmcmc')
        self.assertEqual([ast.unparse(arg) for arg in sampler.args],
            ['data', 'covariance', 'filter_result', 'block_range',
             'frequency_derivative_bounds', 'T_obs', 'fs', 'checkpoints'])

    def test_empty_bank_skips_sampler_without_forcing_a_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory, catalogue_bins=())
            self.assertEqual(result['source_count'], 0)
            self.assertEqual(len(state['sampler']), 0)
            self.assertFalse(state['waveform'] or state['tdi'])
            with h5py.File(output) as saved:
                self.assertNotIn('catalogue_informed_bank', saved.attrs)
                np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_td'][:], original)

    def test_catalogue_predictions_cannot_trigger_an_empty_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory, bins=(), catalogue_bins=(10, 23))
            self.assertEqual(result['source_count'], 0)
            self.assertEqual(len(state['screen']), 1)
            self.assertEqual(len(state['sampler']), 0)
            self.assertFalse(state['waveform'] or state['tdi'])
            self.assertEqual(len(state['search']), 4)
            with h5py.File(output) as saved:
                self.assertEqual(saved['rounds/01'].attrs['visited_block_count'], 4)
                np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_td'][:], original)
    def test_rejected_fit_does_not_stop_other_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory, association_confirmed_counts={0: 0, 1: 1})
            self.assertEqual(result['source_count'], 1)
            self.assertEqual(len(state['tdi']), 1)
            with h5py.File(output) as saved:
                self.assertFalse(saved['rounds/01/blocks/00000'].attrs['accepted_for_subtraction'])
                self.assertTrue(saved['rounds/01/blocks/00001'].attrs['accepted_for_subtraction'])
                np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_td'][:], original - saved['final_subtraction/signal_xyz_td'][:])
    def test_all_rejected_fits_preserve_original(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory, association_confirmed_counts={0: 0, 1: 0})
            self.assertEqual(result['stopping_reason'], 'no_accepted_blocks')
            self.assertFalse(state['waveform'] or state['tdi'])
            with h5py.File(output) as saved:
                np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_td'][:], original)
    def test_sampler_zero_source_model_preserves_the_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory, empty_fit=True)
            self.assertEqual(result['source_count'], 0)
            self.assertFalse(state['tdi'] or state['waveform'])
            with h5py.File(output) as saved:
                np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_td'][:], original)
    def test_blocks_are_rechecked_after_the_global_residual_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, _, output = self.run_case(directory, wait_for_round=(23,))
            self.assertEqual(result['rounds'], 3)
            self.assertEqual(len(state['tdi']), 2)
            self.assertEqual(len(state['screen']), 3)
            with h5py.File(output) as saved:
                self.assertEqual(saved['rounds/01/selected_count'][1], 0)
                self.assertEqual(saved['rounds/01/accepted_count'][1], 0)
                self.assertEqual(saved['rounds/02/selected_count'][1], 1)
                self.assertEqual(saved['rounds/02/accepted_count'][1], 1)
                self.assertIn('00001', saved['blocks'])
    def test_requested_range_keeps_global_indices_and_skips_other_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory,
                subtraction_frequency_range=(.00015, .00025))
            self.assertEqual(result['selected_block_indices'], [1])
            self.assertEqual(result['total_block_count'], 4)
            self.assertEqual(result['source_count'], 1)
            self.assertEqual(len(state['sampler']), 1)
            np.testing.assert_allclose(state['sampler'][0][3],
                [7.60863249122299e-23, 10.**-14.8], rtol=1.e-15, atol=0.)
            np.testing.assert_allclose(state['sampler'][0][4], [.00015, .00025], rtol=0, atol=1e-18)
            np.testing.assert_array_equal(state['search'][0][0][:, 2], [23 / 128000.])
            np.testing.assert_array_equal(state['search'][0][0][:, 0], [2])
            with h5py.File(output) as saved:
                np.testing.assert_array_equal(saved['config/block_indices'][:], np.arange(4))
                np.testing.assert_array_equal(saved['config/selected_block_indices'][:], [1])
                np.testing.assert_array_equal(saved['config/subtraction_frequency_range'][:], [.00015, .00025])
                np.testing.assert_array_equal(saved['final_subtraction/block_index'][:], [1])
                self.assertEqual(list(saved['blocks']), ['00001'])
                self.assertEqual(saved['blocks/00001/block_index'][()], 1)
                np.testing.assert_array_equal(saved['blocks/00001/frequency_range'][:], saved['config/frequency_blocks_hz'][1])
                np.testing.assert_array_equal(saved['rounds/01/selected_count'][:], [0, 1, 0, 0])
                np.testing.assert_array_equal(saved['rounds/01/accepted_count'][:], [0, 1, 0, 0])
                for round_group in saved['rounds'].values():
                    self.assertNotIn('block_status', round_group)
                self.assertEqual(saved['rounds/01'].attrs['visited_block_count'], 1)
                self.assertEqual(saved['rounds/01'].attrs['selected_block_count'], 1)
                # The source in block 0 remains in the shared residual.
                outside = np.tile(np.sin(2 * np.pi * 10 * np.arange(128) / 128), (3, 1))
                np.testing.assert_allclose(saved['final_subtraction/residual_xyz_td'][:], outside, atol=1.e-14)
                np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_td'][:],
                    original - saved['final_subtraction/signal_xyz_td'][:])
    def test_exact_block_edges_do_not_select_floating_point_slivers(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, _, _ = self.run_case(directory, subtraction_frequency_range=(.00015, .00025))
            self.assertEqual(result['selected_block_indices'], [1])
            self.assertEqual(len(state['sampler']), 1)
    def test_source_range_is_lower_inclusive_and_upper_exclusive(self):
        bounds = (.00005, .00015)
        with tempfile.TemporaryDirectory() as directory:
            result, _, _, output = self.run_case(directory, subtraction_frequency_range=bounds)
            self.assertEqual(result['selected_block_indices'], [0])
            self.assertEqual(result['source_count'], 1)
            with h5py.File(output) as saved:
                frequencies = saved['final_subtraction/parameters'][:, 1]
                np.testing.assert_array_equal(frequencies, [10 / 128000.])
                np.testing.assert_array_equal(saved['final_subtraction/block_index'][:], [0])
                for group in saved['blocks'].values():
                    limits = group['frequency_range'][:]
                    self.assertTrue(limits[0] >= bounds[0] - 1e-18 and limits[1] <= bounds[1] + 1e-18)
    def test_changing_range_does_not_renumber_the_grid(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first, _, _, output_a = self.run_case(a, subtraction_frequency_range=(.00015, .00025))
            second, state, original, output_b = self.run_case(b, subtraction_frequency_range=(.00025, .00045))
            self.assertEqual(first['selected_block_indices'], [1])
            self.assertEqual(second['selected_block_indices'], [2, 3])
            self.assertEqual(len(state['sampler']), 0)
            self.assertTrue(all(table.shape == (0, 11) for table, *_ in state['sampler']))
            self.assertFalse(state['waveform'] or state['tdi'])
            with h5py.File(output_a) as left, h5py.File(output_b) as right:
                np.testing.assert_array_equal(left['config/frequency_blocks_hz'][:], right['config/frequency_blocks_hz'][:])
                np.testing.assert_array_equal(left['config/block_indices'][:], right['config/block_indices'][:])
                np.testing.assert_array_equal(right['final_subtraction/residual_xyz_td'][:], original)
    def test_default_and_explicit_full_range_select_all_blocks(self):
        for bounds in (None, (0., .0005)):
            with self.subTest(bounds=bounds), tempfile.TemporaryDirectory() as directory:
                result, _, _, output = self.run_case(directory, subtraction_frequency_range=bounds)
                self.assertEqual(result['selected_block_indices'], [0, 1, 2, 3])
                with h5py.File(output) as saved:
                    np.testing.assert_array_equal(saved['config/block_indices'][:], np.arange(4))
                    np.testing.assert_array_equal(saved['config/selected_block_indices'][:], np.arange(4))
    def test_eleven_column_coordinate_round_trips(self):
        tree = ast.parse((ROOT / 'eric_rjmcmc.py').read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('encode_doppler', 'decode_doppler')]
        scope = dict(np=np)
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'coordinates', 'exec'), scope)
        points = np.array([[.003, -2.e-15, 1., -.2], [.004, 0., 2., .1], [.005, 2.e-12, 3., .4]])
        table = np.arange(len(points) * 11, dtype=float).reshape(-1, 11)
        table[:, 0] = [2111, 3001, 29000000]
        table[:, [2, 3, 6, 7]] = points
        unchanged = table.copy()
        value = scope['encode_doppler'](table, .004, 86400., chirp_coordinate='linear')
        expected = np.column_stack((np.log10(points[:, 0] / .004) * .004 * 86400. * np.log(10.),
            points[:, 1] * 86400.**2, points[:, 2] % (2 * np.pi), np.sin(points[:, 3])))
        np.testing.assert_array_equal(value, expected)
        np.testing.assert_allclose(scope['decode_doppler'](value, .004, 86400., chirp_coordinate='linear'), points, rtol=1.e-14, atol=1.e-28)
        np.testing.assert_array_equal(table, unchanged)
        positive = points[2:]
        old = np.column_stack((np.log10(positive[:, 0] / .004) * .004 * 86400. * np.log(10.), np.log10(positive[:, 1]), positive[:, 2] % (2 * np.pi), np.sin(positive[:, 3])))
        np.testing.assert_array_equal(scope['encode_doppler'](table[2:], .004, 86400.), old)
        np.testing.assert_array_equal(table, unchanged)
        for coordinate in ('linear', 'log10'):
            empty = np.empty((0, 11))
            encoded = scope['encode_doppler'](empty, .004, 86400., chirp_coordinate=coordinate)
            self.assertEqual(encoded.shape, (0, 4))
            self.assertEqual(empty.shape, (0, 11))
            self.assertEqual(scope['decode_doppler'](encoded, .004, 86400., chirp_coordinate=coordinate).shape, (0, 4))
        encoder = next(node for node in functions if node.name == 'encode_doppler')
        self.assertEqual(ast.unparse(encoder.body[1]), 'para_doppler = para_doppler[:, [2, 3, 6, 7]]')
        self.assertFalse(any(isinstance(node, ast.If) for node in ast.walk(encoder)))


if __name__ == '__main__':
    unittest.main()

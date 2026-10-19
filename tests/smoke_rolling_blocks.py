"""Bounded single-A100 test: four real 30-day block fits, serial versus rolling.

Run on gpu1 in lisa_pipeline: python tests/smoke_rolling_blocks.py OUTPUT_DIR REFERENCE_DIR
Reuse the previous frozen 30-day fixture and its existing observation/SNR files.
No observation or catalogue is regenerated. Seeds and timing are test-only.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from functools import partial
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from tempfile import mkdtemp
from unittest.mock import patch
import hashlib
import json
import pickle
import sys
import time
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eric_subtraction_all as pipeline
from eric_subtraction_all import initialize_worker, single_block_process
from script.eric_Fstats import build_response, reconstruct_waveform, physical_amplitudes
from script.dwd_blind_search import F_statistic


def measured_initializer(inputs, shared_arrays, output):
    initialize_worker(inputs, shared_arrays)
    path = output / f'initialized_{os.getpid()}.json'
    assert not path.exists()
    path.write_text(json.dumps(dict(pid=os.getpid(), initialized_once=True)))


def measured_worker(block_index, dwd_template_table, *, work, label):
    started = time.time()
    arrays = (pipeline.lisa_data_fd, pipeline.effective_noise_covariance)
    snapshot = [hashlib.sha256(memoryview(value)).hexdigest() for value in arrays]
    table_snapshot = hashlib.sha256(memoryview(dwd_template_table)).hexdigest()
    seed = 20261013 + int(block_index)
    np.random.seed(seed)
    default_rng = np.random.default_rng
    with patch('numpy.random.default_rng', lambda *a, **kw: default_rng(seed) if not a and not kw else default_rng(*a, **kw)):
        result = single_block_process(block_index, dwd_template_table, work=work, label=label)
    assert snapshot == [hashlib.sha256(memoryview(value)).hexdigest() for value in arrays]
    assert table_snapshot == hashlib.sha256(memoryview(dwd_template_table)).hexdigest()
    timing = dict(block=int(block_index), pid=os.getpid(), start=started,
                  end=time.time(), seconds=time.time() - started)
    timing_file = (work.parent / f'controller_{label}_{block_index}.json' if label.isdigit()
                   else work / f'{label}_{block_index}.json')
    timing_file.write_text(json.dumps(timing))
    return result


if __name__ == '__main__':
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = Path(mkdtemp(prefix='four_blocks_', dir=output))
    reference = Path(sys.argv[2]).resolve()
    snapshot = next(reference.glob('four_blocks_*'))
    with (snapshot / 'shared.pkl').open('rb') as saved:
        (frequency_blocks, freq_block_width, fields, T_obs, fs, F_threshold,
         derivative_bounds, sample_number, maximum_sources, response_times, response_geometry) = pickle.load(saved)
    fft_frequency = np.load(snapshot / 'fft_frequency.npy')
    lisa_data_fd = np.vstack((fft_frequency, np.load(snapshot / 'residual_xyz_fd.npy')))
    effective_noise_covariance = np.load(snapshot / 'effective_noise_covariance.npy')
    dwd_template_table = np.load(snapshot / 'template_dwd.npy')
    inputs = (frequency_blocks, freq_block_width, fields, T_obs, fs, F_threshold,
        np.array([[-1.e-10, -1.e-25], [1.e-25, 1.e-10]]), sample_number, maximum_sources,
        response_times, response_geometry)
    report = dict(scope='Four isolated block fits on one frozen synthetic XYZ snapshot; 30 days, 0.1 Hz, 96 Eryn cycles per fit (32 burn-in), capacity 2, one A100. Fixed test-only seeds. Not a population subtraction or convergence test.',
        controller_sha256=hashlib.sha256((ROOT / 'eric_subtraction_all.py').read_bytes()).hexdigest(),
        work=str(work), runs={}, comparison=[])
    results = {}
    with ExitStack() as handles:
        shared_arrays = {}
        for name, value in [('lisa_data_fd', lisa_data_fd), ('effective_noise_covariance', effective_noise_covariance)]:
            memory = SharedMemory(create=True, size=value.nbytes)
            handles.callback(memory.unlink)
            handles.callback(memory.close)
            np.ndarray(value.shape, dtype=value.dtype, buffer=memory.buf)[:] = value
            shared_arrays[name] = (memory.name, value.shape, value.dtype)
        for label, workers in [('serial', 1), ('rolling', 3)]:
            started = time.perf_counter()
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'),
                    initializer=measured_initializer, initargs=(inputs, shared_arrays, output)) as executor:
                results[label] = list(executor.map(partial(measured_worker, work=work, label=label),
                    range(4), [dwd_template_table[i:i + 1] for i in range(4)], chunksize=1))
            report['runs'][label] = dict(wall_seconds=time.perf_counter() - started,
                jobs=[json.loads((work / f'{label}_{i}.json').read_text()) for i in range(4)])
        crowded_table = np.repeat(dwd_template_table[:1], 5, axis=0)
        started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=1, mp_context=get_context('spawn'),
                initializer=measured_initializer, initargs=(inputs, shared_arrays, output)) as executor:
            capacity_result = executor.submit(partial(measured_worker, work=work, label='crowded'), 0, crowded_table).result()
        capacity_seconds = time.perf_counter() - started
    for memory_name, _, _ in shared_arrays.values():
        try:
            leaked = SharedMemory(name=memory_name)
        except FileNotFoundError:
            continue
        leaked.close()
        raise AssertionError('shared memory was not unlinked')
    report['shared_memory_released'] = True
    assert not list(work.glob('*.npy')) and not (work / 'shared.pkl').exists()
    assert len(list(output.glob('initialized_*.json'))) == 5
    report['reusable_inputs_initialized_once_per_worker'] = True
    for i, ((physical, filtered, bins, posterior), (reference_physical, reference_filtered, reference_bins, reference_posterior)) in enumerate(zip(results['rolling'], results['serial'])):
        np.testing.assert_array_equal(bins, reference_bins)
        np.testing.assert_array_equal(filtered[:, 0], [20001 + i])
        np.testing.assert_array_equal(filtered, reference_filtered)
        assert filtered.shape[1] == 11
        np.testing.assert_allclose(physical, reference_physical, rtol=2.e-10, atol=1.e-30)
        assert physical.shape[1] == 8
        np.testing.assert_allclose(posterior['doppler'], reference_posterior['doppler'], rtol=2.e-10, atol=1.e-14)
        np.testing.assert_allclose(posterior['bases'], reference_posterior['bases'], rtol=2.e-10, atol=1.e-32)
        np.testing.assert_allclose(posterior['fit_result']['coefficients'], reference_posterior['fit_result']['coefficients'], rtol=2.e-10, atol=1.e-10)
        # Legacy generate-in-helper path; step 5 now consumes the step-4 values.
        from script.eric_generate_waveform import GPUResponse
        original_generator = GPUResponse(np.broadcast_to(np.eye(3), (len(bins), 3, 3)),
            np.zeros((3, len(bins)), complex), response_times, response_geometry, fft_frequency[bins], T_obs, fs)
        points = dwd_template_table[i:i + 1, [2, 3, 6, 7]]
        original_bases = original_generator.build(np.repeat(points, 128, axis=0))[:1]
        original = F_statistic(lisa_data_fd[1:, bins], original_bases,
            covariance=effective_noise_covariance[bins], T_obs=T_obs, fs=fs)
        np.testing.assert_array_equal(filtered, dwd_template_table[i:i + 1][original >= F_threshold])
        waveform = reconstruct_waveform(posterior['fit_result']['coefficients'], posterior['bases'])
        old_waveform = reconstruct_waveform(reference_posterior['fit_result']['coefficients'], reference_posterior['bases'])
        report['comparison'].append(dict(block=i, bins=len(bins), fitted_sources=len(posterior['doppler']),
            waveform_relative_l2=float(np.linalg.norm(waveform - old_waveform) / max(np.linalg.norm(old_waveform), 1.e-300)),
            filtered_table_equal=True, reference_threshold_selection_verified=True,
            doppler_max_absolute=float(np.max(abs(posterior['doppler'] - reference_posterior['doppler']))) if len(posterior['doppler']) else 0.))
    jobs = report['runs']['rolling']['jobs']
    events = sorted([(j['start'], 1) for j in jobs] + [(j['end'], -1) for j in jobs])
    report['maximum_active_jobs'] = int(np.max(np.cumsum([delta for _, delta in events])))
    assert report['maximum_active_jobs'] == 3, jobs
    assert jobs[3]['start'] < max(j['end'] for j in jobs[:3]), jobs
    report['fourth_started_before_all_first_three_finished'] = jobs[3]['start'] < max(j['end'] for j in jobs[:3])
    report['status'] = 'passed'
    (output / 'smoke.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)

    # Exercise the real guarded controller, common HDF5 output and TD/TDI barrier.
    # Two rounds is a test-only execution cap, never a production stopping rule.
    from test_subtraction_all import run_pipeline
    import eric_generate_waveform
    with h5py.File(reference / 'controller_observation.h5') as saved:
        detector_config = json.loads(saved['observation'].attrs['detector_config'])
        total_xyz_td = saved['observation/total_xyz_td'][:]
    native_tdi = eric_generate_waveform.tdi_gpu
    tdi_calls = []

    def checked_tdi(*args, **kwargs):
        tdi_calls.append(time.time())
        return native_tdi(*args, **kwargs)

    started = time.perf_counter()
    with patch('eric_generate_waveform.tdi_gpu', checked_tdi):
        result = run_pipeline(Path(detector_config['orbits']), reference / 'controller_snr.h5',
            reference / 'controller_observation.h5', output / 'controller_result.h5',
            frequency_range=(.003, .00304), freq_block_width=freq_block_width,
            sample_number=(32, 64), maximum_sources=2,
            single_block_process=measured_worker, _round_limit=2)
    controller = dict(wall_seconds=time.perf_counter() - started, result=result,
        tdi_calls=tdi_calls, scope='Real controller, native TD waveform/TDI, same four-block synthetic observation; at most two rounds (test-only cap). Not a convergence test.')
    with h5py.File(output / 'controller_result.h5') as saved:
        signal = saved['final_subtraction/signal_xyz_td'][:]
        residual = saved['final_subtraction/residual_xyz_td'][:]
        np.testing.assert_allclose(residual, total_xyz_td - signal, rtol=1.e-12, atol=1.e-35)
        np.testing.assert_array_equal(saved['final_subtraction/lisa_data_td'][0], np.arange(total_xyz_td.shape[-1]) / fs)
        np.testing.assert_array_equal(saved['final_subtraction/lisa_data_fd'][0].real, fft_frequency)
        assert saved['final_subtraction/parameters'].shape[1] == 8
        np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_fd'][:], np.fft.rfft(residual, axis=-1))
        controller['accepted_per_round'] = [group['accepted_count'][:].tolist() for group in saved['rounds'].values()]
        controller['residual_and_fft_identity'] = True
        for label, group in saved['rounds'].items():
            assert len(group['blocks']) == 4
            timings = [json.loads((output / f'controller_{label}_{i}.json').read_text()) for i in range(4)]
            if np.any(group['accepted_count'][:]):
                assert tdi_calls[int(label) - 1] > max(t['end'] for t in timings)
        assert tdi_calls, 'fixture should exercise the actual TD/TDI subtraction'
        controller['round_barrier_checked'] = True
    (output / 'controller_smoke.json').write_text(json.dumps(controller, indent=2) + '\n')
    print(json.dumps(controller, indent=2), flush=True)

    # A bank larger than the model capacity must remain available for births.
    physical, filtered, retained_bins, posterior = capacity_result
    assert len(filtered) == 5
    assert posterior['candidate_centers'].shape == (5, 4)
    assert posterior['config']['maximum_sources'] == 2
    assert posterior['config']['initial_count'] == 2
    assert max(posterior['chains']['0']['source_count']) <= 2
    capacity = dict(seconds=capacity_seconds, candidate_count=5,
        maximum_sources=2, final_sources=len(physical), all_centers_available=True,
        scope='One additional actual 96-cycle Eryn fit, five candidates and capacity two; not convergence.')
    (output / 'capacity_smoke.json').write_text(json.dumps(capacity, indent=2) + '\n')
    print(json.dumps(capacity, indent=2), flush=True)

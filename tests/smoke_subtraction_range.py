"""Native LISA/GPU smoke for fixed block IDs and a requested subtraction range.

Reuses the 30-day native fixture. Short sampler runs validate integration;
they do not assess convergence or annual subtraction throughput.
"""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
from pathlib import Path
import hashlib
import json
import sys
import time
import h5py
import numpy as np
import pytdi.michelson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eric_generate_waveform
import eric_rjmcmc
from test_subtraction_all import run_pipeline as run_subtraction

fixture = ROOT / 'tmp/subtraction_all_20261012/gpu1'
blind_fixture = ROOT / 'tmp/subtraction_blind_20261012'
output = ROOT / 'tmp/subtraction_range_20261012'
output.mkdir(parents=True, exist_ok=True)
report = dict(status='running', scope='One unlisted native DWD plus native noise, 30 days, selected global block IDs, 96-cycle integration smoke.',
    source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ('eric_subtraction_all.py', 'script/dwd_blind_search.py', 'eric_rjmcmc.py', 'eric_generate_waveform.py')})
with h5py.File(fixture / 'observation.h5', 'r') as saved:
    total = saved['observation/total_xyz_td'][:]
    truth = saved['observation/signal_xyz_td'][:]
sampler_calls, tdi_checks = [], []
native_sampler, native_tdi = eric_rjmcmc.run_eryn_rjmcmc, eric_generate_waveform.tdi_gpu


def sampler(*args, **kwargs):
    sampler_calls.append(dict(frequency_range_hz=np.asarray(args[3]).tolist(), initial_count=len(args[2])))
    return native_sampler(*args, **kwargs)


def checked_tdi(measurements, config, *, keep):
    actual = native_tdi(measurements, config, keep=keep)
    reference = np.stack([getattr(pytdi.michelson, axis + '2').build(**config)(measurements)[keep] for axis in 'XYZ'])
    identical = bool(np.array_equal(actual.view('u8'), reference.view('u8')))
    tdi_checks.append(dict(bitwise_identical=identical, max_absolute=float(np.max(abs(actual - reference)))))
    assert identical, tdi_checks[-1]
    return actual


eric_rjmcmc.run_eryn_rjmcmc, eric_generate_waveform.tdi_gpu = sampler, checked_tdi
try:
    np.random.seed(10212)
    started = time.perf_counter()
    excluded = run_subtraction(fixture / 'orbit.h5', blind_fixture / 'empty_snr.h5',
        fixture / 'observation.h5', output / 'excluded.h5', T_obs=30. * 86400., fs=.1,
        subtraction_frequency_range=(.002, .00203), sample_number=(32, 64), maximum_sources=2, maximum_rounds=3)
    excluded_seconds = time.perf_counter() - started
    assert excluded['source_count'] == 0 and not sampler_calls and not tdi_checks, excluded
    with h5py.File(output / 'excluded.h5', 'r') as saved:
        all_edges = saved['config/block_edges_hz'][:]
        all_indices = saved['config/block_indices'][:]
        selected = saved['config/selected_block_indices'][:]
        actual = saved['final_subtraction/residual_xyz_td'][:]
        status = saved['rounds/01/block_status'][:]
        excluded_visited = int(saved['rounds/01'].attrs['visited_block_count'])
    np.testing.assert_array_equal(selected, [199, 200, 201, 202])
    np.testing.assert_array_equal(actual, total)
    assert excluded_visited == len(selected)
    outside = np.ones(len(all_indices), bool)
    outside[selected] = False
    assert np.all(status[outside] == b'outside_subtraction_range')
    report['excluded'] = dict(seconds=excluded_seconds, result=excluded, visited_block_count=excluded_visited,
        outside_status_count=int(outside.sum()), observation_preserved_exactly=True)

    requested = np.array([.0030005, .003008])
    started = time.perf_counter()
    included = run_subtraction(fixture / 'orbit.h5', blind_fixture / 'empty_snr.h5',
        fixture / 'observation.h5', output / 'included.h5', T_obs=30. * 86400., fs=.1,
        subtraction_frequency_range=requested, sample_number=(32, 64), maximum_sources=2, maximum_rounds=3)
    included_seconds = time.perf_counter() - started
    with h5py.File(output / 'included.h5', 'r') as saved:
        included_edges = saved['config/block_edges_hz'][:]
        included_indices = saved['config/block_indices'][:]
        selected = saved['config/selected_block_indices'][:]
        ids = saved['final_subtraction/dwd_id'][:]
        blocks = saved['final_subtraction/block_index'][:]
        parameters = saved['final_subtraction/parameters'][:]
        signal = saved['final_subtraction/signal_xyz_td'][:]
        residual = saved['final_subtraction/residual_xyz_td'][:]
        residual_fd = saved['final_subtraction/residual_xyz_fd'][:]
        limits = saved['blocks/00300/search_frequency_range'][:]
        round_status = [group['block_status'][:] for group in saved['rounds'].values()]
        visited = [int(group.attrs['visited_block_count']) for group in saved['rounds'].values()]
    temporary_directory = included['temporary_directory']
    np.testing.assert_array_equal(included_edges, all_edges)
    np.testing.assert_array_equal(included_indices, all_indices)
    np.testing.assert_array_equal(selected, [300])
    np.testing.assert_array_equal(limits, requested)
    assert included['source_count'] == 1 and np.all(ids < 0), included
    assert np.all(blocks == 300) and np.all((parameters[:, 1] >= requested[0]) & (parameters[:, 1] < requested[1]))
    assert len(sampler_calls) == 1 and len(tdi_checks) == 1
    np.testing.assert_array_equal(sampler_calls[0]['frequency_range_hz'], requested)
    outside = np.arange(len(all_indices)) != 300
    assert all(np.all(status[outside] == b'outside_subtraction_range') for status in round_status)
    assert all(count == 1 for count in visited)
    np.testing.assert_array_equal(residual, total - signal)
    np.testing.assert_array_equal(residual_fd, np.fft.rfft(residual, axis=-1))
    error = float(np.linalg.norm(signal - truth) / np.linalg.norm(truth))
    assert error < .2, error
    assert not Path(temporary_directory).exists()
    report['included'] = dict(seconds=included_seconds, result=included, recovered_id=ids.tolist(),
        recovered_block=blocks.tolist(), fitted_frequency_hz=parameters[:, 1].tolist(),
        visited_block_count_per_round=visited, signal_relative_l2=error)
    report.update(status='passed', sampler_calls=sampler_calls, native_tdi_checks=tdi_checks)
except Exception as error:
    report.update(status='failed', error=repr(error))
    raise
finally:
    eric_rjmcmc.run_eryn_rjmcmc, eric_generate_waveform.tdi_gpu = native_sampler, native_tdi
    (output / 'scientific.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)

"""Real-data screen and unknown-source recovery using an EMPTY SNR catalogue.

Reuses the native LISA 30-day injection/noise fixture from the earlier smoke.
Run on CUDA from the repository; fixture truth is used only for assertions.
The 96-cycle sampler run tests integration, not convergence or completeness.
"""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
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
output = ROOT / 'tmp/subtraction_blind_20261012'
output.mkdir(parents=True, exist_ok=True)
report = dict(status='running', source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    for name in ('eric_subtraction_all.py', 'script/dwd_blind_search.py', 'eric_rjmcmc.py', 'eric_generate_waveform.py')})
try:
    T_obs, fs = 30. * 86400., .1
    with h5py.File(fixture / 'snr.h5') as saved:
        empty = saved['sky/igb/cat'][:0]
    with h5py.File(output / 'empty_snr.h5', 'w') as saved:
        saved.create_dataset('sky/dgb/cat', data=empty)
        saved.create_dataset('sky/igb/cat', data=empty)
        saved.create_dataset('settings/observation_duration_s', data=T_obs)
        saved.create_dataset('settings/detector_sampling_rate_hz', data=fs)
        saved.create_dataset('status/complete', data=True)
    with h5py.File(fixture / 'observation.h5') as saved:
        total = saved['observation/total_xyz_td'][:]
        truth = saved['observation/signal_xyz_td'][:]
        noise = saved['observation/noise_xyz_td'][:]
    with h5py.File(output / 'noise_only.h5', 'w') as saved:
        saved.create_dataset('observation/total_xyz_td', data=noise)
    sampler_calls, tdi_checks = [], []
    native_sampler, native_tdi = eric_rjmcmc.run_eryn_rjmcmc, eric_generate_waveform.tdi_gpu

    def sampler(*args, **kwargs):
        sampler_calls.append(dict(frequency_range=np.asarray(args[3]).tolist(),
            initial_count=len(args[2])))
        return native_sampler(*args, **kwargs)

    def checked_tdi(measurements, config, *, keep):
        actual = native_tdi(measurements, config, keep=keep)
        reference = np.stack([getattr(pytdi.michelson, axis + '2').build(**config)(measurements)[keep] for axis in 'XYZ'])
        identical = bool(np.array_equal(actual.view('u8'), reference.view('u8')))
        tdi_checks.append(dict(bitwise_identical=identical, max_absolute=float(np.max(abs(actual - reference)))))
        assert identical, tdi_checks[-1]
        return actual

    eric_rjmcmc.run_eryn_rjmcmc, eric_generate_waveform.tdi_gpu = sampler, checked_tdi
    np.random.seed(10212)
    started = time.perf_counter()
    noise_result = run_subtraction(fixture / 'orbit.h5', output / 'empty_snr.h5',
        output / 'noise_only.h5', output / 'noise_result.h5', T_obs=T_obs, fs=fs,
        sample_number=(32, 64), maximum_sources=2, maximum_rounds=3)
    report['noise_seconds'] = time.perf_counter() - started
    assert not sampler_calls and not tdi_checks, 'Noise-only fixture unexpectedly fitted; investigate screen.'
    assert noise_result['source_count'] == 0, noise_result
    with h5py.File(output / 'noise_result.h5') as saved:
        report['noise_blocks_screened'] = int(saved['rounds/01'].attrs['data_screened_block_count'])
        report['noise_blocks_below_spectral_threshold'] = int(saved['rounds/01'].attrs['spectral_rejected_block_count'])
        assert report['noise_blocks_screened'] == 5000
        np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_td'][:], noise)
    started = time.perf_counter()
    signal_result = run_subtraction(fixture / 'orbit.h5', output / 'empty_snr.h5',
        fixture / 'observation.h5', output / 'unknown_result.h5', T_obs=T_obs, fs=fs,
        sample_number=(32, 64), maximum_sources=2, maximum_rounds=3)
    report['unknown_source_seconds'] = time.perf_counter() - started
    with h5py.File(output / 'unknown_result.h5') as saved:
        signal = saved['final_subtraction/signal_xyz_td'][:]
        residual = saved['final_subtraction/residual_xyz_td'][:]
        ids = saved['final_subtraction/dwd_id'][:]
        assert signal_result['source_count'] == 1 and np.all(ids < 0), signal_result
        np.testing.assert_array_equal(residual, total - signal)
        np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_fd'][:], np.fft.rfft(residual, axis=-1))
        report['unknown_ids'] = ids.tolist()
        report['unknown_signal_relative_l2'] = float(np.linalg.norm(signal - truth) / np.linalg.norm(truth))
        report['unknown_round_screens'] = [dict(round=name,
            blocks=int(group.attrs['data_screened_block_count']),
            spectral_rejected=int(group.attrs['spectral_rejected_block_count']),
            accepted_blocks=int(group.attrs['accepted_block_count'])) for name, group in saved['rounds'].items()]
        assert all(item['blocks'] == 5000 for item in report['unknown_round_screens'])
        assert report['unknown_signal_relative_l2'] < .2, report['unknown_signal_relative_l2']
        assert not Path(signal_result['temporary_directory']).exists()
    report.update(status='passed', noise_result=noise_result, unknown_result=signal_result,
        sampler_calls=sampler_calls, native_tdi_checks=tdi_checks,
        scope='One unlisted injected DWD plus native noise, 30 days, 5000 bands, 96 sampler cycles; not an exhaustive or converged search.')
finally:
    eric_rjmcmc.run_eryn_rjmcmc, eric_generate_waveform.tdi_gpu = native_sampler, native_tdi
    (output / 'scientific.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)

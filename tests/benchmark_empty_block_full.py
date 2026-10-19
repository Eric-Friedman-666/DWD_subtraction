"""Time blind F statistics and unconditional RJMCMC on native LISA noise.

Run on gpu1 from the repository. This is one 30-day block, not an annual
throughput or convergence test. No catalogue or spectral gate is consulted.
"""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
from pathlib import Path
from tempfile import mkdtemp
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
from eric_rjmcmc import run_eryn_rjmcmc
from script.dwd_blind_search import amplitude_f_statistic, blind_f_statistic_search
from script.eric_Fstats import build_response, noise_inner_product, reconstruct_waveform
from script.eric_preprocess import effective_noise_cov, periodogram

output = ROOT / 'tmp/subtraction_empty_full_20261012'
output.mkdir(parents=True, exist_ok=True)
work = Path(mkdtemp(prefix='run_', dir=output))
report = dict(status='running', scope='One native-noise-only 30-day frequency block; no catalogue or spectral gate; unconditional RJMCMC.',
    run_directory=str(work), source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ('eric_subtraction_all.py', 'script/dwd_blind_search.py', 'eric_rjmcmc.py', 'script/eric_generate_waveform.py')})
started = time.perf_counter()
try:
    T_obs, fs, width = 30. * 86400., .1, 1.e-5
    derivative_bounds = np.array([-3.e-12, 3.e-12])
    block_index = 300
    block_range = 1 / T_obs + np.array([block_index, block_index + 1]) * width
    center = float(block_range.mean())
    with h5py.File(ROOT / 'tmp/subtraction_blind_20261012/noise_only.h5', 'r') as saved:
        noise = saved['observation/total_xyz_td'][:]
    assert noise.shape == (3, int(round(T_obs * fs)))
    fft_frequency = np.fft.rfftfreq(noise.shape[-1], d=1 / fs)
    noise_fd = np.fft.rfft(noise, axis=-1)
    padding = block_range[1] * 1.e-4 + max(abs(derivative_bounds)) * T_obs + 8 / T_obs
    retained_bins = np.flatnonzero((fft_frequency >= block_range[0] - padding)
        & (fft_frequency <= block_range[1] + padding) & (fft_frequency > 0))
    center_bin = int(round(center * T_obs))
    wide_bins = np.arange(max(1, min(center_bin - 2048, retained_bins[0] - 256)),
        min(len(fft_frequency), max(center_bin + 2049, retained_bins[-1] + 257)))
    core_indices = retained_bins - wide_bins[0]
    physical_frequency, wide_frequency = fft_frequency[retained_bins], fft_frequency[wide_bins]
    data = noise_fd[:, retained_bins]
    report.update(duration_days=T_obs / 86400, sampling_rate_hz=fs, block_index=block_index,
        frequency_range_hz=block_range.tolist(), derivative_bounds_hz_per_s=derivative_bounds.tolist(),
        retained_fourier_bins=len(retained_bins), waveform_fourier_bins=len(wide_bins),
        input_and_fft_seconds=time.perf_counter() - started,
        iterations=2000, burn_in=500, production_cycles=1500, maximum_sources=49,
        inverse_temperatures=[1., .8, .6, .4], spectral_gate_used=False, sampler_gate_used=False)

    stage = time.perf_counter()
    covariance = effective_noise_cov(periodogram(noise_fd[:, wide_bins], T_obs, fs),
        neighboring_bins=256, downweight_peaks=True)[core_indices].copy()
    scale = np.trace(covariance, axis1=-2, axis2=-1).real / 3
    covariance += max(float(np.median(scale)) * 1.e-10, 1.e-100) * np.eye(3)
    report['covariance_seconds'] = time.perf_counter() - stage

    stage = time.perf_counter()
    builder, response_times, response_geometry = build_response(work, wide_frequency, T_obs, fs,
        orbit_file=ROOT / 'tmp/subtraction_all_20261012/gpu1/orbit.h5')
    report['geometry_seconds'] = time.perf_counter() - stage
    print('Searching every half-bin carrier, 32 sky points and 9 signed chirps, without a spectral gate.', flush=True)
    stage = time.perf_counter()
    # All retained frequencies are supplied: none are selected by power or truth.
    search = blind_f_statistic_search(data, covariance, builder, core_indices, T_obs, fs,
        block_range, derivative_bounds, physical_frequency,
        f_threshold=16., snr_threshold=7., maximum_sources=49, sky_points=32, chirp_points=9)
    report['f_statistic_seconds'] = time.perf_counter() - stage
    report['f_statistic'] = dict(initial_count=len(search['doppler']),
        maximum_two_f=float(search['maximum_two_f']), finite_grid_trials=int(search['finite_grid_trials']),
        coarse_evaluations=int(search['coarse_evaluations']), refinement_evaluations=int(search['refinement_evaluations']),
        retained_two_f=search['two_f'].tolist())
    (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(f_statistic_seconds=report['f_statistic_seconds'], search=report['f_statistic'])), flush=True)

    np.random.seed(10212)
    stage = time.perf_counter()
    # Unlike the controller, this benchmark runs the sampler even without seeds.
    posterior = run_eryn_rjmcmc(data, covariance, search['doppler'], block_range,
        derivative_bounds, center, T_obs, fs, work / 'checkpoints',
        chain_count=1, inverse_temperatures=(1., .8, .6, .4),
        iterations=2000, burn_in=500, maximum_sources=49, chirp_coordinate='linear',
        response_times=response_times, response_geometry=response_geometry,
        physical_frequency=physical_frequency)
    report['rjmcmc_seconds'] = time.perf_counter() - stage
    chain = posterior['chains']['0']
    nonzero = np.flatnonzero(posterior['count_probability'])
    report['posterior'] = dict(modal_count=int(posterior['modal_count']),
        zero_source_probability=float(posterior['count_probability'][0]),
        count_probability={str(int(n)): float(posterior['count_probability'][n]) for n in nonzero},
        minimum_replica_count=int(chain['trace_counts'].min()), maximum_replica_count=int(chain['trace_counts'].max()),
        cold_zero_cycles=int(np.count_nonzero(chain['source_count'] == 0)),
        waveform_evaluations=int(posterior['waveform_evaluations']), likelihood_calls=int(posterior['likelihood_calls']),
        backend_setup_seconds=float(posterior['backend_setup_seconds']), gpu_device_kind=posterior['gpu_device_kind'],
        independent_chain_convergence_assessed=False)
    doppler = posterior['doppler']
    coefficients = posterior['fit_result']['coefficients']
    bases = builder.build(doppler)[:, :, :, core_indices]
    sources = reconstruct_waveform(coefficients, bases)
    inverse = np.linalg.inv(covariance)
    snr = np.sqrt(np.maximum(noise_inner_product(sources, sources, inverse, T_obs, fs), 0.))
    conditional_f = np.zeros(len(doppler))
    for i in range(len(doppler)):
        conditional_f[i] = amplitude_f_statistic(data - sources.sum(axis=0) + sources[i],
            bases[i:i + 1], covariance, T_obs, fs)['two_f'][0] / 2
    report['fitted_noise_sources'] = dict(snr=snr.tolist(), conditional_f=conditional_f.tolist(),
        passing_snr7_and_f16=int(np.count_nonzero((snr >= 7.) & (conditional_f >= 16.))))
    with h5py.File(work / 'posterior.h5', 'w') as saved:
        saved.create_dataset('count_probability', data=posterior['count_probability'])
        saved.create_dataset('trace_counts', data=chain['trace_counts'])
        saved.create_dataset('production_counts', data=chain['source_count'])
        saved.create_dataset('doppler', data=doppler)
        saved.create_dataset('coefficients', data=coefficients)
        saved.create_dataset('f_statistic_initial_doppler', data=search['doppler'])
        saved.attrs.update(complete=True, spectral_gate_used=False, sampler_gate_used=False,
            iterations=2000, burn_in=500, duration=T_obs, rate=fs)
    report.update(status='passed', total_seconds=time.perf_counter() - started)
except Exception as error:
    report.update(status='failed', error=repr(error), total_seconds=time.perf_counter() - started)
    raise
finally:
    (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)

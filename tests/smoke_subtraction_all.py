"""One injected frequency block, all-band rounds, real Eryn and native LISA on CUDA.

Run from the repository on gpu1/gpu2:
  python tests/smoke_subtraction_all.py tmp/subtraction_all_20261012
The fixture contains one negative-chirp DWD over 30 days at 0.1 Hz. Production
inputs are not altered. The short chain tests integration, not convergence.
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
import lisaorbits
import lisagwresponse
import lisainstrument
import numpy as np
import pytdi
import pytdi.michelson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import eric_generate_waveform
import eric_rjmcmc
from test_subtraction_all import run_pipeline as run_subtraction
from script.eric_preprocess import effective_noise_cov, periodogram, inner_product

OUT = ROOT / (sys.argv[1] if len(sys.argv) > 1 else 'tmp/subtraction_all_20261012')
host = os.uname().nodename
work = OUT / host
work.mkdir(parents=True, exist_ok=True)
report = dict(status='running', host=host, source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    for name in ('eric_subtraction_all.py', 'eric_rjmcmc.py', 'eric_generate_waveform.py')})
try:
    T_obs, fs = 30. * 86400., .1
    N, N_day = int(round(T_obs * fs)), int(round(86400 * fs))
    orbit = work / 'orbit.h5'
    lisaorbits.KeplerianOrbits(L=2.5e9).write(str(orbit), t0=-20e6, dt=86400., size=825, mode='w')
    config = dict(T_obs=T_obs, dt=1 / fs, orbits=str(orbit), orbit_dataset='tcb/ltt',
        include_padding=True, gpu_batch_size=10000, minimum_envelope_samples=512,
        aafilter=('kaiser', 240., .275 * fs, .725 * fs),
        clock_offsets=0., clock_freqoffsets=0., clock_freqlindrifts=0., clock_freqquaddrifts=0.)
    parameters = np.array([[8.e-22, .00300317, -2.e-17, .47, .91, 1.2, -.3, .62]])
    links = eric_generate_waveform.DWD_simulation_gpu(parameters, config)
    response = lisagwresponse.GalacticBinary(A=1., f=.003, ra=0., dec=0.,
        orbits=str(orbit), orbit_interp_order=5, shift_sun='never')
    with h5py.File(work / 'injected_links.h5', 'w') as saved:
        response.init_response_file(saved, 1 / fs, len(links), -2 * N_day / fs, 'tcb')
        saved['tcb/y'][:] = links
    del links
    native_settings = {name: value for name, value in config.items()
        if name not in ('T_obs', 'include_padding', 'gpu_batch_size', 'minimum_envelope_samples')}
    signal_instrument = lisainstrument.Instrument(**native_settings, gws=str(work / 'injected_links.h5'),
        size=N + 2 * N_day, t0=-N_day / fs)
    signal_instrument.disable_all_noises()
    signal_instrument.disable_dopplers()
    signal_data = pytdi.Data.from_instrument(signal_instrument.export_numpy_full())
    noise_instrument = lisainstrument.Instrument(**native_settings, size=N + 2 * N_day,
        t0=-N_day / fs, testmass_asds=3.e-15, oms_asds=(15.e-12, 0., 0., 0., 0., 0.))
    noise_instrument.disable_all_noises(excluding={'test-mass', 'oms'})
    noise_instrument.disable_dopplers()
    noise_data = pytdi.Data.from_instrument(noise_instrument.export_numpy_full())
    keep = slice(N_day, N_day + N)
    native_tdi = [getattr(pytdi.michelson, axis + '2').build(**signal_data.args) for axis in 'XYZ']
    injected = np.stack([operator(signal_data.measurements)[keep] for operator in native_tdi]) / signal_instrument.central_freq
    noise = np.stack([operator(noise_data.measurements)[keep] for operator in native_tdi]) / signal_instrument.central_freq
    observation = injected + noise
    del signal_data, noise_data, signal_instrument, noise_instrument, native_tdi
    frequencies = np.fft.rfftfreq(N, d=1 / fs)
    center = int(round(parameters[0, 1] * T_obs))
    bins = np.arange(max(1, center - 2048), min(len(frequencies), center + 2049))
    covariance = effective_noise_cov(periodogram(np.fft.rfft(noise, axis=-1)[:, bins], T_obs, fs),
                                    neighboring_bins=256, downweight_peaks=True)
    signal_fd = np.fft.rfft(injected, axis=-1)[:, bins]
    absolute_snr = float(np.sqrt(inner_product(signal_fd, signal_fd, covariance, T_obs, fs)))
    assert absolute_snr > 7., absolute_snr
    fields = ('Amplitude', 'Frequency', 'FrequencyDerivative', 'InitialPhase', 'Inclination',
              'EclipticLongitude', 'EclipticLatitude', 'Polarization', 'SNR_absolute', 'SNR_relative')
    table = np.zeros((1, 1), dtype=[(name, 'f8') for name in fields])
    for column, name in enumerate(fields[:8]):
        table[name][:, 0] = parameters[:, column]
    table['SNR_absolute'][0, 0] = absolute_snr
    table['SNR_relative'][0, 0] = absolute_snr
    with h5py.File(work / 'snr.h5', 'w') as saved:
        saved.create_dataset('sky/dgb/cat', data=table[:0])
        saved.create_dataset('sky/igb/cat', data=table)
        saved.create_dataset('settings/observation_duration_s', data=T_obs)
        saved.create_dataset('settings/detector_sampling_rate_hz', data=fs)
        saved.create_dataset('status/complete', data=True)
    with h5py.File(work / 'observation.h5', 'w') as saved:
        saved.create_dataset('observation/total_xyz_td', data=observation)
        saved.create_dataset('observation/signal_xyz_td', data=injected)
        saved.create_dataset('observation/noise_xyz_td', data=noise)
    original_tdi, original_sampler = eric_generate_waveform.tdi_gpu, eric_rjmcmc.run_eryn_rjmcmc
    tdi_comparisons, sampler_calls = [], []

    def checked_tdi(measurements, args, *, keep):
        actual = original_tdi(measurements, args, keep=keep)
        reference = np.stack([getattr(pytdi.michelson, axis + '2').build(**args)(measurements)[keep] for axis in 'XYZ'])
        identical = bool(np.array_equal(actual.view('u8'), reference.view('u8')))
        tdi_comparisons.append(dict(shape=list(actual.shape), bitwise_identical=identical,
            max_absolute=float(np.max(abs(actual - reference)))))
        assert identical, tdi_comparisons[-1]
        return actual

    def checked_sampler(*args, **kwargs):
        assert kwargs['chirp_coordinate'] == 'linear'
        sampler_calls.append(dict(frequency_range=np.asarray(args[3]).tolist(), initial_count=len(args[2]),
            derivative_bounds=np.asarray(args[4]).tolist()))
        return original_sampler(*args, **kwargs)

    eric_generate_waveform.tdi_gpu = checked_tdi
    eric_rjmcmc.run_eryn_rjmcmc = checked_sampler
    np.random.seed(10212)
    started = time.perf_counter()
    result = run_subtraction(orbit, work / 'snr.h5', work / 'observation.h5', work / 'subtraction.h5',
        T_obs=T_obs, fs=fs, sample_number=(32, 64), maximum_sources=2, maximum_rounds=3)
    report['subtraction_seconds'] = time.perf_counter() - started
    report.update(result=result, native_tdi_comparisons=tdi_comparisons, sampler_calls=sampler_calls,
                  injected_absolute_snr=absolute_snr, injected_parameters=parameters.tolist())
    eric_generate_waveform.tdi_gpu, eric_rjmcmc.run_eryn_rjmcmc = original_tdi, original_sampler
    assert result['source_count'] == 1 and result['rounds'] == 2, result
    assert result['stopping_reason'] == 'no_new_candidates'
    assert len(sampler_calls) >= 1 and len(tdi_comparisons) == 1
    with h5py.File(work / 'subtraction.h5', 'r') as saved:
        assert saved.attrs['complete'] and saved.attrs['single_file_complete']
        retained_signal = saved['final_subtraction/signal_xyz_td'][:]
        residual = saved['final_subtraction/residual_xyz_td'][:]
        assert np.array_equal(residual, observation - retained_signal)
        assert np.array_equal(saved['final_subtraction/residual_xyz_fd'][:], np.fft.rfft(residual, axis=-1))
        assert np.array_equal(saved['final_subtraction/dwd_id'][:], [1])
        for group in saved['rounds'].values():
            statuses = group['block_status'][:]
            assert len(statuses) == 5000
            assert group.attrs['visited_block_count'] == 5000
            assert group.attrs['data_screened_block_count'] == 5000
            assert group.attrs['spectral_rejected_block_count'] >= 4990
        report['block_count'] = 5000
        report['spectral_rejected_blocks_per_round'] = [int(group.attrs['spectral_rejected_block_count'])
            for group in saved['rounds'].values()]
        report['negative_chirp_recovered'] = bool(saved['final_subtraction/parameters'][0, 2] < 0)
        # A short chain is allowed to estimate a small positive derivative; record the value.
        report['fitted_chirp'] = float(saved['final_subtraction/parameters'][0, 2])
        report['residual_identity'] = True
        report['fft_identity'] = True
        difference = retained_signal - injected
        report['signal_relative_l2'] = float(np.linalg.norm(difference) / np.linalg.norm(injected))
        report['band_power_before'] = float(np.sum(abs(np.fft.rfft(observation, axis=-1)[:, bins])**2))
        report['band_power_after'] = float(np.sum(abs(np.fft.rfft(residual, axis=-1)[:, bins])**2))
        assert report['band_power_after'] < report['band_power_before']
        assert not Path(result['temporary_directory']).exists()
    report['scope'] = 'Real 96-cycle Eryn integration, one injected DWD, 30 days; convergence not assessed.'
    report['status'] = 'passed'
except Exception as error:
    report.update(status='failed', error=repr(error))
    raise
finally:
    (OUT / (host + '.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)

"""Source-specific isolated native TD/TDI controller smoke (900 or 3600 s).

Only this test's new synthetic files are read. Independent lisagwresponse links
feed a real instrument/TDI observation. The unmodified controller uses the real
sampler, unchanged source-specific dispatcher, winner-only refit, round exclusions and native
combined subtraction. Thirty days/four injected sources is NOT full-year native
accuracy, posterior convergence, blind discovery or 64-concurrent-fit proof.
The original strict >95% per-source native gate is retained, not relaxed.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('DWD_WORKER_GPU_MEM_FRACTION', '.045')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from datetime import datetime, timezone
import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'tests')]
from native_smoke_audit import json_attribute, checked_winner_refit, strict_native_quality_passed


def save(path, value):
    temporary = Path(str(path)+'.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def audited_worker(index, table, *, work, label):
    import numpy as np
    import jax
    import eric_subtraction_all as pipeline
    arrays = (pipeline.lisa_data_fd, pipeline.effective_noise_covariance)
    before = [hashlib.sha256(memoryview(value)).hexdigest() for value in arrays]
    assert (table[:, 10] > 7.).all(), 'Fixed saved SNR selection changed'
    from script.dwd_chirp_policy import frequency_dependent_chirp_bounds, frequency_dependent_padding
    blocks, width, _, duration, rate = pipeline.worker_inputs[:5]
    cap = np.array([[-4.e-12, -7.60863249122299e-23], [7.60863249122299e-23, 4.e-12]])
    expected_bounds = frequency_dependent_chirp_bounds(blocks, cap)
    expected_padding = frequency_dependent_padding(duration, width, blocks, expected_bounds)
    np.testing.assert_array_equal(pipeline.worker_inputs[6], expected_bounds)
    np.testing.assert_array_equal(pipeline.worker_inputs[12], expected_padding)
    full_frequency = pipeline.lisa_data_fd[0].real
    padding = float(expected_padding[index])
    expected_bins = np.flatnonzero((full_frequency >= blocks[index, 0]-padding)
        & (full_frequency <= blocks[index, 1]+padding) & (full_frequency > 0))
    sampler_calls = []
    actual_sampler = pipeline.run_eryn_rjmcmc
    def checked_sampler(data, covariance, filtered, block_range, bounds, T_obs, fs, checkpoints, **settings):
        np.testing.assert_array_equal(bounds, expected_bounds[index])
        np.testing.assert_array_equal(block_range, blocks[index])
        np.testing.assert_array_equal(settings['physical_frequency'], full_frequency[expected_bins])
        assert settings['maximum_sources'] == 100 and settings['iterations'] == 2000
        assert settings['burn_in'] == 500 and settings['amplitude_prior_sigma'] == .1
        sampler_calls.append(dict(derivative_bounds=np.asarray(bounds).tolist(),
            retained_bins_count=len(expected_bins), padding_hz=padding))
        return actual_sampler(data, covariance, filtered, block_range, bounds, T_obs, fs, checkpoints, **settings)
    seed = 944001+int(index)+10000*int(label)
    np.random.seed(seed)
    default_rng = np.random.default_rng
    with patch('numpy.random.default_rng', lambda *a, **kw:
            default_rng(seed) if not a and not kw else default_rng(*a, **kw)), \
            patch.object(pipeline, 'run_eryn_rjmcmc', checked_sampler):
        result = pipeline.single_block_process(index, table, work=work, label=label)
    np.testing.assert_array_equal(result[2], expected_bins)
    after = [hashlib.sha256(memoryview(value)).hexdigest() for value in arrays]
    assert before == after, 'Worker changed shared round residual/covariance'
    audit = dict(block=int(index), round=label, pid=os.getpid(),
        gpu=os.environ.get('DWD_ASSIGNED_GPU', os.environ.get('CUDA_VISIBLE_DEVICES')),
        gpu_slot=(int(os.environ['DWD_GPU_SLOT_INDEX']) if 'DWD_GPU_SLOT_INDEX' in os.environ else None),
        assignment_kind='stable_supervisor_slot' if 'DWD_ASSIGNED_GPU' in os.environ else 'legacy_single_device',
        gpu_memory_fraction_observed=os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'),
        input_ids=table[:, 0].astype(int).tolist(), input_count=len(table),
        screened_count=len(result[1]), shared_snapshot_hashes=before,
        per_bank_derivative_bounds=expected_bounds[index].tolist(),
        per_bank_padding_hz=padding, retained_bins_count=len(expected_bins),
        actual_sampler_call_audits=sampler_calls,
        per_bank_policy_forwarded_to_actual_sampler=True)
    if result[3] is not None:
        assert len(sampler_calls) == 1
        fit = result[3]
        audit.update(initial_count=int(fit['config']['initial_count']),
            outside_prior=int(fit['config']['initial_centers_outside_prior']),
            finite_committed_log_priors=bool(np.isfinite(fit['chains']['0']['cycle_state']['log_prior']).all()),
            maximum_sources=int(fit['config']['maximum_sources']),
            memory_stats=jax.devices('gpu')[0].memory_stats())
    assert len(jax.devices('gpu')) == 1
    if 'DWD_ASSIGNED_GPU' in os.environ:
        assert os.environ['CUDA_VISIBLE_DEVICES'] == os.environ['DWD_ASSIGNED_GPU']
    else:
        assert len(os.environ['DWD_GPU_DEVICES'].split(',')) == 1
        assert os.environ['CUDA_VISIBLE_DEVICES'] == os.environ['DWD_GPU_DEVICES']
    save(Path(work)/f'worker_audit_{label}_{int(index):05d}.json', audit)
    return result


def make_fixture(directory):
    import h5py
    import numpy as np
    import lisaorbits
    import lisagwresponse
    from lisagwresponse.response import LINKS
    import lisainstrument
    import pytdi
    from eric_generate_waveform import tdi_gpu
    from script.tool import _bme_to_icrs
    duration, rate = 30.*86400., .1
    count, day = round(duration*rate), 8640
    orbit = directory/'orbits.h5'
    lisaorbits.KeplerianOrbits(L=2.5e9).write(str(orbit), t0=-20e6, dt=86400., size=825, mode='w')
    table = np.zeros((6, 11))
    table[:, 0] = np.arange(992001, 992007)
    table[:, 1] = 8.e-22
    table[:, 2] = [.00300317, .00900317, .01200317, .04296793987485368, .02100317, .02200317]
    table[:, 3] = [2.e-17, -2.698721932368033e-15, 3.e-14, 2.758292489484066e-12, 1.e-14, 1.e-14]
    table[:, 4:9] = [.47, .91, 1.2, -.3, .62]
    table[:, 9:] = 20.
    table[-2:, 10] = [7., 6.]
    config = dict(T_obs=duration, dt=1/rate, orbits=str(orbit), orbit_dataset='tcb/ltt',
        include_padding=True, gpu_batch_size=128, minimum_envelope_samples=512,
        aafilter=('kaiser', 240., .275*rate, .725*rate), clock_offsets=0.,
        clock_freqoffsets=0., clock_freqlindrifts=0., clock_freqquaddrifts=0.)
    ra, dec, psi = _bme_to_icrs(table[:, 6], table[:, 7], table[:, 8])
    padded_count = count+4*day
    links = np.zeros((padded_count, 6))
    for n, row in enumerate(table[:4]):
        model = lisagwresponse.GalacticBinary(A=row[1], f=row[2], df=row[3], phi0=row[4],
            iota=row[5], ra=ra[n], dec=dec[n], psi=psi[n], t_init=0.,
            orbits=str(orbit), orbit_interp_order=5, shift_sun='never')
        for first in range(0, padded_count, 65536):
            last = min(first+65536, padded_count)
            links[first:last] += model.compute_gw_response((np.arange(first, last)-2*day)/rate, LINKS)
    link_file = directory/'independent_native_links.h5'
    with h5py.File(link_file, 'w') as saved:
        model.init_response_file(saved, 1/rate, len(links), -2*day/rate, 'tcb')
        saved['tcb/y'][:] = links
    settings = {key: value for key, value in config.items()
        if key not in ('T_obs', 'include_padding', 'gpu_batch_size', 'minimum_envelope_samples')}
    instrument = lisainstrument.Instrument(**settings, gws=str(link_file), size=count+2*day, t0=-day/rate)
    instrument.disable_all_noises()
    instrument.disable_dopplers()
    data = pytdi.Data.from_instrument(instrument.export_numpy_full())
    signal = tdi_gpu(data.measurements, data.args, keep=slice(day, day+count))/instrument.central_freq
    del data, instrument, links
    np.random.seed(944001)
    instrument = lisainstrument.Instrument(**settings, size=count+2*day, t0=-day/rate,
        testmass_asds=3.e-15, oms_asds=(15.e-12, 0., 0., 0., 0., 0.))
    instrument.disable_all_noises(excluding={'test-mass', 'oms'})
    instrument.disable_dopplers()
    data = pytdi.Data.from_instrument(instrument.export_numpy_full())
    noise = tdi_gpu(data.measurements, data.args, keep=slice(day, day+count))/instrument.central_freq
    observation = signal+noise
    with h5py.File(directory/'observation.h5', 'w') as saved:
        group = saved.create_group('observation')
        group.attrs.update(duration=duration, rate=rate, detector_config=json.dumps(config))
        group.create_dataset('total_xyz_td', data=observation)
    with h5py.File(directory/'catalogue.h5', 'w') as saved:
        saved.create_dataset('catalogue', data=table)
    return orbit, table, signal, noise, observation, duration, rate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--grid-seconds', type=int, choices=(900, 3600), required=True)
    parser.add_argument('--device-count', type=int, choices=(1, 4), required=True)
    args = parser.parse_args()
    if args.source_root.resolve() != ROOT:
        raise SystemExit('Run the smoke script inside the intended frozen source tree')
    if (args.grid_seconds, args.device_count) not in ((900, 4), (3600, 1)):
        raise SystemExit('Require server3600s/oneGPU or OzSTAR900s/fourGPUs')
    if os.uname().sysname != 'Linux':
        raise SystemExit('No native scientific tests on macOS')
    if (not os.environ.get('SLURM_JOB_ID') or
            os.environ.get('SLURM_JOB_NUM_NODES', os.environ.get('SLURM_NNODES')) != '1'):
        raise SystemExit('Candidate native smoke requires a NEW one-node Slurm allocation; never use running server gpu1')
    allocated = tuple(device.strip() for device in os.environ.get(
        'DWD_NATIVE_ALLOCATED_GPU_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES', '')).split(',')
        if device.strip())
    if len(allocated) != 4 or len(set(allocated)) != 4:
        raise SystemExit('A complete admitted four-GPU allocation must be recorded before selecting a subset')
    devices = tuple(device.strip() for device in os.environ.get('DWD_GPU_DEVICES',
        os.environ.get('CUDA_VISIBLE_DEVICES', '')).split(',') if device.strip())
    assert len(devices) == args.device_count and len(set(devices)) == args.device_count, devices
    assert set(devices).issubset(allocated), (devices, allocated)
    directory = args.directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    os.environ['DWD_GPU_DEVICES'] = ','.join(devices)
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(devices)
    os.environ['DWD_NATIVE_ALLOCATED_GPU_DEVICES'] = ','.join(allocated)
    os.environ['DWD_BLOCK_WORKERS'] = '16'
    report = dict(status='running', operational_status='pending', native_quality_gate='pending',
        scope=__doc__, started_utc=datetime.now(timezone.utc).isoformat(),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), devices=list(devices),
        response_step_seconds=args.grid_seconds, workers_per_gpu=16,
        configured_workers=16*args.device_count, source_root=str(ROOT),
        admitted_allocation_devices=list(allocated), selected_allocation_subset=list(devices),
        hardware_validation='OzSTAR_not_server_GPU1' if args.grid_seconds == 3600 else 'OzSTAR_isolated_four_GPU_allocation',
        server_driver_or_sixteen_way_capacity_proved=False,
        scientific_settings=dict(F_threshold=16., amplitude_prior_sigma=.1,
            burn_in=500, production_cycles=1500, maximum_sources=100, fixed_saved_snr_strictly_above=7.),
        known_previous_300s_native_gate=dict(status='failed', frequency_hz=.04296793987485368,
            power_removed_percent=94.06140412376406, required_strictly_greater_than_percent=95., waived=False),
        preserved_r2_ozstar_smoke=dict(job_id='18803948', status='failed_posthoc_schema_check',
            native_quality_gate='pending_not_measured', frozen_results_unchanged=True),
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ('eric_subtraction_all.py', 'eric_rjmcmc.py', 'script/dwd_worker_supervisor.py',
                         'script/dwd_frequency_support.py', 'script/dwd_reference_cache.py',
                         'script/dwd_chirp_policy.py',
                         'tests/native_smoke_audit.py')})
    save(directory/'smoke.json', report)
    try:
        import numpy as np
        import h5py
        import eric_generate_waveform
        from test_subtraction_all import run_pipeline
        from script.dwd_frequency_support import chirp_frequency_padding, RESPONSE_STEP_SECONDS
        from script.dwd_chirp_policy import (
            frequency_dependent_chirp_bounds, frequency_dependent_padding, bank_configuration_record)
        import script.dwd_reference_cache as cache_module
        assert RESPONSE_STEP_SECONDS == args.grid_seconds
        inputs = directory/'fixture'
        inputs.mkdir()
        orbit, table, signal, noise, observation, duration, rate = make_fixture(inputs)
        print('NATIVE_FIXTURE_COMPLETE', flush=True)
        output = directory/'subtraction.h5'
        native_tdi, native_waveform = eric_generate_waveform.tdi_gpu, eric_generate_waveform.DWD_simulation_gpu
        barriers, waveform_barriers = [], []
        actual_cache, cache_calls = cache_module.precompute_reference_waveforms, []
        def checked_cache(cache_file, selected_blocks, template_tables, frequency_blocks, block_width,
                frequency, duration, rate, response_times, response_geometry, response_type,
                amplitude_coordinates, reconstruct_waveform, **settings):
            cap = np.array([[-4.e-12, -7.60863249122299e-23], [7.60863249122299e-23, 4.e-12]])
            expected_bounds = frequency_dependent_chirp_bounds(frequency_blocks, cap)
            expected_padding = frequency_dependent_padding(duration, block_width, frequency_blocks, expected_bounds)
            np.testing.assert_array_equal(settings['chirp_bounds_by_block'], expected_bounds)
            np.testing.assert_array_equal(settings['frequency_padding_hz'], expected_padding)
            result = actual_cache(cache_file, selected_blocks, template_tables, frequency_blocks,
                block_width, frequency, duration, rate, response_times, response_geometry,
                response_type, amplitude_coordinates, reconstruct_waveform, **settings)
            banks = []
            for index, rows in zip(selected_blocks, template_tables):
                if not len(rows):
                    assert int(index) not in result
                    continue
                padding = float(expected_padding[index])
                expected_bins = np.flatnonzero((frequency >= frequency_blocks[index, 0]-padding)
                    & (frequency <= frequency_blocks[index, 1]+padding) & (frequency > 0))
                np.testing.assert_array_equal(result[int(index)]['retained_bins'], expected_bins)
                np.testing.assert_array_equal(result[int(index)]['dwd_id'], rows[:, 0].astype(int))
                banks.append(dict(block=int(index), derivative_bounds=expected_bounds[index].tolist(),
                    padding_hz=padding, retained_bins_count=len(expected_bins), input_count=len(rows)))
            cache_calls.append(dict(actual_per_bank_arrays_verified=True, banks=banks))
            return result
        def checked_waveform(*args, **kwargs):
            assert not multiprocessing.active_children(), 'Native synthesis raced live workers'
            waveform_barriers.append(time.time())
            return native_waveform(*args, **kwargs)
        def checked_tdi(*args, **kwargs):
            assert not multiprocessing.active_children(), 'Combined subtraction raced live workers'
            barriers.append(time.time())
            return native_tdi(*args, **kwargs)
        with patch('eric_generate_waveform.tdi_gpu', checked_tdi), \
                patch('eric_generate_waveform.DWD_simulation_gpu', checked_waveform), \
                patch.object(cache_module, 'precompute_reference_waveforms', checked_cache):
            result = run_pipeline(orbit, inputs/'catalogue.h5', inputs/'observation.h5', output,
                single_block_process=audited_worker, block_processes=16,
                gpu_devices=devices, _round_limit=3)
        assert not multiprocessing.active_children()
        status = json.loads(output.with_suffix('.status.json').read_text())
        assert status['complete'] and status['configured_workers'] == 16*args.device_count
        with h5py.File(output, 'r') as saved:
            assert saved.attrs['complete']
            ids = saved['final_subtraction/recovered_dwd_id'][:]
            model = saved['final_subtraction/signal_xyz_td'][:]
            residual = saved['final_subtraction/residual_xyz_td'][:]
            np.testing.assert_allclose(residual, observation-model, rtol=1.e-12, atol=1.e-35)
            np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_fd'][:], np.fft.rfft(residual, axis=-1))
            for key, value in (('maximum_sources', 100), ('amplitude_prior_sigma', .1),
                    ('response_step_seconds', args.grid_seconds), ('relative_snr_threshold', 7.), ('f_threshold', 16.)):
                assert saved['config/'+key][()] == value
            np.testing.assert_array_equal(saved['config/sample_number'][:], [500, 1500])
            blocks = saved['config/frequency_blocks_hz'][:]
            bounds = frequency_dependent_chirp_bounds(blocks, saved['config/derivative_bounds'][:])
            pads = frequency_dependent_padding(duration, 1.e-5, blocks, bounds)
            np.testing.assert_array_equal(saved['config/derivative_bounds_by_block'][:], bounds)
            np.testing.assert_array_equal(saved['config/likelihood_frequency_padding_hz'][:], pads)
            expected_bank_record = bank_configuration_record(duration, 1.e-5, blocks,
                saved['config/derivative_bounds'][:], bounds, pads)
            assert json_attribute(saved['config'], 'chirp_bounds_policy_sha256') == expected_bank_record['policy_sha256']
            refits = []
            for round_group in saved['rounds'].values():
                if 'amplitude_refits' not in round_group:
                    continue
                for bank, audit in round_group['amplitude_refits'].items():
                    refits.append(checked_winner_refit(audit))
        np.testing.assert_array_equal(ids, table[:4, 0].astype(int))
        atomic_bank_settings = json.loads((directory/'frequency_banks.settings.json').read_text())
        assert all(atomic_bank_settings[key] == value for key, value in expected_bank_record.items())
        assert status['chirp_bounds_policy_sha256'] == expected_bank_record['policy_sha256']
        assert status['finalized_identified_dwd_count'] == 4
        assert status['finalized_round_counts'].get('02') == 0 and len(barriers) == len(waveform_barriers) == 1
        assert refits and not status.get('worker_recovery_events')
        work = output.with_suffix('.checkpoints')
        audits = [json.loads(path.read_text()) for path in work.glob('worker_audit_*.json')]
        assert audits and all(row['round'] == '01' for row in audits)
        assert len({tuple(row['shared_snapshot_hashes']) for row in audits}) == 1
        assert len(cache_calls) == 1 and len(cache_calls[0]['banks']) == 4
        cache_by_block = {row['block']: row for row in cache_calls[0]['banks']}
        for row in audits:
            cached = cache_by_block[row['block']]
            assert row['per_bank_derivative_bounds'] == cached['derivative_bounds']
            assert row['per_bank_padding_hz'] == cached['padding_hz']
            assert row['retained_bins_count'] == cached['retained_bins_count']
        assert all(row.get('outside_prior', 0) == 0 and row.get('finite_committed_log_priors', True)
            for row in audits)
        assert not list(work.glob('round_02_block_*')), 'Finalized IDs relaunched in next round'
        cache = json.loads((work/'reference_waveforms.status.json').read_text())
        assert cache['complete'] and cache['reference_sources'] == 4 and cache['loaded_blocks'] == 0
        frequency = np.fft.rfftfreq(signal.shape[-1], 1/rate)
        original_fd, remaining_fd = np.fft.rfft(signal, axis=-1), np.fft.rfft(residual-noise, axis=-1)
        bounds = np.array([[-4.e-12, -7.60863249122299e-23], [7.60863249122299e-23, 4.e-12]])
        padding = chirp_frequency_padding(duration, 1.e-5, .05, bounds)
        rows = []
        for row in table[:4]:
            lower = .0001+np.floor((row[2]-.0001)/1.e-5)*1.e-5
            keep = (frequency >= lower-padding) & (frequency <= lower+1.e-5+padding)
            fraction = float(np.sum(abs(remaining_fd[:, keep])**2)/np.sum(abs(original_fd[:, keep])**2))
            rows.append(dict(id=int(row[0]), frequency_hz=float(row[2]),
                identified=bool(int(row[0]) in ids), native_foreground_power_removed_percent=100*(1-fraction)))
        report.update(operational_status='passed', result=result,
            finalized_round_counts=status['finalized_round_counts'], per_source=rows,
            identified_ids=ids.astype(int).tolist(), worker_audits=audits, amplitude_refits=refits,
            cache=cache, subtraction_barriers=len(barriers), native_waveform_barriers=len(waveform_barriers),
            per_bank_cache_call_audits=cache_calls, per_bank_policy_verified_through_actual_calls=True,
            atomic_bank_settings_verified=True,
            atomic_bank_settings_policy_sha256=expected_bank_record['policy_sha256'],
            all_workers_reaped_before_native_synthesis=True, native_residual_identity=True,
            native_rfft_identity=True, strict_snr7_exclusions=True, finalized_ids_excluded_next_round=True,
            native_quality_evaluation_band='fixed_global_reference_support_not_narrow_candidate_window',
            whole_native_foreground_error_fraction=float(np.sum((residual-noise)**2)/np.sum(signal**2)))
        quality_passed = strict_native_quality_passed(rows)
        report.update(native_quality_gate='passed' if quality_passed else 'failed',
            required_native_power_removed_percent_strictly_greater_than=95.,
            status='passed' if quality_passed else 'failed_native_quality')
        save(directory/'smoke.json', report)
        if not quality_passed:
            raise RuntimeError('Operational integration passed but retained native >95% quality gate failed')
    except BaseException as error:
        report.update(error=repr(error), traceback=traceback.format_exc()[-12000:])
        if report['operational_status'] != 'passed':
            report.update(status='failed', operational_status='failed')
        raise
    finally:
        report['completed_utc'] = datetime.now(timezone.utc).isoformat()
        save(directory/'smoke.json', report)
        print('OZSTAR_NATIVE_SMOKE_FINAL', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

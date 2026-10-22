# Step 1. Import packages.
from pathlib import Path
from contextlib import ExitStack
from functools import partial
from time import perf_counter
from multiprocessing import current_process
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.util import Finalize
import json
import os
import sys
import pickle
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import h5py
import numpy as np
from tqdm.auto import tqdm
import lisagwresponse
import lisainstrument
import pytdi
from eric_generate_waveform import DWD_simulation_gpu, tdi_gpu
from script.eric_generate_waveform import GPUResponse
from script.eric_Fstats import (
    build_response, physical_amplitude_coordinates,
    physical_amplitudes, reconstruct_waveform,
)
from script.dwd_blind_search import F_statistic
from script.eric_preprocess import effective_noise_cov, periodogram, inner_product
from script.dwd_bank_association import assign_bank_ids
from script.dwd_amplitude_refit import joint_amplitude_only_refit
from script.dwd_reference_cache import precompute_reference_waveforms
from script.dwd_run_status import write_run_status
from script.dwd_worker_supervisor import run_block_jobs
from script.dwd_frequency_support import chirp_frequency_padding, RESPONSE_STEP_SECONDS
from script.dwd_chirp_policy import (
    frequency_dependent_chirp_bounds, frequency_dependent_padding,
    chirp_bounds_for_bank, bank_padding_values, bank_configuration_record,
)
from eric_rjmcmc import run_eryn_rjmcmc
from script.tool import _save_tree
def initialize_worker(inputs, shared_arrays, gpu_devices=None):
    global worker_inputs, lisa_data_fd, effective_noise_covariance
    if gpu_devices is not None:
        assigned_gpu = os.environ.get('DWD_ASSIGNED_GPU')
        worker_index = int(os.environ.get('DWD_WORKER_SLOT_INDEX',
            current_process()._identity[-1] - 1))
        os.environ['CUDA_VISIBLE_DEVICES'] = (assigned_gpu if assigned_gpu is not None
            else gpu_devices[worker_index % len(gpu_devices)])
        if hasattr(os, 'sched_getaffinity'):
            cpus = sorted(os.sched_getaffinity(0))
            os.sched_setaffinity(0, {cpus[worker_index % len(cpus)]})
        print(f'Worker {os.getpid()} GPU {os.environ["CUDA_VISIBLE_DEVICES"]}', flush=True)
    worker_inputs = inputs
    lisa_data_fd_memory = SharedMemory(name=shared_arrays['lisa_data_fd'][0])
    Finalize(None, lisa_data_fd_memory.close, exitpriority=10)
    lisa_data_fd = np.ndarray(shared_arrays['lisa_data_fd'][1], dtype=shared_arrays['lisa_data_fd'][2],
        buffer=lisa_data_fd_memory.buf)
    effective_noise_covariance_memory = SharedMemory(name=shared_arrays['effective_noise_covariance'][0])
    Finalize(None, effective_noise_covariance_memory.close, exitpriority=10)
    effective_noise_covariance = np.ndarray(shared_arrays['effective_noise_covariance'][1],
        dtype=shared_arrays['effective_noise_covariance'][2], buffer=effective_noise_covariance_memory.buf)
def single_block_process(block_index, dwd_template_table, *, work, label):
    from script.dwd_worker_supervisor import worker_progress
    worker_progress('preparing_bank', force=True, input_sources=int(len(dwd_template_table)))
    checkpoints = work / f'round_{label}_block_{block_index:05d}'
    checkpoints.mkdir(parents=True, exist_ok=True)
    result_file = checkpoints / 'result.pkl'
    if result_file.exists():
        with result_file.open('rb') as saved:
            return pickle.load(saved)
    print(f'Round {label} block {block_index:05d} worker {os.getpid()} GPU {os.environ.get("CUDA_VISIBLE_DEVICES", "0")}', flush=True)
    # Step 4. Select the block and calculate its template waveforms.
    (frequency_blocks, freq_block_width, fields, T_obs, fs, F_threshold,
     frequency_derivative_bounds, sample_number, maximum_sources,
     response_times, response_geometry, amplitude_prior_sigma, frequency_padding_hz) = worker_inputs
    block_range = frequency_blocks[block_index]
    frequency_derivative_bounds = chirp_bounds_for_bank(
        frequency_derivative_bounds, block_index, len(frequency_blocks))
    padding = bank_padding_values(frequency_padding_hz, freq_block_width,
        len(frequency_blocks))[block_index]
    retained_bins = np.flatnonzero((lisa_data_fd[0].real >= block_range[0] - padding)
        & (lisa_data_fd[0].real <= block_range[1] + padding) & (lisa_data_fd[0].real > 0))
    physical_frequency = lisa_data_fd[0, retained_bins].real
    covariance = effective_noise_covariance[retained_bins]
    data = lisa_data_fd[1:, retained_bins]
    response = GPUResponse(np.broadcast_to(np.eye(3), (len(physical_frequency), 3, 3)),
        np.zeros((3, len(physical_frequency)), complex), response_times, response_geometry, physical_frequency, T_obs, fs)
    waveform_table_fd = np.empty((len(dwd_template_table), 4, 3, len(retained_bins)), dtype=complex)
    for first in range(0, len(dwd_template_table), 128):
        worker_progress('waveform_basis', first=int(first), total=int(len(dwd_template_table)))
        points = dwd_template_table[first:first + 128, [2, 3, 6, 7]]
        padded = np.vstack((points, np.repeat(points[-1:], 128 - len(points), axis=0)))
        waveform_table_fd[first:first + len(points)] = response.build(padded)[:len(points)]
    # Step 5. Filter the calculated templates by F-statistic.
    worker_progress('f_screening', force=True)
    F_values = F_statistic(data, waveform_table_fd, covariance=covariance, T_obs=T_obs, fs=fs)
    retained = F_values >= F_threshold
    filter_result = dwd_template_table[retained]
    template_number = len(filter_result)
    del waveform_table_fd, response, F_values
    if template_number == 0:
        with (checkpoints / 'result.partial').open('wb') as saved:
            pickle.dump((np.empty((0, 8)), filter_result, retained_bins, None), saved, protocol=5)
        (checkpoints / 'result.partial').replace(result_file)
        return np.empty((0, 8)), filter_result, retained_bins, None
    # Step 6. Fit the candidate bank and convert recovered physical parameters.
    worker_progress('sampler_setup', force=True, candidates=int(template_number))
    posterior = run_eryn_rjmcmc(data, covariance, filter_result, block_range,
        frequency_derivative_bounds, T_obs, fs, checkpoints,
        iterations=sum(sample_number), burn_in=sample_number[0],
        maximum_sources=maximum_sources, amplitude_prior_sigma=amplitude_prior_sigma,
        response_times=response_times, response_geometry=response_geometry, physical_frequency=physical_frequency)
    physical = physical_amplitudes(posterior['fit_result']['coefficients'])
    dwd_recover_table = np.column_stack((physical['A'], posterior['doppler'][:, 0], posterior['doppler'][:, 1],
        physical['phi0'], physical['iota'], posterior['doppler'][:, 2], posterior['doppler'][:, 3], physical['psi_bme']))
    worker_progress('saving_result', force=True)
    with (checkpoints / 'result.partial').open('wb') as saved:
        pickle.dump((dwd_recover_table, filter_result, retained_bins, posterior), saved, protocol=5)
    (checkpoints / 'result.partial').replace(result_file)
    return dwd_recover_table, filter_result, retained_bins, posterior
if __name__ == '__main__':
    # Step 2. Configure the subtraction.
    orbit_file = Path('./data/dwd_observation_full_catalogue.orbits.h5')
    snr_file = Path('./data/ldc_dwd_data.h5')
    observation_file = Path('./data/dwd_observation_full_catalogue.h5')
    output_file = Path(os.environ.get('DWD_SUBTRACTION_OUTPUT', './data/dwd_subtraction_all.h5'))
    frequency_range = (1.e-4, .05)
    freq_block_width = 1.e-5
    frequency_derivative_bounds = np.array([[-4.e-12, -7.60863249122299e-23],
                                          [7.60863249122299e-23, 4.e-12]])
    SNR_r_threshold = 7.
    F_threshold = 16.
    maximum_sources = 100
    amplitude_prior_sigma = .1
    refit_identified_amplitudes = True
    sample_number = (500, 1500)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    status_file = output_file.with_suffix('.status.json')
    run_status = dict(phase='loading_observation', current_round=None, last_completed_round=0,
        complete=False, relative_snr_threshold=SNR_r_threshold,
        block_processes_per_gpu=int(os.environ.get('DWD_BLOCK_WORKERS', '3')),
        worker_progress_timeout_s=float(os.environ.get('DWD_WORKER_PROGRESS_TIMEOUT', '600')),
        worker_max_attempts=int(os.environ.get('DWD_WORKER_MAX_ATTEMPTS', '3')),
        worker_gpu_memory_fraction=float(os.environ.get('DWD_WORKER_GPU_MEM_FRACTION', '.045')),
        gpu_devices=tuple(os.environ.get('DWD_GPU_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES', '0')).split(',')),
        worker_execution='supervised_single_use_spawn',
        finalized_identified_dwd_count=0, finalized_round_counts={})
    run_status['configured_workers'] = run_status['block_processes_per_gpu'] * len(run_status['gpu_devices'])
    write_run_status(status_file, **run_status)
    with h5py.File(observation_file, 'r') as saved:
        T_obs = saved['observation'].attrs['duration']
        fs = saved['observation'].attrs['rate']
        detector_config = json.loads(saved['observation'].attrs['detector_config'])
    detector_config['orbits'] = str(orbit_file)
    # Step 3. Read the observation and prepare shared inputs.
    N = int(round(T_obs * fs))
    with h5py.File(observation_file, 'r') as saved:
        lisa_data_td = saved['observation/total_xyz_td'][:]
    lisa_data_fd = np.vstack((np.fft.rfftfreq(N, d=1 / fs), np.fft.rfft(lisa_data_td, axis=-1)))
    with h5py.File(snr_file, 'r') as saved:
        dwd_template_table = saved['catalogue'][saved['catalogue'][:, 10] > SNR_r_threshold]
    frequency_blocks = np.arange(frequency_range[0], frequency_range[1] - freq_block_width * 1.e-10, freq_block_width)[:, None] + np.array([0., freq_block_width])
    frequency_blocks[-1, 1] = frequency_range[1]
    block_count = len(frequency_blocks)
    block_indices = np.arange(block_count, dtype=np.int64)
    fields = dict(A=1, f=2, df=3, phi0=4, iota=5, longitude=6, latitude=7, psi_bme=8)
    block_range = frequency_blocks[block_indices[0]]
    frequency_derivative_bounds_by_block = frequency_dependent_chirp_bounds(
        frequency_blocks, frequency_derivative_bounds)
    frequency_padding_hz = frequency_dependent_padding(
        T_obs, freq_block_width, frequency_blocks, frequency_derivative_bounds_by_block)
    bank_configuration = bank_configuration_record(T_obs, freq_block_width,
        frequency_blocks, frequency_derivative_bounds,
        frequency_derivative_bounds_by_block, frequency_padding_hz)
    write_run_status(output_file.parent / 'frequency_banks.settings.json',
        **bank_configuration)
    run_status.update(chirp_bounds_policy=bank_configuration['policy']['name'],
        chirp_bounds_policy_sha256=bank_configuration['policy_sha256'])
    padding = frequency_padding_hz[block_indices[0]]
    retained_bins = np.flatnonzero((lisa_data_fd[0].real >= block_range[0] - padding)
        & (lisa_data_fd[0].real <= block_range[1] + padding) & (lisa_data_fd[0].real > 0))
    _, response_times, response_geometry = build_response(
        output_file.parent, lisa_data_fd[0, retained_bins].real, T_obs, fs, orbit_file=orbit_file)
    configuration = dict(duration=T_obs, rate=fs, frequency_range=frequency_range,
        frequency_blocks_hz=frequency_blocks, block_indices=block_indices, frequency_block_width=freq_block_width,
        subtraction_frequency_range=np.array(frequency_range), selected_block_indices=block_indices,
        derivative_bounds=frequency_derivative_bounds,
        derivative_bounds_by_block=frequency_derivative_bounds_by_block,
        chirp_bounds_policy=bank_configuration['policy'],
        chirp_bounds_policy_sha256=bank_configuration['policy_sha256'],
        chirp_coordinate='linear', f_threshold=F_threshold,
        chirp_prior='positive_log_uniform' if frequency_derivative_bounds.ndim == 1 else 'signed_log_uniform',
        likelihood_frequency_padding_hz=frequency_padding_hz,
        response_step_seconds=RESPONSE_STEP_SECONDS,
        relative_snr_threshold=SNR_r_threshold, sample_number=sample_number, maximum_sources=maximum_sources,
        amplitude_prior_sigma=amplitude_prior_sigma, block_processes_per_gpu=run_status['block_processes_per_gpu'],
        worker_gpu_memory_fraction=run_status['worker_gpu_memory_fraction'],
        amplitude_refit='joint_nonnegative_winner_only_fixed_shape_scale' if refit_identified_amplitudes else 'none',
        gpu_devices=','.join(run_status['gpu_devices']), strong_reporting_snr_threshold=7.,
        relative_snr_source='fixed_saved_catalogue', identification_template_scope='remaining_same_block_input_bank', subtraction_policy='identified_only', exclude_recovered_templates=True)
    with h5py.File(output_file, 'a') as saved:
        if saved.attrs.get('complete', False):
            print(f'Already complete: {output_file}', flush=True)
            sys.exit(0)
        if 'config' not in saved:
            _save_tree(saved.create_group('config'), configuration)
            saved.create_group('rounds')
    # Step 7. Fit all blocks, keep identified DWDs, subtract once, and repeat.
    work = output_file.with_suffix('.checkpoints').resolve()
    work.mkdir(parents=True, exist_ok=True)
    run_status.update(phase='reference_cache', selected_block_count=block_count)
    write_run_status(status_file, **run_status)
    reference_template_tables = [dwd_template_table[
        (dwd_template_table[:, fields['f']] >= frequency_blocks[index, 0]) & (
        (dwd_template_table[:, fields['f']] < frequency_blocks[index, 1]) |
        ((index == block_count - 1) & (dwd_template_table[:, fields['f']] == frequency_blocks[index, 1])))]
        for index in block_indices]
    reference_waveform_cache = precompute_reference_waveforms(work / 'reference_waveforms.h5',
        block_indices, reference_template_tables, frequency_blocks, freq_block_width,
        lisa_data_fd[0].real, T_obs, fs, response_times, response_geometry,
        GPUResponse, physical_amplitude_coordinates, reconstruct_waveform,
        frequency_padding_hz=frequency_padding_hz,
        chirp_bounds_by_block=frequency_derivative_bounds_by_block)
    del reference_template_tables
    accepted_parameters, accepted_ids, accepted_blocks = [], [], []
    with h5py.File(output_file, 'r') as saved:
        round_number = int(saved.attrs.get('last_completed_round', 0))
        for number in range(1, round_number + 1):
            group = saved[f'rounds/{number:02d}']
            accepted_parameters.append(group['parameters'][:])
            accepted_ids.append(group['dwd_id'][:])
            accepted_blocks.append(group['block_index'][:])
            if len(group['parameters']):
                lisa_data_td -= group['signal_xyz_td'][:]
    lisa_data_fd[1:] = np.fft.rfft(lisa_data_td, axis=-1)
    recovered_ids = np.concatenate(accepted_ids) if accepted_ids else np.empty(0, dtype=np.int64)
    run_status.update(last_completed_round=round_number,
        finalized_identified_dwd_count=int(len(np.unique(recovered_ids))))
    dwd_template_table = dwd_template_table[~np.isin(dwd_template_table[:, 0], recovered_ids)]
    neighboring_bins, chunk_size = 256, 16384
    with ExitStack() as shared_memory:
        lisa_data_fd_memory = SharedMemory(create=True, size=lisa_data_fd.nbytes)
        shared_memory.callback(lisa_data_fd_memory.unlink)
        shared_memory.callback(lisa_data_fd_memory.close)
        shared_memory.callback(globals().pop, 'lisa_data_fd', None)
        lisa_data_fd_memory.buf[:lisa_data_fd.nbytes] = memoryview(lisa_data_fd).cast('B')
        lisa_data_fd = np.ndarray(lisa_data_fd.shape, dtype=lisa_data_fd.dtype, buffer=lisa_data_fd_memory.buf)
        effective_noise_covariance_memory = SharedMemory(create=True, size=lisa_data_fd.shape[1] * 9 * np.dtype(complex).itemsize)
        shared_memory.callback(effective_noise_covariance_memory.unlink)
        shared_memory.callback(effective_noise_covariance_memory.close)
        shared_memory.callback(globals().pop, 'effective_noise_covariance', None)
        effective_noise_covariance = np.ndarray((lisa_data_fd.shape[1], 3, 3), dtype=complex,
            buffer=effective_noise_covariance_memory.buf)
        worker_inputs = (frequency_blocks, freq_block_width, fields, T_obs, fs, F_threshold,
            frequency_derivative_bounds_by_block, sample_number, maximum_sources,
            response_times, response_geometry, amplitude_prior_sigma, frequency_padding_hz)
        shared_arrays = dict(lisa_data_fd=(lisa_data_fd_memory.name, lisa_data_fd.shape, lisa_data_fd.dtype),
            effective_noise_covariance=(effective_noise_covariance_memory.name,
                effective_noise_covariance.shape, effective_noise_covariance.dtype))
        worker_count = run_status['block_processes_per_gpu'] * len(run_status['gpu_devices'])
        while True:
            round_start = perf_counter()
            round_number += 1
            label = f'{round_number:02d}'
            print(f'Round {label}', flush=True)
            run_status.update(phase='covariance', current_round=round_number,
                nonempty_input_block_indices=None, empty_input_block_count=None)
            write_run_status(status_file, **run_status)
            for first in tqdm(range(1, lisa_data_fd.shape[1], chunk_size), desc='Covariance', unit='chunk'):
                last = min(first + chunk_size, lisa_data_fd.shape[1])
                lower, upper = max(1, first - neighboring_bins), min(lisa_data_fd.shape[1], last + neighboring_bins)
                psd = periodogram(lisa_data_fd[1:, lower:upper], T_obs, fs)
                local = effective_noise_cov(psd, neighboring_bins=neighboring_bins, downweight_peaks=True)
                scale = np.trace(local, axis1=-2, axis2=-1).real / 3
                floor = max(float(np.median(scale)) * 1.e-10, 1.e-100)
                local += floor * np.eye(3)
                effective_noise_covariance[first:last] = local[first - lower:last - lower]
            effective_noise_covariance[0] = effective_noise_covariance[1]
            block_template_tables = [dwd_template_table[
                (dwd_template_table[:, fields['f']] >= frequency_blocks[index, 0]) & (
                (dwd_template_table[:, fields['f']] < frequency_blocks[index, 1]) |
                ((index == block_count - 1) & (dwd_template_table[:, fields['f']] == frequency_blocks[index, 1])))]
                for index in block_indices]
            nonempty_indices = [int(index) for index, table in zip(block_indices, block_template_tables) if len(table)]
            run_status.update(phase='fitting_blocks', nonempty_input_block_indices=nonempty_indices,
                empty_input_block_count=block_count - len(nonempty_indices))
            write_run_status(status_file, **run_status)
            block_jobs = ((index, table) for index, table in zip(block_indices, block_template_tables)
                if len(table))
            function = partial(single_block_process, work=work, label=label)
            run_block_jobs(function, block_jobs, max_workers=worker_count,
                initializer=initialize_worker, initargs=(worker_inputs, shared_arrays, run_status['gpu_devices']),
                work=work, label=label, progress_timeout=run_status['worker_progress_timeout_s'],
                max_attempts=run_status['worker_max_attempts'], startup_ready_phase='f_screening',
                status_file=status_file, run_status=run_status,
                gpu_devices=run_status['gpu_devices'], workers_per_gpu=run_status['block_processes_per_gpu'],
                worker_gpu_memory_fraction=run_status['worker_gpu_memory_fraction'])
            run_status.update(phase='matching')
            write_run_status(status_file, **run_status)
            template_number, recovered_number, identified_number, ghost_number = np.zeros((4, block_count), dtype=np.int64)
            proposals, proposal_ids, proposal_blocks = [], [], []
            amplitude_refits = {}
            for block_index, table in zip(block_indices, block_template_tables):
                if not len(table):
                    continue
                with (work / f'round_{label}_block_{block_index:05d}' / 'result.pkl').open('rb') as saved:
                    dwd_recover_table, filter_result, retained_bins, posterior = pickle.load(saved)
                template_number[block_index] = len(filter_result)
                recovered_number[block_index] = len(dwd_recover_table)
                if not len(dwd_recover_table):
                    continue
                covariance = effective_noise_covariance[retained_bins]
                new_sources = reconstruct_waveform(posterior['fit_result']['coefficients'], posterior['bases'])
                truth_rows = table
                truth_ids = truth_rows[:, 0].astype(np.int64)
                reference_bank = reference_waveform_cache[block_index]
                remaining = np.isin(reference_bank['dwd_id'], truth_ids)
                truth_waveforms = reference_bank['waveforms'][remaining]
                truth_association = assign_bank_ids(new_sources, truth_waveforms, truth_ids,
                    np.linalg.inv(covariance), T_obs, fs, minimum_overlap=.9)
                identified = truth_association['matched_dwd_id'] >= 0
                identified_number[block_index] = identified.sum()
                ghost_number[block_index] = len(dwd_recover_table) - identified.sum()
                if refit_identified_amplitudes and identified.any():
                    scales, refit_audit = joint_amplitude_only_refit(
                        lisa_data_fd[1:, retained_bins], new_sources,
                        truth_association, covariance, T_obs, fs, inner_product=inner_product)
                    # Fit winners jointly; unmatched/duplicate components never
                    # enter the minimization. Only winners' overall amplitudes
                    # change, never their shapes or the saved worker checkpoint.
                    dwd_recover_table = dwd_recover_table.copy()
                    dwd_recover_table[identified, 0] *= scales[identified]
                    amplitude_refits[f'block_{block_index:05d}'] = refit_audit
                proposals.append(dwd_recover_table[identified])
                proposal_ids.append(truth_association['matched_dwd_id'][identified])
                proposal_blocks.append(np.full(identified.sum(), block_index, dtype=np.int64))
                del new_sources, truth_waveforms, truth_rows
            dwd_recover_table = np.concatenate(proposals) if proposals else np.empty((0, 8))
            round_ids = np.concatenate(proposal_ids) if proposal_ids else np.empty(0, dtype=np.int64)
            round_blocks = np.concatenate(proposal_blocks) if proposal_blocks else np.empty(0, dtype=np.int64)
            round_summary = dict(parameters=dwd_recover_table, dwd_id=round_ids, block_index=round_blocks,
                selected_count=template_number, recovered_count=recovered_number,
                identified_count=identified_number, ghost_count=ghost_number, accepted_count=identified_number,
                identified_dwd_count=len(np.unique(round_ids)), amplitude_refits=amplitude_refits)
            run_status.update(phase='subtracting' if len(dwd_recover_table) else 'finalizing_round')
            write_run_status(status_file, **run_status)
            if len(dwd_recover_table):
                N_day = int(round(86400 * fs))
                links = DWD_simulation_gpu(dwd_recover_table, detector_config)
                response_model = lisagwresponse.GalacticBinary(A=1., f=.003, ra=0., dec=0.,
                    orbits=str(orbit_file), orbit_interp_order=5, shift_sun='never')
                links_file = work / 'recovered_links.h5'
                with h5py.File(links_file, 'w') as saved:
                    response_model.init_response_file(saved, detector_config['dt'], len(links), -2 * N_day / fs, 'tcb')
                    saved['tcb/y'][:] = links
                del links
                instrument = lisainstrument.Instrument(
                    **{name: value for name, value in detector_config.items()
                       if name not in ('T_obs', 'gpu_batch_size', 'minimum_envelope_samples', 'include_padding')},
                    gws=str(links_file), size=N + 2 * N_day, t0=-N_day / fs)
                instrument.disable_all_noises()
                instrument.disable_dopplers()
                data = pytdi.Data.from_instrument(instrument.export_numpy_full())
                new_xyz_td = tdi_gpu(data.measurements, data.args, keep=slice(N_day, N_day + N)) / instrument.central_freq
                lisa_data_td -= new_xyz_td
                lisa_data_fd[1:] = np.fft.rfft(lisa_data_td, axis=-1)
                round_summary['signal_xyz_td'] = new_xyz_td
                del data, instrument, response_model
                accepted_parameters.append(dwd_recover_table)
                accepted_ids.append(round_ids)
                accepted_blocks.append(round_blocks)
                recovered_ids = np.concatenate(accepted_ids)
                dwd_template_table = dwd_template_table[~np.isin(dwd_template_table[:, 0], round_ids)]
            with h5py.File(output_file, 'a') as saved:
                if label in saved['rounds']:
                    del saved[f'rounds/{label}']
                _save_tree(saved['rounds'].create_group(label), round_summary)
                elapsed_seconds = perf_counter() - round_start
                saved[f'rounds/{label}'].create_dataset('elapsed_seconds', data=elapsed_seconds)
                saved.attrs['last_completed_round'] = round_number
            run_status['finalized_round_counts'][label] = int(round_summary['identified_dwd_count'])
            run_status.update(phase='round_complete', last_completed_round=round_number,
                finalized_identified_dwd_count=int(len(np.unique(recovered_ids))),
                last_round_subtracted_components=int(len(dwd_recover_table)))
            write_run_status(status_file, **run_status)
            print(f'Round {label}: {elapsed_seconds:.1f} s, identified {round_summary["identified_dwd_count"]} DWDs, '
                f'subtracted {len(dwd_recover_table)} components, cumulative unique DWDs {len(np.unique(recovered_ids))}', flush=True)
            if not len(dwd_recover_table):
                break
        dwd_recover_table = np.concatenate(accepted_parameters) if accepted_parameters else np.empty((0, 8))
        run_status.update(phase='writing_final_outputs')
        write_run_status(status_file, **run_status)
        with h5py.File(observation_file, 'r') as saved:
            signal_xyz_td = saved['observation/total_xyz_td'][:]
        signal_xyz_td -= lisa_data_td
        final_data = dict(signal_xyz_td=signal_xyz_td, residual_xyz_td=lisa_data_td,
            signal_xyz_fd=np.fft.rfft(signal_xyz_td, axis=-1), residual_xyz_fd=lisa_data_fd[1:],
            lisa_data_td=np.vstack((np.arange(N) / fs, lisa_data_td)), lisa_data_fd=lisa_data_fd,
            parameters=dwd_recover_table, dwd_id=recovered_ids, recovered_dwd_id=np.unique(recovered_ids),
            block_index=np.concatenate(accepted_blocks) if accepted_blocks else np.empty(0, dtype=np.int64))
        stopping_reason = 'no_new_candidates' if not np.any(template_number) else 'no_identified_components'
        with h5py.File(output_file, 'a') as saved:
            if 'final_subtraction' in saved:
                del saved['final_subtraction']
            _save_tree(saved.create_group('final_subtraction'), final_data)
            saved.attrs.update(complete=True, stopping_reason=stopping_reason)
        run_status.update(phase='complete', complete=True, stopping_reason=stopping_reason)
        write_run_status(status_file, **run_status)
        pipeline_result = dict(output_file=str(output_file), rounds=round_number, source_count=len(dwd_recover_table),
            selected_block_indices=block_indices.tolist(), total_block_count=block_count,
            stopping_reason=stopping_reason)

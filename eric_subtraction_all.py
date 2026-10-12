'''Step 1: import all the necessary packages.'''
from pathlib import Path
from tempfile import mkdtemp
import hashlib
import json
import shutil
import h5py
import numpy as np
import lisagwresponse
import lisainstrument
import pytdi
from eric_generate_waveform import DWD_simulation_gpu, tdi_gpu
from script.eric_Fstats import (
    build_response, physical_amplitude_coordinates,
    physical_amplitudes, reconstruct_waveform, noise_inner_product,
)
from script.dwd_blind_search import (
    residual_covariance, template_f_statistic_search, amplitude_f_statistic,
)
from script.dwd_bank_association import assign_bank_ids
from eric_rjmcmc import run_eryn_rjmcmc
from script.oneblock_observation import read_checkpoint, write_checkpoint
from script.tool import _save_tree, single_cpu

'''Step 2: define the parameters.'''
orbit_file = Path('./data/dwd_observation_full_catalogue.orbits.h5')
snr_file = Path('./data/ldc_dwd_data.h5')
observation_file = Path('./data/dwd_observation_full_catalogue.h5')
output_file = Path('./data/dwd_subtraction_all.h5')
frequency_range = (0.0001, 0.05)
f_derivative_pn = [-25, -10]
freq_block_width = 1e-05
subtraction_frequency_range = None
frequency_derivative_bounds = 10. ** f_derivative_pn[1] * np.array([-1., 1.])
F_threshold = 16.0
SNR_r_threshold = 7.0
sample_number = (500, 1500)
maximum_sources = 64
with h5py.File(observation_file, 'r') as saved:
    T_obs = saved['observation'].attrs['duration']
    fs = saved['observation'].attrs['rate']
    total_xyz_td = saved['observation/total_xyz_td'][:]
    detector_config = json.loads(saved['observation'].attrs['detector_config'])
detector_config['orbits'] = str(orbit_file)
N, N_day = int(round(T_obs * fs)), int(round(86400 * fs))
frequency_blocks = np.arange(
    frequency_range[0], frequency_range[1] - freq_block_width * 1.e-10, freq_block_width,
)[:, None] + np.array([0., freq_block_width])
frequency_blocks[-1, 1] = frequency_range[1]
block_count = len(frequency_blocks)
block_indices = np.arange(block_count, dtype=np.int64)
subtraction_range = np.array([frequency_range[0], frequency_range[1]]) if subtraction_frequency_range is None else np.asarray(subtraction_frequency_range, dtype=float)
selected_block_indices = block_indices[
    (frequency_blocks[:, 0] >= subtraction_range[0] - freq_block_width * 1.e-10)
    & (frequency_blocks[:, 1] <= subtraction_range[1] + freq_block_width * 1.e-10)]
selected_block_count = len(selected_block_indices)
fields = dict(A=1, f=2, df=3, phi0=4, iota=5, longitude=6, latitude=7, psi_bme=8)
# Step 3. read the candidate bank and prepare shared state.
with h5py.File(snr_file, 'r') as saved:
    template_dwd = saved['catalogue'][saved['catalogue'][:, 10] > SNR_r_threshold]
total_xyz_fd, fft_frequency = np.fft.rfft(total_xyz_td, axis=-1), np.fft.rfftfreq(N, d=1 / fs)
work = Path(mkdtemp(prefix='.dwd_all_', dir=output_file.parent)).resolve()
configuration = dict(duration=T_obs, rate=fs, frequency_range=np.array([frequency_range[0], frequency_range[1]]),
    frequency_blocks_hz=frequency_blocks, block_indices=block_indices, frequency_block_width=freq_block_width,
    subtraction_frequency_range=subtraction_range, selected_block_indices=selected_block_indices,
    derivative_bounds=frequency_derivative_bounds,
    chirp_coordinate='linear', f_threshold=F_threshold,
    relative_snr_threshold=SNR_r_threshold, sample_number=np.asarray(sample_number),
    minimum_sampler_capacity=maximum_sources)
with h5py.File(output_file, 'w') as saved:
    _save_tree(saved.create_group('config'), configuration)
    saved.create_group('blocks')
    saved.create_group('rounds')
# Step 4. start a global round and estimate covariance.
round_number, last_accepted_round = 0, 0
print(f'Subtraction range {subtraction_range[0]:.8g}--{subtraction_range[1]:.8g} Hz: '
      f'selected {selected_block_count} of {block_count} blocks, indices '
      f'{selected_block_indices[0]}--{selected_block_indices[-1]} (zero-based).', flush=True)
with single_cpu():
    while True:
        round_number += 1
        label = f'{round_number:02d}'
        print(f'Global round {round_number}: visit {selected_block_count} selected frequency blocks.', flush=True)
        # This complete Fourier snapshot is unchanged throughout the block loop.
        round_residual_fd = total_xyz_fd if round_number == 1 else residual_xyz_fd
        round_residual_fd.setflags(write=False)
        round_digest = hashlib.sha256(round_residual_fd.tobytes()).hexdigest()
        covariance_full = residual_covariance(round_residual_fd, T_obs, fs)
        block_order = selected_block_indices
        status = np.full(block_count, 'outside_subtraction_range', dtype='S32')
        status[selected_block_indices] = 'pending'
        with h5py.File(output_file, 'a') as saved:
            group = saved['rounds'].create_group(label)
            group.create_group('blocks')
            group.attrs.update(complete=False, committed=False, residual_input_sha256=round_digest)
        # Step 5. evaluate each selected block against its recorded templates.
        selected_counts = np.zeros(block_count, dtype=np.int64)
        accepted_counts = np.zeros(block_count, dtype=np.int64)
        proposals, proposal_doppler, proposal_coefficients, proposal_ids, proposal_blocks = [], [], [], [], []
        for block_index in block_order:
            key = f'{block_index:05d}'
            block_range = frequency_blocks[block_index]
            center = float(block_range.mean())
            block_derivative_bounds = np.asarray(frequency_derivative_bounds, dtype=float)
            padding = block_range[1] * 1.e-4 + max(abs(block_derivative_bounds)) * T_obs + 8 / T_obs
            retained_bins = np.flatnonzero((fft_frequency >= block_range[0] - padding)
                                          & (fft_frequency <= block_range[1] + padding) & (fft_frequency > 0))
            center_bin = int(round(center * T_obs))
            wide_bins = np.arange(max(1, min(center_bin - 2048, retained_bins[0] - 256)),
                min(len(fft_frequency), max(center_bin + 2049, retained_bins[-1] + 257)))
            core_indices = retained_bins - wide_bins[0]
            physical_frequency, wide_frequency = fft_frequency[retained_bins], fft_frequency[wide_bins]
            if round_number == 1 and block_index == block_order[0]:
                builder, response_times, response_geometry = build_response(work, wide_frequency, T_obs, fs, orbit_file=orbit_file)
                builder_type = type(builder)
            else:
                builder = builder_type(np.broadcast_to(np.eye(3), (len(wide_frequency), 3, 3)),
                    np.zeros((3, len(wide_frequency)), complex), response_times, response_geometry, wide_frequency, T_obs, fs)
            template_data = dict(derivative_bounds=block_derivative_bounds,
                retained_bins=retained_bins, wide_bins=wide_bins, core_indices=core_indices,
                block_index=block_index, frequency_range=block_range,
                physical_frequency_hz=physical_frequency)
            with h5py.File(output_file, 'a') as saved:
                if key not in saved['blocks']:
                    _save_tree(saved['blocks'].create_group(key), template_data)
            covariance = covariance_full[retained_bins].copy()
            inverse = np.linalg.inv(covariance)
            covariance.setflags(write=False)
            data = round_residual_fd[:, retained_bins]
            target = (template_dwd[:, fields['f']] >= block_range[0]) & (
                (template_dwd[:, fields['f']] < block_range[1]) |
                ((block_index == block_count - 1) & (template_dwd[:, fields['f']] == block_range[1])))
            search = template_f_statistic_search(data, covariance, builder, core_indices, T_obs, fs,
                template_dwd[target], f_threshold=F_threshold)
            search['template_dwd_id'] = template_dwd[target, 0].astype(np.int64)
            selected = search['template_dwd']
            selected_counts[block_index] = len(selected)
            block_data = dict(fixed_covariance=covariance,
                screening=dict(search=search, residual_input_xyz_fd=data))
            block_attributes = dict(residual_input_sha256=round_digest,
                block_index=block_index,
                fixed_covariance_sha256=hashlib.sha256(covariance.tobytes()).hexdigest(),
                proposed_for_subtraction=False, accepted_for_subtraction=False)
            status[block_index] = 'no_new_candidates'
            # Step 6. run RJMCMC and assess fitted sources.
            if len(selected):
                print(f'  Block {key}, {block_range[0]:.8f}--{block_range[1]:.8f} Hz: fit {len(selected)} candidates.', flush=True)
                checkpoints = work / f'round_{label}_block_{key}'
                posterior = run_eryn_rjmcmc(data, covariance, selected, block_range,
                    block_derivative_bounds, center, T_obs, fs, checkpoints,
                    chain_count=1, inverse_temperatures=(1., .8, .6, .4),
                    iterations=sum(sample_number), burn_in=sample_number[0],
                    maximum_sources=max(maximum_sources, len(selected)), chirp_coordinate='linear',
                    response_times=response_times, response_geometry=response_geometry, physical_frequency=physical_frequency)
                doppler, coefficients = posterior['doppler'], posterior['fit_result']['coefficients']
                new_bases = builder.build(doppler)[:, :, :, core_indices]
                new_sources = reconstruct_waveform(coefficients, new_bases)
                source_snr = np.sqrt(np.maximum(noise_inner_product(new_sources, new_sources, inverse, T_obs, fs), 0.))
                assessment_data = data.copy()
                # Fits share the round snapshot, but acceptance is conditional on
                # already proposed components to avoid subtracting a sideband twice.
                if proposal_doppler:
                    points, amplitudes = np.concatenate(proposal_doppler), np.concatenate(proposal_coefficients)
                    nearby = abs(points[:, 0] - center) <= .5 * np.ptp(block_range) + 2 * padding
                    if np.any(nearby):
                        pending = reconstruct_waveform(amplitudes[nearby],
                            builder.build(points[nearby])[:, :, :, core_indices])
                        assessment_data -= pending.sum(axis=0)
                conditional_f = np.zeros(len(doppler))
                for i in range(len(doppler)):
                    conditional_data = assessment_data - new_sources.sum(axis=0) + new_sources[i]
                    conditional_f[i] = amplitude_f_statistic(conditional_data, new_bases[i:i + 1],
                        covariance, T_obs, fs)['two_f'][0] / 2
                owned = (doppler[:, 0] >= block_range[0]) & ((doppler[:, 0] < block_range[1])
                    | ((block_range[1] == frequency_range[1]) & (doppler[:, 0] == frequency_range[1])))
                duplicate = np.zeros(len(doppler), dtype=bool)
                previous_doppler = proposal_doppler if round_number == 1 else accepted_doppler + proposal_doppler
                previous_coefficients = proposal_coefficients if round_number == 1 else accepted_coefficients + proposal_coefficients
                if previous_doppler:
                    points, amplitudes = np.concatenate(previous_doppler), np.concatenate(previous_coefficients)
                    nearby = abs(points[:, 0] - center) <= .5 * np.ptp(block_range) + 2 * padding
                    if np.any(nearby):
                        previous = reconstruct_waveform(amplitudes[nearby],
                            builder.build(points[nearby])[:, :, :, core_indices])
                        overlap = noise_inner_product(new_sources[:, None], previous[None, :], inverse, T_obs, fs)
                        norm = np.sqrt(np.maximum(noise_inner_product(new_sources, new_sources, inverse, T_obs, fs)[:, None]
                            * noise_inner_product(previous, previous, inverse, T_obs, fs)[None, :], 1.e-300))
                        duplicate |= np.any(overlap / norm > .9, axis=1)
                retained = owned & ~duplicate & np.isfinite(source_snr) & (source_snr > SNR_r_threshold) & (conditional_f >= F_threshold)
                # Deduplicate the fitted components using measured waveform overlap.
                for i in np.argsort(source_snr)[::-1]:
                    if not retained[i]:
                        continue
                    for j in np.flatnonzero(retained):
                        if j == i or source_snr[j] > source_snr[i] or (source_snr[j] == source_snr[i] and j < i):
                            continue
                        overlap = noise_inner_product(new_sources[i], new_sources[j], inverse, T_obs, fs)
                        if overlap / max(source_snr[i] * source_snr[j], 1.e-300) > .9:
                            retained[j], duplicate[j] = False, True
                retained_sum = new_sources[retained].sum(axis=0)
                improvement = float(noise_inner_product(assessment_data, assessment_data, inverse, T_obs, fs)
                    - noise_inner_product(assessment_data - retained_sum, assessment_data - retained_sum, inverse, T_obs, fs))
                accepted = np.any(retained) and improvement > 0
                status[block_index] = 'proposed' if accepted else 'fit_rejected_by_data'
                # Step 7. associate recovered sources with catalogue labels.
                # Catalogue reads happen after inference and have no acceptance role.
                if round_number == 1 and np.count_nonzero(selected_counts) == 1:
                    with h5py.File(snr_file, 'r') as saved:
                        truth_frequency = saved['catalogue'][:, fields['f']]
                        chirp = saved['catalogue'][:, fields['df']]
                    maximum_drift = float(np.max(abs(chirp) * T_obs + abs(truth_frequency) * 1.e-4 + 8 / T_obs))
                    truth_order = np.argsort(truth_frequency, kind='stable')
                    truth_frequency = truth_frequency[truth_order]
                # Truth is consulted only after fitting, and cached on disk for this block.
                truth_cache = work / f'truth_{key}.h5'
                if truth_cache.exists():
                    with h5py.File(truth_cache, 'r') as saved:
                        truth_ids = saved['dwd_id'][:]
                        truth_waveforms = saved['source_xyz_fd'][:]
                else:
                    truth_low, truth_high = block_range + np.array([-2304., 2304.]) / T_obs
                    first, last = np.searchsorted(truth_frequency, [truth_low - maximum_drift, truth_high + maximum_drift])
                    nearby_indices = np.sort(truth_order[first:last])
                    with h5py.File(snr_file, 'r') as saved:
                        records = saved['catalogue'][nearby_indices]
                    frequency = records[:, fields['f']]
                    end = frequency + records[:, fields['df']] * T_obs
                    margin = np.maximum(frequency, end) * 1.e-4 + 8 / T_obs
                    nearby = (np.minimum(frequency, end) - margin <= truth_high) & (np.maximum(frequency, end) + margin >= truth_low)
                    truth_rows = records[nearby]
                    truth_ids = truth_rows[:, 0].astype(np.int64)
                    truth_doppler = truth_rows[:, [fields[name] for name in ('f', 'df', 'longitude', 'latitude')]]
                    cos_iota = np.cos(truth_rows[:, fields['iota']])
                    truth_coefficients = physical_amplitude_coordinates(truth_rows[:, fields['A']] * (1 + cos_iota**2),
                        2 * truth_rows[:, fields['A']] * cos_iota, truth_rows[:, fields['psi_bme']], truth_rows[:, fields['phi0']]) / 1.e-22
                    truth_waveforms = np.empty((len(truth_ids), 3, len(retained_bins)), dtype=complex)
                    for first in range(0, len(truth_ids), 32):
                        truth_waveforms[first:first + 32] = reconstruct_waveform(truth_coefficients[first:first + 32],
                            builder.build(truth_doppler[first:first + 32])[:, :, :, core_indices])
                    with h5py.File(truth_cache, 'w') as saved:
                        saved.create_dataset('dwd_id', data=truth_ids)
                        saved.create_dataset('source_xyz_fd', data=truth_waveforms)
                truth_association = assign_bank_ids(new_sources, truth_waveforms, truth_ids, inverse, T_obs, fs, minimum_overlap=.9)
                matched = truth_association['matched_dwd_id'].copy()
                physical = physical_amplitudes(coefficients)
                parameters = np.column_stack((physical['A'], doppler[:, 0], doppler[:, 1], physical['phi0'],
                    physical['iota'], doppler[:, 2], doppler[:, 3], physical['psi_bme']))
                block_data.update(catalogue_association=dict(matched_dwd_id=matched,
                    best_overlap=truth_association['best_overlap'], scope='Post-fit labels only; unmatched fits may be subtracted.'),
                    # Step 8. record the accepted fits.
                    acceptance=dict(source_snr=source_snr, conditional_f=conditional_f,
                        owned_by_block=owned, duplicate_previous_or_this_round=duplicate,
                        retained=retained, likelihood_improvement=improvement),
                    recovery=dict(doppler=doppler, coefficients=coefficients, parameters=parameters, source_xyz_fd=new_sources),
                    rjmcmc={name: value for name, value in posterior.items() if name not in ('bases', 'fit_result', 'checkpoint_file')})
                block_attributes.update(proposed_for_subtraction=accepted,
                    source_xyz_fd_scope='Frequency-domain fitting and post-fit association only.')
                if accepted:
                    retained_ids = matched[retained].copy()
                    if round_number == 1 and not np.any(accepted_counts):
                        next_unknown_id = -1
                    if round_number == 1:
                        used_labels = np.concatenate(proposal_ids) if proposal_ids else np.empty(0, dtype=np.int64)
                    else:
                        used_labels = np.concatenate(accepted_ids + proposal_ids)
                    for i in range(len(retained_ids)):
                        if retained_ids[i] < 0 or retained_ids[i] in used_labels or retained_ids[i] in retained_ids[:i]:
                            retained_ids[i] = next_unknown_id
                            next_unknown_id -= 1
                    proposals.append(parameters[retained])
                    proposal_doppler.append(doppler[retained])
                    proposal_coefficients.append(coefficients[retained])
                    proposal_ids.append(retained_ids)
                    proposal_blocks.append(np.full(np.count_nonzero(retained), block_index, dtype=np.int64))
                    accepted_counts[block_index] = np.count_nonzero(retained)
                snapshots = {}
                for path in sorted(checkpoints.glob('*.h5')):
                    with h5py.File(path, 'r') as saved:
                        snapshots[path.stem] = read_checkpoint(saved)
                with h5py.File(output_file, 'a') as saved:
                    group = saved[f'rounds/{label}/blocks'].create_group(key)
                    _save_tree(group, block_data)
                    group.attrs.update(block_attributes)
                    checkpoint_group = group.create_group('checkpoints')
                    for name, snapshot in snapshots.items():
                        write_checkpoint(checkpoint_group.create_group(name), snapshot)
                    group['rjmcmc'].attrs['checkpoint_group'] = f'/rounds/{label}/blocks/{key}/checkpoints'
            else:
                with h5py.File(output_file, 'a') as saved:
                    group = saved[f'rounds/{label}/blocks'].create_group(key)
                    _save_tree(group, block_data)
                    group.attrs.update(block_attributes)
        # Step 9. subtract the combined time-domain TDI signal.
        if round_number == 1:
            accepted_parameters, accepted_doppler, accepted_coefficients, accepted_ids, accepted_blocks = [], [], [], [], []
            accumulated_xyz_td = np.zeros_like(total_xyz_td)
            residual_xyz_td, residual_xyz_fd = total_xyz_td.copy(), total_xyz_fd
        # Only here may the residual change: one waveform/TDI sum for all accepted blocks.
        if proposals:
            parameters = np.concatenate(proposals)
            print(f'  Subtract {len(parameters)} fitted DWDs from {np.count_nonzero(accepted_counts)} blocks together.', flush=True)
            links = DWD_simulation_gpu(parameters, detector_config)
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
            accumulated_xyz_td += new_xyz_td
            residual_xyz_td = total_xyz_td - accumulated_xyz_td
            residual_xyz_fd = np.fft.rfft(residual_xyz_td, axis=-1)
            del data, instrument, response_model, new_xyz_td
            accepted_parameters.append(parameters)
            accepted_doppler.extend(proposal_doppler)
            accepted_coefficients.extend(proposal_coefficients)
            accepted_ids.extend(proposal_ids)
            accepted_blocks.extend(proposal_blocks)
            last_accepted_round = round_number
            status[status == b'proposed'] = 'accepted'
        # Step 10. record the round and decide whether to continue.
        recovered_ids = np.concatenate(accepted_ids) if accepted_ids else np.empty(0, dtype=np.int64)
        round_summary = dict(block_status=status, selected_count=selected_counts, accepted_count=accepted_counts,
            recovered_dwd_id=recovered_ids)
        with h5py.File(output_file, 'a') as saved:
            group = saved[f'rounds/{label}']
            _save_tree(group, round_summary)
            group.attrs.update(complete=True, committed=bool(proposals), visited_block_count=selected_block_count,
                selected_block_count=selected_block_count,
                accepted_block_count=int(np.count_nonzero(accepted_counts)),
                residual_output_sha256=hashlib.sha256(residual_xyz_fd.tobytes()).hexdigest())
            for block_index in np.flatnonzero(accepted_counts):
                saved[f'rounds/{label}/blocks/{block_index:05d}'].attrs.update(accepted_for_subtraction=True,
                    subtraction_backend='DWD_simulation_gpu + native Instrument + tdi_gpu')
            saved.attrs.update(last_completed_round=round_number, last_accepted_round=last_accepted_round)
        if not proposals:
            stopping_reason = 'no_new_candidates' if not np.any(selected_counts) else 'no_accepted_blocks'
            break
# Step 11. save the final results.
parameters = np.concatenate(accepted_parameters) if accepted_parameters else np.empty((0, 8))
ids = np.concatenate(accepted_ids) if accepted_ids else np.empty(0, dtype=np.int64)
final_data = dict(signal_xyz_td=accumulated_xyz_td, residual_xyz_td=residual_xyz_td,
    signal_xyz_fd=np.fft.rfft(accumulated_xyz_td, axis=-1), residual_xyz_fd=residual_xyz_fd,
    parameters=parameters, dwd_id=ids,
    doppler=np.concatenate(accepted_doppler) if accepted_doppler else np.empty((0, 4)),
    coefficients=np.concatenate(accepted_coefficients) if accepted_coefficients else np.empty((0, 4)),
    block_index=np.concatenate(accepted_blocks) if accepted_blocks else np.empty(0, dtype=np.int64))
with h5py.File(output_file, 'a') as saved:
    _save_tree(saved.create_group('final_subtraction'), final_data)
    saved['final_subtraction'].attrs['accepted_round'] = last_accepted_round
    saved.attrs.update(complete=True, single_file_complete=True, stopping_reason=stopping_reason,
        stopping_condition_met=True, iteration_stable=False)
shutil.rmtree(work)
print(f'Saved {output_file}: {round_number} global rounds, {len(parameters)} accepted components; {stopping_reason}.', flush=True)
pipeline_result = dict(output_file=str(output_file), rounds=round_number, source_count=len(parameters),
            selected_block_indices=selected_block_indices.tolist(), total_block_count=block_count,
            last_accepted_round=last_accepted_round, stopping_reason=stopping_reason)

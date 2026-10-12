"""Main DWD workflow: screen, fit, subtract, and repeat on the residual."""

# Run Steps 1-4 and 6-8 once; repeat Steps 5 and 9-11 on the residual.

# Step 1. Import packages
from __future__ import annotations
from functools import partial
import os
for thread_variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(thread_variable,'1')
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path
from tempfile import mkdtemp
import h5py
import lisagwresponse
import lisainstrument
import numpy as np
import pytdi
from tqdm.auto import tqdm
from eric_generate_waveform import DWD_simulation_gpu, tdi_gpu
from script.eric_preprocess import effective_noise_cov, periodogram, relative_covariance_change
from script.eric_Fstats import (
    build_response, gpu_amplitude_bases_fd, physical_amplitudes, f_statistic,
    noise_inner_product, physical_amplitude_coordinates, reconstruct_waveform,
)
from script.dwd_bank_association import assign_bank_ids
from script.tool import _save_tree
from script.oneblock_godview import PopulationGodView
from script.eric_rjmcmc import run_eryn_rjmcmc
from script.tool import single_cpu
from script.oneblock_gpu_monitor import GPUUsageRecorder
from script.oneblock_observation import read_checkpoint, write_checkpoint
# Step 2. Configuration and parameter settings
orbit_file = Path('./data/dwd_observation_full_catalogue.orbits.h5')
snr_file = Path('./data/ldc_dwd_data.h5')
observation_file = Path('./data/dwd_observation_full_catalogue.h5')
output_file = Path('./data/dwd_1block_full_catalogue.h5')
T_obs = 360.0 * 86400.0
fs = 0.1
freq_block_center = 0.0032922542578130718
freq_block_width = 10.0e-6
freq_block_range = np.array([
    freq_block_center - 0.5 * freq_block_width,
    freq_block_center + 0.5 * freq_block_width,
])
frequency_derivative_bounds = (7.60863249122299e-23,10.0**-14.8)
F_threshold = 16.0
SNR_a_threshold = 7.0
SNR_r_threshold = 7.0
sample_number = [500, 1500]
ghost_fraction_threshold = 0.30
parameter_fields = {
    'A': 'Amplitude', 'f': 'Frequency', 'df': 'FrequencyDerivative',
    'phi0': 'InitialPhase', 'iota': 'Inclination',
    'longitude': 'EclipticLongitude', 'latitude': 'EclipticLatitude',
    'psi_bme': 'Polarization',
}
N = int(round(T_obs * fs))
N_day = int(round(86400 * fs))
detector_config = dict(
    T_obs=T_obs, dt=1 / fs, orbits=str(orbit_file), orbit_dataset='tcb/ltt',
    gpu_batch_size=10000, minimum_envelope_samples=512, include_padding=True,
    aafilter=('kaiser', 240., .275 * fs, .725 * fs),
    clock_offsets=0., clock_freqoffsets=0., clock_freqlindrifts=0., clock_freqquaddrifts=0.,
)

# -----------------------------------------------------------------------
# Step 3. Read the catalogue and select DWDs in this frequency block
# -----------------------------------------------------------------------

print('Step 3/11  Reading the DWD catalogue in this frequency block',flush=True)
with h5py.File(snr_file, 'r') as hdf5:
    dgb = hdf5['sky/dgb/cat'][:, 0]
    igb = hdf5['sky/igb/cat'][:, 0]
    snr_duration = hdf5['settings/observation_duration_s'][()]
    snr_rate = hdf5['settings/detector_sampling_rate_hz'][()]
    snr_complete = hdf5['status/complete'][()]
if not snr_complete or snr_duration != T_obs or snr_rate != fs:
    raise ValueError('The SNR result must be complete and match T_obs and fs.')
catalogue = np.empty(len(dgb) + len(igb), dtype=[('ID', 'i8')] + [
    (field, 'f8') for field in parameter_fields.values()
] + [('SNR_absolute', 'f8'), ('SNR_relative', 'f8')])
catalogue['ID'] = np.arange(1, len(catalogue) + 1, dtype=np.int64)
for field in (*parameter_fields.values(), 'SNR_absolute', 'SNR_relative'):
    catalogue[field] = np.concatenate((dgb[field], igb[field]))
del dgb, igb
frequency = catalogue['Frequency']
target_dwd_index = np.flatnonzero(
    (frequency >= freq_block_range[0]) & (frequency <= freq_block_range[1])
    & (catalogue['SNR_absolute'] > SNR_a_threshold)
)
target_dwd = catalogue[target_dwd_index]
del catalogue, frequency
order = np.argsort(target_dwd['Frequency'])
target_dwd = target_dwd[order]
target_dwd_index = target_dwd_index[order]
bank_ids = target_dwd['ID'].astype(np.int64)
bank_parameters = {name:np.asarray(target_dwd[field],dtype=np.float64) for name,field in parameter_fields.items()}
bank_doppler = np.column_stack([bank_parameters[name] for name in ('f','df','longitude','latitude')])

print(f'     Results: {output_file}',flush=True)
output_file.parent.mkdir(parents=True,exist_ok=True)
work = Path(mkdtemp(prefix='.dwd_1block_',dir=output_file.parent)).resolve()

telemetry_file = work/'gpu_recovery.csv'
with single_cpu(),GPUUsageRecorder(telemetry_file) as monitor:
    started = time.perf_counter()

    # ---------------------------------------------------------------
    # Step 4. Read the observation, calculate FFT, and select the frequency block
    # ---------------------------------------------------------------

    print(f'Step 4/11  Read the saved observation from {observation_file}; simulation skipped',flush=True)
    with h5py.File(observation_file,'r') as saved:
        total_xyz_td = saved['observation/total_xyz_td'][:]
    if total_xyz_td.shape != (3, N):
        raise ValueError('The observation must contain three XYZ streams matching T_obs and fs.')

    # Continue Step 4: keep native Fourier coefficients and a wider covariance window.
    print('Step 4/11  Native raw Fourier coefficients; no smoothing or rebinning of data',flush=True)
    full_fft = np.fft.rfft(total_xyz_td,axis=-1)
    fft_frequency = np.fft.rfftfreq(total_xyz_td.shape[-1],d=1/fs)
    padding = (freq_block_range[1]*1e-4
               +frequency_derivative_bounds[1]*T_obs+8/T_obs)
    retained_bins = np.flatnonzero(
        (fft_frequency>=freq_block_range[0]-padding)&(fft_frequency<=freq_block_range[1]+padding)
    )
    center_bin = int(round(freq_block_center*T_obs))
    wide_bins = np.arange(center_bin-2048,center_bin+2049)
    core_indices = retained_bins-wide_bins[0]
    physical_frequency = fft_frequency[retained_bins]
    wide_frequency = fft_frequency[wide_bins]
    original_total_xyz_fd = full_fft[:,retained_bins]
    original_total_xyz_fd_wide = full_fft[:,wide_bins]
    del full_fft

    # ---------------------------------------------------------------
    # Steps 6-8. Prepare GPU waveforms and the unchanged amplitude-basis bank
    # ---------------------------------------------------------------
    # This GPU route reconstructs bank waveforms from their four amplitude bases,
    # rather than repeating the TD-waveform/FFT example in the Markdown outline.

    # Step 8. Prepare four amplitude-basis templates per candidate
    builder,response_times,response_geometry = build_response(
        work,wide_frequency,T_obs,fs,orbit_file=orbit_file,
    )

    candidate_amplitude_bases = partial(
        gpu_amplitude_bases_fd, builder=builder, core_indices=core_indices)

    print(f'Steps 6-8/11  Generate {len(bank_ids)} XYZ templates on GPU and store them in /templates',flush=True)
    template_started = time.perf_counter()
    bank_bases = candidate_amplitude_bases(bank_doppler)

    # Step 6. Generate selected DWD waveforms from the GPU amplitude bases
    bank_amplitude = np.asarray(bank_parameters['A'],dtype=np.float64)
    bank_cos_iota = np.cos(np.asarray(bank_parameters['iota'],dtype=np.float64))
    bank_coefficients = physical_amplitude_coordinates(
        bank_amplitude*(1+bank_cos_iota**2),2*bank_amplitude*bank_cos_iota,
        bank_parameters['psi_bme'],bank_parameters['phi0'])/1e-22
    bank_waveforms = reconstruct_waveform(bank_coefficients,bank_bases)
    template_seconds = time.perf_counter()-template_started
    # This helper reads injection truth only when called after RJMCMC.
    godview = PopulationGodView(work,builder,core_indices,T_obs,fs,
        snr_file=snr_file,frequency_range=freq_block_range)
    excluded_bank_ids = np.empty(0,dtype=np.int64)
    previously_recovered_ids = np.empty(0,dtype=np.int64)
    accumulated_xyz_td = np.zeros_like(total_xyz_td)
    accumulated_xyz_fd_wide = np.zeros_like(original_total_xyz_fd_wide)
    retained_residual_xyz_fd_wide = original_total_xyz_fd_wide.copy()
    previous_doppler = np.empty((0,4))
    previous_coefficients = np.empty((0,4))
    previous_sources = np.empty((0,3,len(physical_frequency)),dtype=complex)
    previous_covariance = None
    stable_rounds = 0
    stopped = False
    stopping_reason = ''
    last_accepted_round = 0

    # Step 7. Keep an unchanged bank; select candidates separately in each round
    initial_data = dict(
            config=dict(duration=T_obs,rate=fs,center=freq_block_center,
                frequency_range=freq_block_range,derivative_bounds=frequency_derivative_bounds,
                frequency_hz=physical_frequency,wide_frequency_hz=wide_frequency,
                core_indices=core_indices,retained_bins=retained_bins,
                iterations=sum(sample_number),burn_in=sample_number[0],maximum_sources=49,
                maximum_rounds=0,
                ghost_fraction_threshold=ghost_fraction_threshold,recovery_overlap_threshold=.9,
                f_threshold=F_threshold,absolute_snr_threshold=SNR_a_threshold,relative_snr_threshold=SNR_r_threshold,
                exclude_recovered_templates=True,new_sources_only=True),
            provenance=dict(orbit_file=str(orbit_file.resolve()),snr_file=str(snr_file.resolve()),reference=str(observation_file.resolve()),observation_file=str(observation_file.resolve()),observation_group='/observation',input_schema='dwd-oneblock-observation:1',simulation_reused=True),
            data=dict(original_total_xyz_fd=original_total_xyz_fd,original_total_xyz_fd_wide=original_total_xyz_fd_wide),
            templates=dict(dwd_id=bank_ids,catalogue_index=target_dwd_index,
                parameters=bank_parameters,doppler=bank_doppler,
                saved_snr_absolute=target_dwd['SNR_absolute'],saved_snr_relative=target_dwd['SNR_relative'],
                bases=bank_bases,coefficients=bank_coefficients,source_xyz_fd=bank_waveforms,
                generation_seconds=template_seconds,generation_backend=type(builder).__name__),
        )
    with h5py.File(output_file,'w') as saved:
        _save_tree(saved,initial_data)
        saved.create_group('rounds')
        saved.attrs.update(complete=False,single_file_complete=False,last_accepted_round=0,
            target='DWD recovery conditional on a fixed, data-estimated effective covariance.',
            covariance_updated_between_rounds_only=True,posterior_samples_pooled_across_rounds=False,
            unlimited_rounds=True,
            catalogue_informed_bank=True,noise_parameters_inferred=False,
            independent_chain_convergence_assessed=False,godview_stopping=True,
            source_inference_scope='Additional DWDs conditional on fixed previously accepted waveforms.')

    # Only the main workflow owns the outer round loop and termination policy.
    round_number = 0
    while True:
        round_number += 1
        label = f'{round_number:02d}'

        # -----------------------------------------------------------
        # Step 5. Re-estimate effective covariance from this round's residual
        # -----------------------------------------------------------

        print(f'Step 5/11  Round {round_number}: fixed +/-256-bin covariance; '
              f'SNR_r > {SNR_r_threshold:g}, F >= {F_threshold:g}',flush=True)
        residual_xyz_fd_wide = retained_residual_xyz_fd_wide
        residual_input_xyz_fd = residual_xyz_fd_wide[:,core_indices].copy()
        psd = periodogram(residual_xyz_fd_wide,T_obs,fs)
        estimate = dict(covariance=effective_noise_cov(psd,neighboring_bins=256,downweight_peaks=True),
                        half_width=256,downweight_peaks=True)
        fixed_covariance = estimate['covariance'][core_indices].copy()
        fixed_inverse = np.linalg.inv(fixed_covariance)
        digest = hashlib.sha256(fixed_covariance.tobytes()).hexdigest()
        fixed_covariance.setflags(write=False)

        # -----------------------------------------------------------
        # Step 9. Re-screen the unchanged bank on the current residual
        # -----------------------------------------------------------

        eligible = ~np.isin(bank_ids,excluded_bank_ids)
        # Relative SNR includes the effective instrument + unresolved background.
        relative_snr = np.sqrt(np.maximum(noise_inner_product(
            bank_waveforms,bank_waveforms,fixed_inverse,T_obs,fs,
        ),0.0))
        two_f = np.full(len(bank_ids),np.nan)
        if np.any(eligible):
            two_f[eligible] = f_statistic(
                residual_xyz_fd_wide[:,core_indices],bank_bases[eligible],fixed_covariance,
                T_obs,fs,
            ).two_f
        selection = eligible & (relative_snr>SNR_r_threshold) & (two_f>=2*F_threshold)
        selected_candidates = np.flatnonzero(selection)
        print(f'Step 9/11  Round {round_number}: {len(selected_candidates)} selected; '
              f'{(~eligible).sum()} already-matched templates excluded',flush=True)

        # Stop before RJMCMC when no unused template passes the threshold.
        if len(selected_candidates)==0:
            stopped = True
            stopping_reason = 'no_new_candidates'
            terminal_data = dict(attempted_round=round_number,F=two_f/2,two_F=two_f,relative_snr=relative_snr,
                    selected=selection,eligible=eligible,excluded_dwd_id=excluded_bank_ids,
                    fixed_covariance=fixed_covariance,estimator=estimate,
                    residual_xyz_fd=residual_xyz_fd_wide[:,core_indices])
            with h5py.File(output_file,'a') as saved:
                terminal = saved.create_group('termination_screening')
                _save_tree(terminal,terminal_data)
                terminal.attrs['fixed_covariance_sha256'] = digest
            print('Stopping: no new candidates; keep the previous accepted subtraction.',flush=True)
            break

        # -----------------------------------------------------------
        # Step 10. Fit additional sources jointly; covariance stays fixed in this run
        # -----------------------------------------------------------

        initial_catalogue = bank_doppler[selected_candidates]
        print(f'Step 10/11  Round {round_number}: {len(initial_catalogue)} new candidates; '
              f'{len(previous_doppler)} earlier components frozen; one Eryn ladder, four replicas',flush=True)
        checkpoint_directory = work/f'round_{label}_checkpoints'
        posterior = run_eryn_rjmcmc(
            residual_xyz_fd_wide[:,core_indices],fixed_covariance,
            initial_catalogue,freq_block_range,frequency_derivative_bounds,
            freq_block_center,T_obs,fs,checkpoint_directory,
            chain_count=1,inverse_temperatures=(1.,.8,.6,.4),
            iterations=sum(sample_number),burn_in=sample_number[0],maximum_sources=49,
            response_times=response_times,response_geometry=response_geometry,
            physical_frequency=physical_frequency,
        )
        new_coefficients = posterior['fit_result']['coefficients']
        new_sources_wide = reconstruct_waveform(new_coefficients,builder.build(posterior['doppler']))
        new_sources = new_sources_wide[:,:,core_indices]
        new_wave = new_sources_wide.sum(axis=0)
        recovered_relative_snr = np.sqrt(np.maximum(noise_inner_product(
            new_sources,new_sources,fixed_inverse,T_obs,fs,
        ),0.0))
        new_recovery = dict(doppler=posterior['doppler'],coefficients=new_coefficients,source_xyz_fd=new_sources,
            signal_xyz_fd=new_wave[:,core_indices],signal_xyz_fd_wide=new_wave,relative_snr=recovered_relative_snr)

        # -----------------------------------------------------------
        # Step 11. Reject additions if ghosts exceed 30% or there are no new matches
        # -----------------------------------------------------------

        print(f'Step 11/11  Round {round_number}: identify new sources and rebuild the original-data residual',flush=True)
        doppler = np.concatenate((previous_doppler,posterior['doppler']))
        coefficients = np.concatenate((previous_coefficients,new_coefficients))
        source_xyz_fd = np.concatenate((previous_sources,new_sources))
        signal_xyz_fd_wide = accumulated_xyz_fd_wide+new_wave
        residual_xyz_fd_wide = original_total_xyz_fd_wide-signal_xyz_fd_wide
        residual_xyz_fd = residual_xyz_fd_wide[:,core_indices]
        bank_association = assign_bank_ids(source_xyz_fd,bank_waveforms,bank_ids,
            fixed_inverse,T_obs,fs,minimum_overlap=.9)
        residual_two_f = f_statistic(residual_xyz_fd,bank_bases,fixed_covariance,
            T_obs,fs).two_f

        # God-view is used only after fitting, for the simulation stopping checks.
        ghost_assessment = godview(source_xyz_fd,fixed_covariance)
        matched = ghost_assessment['matched_dwd_id'][len(previous_doppler):]
        duplicate = np.isin(matched,previously_recovered_ids) & (matched>=0)
        matched = np.where(duplicate,-1,matched)
        ghost_count = int((matched<0).sum())
        ghost_fraction = ghost_count/max(len(matched),1)
        new_recovered_ids = np.setdiff1d(matched[matched>=0],previously_recovered_ids)
        ghost_assessment.update(new_recovered_dwd_id=new_recovered_ids,
            new_recovered_count=len(new_recovered_ids),previously_recovered_count=len(previously_recovered_ids))
        new_source_assessment = dict(matched_dwd_id=matched,ghost_count=ghost_count,source_count=len(matched),
            ghost_fraction=ghost_fraction,duplicate_previous_detection=duplicate,
            scope='Post-fit stopping assessment of additions only; earlier sources cannot dilute the ghost fraction.')

        # Reject this round if ghosts exceed 30%, or it adds no new recovered ID.
        if ghost_fraction>ghost_fraction_threshold:
            stopped = True
            stopping_reason = 'ghost_fraction_exceeded'
        elif len(new_recovered_ids)==0:
            stopped = True
            stopping_reason = 'no_new_recovered_dwd'
        accepted = not stopped
        print(f'     New recovered DWDs: {len(new_recovered_ids)}; ghosts: '
              f'{ghost_count}/{len(matched)} ({ghost_fraction:.1%})',flush=True)

        # Generate time-domain waveforms and apply TDI only for accepted subtraction.
        if accepted:
            new_physical = physical_amplitudes(new_coefficients)
            identified_dwd_parameters = np.column_stack((
                new_physical['A'], posterior['doppler'][:,0], posterior['doppler'][:,1],
                new_physical['phi0'], new_physical['iota'], posterior['doppler'][:,2],
                posterior['doppler'][:,3], new_physical['psi_bme'],
            ))
            links = DWD_simulation_gpu(identified_dwd_parameters, detector_config)
            links_file = work / 'recovered_links.h5'
            response_model = lisagwresponse.GalacticBinary(
                A=1., f=.003, ra=0., dec=0., orbits=str(orbit_file),
                orbit_interp_order=5, shift_sun='never',
            )
            with h5py.File(links_file, 'w') as saved:
                response_model.init_response_file(saved, detector_config['dt'], len(links), -2 * N_day / fs, 'tcb')
                saved['tcb/y'][:] = links
            del links
            signal_instrument = lisainstrument.Instrument(
                **{key: value for key, value in detector_config.items()
                   if key not in ('T_obs', 'gpu_batch_size', 'minimum_envelope_samples', 'include_padding')},
                gws=str(links_file), size=N + 2 * N_day, t0=-N_day / fs,
            )
            signal_instrument.disable_all_noises()
            signal_instrument.disable_dopplers()
            signal_data = pytdi.Data.from_instrument(signal_instrument.export_numpy_full())
            new_signal_xyz_td = tdi_gpu(
                signal_data.measurements, signal_data.args, keep=slice(N_day, N_day + N),
            ) / signal_instrument.central_freq
            signal_xyz_td = accumulated_xyz_td + new_signal_xyz_td
            residual_xyz_td = total_xyz_td - signal_xyz_td
            signal_xyz_fd_wide = np.fft.rfft(signal_xyz_td,axis=-1)[:,wide_bins]
            residual_xyz_fd_wide = np.fft.rfft(residual_xyz_td,axis=-1)[:,wide_bins]
            residual_xyz_fd = residual_xyz_fd_wide[:,core_indices]
            new_recovery.update(parameters=identified_dwd_parameters,
                signal_xyz_fd=np.fft.rfft(new_signal_xyz_td,axis=-1)[:,retained_bins],
                signal_xyz_fd_wide=np.fft.rfft(new_signal_xyz_td,axis=-1)[:,wide_bins])
            residual_two_f = f_statistic(residual_xyz_fd,bank_bases,fixed_covariance,T_obs,fs).two_f
            del signal_data, signal_instrument, response_model, new_signal_xyz_td

        change = signal_xyz_fd_wide[:,core_indices]-accumulated_xyz_fd_wide[:,core_indices]
        norm = lambda value:float(noise_inner_product(value,value,fixed_inverse,T_obs,fs))
        waveform_change = norm(change)/max(norm(signal_xyz_fd_wide[:,core_indices]),np.finfo(float).tiny)
        covariance_change = (np.inf if previous_covariance is None else
                             relative_covariance_change(previous_covariance,fixed_covariance))
        stable = (len(previous_doppler)==len(doppler) and covariance_change<.05
                  and waveform_change<1e-3 and np.max(residual_two_f)<2*F_threshold)
        stable_rounds = stable_rounds+1 if stable else 0
        metrics = dict(covariance_relative_change=covariance_change,waveform_relative_change=waveform_change,
            residual_norm=norm(residual_xyz_fd),original_norm=norm(original_total_xyz_fd),
            max_residual_F=float(np.max(residual_two_f)/2),stable_rounds=stable_rounds,
            stopping_conditions_met=stopped,catalogue_count=len(doppler))
        checkpoint_data = {}
        for path in sorted(checkpoint_directory.glob('*.h5')):
            with h5py.File(path,'r') as source:
                checkpoint_data[path.stem] = read_checkpoint(source)
        checkpoint_group = json.dumps(f'/rounds/{label}/checkpoints')
        recovered_parameters = physical_amplitudes(coefficients)
        round_data = dict(estimator=estimate,fixed_covariance=fixed_covariance,
                screening=dict(F=two_f/2,two_F=two_f,relative_snr=relative_snr,
                    selected=selection,selected_indices=selected_candidates,
                    eligible=eligible,excluded_dwd_id=excluded_bank_ids,
                    residual_input_xyz_fd=residual_input_xyz_fd),
                initial_catalogue=initial_catalogue,bank_association=bank_association,godview=ghost_assessment,
                new_source_assessment=new_source_assessment,new_recovery=new_recovery,
                rjmcmc={key:value for key,value in posterior.items() if key not in ('bases','fit_result','checkpoint_file')},
                recovery=dict(doppler=doppler,coefficients=coefficients,source_xyz_fd=source_xyz_fd,
                    signal_xyz_fd_wide=signal_xyz_fd_wide,residual_xyz_fd=residual_xyz_fd,
                    residual_xyz_fd_wide=residual_xyz_fd_wide,residual_bank_two_f=residual_two_f,
                    parameters=recovered_parameters),metrics=metrics)
        round_attributes = dict(complete=True,fixed_covariance_sha256=digest,accepted_for_subtraction=accepted,
            stopping_reason=stopping_reason if stopped else '',sampler_count_scope='new_sources',
            frozen_source_count=len(previous_doppler),
            source_xyz_fd_scope='Frequency-domain fit model used for association and stopping checks.',
            subtraction_backend='DWD_simulation_gpu + native Instrument + tdi_gpu' if accepted else 'none')
        if accepted:
            last_accepted_round = round_number
        with h5py.File(output_file,'a') as saved:
            group = saved['rounds'].create_group(label)
            _save_tree(group,round_data)
            checkpoints = group.create_group('checkpoints')
            for name, snapshot in checkpoint_data.items():
                write_checkpoint(checkpoints.create_group(name), snapshot)
            group['rjmcmc'].attrs['checkpoint_group'] = checkpoint_group
            group.attrs.update(round_attributes)
            saved.attrs['last_completed_round'] = round_number
            if accepted:
                saved.attrs['last_accepted_round'] = last_accepted_round
            saved.flush()

        if stopped:
            print(f'Stopping: {stopping_reason}; round saved but not subtracted. Keep the previous accepted result.',flush=True)
            break

        # Commit only accepted additions; rejected rounds never change the residual state.
        accumulated_xyz_td = signal_xyz_td
        accumulated_xyz_fd_wide = signal_xyz_fd_wide
        retained_residual_xyz_fd_wide = residual_xyz_fd_wide
        previous_doppler = doppler
        previous_coefficients = coefficients
        previous_sources = source_xyz_fd
        previous_covariance = fixed_covariance
        matched_bank_ids = bank_association['matched_dwd_id']
        excluded_bank_ids = np.union1d(excluded_bank_ids,matched_bank_ids[matched_bank_ids>=0])
        previously_recovered_ids = np.union1d(previously_recovered_ids,new_recovered_ids)

    final_data = dict(
            signal_xyz_td=accumulated_xyz_td,residual_xyz_td=total_xyz_td-accumulated_xyz_td,
            doppler=previous_doppler,signal_xyz_fd=accumulated_xyz_fd_wide[:,core_indices],
            signal_xyz_fd_wide=accumulated_xyz_fd_wide,
            residual_xyz_fd=retained_residual_xyz_fd_wide[:,core_indices],
            residual_xyz_fd_wide=retained_residual_xyz_fd_wide,
        )
    recovery_wall_seconds = time.perf_counter()-started
    with h5py.File(output_file,'a') as saved:
        _save_tree(saved.create_group('final_subtraction'),final_data)
        saved['final_subtraction'].attrs['accepted_round'] = last_accepted_round
        saved.attrs.update(complete=True,iteration_stable=False,stopping_condition_met=stopped,
            recovery_wall_seconds=recovery_wall_seconds,stopping_reason=stopping_reason)
# Finish the recorder before embedding its table and removing the temporary files.
monitor.thread.join()
with open(telemetry_file,newline='') as stream:
    reader = csv.DictReader(stream)
    columns = reader.fieldnames
    raw_rows = list(reader)
rows = [[float(row[key]) if row[key] not in ('','-','N/A') else np.nan for key in columns]
        for row in raw_rows]
usage = np.asarray(rows,dtype=float).reshape(-1,len(columns))
with h5py.File(output_file,'a') as saved:
    telemetry = saved.create_dataset('gpu_usage',data=usage)
    telemetry.attrs['columns'] = columns
    saved.attrs['single_file_complete'] = True
shutil.rmtree(work)
print(f'Final retained round: {last_accepted_round}; stopping reason: {stopping_reason}',flush=True)
print(f'Saved: {output_file}',flush=True)

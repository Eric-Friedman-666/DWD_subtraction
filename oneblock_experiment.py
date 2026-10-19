"""Legacy one-block workflow; all detection/subtraction rounds live outside script/."""
import hashlib
import json
import time
from pathlib import Path
import h5py
import numpy as np
from script.eric_preprocess import effective_noise_cov, periodogram, relative_covariance_change
from script.eric_Fstats import (
    build_response, f_statistic, noise_inner_product,
    physical_amplitude_coordinates, physical_amplitudes, reconstruct_waveform,
)
from script.dwd_bank_association import assign_bank_ids
from script.tool import _save_tree


def iterative_recovery(output,template_file,duration,rate,center,frequency_range,
                       derivative_bounds,maximum_rounds=5,iterations=2500,burn_in=500,
                       termination_policy='stability',ghost_fraction_threshold=.30,
                       f_threshold=64.,exclude_recovered_templates=False,new_sources_only=False,
                       result_file=None,provenance=None):
    from script.eric_rjmcmc import run_eryn_rjmcmc
    from script.tool import single_cpu
    if not np.isfinite(f_threshold) or f_threshold<=0:
        raise ValueError('The screening F threshold must be finite and positive.')
    if new_sources_only and not exclude_recovered_templates:
        raise ValueError('New-source-only rounds require exclusion of already-matched templates.')
    output = Path(output)
    single_file_output = result_file is not None
    result_file = Path(result_file) if single_file_output else output/'experiment.h5'
    with single_cpu():
        started = time.perf_counter()
        print('3/6  Native raw Fourier coefficients; no smoothing or rebinning of data',flush=True)
        with h5py.File(output/'observation.h5') as saved:
            assert saved.attrs['complete']
            total = saved['total_xyz_td'][:]
        full_fft = np.fft.rfft(total,axis=-1)
        count = total.shape[-1]
        del total
        grid = np.fft.rfftfreq(count,d=1/rate)
        padding = frequency_range[1]*1e-4+derivative_bounds[1]*duration+8/duration
        bins = np.flatnonzero((grid>=frequency_range[0]-padding)&(grid<=frequency_range[1]+padding))
        wide_bins = np.arange(int(round(center*duration))-2048,int(round(center*duration))+2049)
        core_indices = bins-wide_bins[0]
        frequency,wide_frequency = grid[bins],grid[wide_bins]
        original,original_wide = full_fft[:,bins],full_fft[:,wide_bins]
        del full_fft
        builder,times,geometry = build_response(output,wide_frequency,duration,rate)
        godview = None
        stop_on_no_new_recovered = termination_policy=='candidates_ghosts_or_no_new_recovered'
        if termination_policy in ('candidates_or_ghosts','candidates_ghosts_or_no_new_recovered'):
            from script.oneblock_godview import PopulationGodView,too_many_ghosts,new_recovered_dwd_ids
            godview = PopulationGodView(output,builder,core_indices,duration,rate)
        elif termination_policy!='stability':
            raise ValueError('Unknown stopping policy.')
        with h5py.File(template_file) as saved:
            bank_ids = saved['templates/dwd_id'][:]
            bank_doppler = np.column_stack([saved['templates/'+key][:] for key in ('f','df','longitude','latitude')])
            if exclude_recovered_templates:
                bank_amplitude = np.asarray(saved['templates/A'][:],dtype=np.float64)
                bank_cos_iota = np.cos(np.asarray(saved['templates/iota'][:],dtype=np.float64))
                bank_coefficients = physical_amplitude_coordinates(
                    bank_amplitude*(1+bank_cos_iota**2),2*bank_amplitude*bank_cos_iota,
                    saved['templates/psi_bme'][:],saved['templates/phi0'][:])/1e-22
        bank_wide = builder.build(bank_doppler)
        bank_bases = bank_wide[:,:,:,core_indices]
        if exclude_recovered_templates:
            bank_waveforms = reconstruct_waveform(bank_coefficients,bank_bases)
        excluded_bank_ids = np.empty(0,dtype=np.int64)
        previous_wave = np.zeros_like(original_wide)
        previous_doppler = np.empty((0,4))
        previous_coefficients = np.empty((0,4))
        previous_sources = np.empty((0,3,len(frequency)),dtype=complex)
        previous_covariance = None
        previously_recovered_ids = np.empty(0,dtype=np.int64)
        stable_rounds = 0
        if single_file_output or not result_file.exists():
            with h5py.File(result_file,'x') as saved:
                _save_tree(saved,dict(config=dict(duration=duration,rate=rate,center=center,
                    frequency_range=frequency_range,derivative_bounds=derivative_bounds,frequency_hz=frequency,
                    wide_frequency_hz=wide_frequency,core_indices=core_indices,retained_bins=bins,
                    iterations=iterations,burn_in=burn_in,maximum_sources=49,maximum_rounds=maximum_rounds,
                    stable_covariance_relative_change=.05,stable_waveform_change=1e-3,
                    termination_policy=termination_policy,ghost_fraction_threshold=ghost_fraction_threshold,
                    recovery_overlap_threshold=.9,f_threshold=f_threshold,
                    exclude_recovered_templates=exclude_recovered_templates,new_sources_only=new_sources_only),
                    data=dict(original_total_xyz_fd=original,original_total_xyz_fd_wide=original_wide),
                    templates=dict(dwd_id=bank_ids,doppler=bank_doppler,bases=bank_bases)))
                saved.create_group('rounds')
                if provenance is not None:
                    _save_tree(saved.create_group('provenance'),provenance)
                if single_file_output:
                    saved.attrs['single_file_complete'] = False
                saved.attrs.update(complete=False,target='DWD recovery conditional on a fixed, data-estimated effective covariance.',
                    covariance_updated_between_rounds_only=True,posterior_samples_pooled_across_rounds=False,
                    catalogue_informed_bank=True,noise_parameters_inferred=False,
                    independent_chain_convergence_assessed=False,last_accepted_round=0,
                    godview_stopping=godview is not None)
                saved.attrs['source_inference_scope'] = ('Additional DWDs conditional on fixed previously accepted waveforms.'
                    if new_sources_only else 'Joint refinement of the complete catalogue.')
        with h5py.File(result_file) as saved:
            saved_threshold = float(saved['config/f_threshold'][()]) if 'config/f_threshold' in saved else 64.
            if saved_threshold!=f_threshold:
                raise ValueError('A different threshold requires a separate experiment directory.')
            encoded_policy = saved['config'].attrs.get('termination_policy')
            saved_policy = (json.loads(encoded_policy) if encoded_policy is not None else
                            ('candidates_or_ghosts' if saved.attrs.get('godview_stopping',False) else 'stability'))
            if saved_policy!=termination_policy:
                raise ValueError('A different stopping policy requires a separate experiment directory.')
            saved_exclusion = bool(saved['config/exclude_recovered_templates'][()]) if 'config/exclude_recovered_templates' in saved else False
            if saved_exclusion!=exclude_recovered_templates:
                raise ValueError('A different template-exclusion policy requires a separate experiment directory.')
            saved_new_only = bool(saved['config/new_sources_only'][()]) if 'config/new_sources_only' in saved else False
            if saved_new_only!=new_sources_only:
                raise ValueError('A different source-refinement policy requires a separate experiment directory.')
        stopped = False
        stopping_reason = 'maximum_rounds_reached'
        for iteration in range(1,maximum_rounds+1):
            label = f'{iteration:02d}'
            with h5py.File(result_file) as saved:
                if label in saved['rounds']:
                    prior = saved['rounds/'+label]
                    if not prior.attrs.get('complete',False):
                        raise RuntimeError('Partial saved round retained; resume its checkpoint explicitly.')
                    if prior.attrs.get('accepted_for_subtraction',True):
                        previous_wave = prior['recovery/signal_xyz_fd_wide'][:]
                        previous_doppler = prior['recovery/doppler'][:]
                        if new_sources_only:
                            previous_coefficients = prior['recovery/coefficients'][:]
                            previous_sources = prior['recovery/source_xyz_fd'][:]
                        previous_covariance = prior['fixed_covariance'][:]
                        if exclude_recovered_templates:
                            matched = prior['bank_association/matched_dwd_id'][:]
                            matched = matched[matched>=0]
                            excluded_bank_ids = (np.union1d(excluded_bank_ids,matched) if new_sources_only else np.unique(matched))
                        if stop_on_no_new_recovered:
                            recovered_ids = prior['godview/matched_dwd_id'][:]
                            previously_recovered_ids = np.union1d(previously_recovered_ids,recovered_ids[recovered_ids>=0])
                    stable_rounds = int(prior['metrics/stable_rounds'][()])
                    if bool(prior['metrics/stopping_conditions_met'][()]):
                        stopped = True
                        stopping_reason = str(prior.attrs.get('stopping_reason',
                            saved.attrs.get('stopping_reason','catalogue_and_background_stable')))
                        break
                    continue
            print(f'4/6  Round {iteration}: estimate and freeze +/-256-bin covariance, then F >= {f_threshold:g} screening',flush=True)
            residual_wide = original_wide-previous_wave
            estimate = dict(covariance=effective_noise_cov(periodogram(residual_wide,duration,rate),
                            neighboring_bins=256,downweight_peaks=True),half_width=256,downweight_peaks=True)
            covariance = estimate['covariance'][core_indices].copy()
            inverse = np.linalg.inv(covariance)
            digest = hashlib.sha256(covariance.tobytes()).hexdigest()
            eligible = ~np.isin(bank_ids,excluded_bank_ids)
            two_f = np.full(len(bank_ids),np.nan)
            if np.any(eligible):
                two_f[eligible] = f_statistic(residual_wide[:,core_indices],bank_bases[eligible],covariance,duration,rate).two_f
            selection = eligible & (two_f>=2*f_threshold)
            selected = np.flatnonzero(selection)
            if exclude_recovered_templates:
                print(f'New-candidate bank: {eligible.sum()} eligible, {(~eligible).sum()} already-matched templates excluded',flush=True)
            if godview is not None and not len(selected):
                stopped,stopping_reason = True,'no_new_candidates'
                with h5py.File(result_file,'a') as saved:
                    if 'termination_screening' not in saved:
                        terminal = saved.create_group('termination_screening')
                        _save_tree(terminal,dict(attempted_round=iteration,F=two_f/2,two_F=two_f,
                            selected=selection,eligible=eligible,excluded_dwd_id=excluded_bank_ids,
                            fixed_covariance=covariance,estimator=estimate,
                            residual_xyz_fd=residual_wide[:,core_indices]))
                        terminal.attrs['fixed_covariance_sha256'] = digest
                print(f'No new F >= {f_threshold:g} candidates: retain the last accepted RJMCMC catalogue; no empty follow-up round.',flush=True)
                break
            initial = (bank_doppler[selected] if new_sources_only else np.concatenate((previous_doppler,bank_doppler[selected])))
            if len(initial)>49:
                raise RuntimeError('Initial catalogue exceeds the documented N<=49 prior; no candidates were silently pruned.')
            if new_sources_only:
                print(f'Round {iteration}: {len(selected)} new candidates active; {len(previous_doppler)} earlier components frozen and subtracted; N counts new sources only',flush=True)
            else:
                print(f'Round {iteration}: {len(selected)} new threshold candidates, {len(previous_doppler)} carried components; all {len(initial)} active',flush=True)
            print('5/6  One Eryn ladder, four replicas, continuous Doppler and birth/death updates',flush=True)
            sampling_data = residual_wide[:,core_indices] if new_sources_only else original
            posterior = run_eryn_rjmcmc(sampling_data,covariance,
                initial,frequency_range,derivative_bounds,center,duration,rate,
                output/f'round_{label}_checkpoints',chain_count=1,iterations=iterations,burn_in=burn_in,
                maximum_sources=49,response_times=times,response_geometry=geometry,physical_frequency=frequency)
            assert posterior['fixed_covariance_sha256']==digest
            coefficients = posterior['fit_result']['coefficients']
            source_wide = reconstruct_waveform(coefficients,builder.build(posterior['doppler']))
            source_core = source_wide[:,:,core_indices]
            doppler = posterior['doppler']
            wave_wide = source_wide.sum(axis=0)
            new_recovery = {}
            if new_sources_only:
                new_recovery = dict(doppler=doppler,coefficients=coefficients,source_xyz_fd=source_core,
                    signal_xyz_fd=wave_wide[:,core_indices],signal_xyz_fd_wide=wave_wide)
                doppler = np.concatenate((previous_doppler,doppler))
                coefficients = np.concatenate((previous_coefficients,coefficients))
                source_core = np.concatenate((previous_sources,source_core))
                wave_wide = previous_wave+wave_wide
            bank_association = {}
            if exclude_recovered_templates:
                # Identify against the bank only; injection labels never decide this mask.
                bank_association = assign_bank_ids(source_core,bank_waveforms,bank_ids,
                    inverse,duration,rate,minimum_overlap=.9)
            residual_wide = original_wide-wave_wide
            residual = residual_wide[:,core_indices]
            residual_two_f = f_statistic(residual,bank_bases,covariance,duration,rate).two_f
            change = wave_wide[:,core_indices]-previous_wave[:,core_indices]
            norm = lambda value:float(noise_inner_product(value,value,inverse,duration,rate))
            waveform_change = norm(change)/max(norm(wave_wide[:,core_indices]),np.finfo(float).tiny)
            covariance_change = (np.inf if previous_covariance is None else relative_covariance_change(previous_covariance,covariance))
            stable = (len(previous_doppler)==len(doppler) and covariance_change<.05
                      and waveform_change<1e-3 and np.max(residual_two_f)<2*f_threshold)
            stable_rounds = stable_rounds+1 if stable else 0
            stopped = stable_rounds>=2
            accepted,ghost_assessment,new_source_assessment = True,{},{}
            if godview is not None:
                ghost_assessment = godview(source_core,covariance)
                round_assessment = ghost_assessment
                if new_sources_only:
                    matched = ghost_assessment['matched_dwd_id'][len(previous_doppler):]
                    duplicate = np.isin(matched,previously_recovered_ids) & (matched>=0)
                    matched = np.where(duplicate,-1,matched)
                    ghosts = int((matched<0).sum())
                    new_source_assessment = dict(matched_dwd_id=matched,ghost_count=ghosts,source_count=len(matched),
                        ghost_fraction=ghosts/max(len(matched),1),duplicate_previous_detection=duplicate,
                        scope='Post-fit stopping assessment of additions only; earlier sources cannot dilute the ghost fraction.')
                    round_assessment = new_source_assessment
                stopped = too_many_ghosts(round_assessment['ghost_count'],round_assessment['source_count'],ghost_fraction_threshold)
                if stopped:
                    stopping_reason = 'ghost_fraction_exceeded'
                if stop_on_no_new_recovered:
                    new_ids = new_recovered_dwd_ids(round_assessment['matched_dwd_id'],previously_recovered_ids)
                    ghost_assessment.update(new_recovered_dwd_id=new_ids,new_recovered_count=len(new_ids),
                        previously_recovered_count=len(previously_recovered_ids))
                    print(f'Post-RJMCMC new recovered DWDs: {len(new_ids)} (unique matched IDs, excluding ghosts)',flush=True)
                    if not stopped and not len(new_ids):
                        stopped,stopping_reason = True,'no_new_recovered_dwd'
                accepted = not stopped
                print(f'Post-RJMCMC {"new-source " if new_sources_only else ""}god-view: {round_assessment["ghost_count"]}/{round_assessment["source_count"]} ghosts '
                      f'({round_assessment["ghost_fraction"]:.1%}); fixed stop cutoff > {ghost_fraction_threshold:.0%}; '
                      f'{"accept round" if accepted else "reject round and keep previous accepted subtraction"}',flush=True)
                if stopped:
                    print(f'Stopping: {stopping_reason}; this confirmation round is saved but not subtracted.',flush=True)
            elif stopped:
                stopping_reason = 'catalogue_and_background_stable'
            metrics = dict(covariance_relative_change=covariance_change,waveform_relative_change=waveform_change,
                           residual_norm=norm(residual),original_norm=norm(original),max_residual_F=float(np.max(residual_two_f)/2),
                           stable_rounds=stable_rounds,stopping_conditions_met=stopped,catalogue_count=len(doppler))
            count_message = (f'modal N_new={posterior["modal_count"]}, proposed cumulative N={len(doppler)}'
                if new_sources_only else f'modal N={posterior["modal_count"]}')
            print(f'6/6  Rebuild residual from original observation: {count_message}, max residual F={metrics["max_residual_F"]:.3f}, covariance change={covariance_change:.3g}',flush=True)
            with h5py.File(result_file,'a') as saved:
                group = saved['rounds'].create_group(label)
                _save_tree(group,dict(estimator=estimate,fixed_covariance=covariance,
                    screening=dict(F=two_f/2,two_F=two_f,selected=selection,selected_indices=selected,
                                   eligible=eligible,excluded_dwd_id=excluded_bank_ids,
                                   residual_input_xyz_fd=(original_wide-previous_wave)[:,core_indices]),
                    initial_catalogue=initial,bank_association=bank_association,godview=ghost_assessment,
                    new_source_assessment=new_source_assessment,new_recovery=new_recovery,
                    rjmcmc={key:value for key,value in posterior.items() if key not in ('bases','fit_result')},
                    recovery=dict(doppler=doppler,coefficients=coefficients,source_xyz_fd=source_core,
                        signal_xyz_fd_wide=wave_wide,residual_xyz_fd=residual,residual_xyz_fd_wide=residual_wide,
                        residual_bank_two_f=residual_two_f,parameters=physical_amplitudes(coefficients)),metrics=metrics))
                if single_file_output:
                    # Preserve the native Eryn backend and progress state inside this round.
                    checkpoints = group.create_group('checkpoints')
                    for path in sorted((output/f'round_{label}_checkpoints').glob('*.h5')):
                        with h5py.File(path,'r') as source:
                            target = checkpoints.create_group(path.stem)
                            target.attrs.update(dict(source.attrs))
                            for key in source:
                                source.copy(key,target)
                    group['rjmcmc'].attrs.pop('checkpoint_file',None)
                    group['rjmcmc'].attrs['checkpoint_group'] = json.dumps(checkpoints.name)
                group.attrs.update(complete=True,fixed_covariance_sha256=digest,accepted_for_subtraction=accepted,
                                   stopping_reason=stopping_reason if stopped else '',
                                   sampler_count_scope='new_sources' if new_sources_only else 'complete_catalogue',
                                   frozen_source_count=len(previous_doppler) if new_sources_only else 0)
                saved.attrs['last_completed_round'] = iteration
                if accepted:
                    saved.attrs['last_accepted_round'] = iteration
                saved.flush()
            if accepted:
                previous_wave,previous_doppler,previous_covariance = wave_wide,doppler,covariance
                if new_sources_only:
                    previous_coefficients,previous_sources = coefficients,source_core
                if exclude_recovered_templates:
                    matched = bank_association['matched_dwd_id']
                    matched = matched[matched>=0]
                    excluded_bank_ids = (np.union1d(excluded_bank_ids,matched) if new_sources_only else np.unique(matched))
                if stop_on_no_new_recovered:
                    previously_recovered_ids = np.union1d(previously_recovered_ids,new_ids)
            if stopped:
                break
        with h5py.File(result_file,'a') as saved:
            saved.attrs.update(complete=True,iteration_stable=stopped if godview is None else False,
                stopping_condition_met=stopped,recovery_wall_seconds=time.perf_counter()-started,stopping_reason=stopping_reason)
            if godview is not None and 'final_subtraction' not in saved:
                final = saved.create_group('final_subtraction')
                _save_tree(final,dict(doppler=previous_doppler,signal_xyz_fd=previous_wave[:,core_indices],
                    signal_xyz_fd_wide=previous_wave,residual_xyz_fd=original-previous_wave[:,core_indices],
                    residual_xyz_fd_wide=original_wide-previous_wave))
                final.attrs['accepted_round'] = saved.attrs['last_accepted_round']
        print('Saved',result_file,flush=True)

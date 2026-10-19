"""Separate parallel-tempering experiment; production prior, likelihood and moves are reused."""
from pathlib import Path
import hashlib
import numpy as np
from scipy.linalg import solve_triangular
from tqdm import tqdm
from eryn.backends import HDFBackend
from eryn.ensemble import EnsembleSampler
from eryn.state import State
from script.tool import single_cpu
from eric_rjmcmc import (CataloguePrior, BirthDistribution, BirthDeathMove, DWDMove,
    Likelihood, GPUResponse, encode_doppler, decode_doppler)


def run_eryn_rjmcmc(data, covariance, dwd_template_table,
        frequency_range, derivative_bounds, T_obs, rate,
        checkpoint_directory, *, response_times, response_geometry, physical_frequency,
        iterations, burn_in, maximum_sources=64, amplitude_prior_sigma=10.):
    """Test the same fit with four fixed temperatures; report only beta=1 samples."""
    with single_cpu():
        betas = np.array([1., .8, .6, .4])
        f_center = np.mean(frequency_range)
        checkpoint_file = Path(checkpoint_directory)/'eryn_chain_00.h5'
        chirp_bounds = np.asarray(derivative_bounds)*T_obs**2
        chirp_log_uniform = chirp_bounds.ndim == 1
        chirp_minimum = 0. if chirp_log_uniform else abs(chirp_bounds[0, 1])
        encoded_para_bounds = np.array([
            np.log10(np.asarray(frequency_range)/f_center)*f_center*T_obs*np.log(10.),
            chirp_bounds.ravel()[[0, -1]], [0., 2*np.pi], [-1., 1.]])
        encoded_para_centers = encode_doppler(dwd_template_table, f_center, T_obs, chirp_coordinate='linear')
        engine = GPUResponse(covariance, data, response_times, response_geometry,
                             physical_frequency, T_obs, rate)
        likelihood = Likelihood(engine, f_center, T_obs, amplitude_prior_sigma,
                                maximum_sources=maximum_sources, chirp_coordinate='linear')
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        prior = CataloguePrior(encoded_para_bounds, maximum_sources, chirp_minimum,
                               chirp_log_uniform=chirp_log_uniform)
        within = DWDMove({'dwd': np.diag(np.array([.025, .3, .015, .015])**2)})
        births = BirthDistribution(encoded_para_centers, encoded_para_bounds, chirp_minimum,
                                   chirp_log_uniform=chirp_log_uniform)
        jump = BirthDeathMove({'dwd': births}, nleaves_min={'dwd': 0},
                             nleaves_max={'dwd': maximum_sources})
        backend = HDFBackend(str(checkpoint_file))
        sampler = EnsembleSampler(
            1, {'dwd': 4}, likelihood, {'dwd': prior, 'all_models_together': prior},
            branch_names=['dwd'], nleaves_min={'dwd': 0}, nleaves_max={'dwd': maximum_sources},
            moves=within, rj_moves=jump, pool=None, backend=backend,
            tempering_kwargs=dict(betas=betas, adaptive=False, permute=False),
            fill_zero_leaves_val=likelihood(np.empty((0, 4))),
        )
        para_doppler = np.zeros((len(betas), 1, maximum_sources, 4))
        active = np.zeros(para_doppler.shape[:-1], dtype=bool)
        initial_count = min(len(encoded_para_centers), maximum_sources)
        para_doppler[:, 0, :initial_count], active[:, 0, :initial_count] = encoded_para_centers[:initial_count], True
        state = State({'dwd': para_doppler}, inds={'dwd': active})
        initial_logm = likelihood(encoded_para_centers[:initial_count])
        initial_prior = prior.logpdf(state.branches_coords, state.branches_inds)[0, 0]
        para_doppler_draws, a_draws, count_draws, logm_draws = [], [], [], []
        amplitude_rng = np.random.default_rng()
        progress = tqdm(sampler.sample(state, iterations=iterations), total=iterations,
                        desc='Eryn RJMCMC', unit='cycle', dynamic_ncols=True)
        for cycle, state in enumerate(progress, 1):
            progress.set_postfix(N=state.branches_inds['dwd'].sum(axis=-1)[:, 0].tolist(), refresh=False)
            if cycle <= burn_in:
                continue
            para_doppler = state.branches_coords['dwd'][0, 0, state.branches_inds['dwd'][0, 0]]
            logm, A, lower = likelihood.details(para_doppler)
            draw = A+(solve_triangular(lower.T, amplitude_rng.normal(size=len(A)),
                                       lower=False) if len(A) else A)
            order = np.argsort(para_doppler[:, 0])
            padded_para_doppler, padded_a = np.full((maximum_sources, 4), np.nan), np.full((maximum_sources, 4), np.nan)
            padded_para_doppler[:len(para_doppler)], padded_a[:len(para_doppler)] = para_doppler[order], draw.reshape(-1, 4)[order]
            para_doppler_draws.append(padded_para_doppler)
            a_draws.append(padded_a)
            count_draws.append(len(para_doppler))
            logm_draws.append(logm)
        probability = np.bincount(count_draws, minlength=maximum_sources+1)/len(count_draws)
        modal = int(np.argmax(probability))
        selected = np.flatnonzero(np.asarray(count_draws) == modal)
        best = selected[np.argmax(np.asarray(logm_draws)[selected])]
        doppler = decode_doppler(para_doppler_draws[best][:modal], f_center, T_obs,
                                 chirp_coordinate='linear')
        bases = engine.build(doppler)
        log_marginal, mean, _ = likelihood.details(para_doppler_draws[best][:modal])
        counts = np.r_[initial_count, backend.get_nleaves()['dwd'][:, 0, 0]]
        logm = np.r_[initial_logm, backend.get_log_like()[:, 0, 0]]
        logp = np.r_[initial_prior, backend.get_log_prior()[:, 0, 0]]
        cycles = np.arange(iterations+1)
        chain = dict(sampled_q=np.array(para_doppler_draws), sampled_amplitudes=np.array(a_draws),
            source_count=np.array(count_draws), log_marginal=np.array(logm_draws),
            cycle_state=dict(cycle=cycles, is_burn_in=cycles <= burn_in, source_count=counts,
                log_marginal=logm, log_prior=logp, log_posterior=logm+logp))
        return dict(doppler=doppler, bases=bases,
            fit_result=dict(coefficients=mean.reshape(-1, 4), log_marginal=log_marginal),
            modal_count=modal, count_probability=probability,
            count_support=np.arange(maximum_sources+1), chains={'0': chain},
            tempering=dict(inverse_temperatures=betas,
                cycle_source_count=np.vstack((np.full(len(betas), initial_count), backend.get_nleaves()['dwd'][:, :, 0])),
                cycle_log_marginal=np.vstack((np.full(len(betas), initial_logm), backend.get_log_like()[:, :, 0])),
                cycle_log_prior=np.vstack((np.full(len(betas), initial_prior), backend.get_log_prior()[:, :, 0])),
                backend_swap_acceptances=np.asarray(backend.swaps_accepted),
                gaussian_acceptances=np.asarray(backend.accepted),
                rj_acceptances=np.asarray(backend.rj_accepted)),
            candidate_centers=encoded_para_centers,
            config=dict(bounds=encoded_para_bounds, chirp_minimum=chirp_minimum,
                chirp_log_uniform=chirp_log_uniform, chirp_coordinate='linear',
                inverse_temperatures=betas, adaptive_temperatures=False,
                maximum_sources=maximum_sources, center=f_center, duration=T_obs, rate=rate,
                iterations=iterations, burn_in=burn_in, initial_count=initial_count,
                amplitude_prior_sigma=amplitude_prior_sigma),
            checkpoint_file=str(checkpoint_file),
            fixed_covariance_sha256=hashlib.sha256(covariance.tobytes()).hexdigest())

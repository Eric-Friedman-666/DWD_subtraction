"""DWD priors, amplitude-marginal likelihood, and GPU-only Eryn sampling."""

from functools import lru_cache
from importlib.metadata import version
from pathlib import Path
import hashlib
import time

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import comb, gammaln
from tqdm.auto import tqdm
from eryn.backends import HDFBackend
from eryn.ensemble import EnsembleSampler
from eryn.moves import DistributionGenerateRJ, GaussianMove
from eryn.prior import ProbDistContainer
from eryn.state import State

from script.eric_generate_waveform import GPUResponse
from script.eric_preprocess import inner_product
from script.tool import single_cpu


# 1. Priors and sampled coordinates
def encode_doppler(para_doppler, f_center, T_obs, *, chirp_coordinate='log10'):
    para_doppler = np.asarray(para_doppler)
    if para_doppler.dtype.names:
        para_doppler = np.column_stack([para_doppler[name] for name in
            ('Frequency', 'FrequencyDerivative', 'EclipticLongitude', 'EclipticLatitude')])
    elif para_doppler.ndim == 2 and para_doppler.shape[1] == 11:
        para_doppler = para_doppler[:, [2, 3, 6, 7]]
    else:
        para_doppler = para_doppler.reshape(-1, 4)
    chirp = (para_doppler[:, 1]*T_obs**2 if chirp_coordinate == 'linear'
             else np.log10(para_doppler[:, 1]))
    return np.column_stack((np.log10(para_doppler[:, 0]/f_center)*f_center*T_obs*np.log(10.),
                            chirp, para_doppler[:, 2] % (2*np.pi),
                            np.sin(para_doppler[:, 3])))


def decode_doppler(para_doppler, f_center, T_obs, *, chirp_coordinate='log10'):
    para_doppler = np.asarray(para_doppler).reshape(-1, 4)
    chirp = (para_doppler[:, 1]/T_obs**2 if chirp_coordinate == 'linear'
             else 10.**para_doppler[:, 1])
    return np.column_stack((f_center*10.**(para_doppler[:, 0]/(f_center*T_obs*np.log(10.))),
                            chirp, para_doppler[:, 2] % (2*np.pi),
                            np.arcsin(np.clip(para_doppler[:, 3], -1., 1.))))


class CataloguePrior:
    def __init__(self, bounds, maximum=49):
        self.bounds, self.count_maximum = np.asarray(bounds), maximum
        self.V = np.prod(self.bounds[:, 1] - self.bounds[:, 0])
        self.DWD_para_order = [0, 1, 2, 3]
        self.key_order = self.DWD_para_order

    def logpdf(self, coords, inds, **kwargs):
        para_doppler, active = coords['dwd'], inds['dwd']
        N = active.sum(axis=-1)
        M = self.count_maximum
        V = self.V
        value = -N * np.log(V) - np.log(comb(M, N))
        outside = np.any((para_doppler < self.bounds[:, 0]) | (para_doppler > self.bounds[:, 1])
                         | ~np.isfinite(para_doppler), axis=-1) & active
        return np.where(outside.any(axis=-1), -np.inf, value)


class BirthDistribution:
    """90% initial-candidate boxes; 10% full-prior support."""
    def __init__(self, centers, bounds):
        self.centers, self.bounds = centers, bounds
        self.lower_limit = np.maximum(centers-[.2, 0., .2, .2], bounds[:, 0])
        self.upper_limit = np.minimum(centers+[.2, 0., .2, .2], bounds[:, 1])
        self.lower_limit[:, 1], self.upper_limit[:, 1] = bounds[1]
        self.lower_limit[:, 2], self.upper_limit[:, 2] = centers[:, 2]-.2, centers[:, 2]+.2

    def logpdf(self, para_doppler):
        para_doppler = np.asarray(para_doppler).reshape(-1, 4)
        broad = 1./np.prod(self.bounds[:, 1]-self.bounds[:, 0])
        if not len(self.centers):
            return np.full(len(para_doppler), np.log(broad))
        delta = (para_doppler[:, None, 2]-self.centers[:, 2]+np.pi) % (2*np.pi)-np.pi
        inside = np.all((para_doppler[:, None, [0, 1, 3]] >= self.lower_limit[:, [0, 1, 3]])
                        & (para_doppler[:, None, [0, 1, 3]] <= self.upper_limit[:, [0, 1, 3]]), axis=-1)
        inside &= np.abs(delta) <= .5*(self.upper_limit[:, 2]-self.lower_limit[:, 2])
        density = .1*broad+.9*(inside/np.prod(self.upper_limit-self.lower_limit, axis=1)).mean(axis=-1)
        return np.log(density)

    def rvs(self, size=1):
        shape = (size,) if isinstance(size, int) else tuple(size)
        para_doppler = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1], size=shape+(4,))
        if len(self.centers):
            flat = para_doppler.reshape(-1, 4)
            selected = np.flatnonzero(np.random.random(len(flat)) >= .1)
            candidate = np.random.randint(len(self.centers), size=len(selected))
            flat[selected] = np.random.uniform(self.lower_limit[candidate], self.upper_limit[candidate])
        para_doppler[..., 2] %= 2*np.pi
        return para_doppler


# 2. Likelihood
def marginal_likelihood(data, bases, noise_covariance, log_determinant, T_obs, rate,
                        amplitude_prior_sigma=10.0):
    """Full 4N Gaussian integral using the noise covariance, not its inverse."""
    data_norm = inner_product(data, data, noise_covariance, T_obs, rate)
    if len(bases):
        h = bases.reshape(len(bases)*4, *data.shape)
        x = np.asarray([inner_product(a, data, noise_covariance, T_obs, rate) for a in h])
        M = np.asarray([[inner_product(a, b, noise_covariance, T_obs, rate)
                         for b in h] for a in h])
        M = 0.5 * (M + M.T)
        precision = M + np.eye(len(M)) / amplitude_prior_sigma**2
        chol, lower = cho_factor(precision, lower=True, check_finite=False)
        mean = cho_solve((chol, lower), x, check_finite=False)
        covariance = cho_solve((chol, lower), np.eye(len(M)), check_finite=False)
        penalty = np.log(np.diag(chol)).sum() + len(M)*np.log(amplitude_prior_sigma)
        improvement = 0.5 * x @ mean - penalty
        coefficients = mean.reshape(-1, 4)
        model = np.einsum('am,amck->ck', coefficients, bases, optimize=True)
    else:
        M = covariance = np.empty((0, 0))
        x = np.empty(0)
        coefficients = np.empty((0, 4))
        improvement = penalty = 0.0
        model = np.zeros_like(data)
    residual = data - model
    residual_norm = inner_product(residual, residual, noise_covariance, T_obs, rate)
    return dict(log_marginal=float(-0.5*data_norm-log_determinant+improvement),
                log_likelihood=float(-0.5*residual_norm-log_determinant),
                coefficients=coefficients, amplitude_covariance=covariance,
                projection=x, data_norm=float(data_norm), log_determinant=float(log_determinant),
                amplitude_matrix=M, log_occam_penalty=float(penalty),
                reconstructed=model, residual=residual, residual_norm=float(residual_norm))


class Likelihood:
    def __init__(self, engine, f_center, T_obs, sigma=10., *, maximum_sources=49, chirp_coordinate='log10'):
        self.f_center, self.T_obs, self.sigma = f_center, T_obs, sigma
        self.chirp_coordinate = chirp_coordinate
        self.engine, self.count_maximum = engine, maximum_sources
        self.white_data = engine.white_data
        self.data_norm = float(self.white_data @ self.white_data)
        self.waveform_evaluations, self.calls = 0, 0
        self.source = lru_cache(maxsize=256)(self._source)
        self.cached_details = lru_cache(maxsize=64)(self._details)
        self.integral = jax.jit(self._integral)
        self.zero = jax.device_put(np.zeros((4, self.white_data.size)), engine.device)

    def _source(self, key):
        para_doppler = np.frombuffer(key, dtype=np.float64)
        point = decode_doppler(para_doppler[None], self.f_center, self.T_obs,
                               chirp_coordinate=self.chirp_coordinate)[0]
        self.waveform_evaluations += 1
        return self.engine.white_single(jax.device_put(point, self.engine.device),
                                        *self.engine.arguments)

    def _integral(self, white):
        gram = white @ white.T
        lower = jnp.linalg.cholesky(gram+jnp.eye(len(white))/self.sigma**2)
        x = white @ self.white_data
        A = jsl.cho_solve((lower, True), x)
        logm = (-.5*self.data_norm+.5*x@A
                -jnp.log(jnp.diag(lower)).sum()-len(white)*jnp.log(self.sigma))
        return logm, A, lower

    def _details(self, key):
        para_doppler = np.frombuffer(key, dtype=np.float64).reshape(-1, 4)
        if not len(para_doppler):
            return -.5*self.data_norm, np.empty(0), np.empty((0, 0))
        white = [self.source(point.tobytes()) for point in para_doppler]
        white += [self.zero]*(self.count_maximum-len(para_doppler))
        logm, A, lower = self.integral(jnp.concatenate(white, axis=0))
        return float(logm), np.asarray(A)[:4*len(para_doppler)], np.asarray(lower)[:4*len(para_doppler), :4*len(para_doppler)]

    def details(self, para_doppler):
        return self.cached_details(np.ascontiguousarray(para_doppler, dtype=np.float64).tobytes())

    def __call__(self, para_doppler):
        self.calls += 1
        return self.details(para_doppler)[0]


# 3. Eryn samplers
class BirthDeathMove(DistributionGenerateRJ):
    def get_proposal(self, coords, inds, *args, **kwargs):
        para_doppler, new_inds, factors = super().get_proposal(coords, inds, *args, **kwargs)
        n, new_n = inds['dwd'].sum(axis=-1), new_inds['dwd'].sum(axis=-1)
        maximum = inds['dwd'].shape[-1]
        # Slot-selection ratio paired with CataloguePrior's uniform count prior.
        factors += (gammaln(n+1)+gammaln(maximum-n+1)
                    -gammaln(new_n+1)-gammaln(maximum-new_n+1))
        return para_doppler, new_inds, factors


class DWDMove(GaussianMove):
    def get_proposal(self, coords, random, branches_inds=None, **kwargs):
        active = branches_inds['dwd']
        slot = np.argmax(np.where(active, random.random(active.shape), -1.), axis=-1)
        selected = np.zeros_like(active)
        np.put_along_axis(selected, slot[..., None], active.any(axis=-1)[..., None], axis=-1)
        para_doppler, factors = super().get_proposal(coords, random, branches_inds={'dwd': selected})
        beta = self.temperature_control.betas[:, None, None, None]
        para_doppler['dwd'] = coords['dwd']+(para_doppler['dwd']-coords['dwd'])/np.sqrt(beta)
        para_doppler['dwd'][..., 2] %= 2*np.pi
        return para_doppler, factors


def run_eryn_rjmcmc(data, covariance, bank_doppler,
        frequency_range, derivative_bounds, f_center, T_obs, rate,
        checkpoint_directory, *, response_times, response_geometry, physical_frequency,
        chain_count=1, inverse_temperatures=(1., .8, .6, .4),
        iterations=6000, burn_in=1000, maximum_sources=49, amplitude_prior_sigma=10.,
        chirp_coordinate='log10'):
    """Fit one residual with fixed covariance on GPU; the caller owns outer rounds."""
    with single_cpu():
        started = time.perf_counter()
        maximum, sigma = maximum_sources, amplitude_prior_sigma
        betas = np.asarray(inverse_temperatures)
        checkpoint_file = Path(checkpoint_directory)/'eryn_chain_00.h5'
        covariance = np.array(covariance, dtype=np.complex128, copy=True)
        covariance_sha = hashlib.sha256(covariance.tobytes()).hexdigest()
        scale = f_center*T_obs*np.log(10.)
        chirp_bounds = np.asarray(derivative_bounds, dtype=float)
        chirp_bounds = chirp_bounds*T_obs**2 if chirp_coordinate == 'linear' else np.log10(chirp_bounds)
        bounds = np.array([np.log10(np.asarray(frequency_range)/f_center)*scale,
                           chirp_bounds, [0., 2*np.pi], [-1., 1.]])
        centers = encode_doppler(bank_doppler, f_center, T_obs, chirp_coordinate=chirp_coordinate)

        engine = GPUResponse(covariance, data, response_times, response_geometry,
                             physical_frequency, T_obs, rate)
        likelihood = Likelihood(engine, f_center, T_obs, sigma, maximum_sources=maximum,
                                chirp_coordinate=chirp_coordinate)
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        prior = CataloguePrior(bounds, maximum)
        within = DWDMove({'dwd': np.diag(np.array([.025, .3, .015, .015])**2)})
        births = ProbDistContainer({(0, 1, 2, 3): BirthDistribution(centers, bounds)})
        jump = BirthDeathMove({'dwd': births}, nleaves_min={'dwd': 0},
                             nleaves_max={'dwd': maximum}, prevent_swaps=True)
        backend = HDFBackend(str(checkpoint_file))
        sampler = EnsembleSampler(
            1, {'dwd': 4}, likelihood, {'dwd': prior, 'all_models_together': prior},
            branch_names=['dwd'], nleaves_min={'dwd': 0}, nleaves_max={'dwd': maximum},
            moves=within, rj_moves=jump, pool=None, backend=backend,
            tempering_kwargs=dict(betas=betas, adaptive=False, permute=False),
            fill_zero_leaves_val=likelihood(np.empty((0, 4))),
        )
        para_doppler = np.zeros((len(betas), 1, maximum, 4))
        active = np.zeros(para_doppler.shape[:-1], dtype=bool)
        para_doppler[:, 0, :len(centers)], active[:, 0, :len(centers)] = centers, True
        state = State({'dwd': para_doppler}, inds={'dwd': active})
        initial_counts = active[:, 0].sum(axis=-1)
        initial_logm = np.full(len(betas), likelihood(centers))
        initial_prior = prior.logpdf(state.branches_coords, state.branches_inds)[:, 0]
        setup_seconds = time.perf_counter()-started

        para_doppler_draws, a_draws, count_draws, logm_draws = [], [], [], []
        amplitude_rng = np.random.default_rng()
        progress = tqdm(sampler.sample(state, iterations=iterations), total=iterations,
                        desc='Eryn RJMCMC', unit='cycle', dynamic_ncols=True)
        for cycle, state in enumerate(progress, 1):
            progress.set_postfix(N=state.branches_inds['dwd'][:, 0].sum(axis=-1).tolist(),
                                 refresh=False)
            if cycle <= burn_in:
                continue
            para_doppler = state.branches_coords['dwd'][0, 0, state.branches_inds['dwd'][0, 0]]
            logm, A, lower = likelihood.details(para_doppler)
            draw = A+(solve_triangular(lower.T, amplitude_rng.normal(size=len(A)),
                                       lower=False) if len(A) else A)
            order = np.argsort(para_doppler[:, 0])
            padded_para_doppler, padded_a = np.full((maximum, 4), np.nan), np.full((maximum, 4), np.nan)
            padded_para_doppler[:len(para_doppler)], padded_a[:len(para_doppler)] = para_doppler[order], draw.reshape(-1, 4)[order]
            para_doppler_draws.append(padded_para_doppler)
            a_draws.append(padded_a)
            count_draws.append(len(para_doppler))
            logm_draws.append(logm)

        probability = np.bincount(count_draws, minlength=maximum+1)/len(count_draws)
        modal = int(np.argmax(probability))
        selected = np.flatnonzero(np.asarray(count_draws) == modal)
        best = selected[np.argmax(np.asarray(logm_draws)[selected])]
        doppler = decode_doppler(para_doppler_draws[best][:modal], f_center, T_obs,
                                 chirp_coordinate=chirp_coordinate)
        bases = engine.build(doppler)
        fit = marginal_likelihood(data, bases, covariance, 0., T_obs, rate, sigma)
        counts = np.vstack((initial_counts, backend.get_nleaves()['dwd'][:, :, 0]))
        logm = np.vstack((initial_logm, backend.get_log_like()[:, :, 0]))
        logp = np.vstack((initial_prior, backend.get_log_prior()[:, :, 0]))
        chain = dict(sampled_q=np.array(para_doppler_draws), sampled_amplitudes=np.array(a_draws),
            source_count=np.array(count_draws), log_marginal=np.array(logm_draws),
            trace_counts=counts[1:], trace_logm=logm[1:], initial_replica_counts=initial_counts,
            cycle_state=dict(cycle=np.arange(iterations+1),
                is_burn_in=np.arange(iterations+1) <= burn_in, source_count=counts,
                log_marginal=logm, log_prior=logp, log_posterior=logm*betas+logp))
        result = dict(doppler=doppler, bases=bases, fit_result=fit, modal_count=modal,
            count_probability=probability, count_support=np.arange(maximum+1), chains={'0': chain},
            candidate_centers=centers, initial_replica_counts=initial_counts,
            config=dict(bounds=bounds, maximum_sources=maximum, center=f_center, duration=T_obs,
                rate=rate, iterations=iterations, burn_in=burn_in, thin=1, chain_count=1,
                inverse_temperatures=betas, amplitude_prior_sigma=sigma,
                chirp_coordinate=chirp_coordinate,
                initialization='all_candidates', initial_selection=None,
                use_gpu=True, backend='gpu',
                independent_chain_convergence_assessed=False),
            engine='Eryn '+version('eryn'), checkpoint_file=str(checkpoint_file),
            fixed_covariance_sha256=covariance_sha, backend_setup_seconds=setup_seconds,
            runtime_seconds=time.perf_counter()-started,
            waveform_evaluations=likelihood.waveform_evaluations, likelihood_calls=likelihood.calls,
            count_prior='Uniform N=0..maximum_sources; exchangeable active slots.',
            amplitude_prior=f'Normalized zero-mean Gaussian sigma={sigma} on each A_mu/1e-22.',
            proposal='Eryn Gaussian MH and distribution RJ; single-source updates; 90% candidate-box births.',
            estimate='Best cold marginal likelihood at modal N; conditional-mean amplitudes.')
        result.update(gpu_device=str(engine.device), gpu_device_kind=engine.device.device_kind)
        return result

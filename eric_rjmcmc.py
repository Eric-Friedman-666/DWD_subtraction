"""DWD priors, amplitude-marginal likelihood, and GPU-only Eryn sampling."""
from functools import lru_cache
from pathlib import Path
import hashlib
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
from eryn.state import State
from script.eric_generate_waveform import GPUResponse
from script.eric_preprocess import inner_product
from script.tool import single_cpu
def encode_doppler(para_doppler, f_center, T_obs, *, chirp_coordinate='log10'):
    para_doppler = np.asarray(para_doppler)
    para_doppler = para_doppler[:, [2, 3, 6, 7]]
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
    @staticmethod
    def chirp_log_volume(bounds, minimum):
        """Normalized log-magnitude support; either positive or both signs."""
        lower, upper = bounds
        if not (np.isfinite([lower, upper, minimum]).all() and minimum >= 0.):
            raise ValueError('Log-chirp bounds and minimum must be finite, with minimum >= 0.')
        if minimum:
            if not (0. < minimum < min(-lower, upper)):
                raise ValueError('Signed log-chirp bounds must straddle a positive minimum magnitude.')
            return np.log(-lower / minimum) + np.log(upper / minimum)
        if not (0. < lower < upper):
            raise ValueError('Positive log-chirp bounds must satisfy 0 < lower < upper.')
        return np.log(upper / lower)
    def __init__(self, encoded_para_bounds, dwd_block_max=64, chirp_minimum=0., *, chirp_log_uniform=False):
        self.encoded_para_bounds, self.dwd_block_max = np.asarray(encoded_para_bounds, dtype=float), dwd_block_max
        self.chirp_minimum = chirp_minimum
        self.chirp_log_uniform = chirp_log_uniform
        encoded_para_widths = self.encoded_para_bounds[:, 1] - self.encoded_para_bounds[:, 0]
        encoded_para_widths[1] -= 2 * chirp_minimum
        if chirp_log_uniform:
            encoded_para_widths[1] = self.chirp_log_volume(self.encoded_para_bounds[1], chirp_minimum)
        self.V = np.prod(encoded_para_widths)
        self.key_order = [0, 1, 2, 3]
    def logpdf(self, encoded_dwd_parameters, inds, **kwargs):
        para_doppler, active = encoded_dwd_parameters['dwd'], inds['dwd']
        N = active.sum(axis=-1)
        M = self.dwd_block_max
        V = self.V
        value = -N * np.log(V) - np.log(comb(M, N))
        if self.chirp_log_uniform:
            magnitude = abs(para_doppler[..., 1])
            value -= np.log(np.where(active & (magnitude > 0), magnitude, 1.)).sum(axis=-1)
        outside = (
            np.any((para_doppler < self.encoded_para_bounds[:, 0])
                   | (para_doppler > self.encoded_para_bounds[:, 1])
                   | ~np.isfinite(para_doppler), axis=-1)
            | (abs(para_doppler[..., 1]) < self.chirp_minimum)
        ) & active
        return np.where(outside.any(axis=-1), -np.inf, value)
class BirthDistribution:
    def __init__(self, encoded_para_centers, encoded_para_bounds, chirp_minimum=0., *, chirp_log_uniform=False):
        self.encoded_para_centers, self.encoded_para_bounds = encoded_para_centers, encoded_para_bounds
        self.chirp_minimum = chirp_minimum
        self.chirp_log_uniform = chirp_log_uniform
        if chirp_log_uniform:
            self.chirp_log_width = CataloguePrior.chirp_log_volume(encoded_para_bounds[1], chirp_minimum)
        self.nearby_lower_limit = np.maximum(encoded_para_centers-[.2, 0., .2, .2], encoded_para_bounds[:, 0])
        self.nearby_upper_limit = np.minimum(encoded_para_centers+[.2, 0., .2, .2], encoded_para_bounds[:, 1])
        self.nearby_lower_limit[:, 1], self.nearby_upper_limit[:, 1] = encoded_para_bounds[1]
        self.nearby_lower_limit[:, 2], self.nearby_upper_limit[:, 2] = encoded_para_centers[:, 2]-.2, encoded_para_centers[:, 2]+.2
    def log_psd_bd(self, para_doppler):
        para_doppler = np.asarray(para_doppler).reshape(-1, 4)
        encoded_para_widths = self.encoded_para_bounds[:, 1]-self.encoded_para_bounds[:, 0]
        encoded_para_widths[1] -= 2 * self.chirp_minimum
        broad = 1./np.prod(encoded_para_widths)
        delta = (para_doppler[:, None, 2]-self.encoded_para_centers[:, 2]+np.pi) % (2*np.pi)-np.pi
        inside = np.all((para_doppler[:, None, [0, 1, 3]] >= self.nearby_lower_limit[:, [0, 1, 3]])
                        & (para_doppler[:, None, [0, 1, 3]] <= self.nearby_upper_limit[:, [0, 1, 3]]), axis=-1)
        inside &= np.abs(delta) <= .5*(self.nearby_upper_limit[:, 2]-self.nearby_lower_limit[:, 2])
        encoded_para_widths = self.nearby_upper_limit-self.nearby_lower_limit
        encoded_para_widths[:, 1] -= 2 * self.chirp_minimum
        density = .1*broad+.9*(inside/np.prod(encoded_para_widths, axis=1)).mean(axis=-1)
        chirp = para_doppler[:, 1]
        outside = abs(chirp) < self.chirp_minimum
        if self.chirp_log_uniform:
            lower, upper = self.encoded_para_bounds[1]
            magnitude = abs(chirp)
            density *= (upper-lower-2*self.chirp_minimum) / (
                np.where((magnitude > 0) & np.isfinite(magnitude), magnitude, 1.) * self.chirp_log_width)
            outside |= (chirp < lower) | (chirp > upper) | ~np.isfinite(chirp)
        return np.where(outside, -np.inf, np.log(density))
    def parameters_proposal_new_dwd(self, size=1):
        shape = (size,) if isinstance(size, int) else tuple(size)
        para_doppler = np.random.uniform(self.encoded_para_bounds[:, 0], self.encoded_para_bounds[:, 1], size=shape+(4,))
        flat = para_doppler.reshape(-1, 4)
        selected = np.flatnonzero(np.random.random(len(flat)) >= .1)
        candidate = np.random.randint(len(self.encoded_para_centers), size=len(selected))
        flat[selected] = np.random.uniform(self.nearby_lower_limit[candidate], self.nearby_upper_limit[candidate])
        if self.chirp_log_uniform:
            lower, upper = self.encoded_para_bounds[1]
            log_draw = (para_doppler[..., 1]-lower)/(upper-lower) * self.chirp_log_width
            if self.chirp_minimum:
                negative_width = np.log(-lower/self.chirp_minimum)
                para_doppler[..., 1] = np.where(log_draw < negative_width,
                    -self.chirp_minimum * np.exp(log_draw),
                    self.chirp_minimum * np.exp(log_draw-negative_width))
            else:
                para_doppler[..., 1] = lower * np.exp(log_draw)
        elif self.chirp_minimum:
            chirp = para_doppler[..., 1]
            para_doppler[..., 1] = (chirp * (1 - self.chirp_minimum / self.encoded_para_bounds[1, 1])
                + np.where(chirp >= 0, self.chirp_minimum, -self.chirp_minimum))
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
        penalty = np.log(np.diag(chol)).sum() + len(M)*np.log(amplitude_prior_sigma)
        improvement = 0.5 * x @ mean - penalty
    else:
        improvement = 0.0
    return float(-0.5 * data_norm - log_determinant + improvement)
class Likelihood:
    def __init__(self, engine, f_center, T_obs, sigma=10., *, maximum_sources=64, chirp_coordinate='log10'):
        self.f_center, self.T_obs, self.sigma = f_center, T_obs, sigma
        self.chirp_coordinate = chirp_coordinate
        self.engine, self.count_maximum = engine, maximum_sources
        self.white_data = engine.white_data
        self.data_norm = float(self.white_data @ self.white_data)
        self.source = lru_cache(maxsize=256)(self._source)
        self.cached_details = lru_cache(maxsize=64)(self._details)
        self.integral = jax.jit(self._integral)
        self.zero = jax.device_put(np.zeros((4, self.white_data.size)), engine.device)
    def _source(self, key):
        para_doppler = np.frombuffer(key, dtype=np.float64)
        point = decode_doppler(para_doppler[None], self.f_center, self.T_obs,
                               chirp_coordinate=self.chirp_coordinate)[0]
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
        return self.details(para_doppler)[0]
# 3. Eryn samplers
class BirthDeathMove(DistributionGenerateRJ):
    def __init__(self, generate_dist, *args, **kwargs):
        self.generate_dist = generate_dist
        super(DistributionGenerateRJ, self).__init__(*args, **kwargs)
    def get_proposal(self, coords, inds, nleaves_min_all, nleaves_max_all, random, **kwargs):
        inds_for_change = self.get_model_change_proposal(
            inds['dwd'], random, nleaves_min_all['dwd'], nleaves_max_all['dwd'])
        para_doppler = {'dwd': coords['dwd'].copy()}
        new_inds = {'dwd': inds['dwd'].copy()}
        factors = np.zeros(inds['dwd'].shape[:2])
        inds_here = tuple(inds_for_change['-1'].T)
        new_inds['dwd'][inds_here] = False
        factors[inds_here[:2]] += self.generate_dist['dwd'].log_psd_bd(
            para_doppler['dwd'][inds_here])
        inds_here = tuple(inds_for_change['+1'].T)
        new_inds['dwd'][inds_here] = True
        para_doppler['dwd'][inds_here] = self.generate_dist['dwd'].parameters_proposal_new_dwd(size=len(inds_here[0]))
        factors[inds_here[:2]] -= self.generate_dist['dwd'].log_psd_bd(
            para_doppler['dwd'][inds_here])
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
        para_doppler['dwd'][..., 2] %= 2*np.pi
        return para_doppler, factors
def run_eryn_rjmcmc(data, covariance, dwd_template_table,
        frequency_range, derivative_bounds, T_obs, rate,
        checkpoint_directory, *, response_times, response_geometry, physical_frequency,
        iterations, burn_in, maximum_sources=64, amplitude_prior_sigma=10.):
    """Fit a nonempty eleven-column table with one beta=1 chain on GPU."""
    with single_cpu():
        from script.dwd_worker_supervisor import worker_progress
        f_center = np.mean(frequency_range)
        checkpoint_file = Path(checkpoint_directory)/'eryn_chain_00.h5'
        chirp_bounds = np.asarray(derivative_bounds)*T_obs**2
        if chirp_bounds.shape == (2,):
            chirp_minimum = 0.
        elif chirp_bounds.shape == (2, 2) and chirp_bounds[1, 0] == -chirp_bounds[0, 1]:
            chirp_minimum = abs(chirp_bounds[0, 1])
        else:
            raise ValueError('Signed chirp bounds require matching negative/positive minimum magnitudes.')
        chirp_log_uniform = True
        CataloguePrior.chirp_log_volume(chirp_bounds.ravel()[[0, -1]], chirp_minimum)
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
            tempering_kwargs=dict(betas=np.ones(1), adaptive=False, permute=False),
            fill_zero_leaves_val=likelihood(np.empty((0, 4))),
        )
        para_doppler = np.zeros((1, 1, maximum_sources, 4))
        active = np.zeros(para_doppler.shape[:-1], dtype=bool)
        initial_encoded_para_centers = encoded_para_centers[np.isfinite(prior.logpdf(
            {'dwd': encoded_para_centers[:, None, :]},
            {'dwd': np.ones((len(encoded_para_centers), 1), dtype=bool)}))]
        initial_count = min(len(initial_encoded_para_centers), maximum_sources)
        para_doppler[0, 0, :initial_count], active[0, 0, :initial_count] = initial_encoded_para_centers[:initial_count], True
        state = State({'dwd': para_doppler}, inds={'dwd': active})
        initial_logm = likelihood(initial_encoded_para_centers[:initial_count])
        initial_prior = prior.logpdf(state.branches_coords, state.branches_inds)[0, 0]
        para_doppler_draws, a_draws, count_draws, logm_draws = [], [], [], []
        amplitude_rng = np.random.default_rng()
        completed_cycles = backend.iteration
        if completed_cycles:
            state = backend.get_last_sample()
        progress = tqdm(sampler.sample(state, iterations=iterations-completed_cycles),
                        initial=completed_cycles, total=iterations,
                        desc='Eryn RJMCMC', unit='cycle', dynamic_ncols=True)
        for cycle, state in enumerate(progress, start=completed_cycles + 1):
            worker_progress('sampling', cycle=int(cycle), total=int(iterations))
            progress.set_postfix(N=state.branches_inds['dwd'].sum(), refresh=False)
        # Read the same retained states after sampling, including any resumed prefix.
        retained_coords = backend.get_chain()['dwd'][burn_in:iterations]
        retained_inds = backend.get_inds()['dwd'][burn_in:iterations]
        for draw_index, (coordinates, active) in enumerate(zip(retained_coords, retained_inds)):
            worker_progress('posterior_reconstruction', draw=int(draw_index), total=len(retained_coords))
            para_doppler = coordinates[0, 0, active[0, 0]]
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
        worker_progress('final_waveform', force=True)
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
            candidate_centers=encoded_para_centers,
            config=dict(bounds=encoded_para_bounds, chirp_minimum=chirp_minimum,
                chirp_log_uniform=chirp_log_uniform, chirp_coordinate='linear',
                maximum_sources=maximum_sources, center=f_center, duration=T_obs, rate=rate,
                iterations=iterations, burn_in=burn_in, initial_count=initial_count,
                initial_centers_outside_prior=len(encoded_para_centers)-len(initial_encoded_para_centers),
                amplitude_prior_sigma=amplitude_prior_sigma),
            checkpoint_file=str(checkpoint_file),
            fixed_covariance_sha256=hashlib.sha256(covariance.tobytes()).hexdigest())

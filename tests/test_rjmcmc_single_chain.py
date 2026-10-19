"""One-replica mechanics using the installed Eryn, not a convergence test."""
import ast
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from scipy.special import comb, gammaln

try:
    from eryn.backends import HDFBackend
    from eryn.ensemble import EnsembleSampler
    from eryn.moves import DistributionGenerateRJ, GaussianMove
    from eryn.state import State
except ModuleNotFoundError:
    EnsembleSampler = None

ROOT = Path(__file__).resolve().parents[1]


class RecordingRandom:
    """Observe acceptance draws while forwarding all RNG operations unchanged."""
    def __init__(self, random):
        self._random = random
        self.draws = []

    def __getattr__(self, name):
        return getattr(self._random, name)

    def rand(self, *shape):
        value = self._random.rand(*shape)
        self.draws.append(value.copy())
        return value


@unittest.skipIf(EnsembleSampler is None, 'Run with installed Eryn in lisa_pipeline.')
class SingleChainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse((ROOT / 'eric_rjmcmc.py').read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name in ('CataloguePrior', 'BirthDistribution', 'BirthDeathMove', 'DWDMove')]
        cls.scope = dict(np=np, comb=comb, gammaln=gammaln,
                         DistributionGenerateRJ=DistributionGenerateRJ, GaussianMove=GaussianMove)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'eric_rjmcmc.py', 'exec'), cls.scope)
        cls.bounds = np.array([[0., 2.], [-3., 3.], [0., 2*np.pi], [-1., 1.]])

    def sampler(self, backend=None):
        prior = self.scope['CataloguePrior'](self.bounds, dwd_block_max=4, chirp_minimum=1.)
        birth = self.scope['BirthDistribution'](np.array([[1., 2., 1., 0.]]), self.bounds, 1.)
        within = self.scope['DWDMove']({'dwd': np.diag(np.array([.025, .3, .015, .015])**2)})
        jump = self.scope['BirthDeathMove']({'dwd': birth}, nleaves_min={'dwd': 0},
                                          nleaves_max={'dwd': 4}, prevent_swaps=True)
        sampler = EnsembleSampler(1, {'dwd': 4}, lambda q: 0.,
            {'dwd': prior, 'all_models_together': prior}, branch_names=['dwd'],
            nleaves_min={'dwd': 0}, nleaves_max={'dwd': 4}, moves=within, rj_moves=jump,
            tempering_kwargs=dict(betas=np.ones(1), adaptive=False, permute=False),
            backend=backend, fill_zero_leaves_val=0.)
        return sampler, within, jump

    def test_gaussian_step_is_unscaled_at_beta_one(self):
        sampler, within, _ = self.sampler()
        self.assertIsNotNone(within.temperature_control)
        np.testing.assert_array_equal(within.temperature_control.betas, [1.])
        self.assertEqual(sampler.ntemps, 1)
        coords = np.broadcast_to([1., 2., 1., 0.], (1, 1, 4, 4)).copy()
        for count in (0, 1, 4):
            active = np.arange(4)[None, None, :] < count
            actual, factors = within.get_proposal({'dwd': coords}, np.random.RandomState(713),
                                                  branches_inds={'dwd': active})
            random = np.random.RandomState(713)
            slot = np.argmax(np.where(active, random.random(active.shape), -1.), axis=-1)
            selected = np.zeros_like(active)
            np.put_along_axis(selected, slot[..., None], active.any(axis=-1)[..., None], axis=-1)
            base = GaussianMove({'dwd': np.diag(np.array([.025, .3, .015, .015])**2)})
            expected, expected_factors = base.get_proposal({'dwd': coords}, random,
                                                          branches_inds={'dwd': selected})
            expected['dwd'][..., 2] %= 2*np.pi
            np.testing.assert_array_equal(actual['dwd'], expected['dwd'])
            np.testing.assert_array_equal(factors, expected_factors)
            np.testing.assert_array_equal(actual['dwd'][~selected], coords[~selected])

    def test_one_replica_backend_stage_order_and_ordinary_acceptance(self):
        for initial_count in (0, 1):
            with self.subTest(initial_count=initial_count), TemporaryDirectory() as directory:
                np.random.seed(713)
                backend = HDFBackend(str(Path(directory)/'chain.h5'))
                sampler, within, jump = self.sampler(backend)
                events, decisions, counts = [], [], [initial_count]

                def observe(move, name, original, model, state):
                    events.append(name)
                    previous_like, previous_prior = state.log_like.copy(), state.log_prior.copy()
                    old_count = state.branches_inds['dwd'].sum(axis=-1)
                    random = RecordingRandom(model.random)
                    proxy = SimpleNamespace(random=random, compute_log_like_fn=model.compute_log_like_fn,
                                            compute_log_prior_fn=model.compute_log_prior_fn)
                    proposal = {}
                    get_proposal = move.get_proposal

                    def capture(*args, **kwargs):
                        result = get_proposal(*args, **kwargs)
                        proposal['factors'] = result[-1]
                        proposal['raw_factors'] = result[-1].copy()
                        if name == 'rj':
                            proposal['new_count'] = result[1]['dwd'].sum(axis=-1)
                        return result

                    with patch.object(move, 'get_proposal', side_effect=capture), \
                         patch.object(move, 'compute_log_posterior', wraps=move.compute_log_posterior) as posterior:
                        result, accepted = original(proxy, state)
                    for call in posterior.call_args_list:
                        np.testing.assert_array_equal(move.compute_log_posterior(*call.args),
                                                      call.args[0] + call.args[1])
                    if posterior.call_args_list:
                        new_like, new_prior = posterior.call_args_list[0].args
                        difference = (proposal['factors'] + new_like + new_prior
                                      - previous_like - previous_prior)
                        expected = difference > np.log(random.draws[-1])
                        np.testing.assert_array_equal(accepted, expected)
                        decisions.append(name)
                    else:
                        self.assertEqual(name, 'gaussian')
                        self.assertEqual(old_count.sum(), 0)
                        self.assertFalse(accepted.any())
                    if name == 'rj':
                        new_count = proposal['new_count']
                        edge = (np.isin(old_count, [0, 4]).astype(float)
                                - np.isin(new_count, [0, 4]).astype(float))*np.log(.5)
                        np.testing.assert_allclose(proposal['factors'], proposal['raw_factors']+edge,
                                                   rtol=0., atol=1.e-15)
                    return result, accepted

                original_within, original_jump = within.propose, jump.propose
                coords = np.zeros((1, 1, 4, 4))
                coords[..., :initial_count, :] = [1., 2., 1., 0.]
                active = np.arange(4)[None, None, :] < initial_count
                with patch.object(within, 'propose', side_effect=lambda m, s: observe(
                        within, 'gaussian', original_within, m, s)), \
                     patch.object(jump, 'propose', side_effect=lambda m, s: observe(
                        jump, 'rj', original_jump, m, s)), \
                     patch.object(sampler.temperature_control, 'do_swaps_indexing',
                                  wraps=sampler.temperature_control.do_swaps_indexing) as swap:
                    for state in sampler.sample(State({'dwd': coords}, inds={'dwd': active}), iterations=64):
                        self.assertEqual(state.branches_coords['dwd'].shape, (1, 1, 4, 4))
                        self.assertEqual(state.log_like.shape, (1, 1))
                        self.assertTrue(np.isfinite(state.log_like).all())
                        self.assertTrue(np.isfinite(state.log_prior).all())
                        counts.append(int(state.branches_inds['dwd'].sum()))
                    self.assertEqual(swap.call_count, 0)
                self.assertEqual(events, ['gaussian', 'rj']*64)
                self.assertIn('gaussian', decisions)
                self.assertEqual(decisions.count('rj'), 64)
                np.testing.assert_array_equal(sampler.temperature_control.betas, [1.])
                self.assertEqual(sampler.temperature_control.swaps_proposed.shape, (0,))
                self.assertEqual(sampler.temperature_control.swaps_accepted.shape, (0,))
                self.assertTrue(np.any(np.diff(counts)>0), counts)
                self.assertTrue(np.any(np.diff(counts)<0), counts)
                self.assertGreaterEqual(min(counts), 0)
                self.assertLessEqual(max(counts), 4)
                self.assertEqual(backend.get_chain()['dwd'].shape, (64, 1, 1, 4, 4))
                self.assertEqual(backend.get_inds()['dwd'].shape, (64, 1, 1, 4))
                self.assertEqual(backend.get_log_like().shape, (64, 1, 1))
                self.assertEqual(backend.get_log_prior().shape, (64, 1, 1))
                np.testing.assert_array_equal(backend.get_betas(), np.ones((64, 1)))


if __name__ == '__main__':
    unittest.main()

"""Exercise the project birth/death path with the installed Eryn implementation."""
import ast
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from scipy.special import comb, gammaln

try:
    from eryn.ensemble import EnsembleSampler
    from eryn.moves import DistributionGenerateRJ, GaussianMove
    from eryn.state import State
except ModuleNotFoundError:
    EnsembleSampler = None

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipIf(EnsembleSampler is None, 'Installed Eryn is required; run in lisa_pipeline on server.')
class BirthDeathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tree = ast.parse((ROOT / 'eric_rjmcmc.py').read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name in ('CataloguePrior', 'BirthDistribution', 'BirthDeathMove', 'DWDMove')]
        cls.scope = dict(np=np, comb=comb, gammaln=gammaln,
                         DistributionGenerateRJ=DistributionGenerateRJ, GaussianMove=GaussianMove)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'eric_rjmcmc.py', 'exec'), cls.scope)
        cls.encoded_para_bounds = np.array([[0., 2.], [-3., 3.], [0., 2 * np.pi], [-1., 1.]])

    def test_actual_eryn_birth_and_death_call_descriptive_methods(self):
        for centers in (np.array([[1., 2., .02, 0.]]), np.array([[1., 2., .02, 0.], [1.1, -2., 6., .1]])):
            for fix_change in (None, 1, -1):
                birth = self.scope['BirthDistribution'](centers, self.encoded_para_bounds, 1.)
                self.assertFalse(hasattr(birth, 'logpdf'))
                self.assertFalse(hasattr(birth, 'rvs'))
                self.assertFalse(hasattr(birth, 'parameters_proposal_log_pdf'))
                move = self.scope['BirthDeathMove']({'dwd': birth}, nleaves_min={'dwd': 0},
                    nleaves_max={'dwd': 4}, fix_change=fix_change, prevent_swaps=True)
                coords = np.broadcast_to([1., 2., .02, 0.], (1, 5, 4, 4)).copy()
                inds = np.arange(4)[None, None, :] < np.arange(5)[None, :, None]
                original_coords, original_inds = coords.copy(), inds.copy()
                np.random.seed(20261014)
                with patch.object(birth, 'log_psd_bd', wraps=birth.log_psd_bd) as density, \
                     patch.object(birth, 'parameters_proposal_new_dwd', wraps=birth.parameters_proposal_new_dwd) as draw:
                    proposed, new_inds, factors = move.get_proposal({'dwd': coords}, {'dwd': inds},
                        {'dwd': 0}, {'dwd': 4}, np.random.RandomState(713))
                    self.assertEqual(density.call_count, 2)
                    self.assertEqual(draw.call_count, 1)
                before, after = inds.sum(axis=-1), new_inds['dwd'].sum(axis=-1)
                np.testing.assert_array_equal(abs(after - before), 1)
                self.assertEqual(after[0, 0], 1)
                self.assertEqual(after[0, -1], 3)
                self.assertTrue(np.all(np.isfinite(factors)))
                np.testing.assert_array_equal(coords, original_coords)
                np.testing.assert_array_equal(inds, original_inds)
                expected = np.zeros_like(factors)
                for walker in range(5):
                    changed = np.flatnonzero(inds[0, walker] != new_inds['dwd'][0, walker])[0]
                    value = (proposed['dwd'][0, walker, changed] if after[0, walker] > before[0, walker]
                             else coords[0, walker, changed])
                    sign = -1 if after[0, walker] > before[0, walker] else 1
                    expected[0, walker] = sign * birth.log_psd_bd(value)[0]
                expected += (gammaln(before + 1) + gammaln(4 - before + 1)
                             - gammaln(after + 1) - gammaln(4 - after + 1))
                np.testing.assert_array_equal(factors, expected)

    def test_actual_eryn_sampler_accepts_births_and_deaths(self):
        np.random.seed(713)
        prior = self.scope['CataloguePrior'](self.encoded_para_bounds, dwd_block_max=4, chirp_minimum=1.)
        birth = self.scope['BirthDistribution'](np.array([[1., 2., .02, 0.]]), self.encoded_para_bounds, 1.)
        jump = self.scope['BirthDeathMove']({'dwd': birth}, nleaves_min={'dwd': 0},
            nleaves_max={'dwd': 4}, prevent_swaps=True)
        within = self.scope['DWDMove']({'dwd': np.eye(4) * .01})
        sampler = EnsembleSampler(1, {'dwd': 4}, lambda parameters: 0.,
            {'dwd': prior, 'all_models_together': prior}, branch_names=['dwd'],
            nleaves_min={'dwd': 0}, nleaves_max={'dwd': 4}, moves=within, rj_moves=jump,
            tempering_kwargs=dict(betas=np.ones(1), adaptive=False, permute=False),
            fill_zero_leaves_val=0.)
        coords = np.broadcast_to([1., 2., 1., 0.], (1, 1, 4, 4)).copy()
        inds = np.zeros((1, 1, 4), bool)
        inds[..., 0] = True
        counts = [1]
        for state in sampler.sample(State({'dwd': coords}, inds={'dwd': inds}), iterations=64):
            counts.append(int(state.branches_inds['dwd'].sum()))
            self.assertTrue(np.isfinite(state.log_prior).all())
        self.assertTrue(np.any(np.diff(counts) > 0), counts)
        self.assertTrue(np.any(np.diff(counts) < 0), counts)
        self.assertGreaterEqual(min(counts), 0)
        self.assertLessEqual(max(counts), 4)

    def test_nonempty_bank_birth_from_zero_model_and_zero_size_batches(self):
        birth = self.scope['BirthDistribution'](np.array([[1., 2., .02, 0.]]), self.encoded_para_bounds, 1.)
        for size in (0, (2, 0)):
            empty = birth.parameters_proposal_new_dwd(size)
            self.assertEqual(empty.shape, ((size,) if isinstance(size, int) else size) + (4,))
            self.assertEqual(birth.log_psd_bd(empty).shape, (0,))
        move = self.scope['BirthDeathMove']({'dwd': birth}, nleaves_min={'dwd': 0},
            nleaves_max={'dwd': 4}, prevent_swaps=True)
        coords = np.zeros((1, 1, 4, 4))
        active = np.zeros((1, 1, 4), bool)
        proposed, new_active, factors = move.get_proposal({'dwd': coords}, {'dwd': active},
            {'dwd': 0}, {'dwd': 4}, np.random.RandomState(713))
        self.assertEqual(new_active['dwd'].sum(), 1)
        self.assertTrue(np.isfinite(factors).all())
        prior = self.scope['CataloguePrior'](self.encoded_para_bounds, dwd_block_max=4, chirp_minimum=1.)
        np.testing.assert_array_equal(prior.logpdf({'dwd': coords}, {'dwd': active}), 0.)
        self.assertTrue(np.isfinite(prior.logpdf(proposed, new_active)).all())


if __name__ == '__main__':
    unittest.main()

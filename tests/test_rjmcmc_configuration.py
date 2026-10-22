"""Prior support and capped initialization, using the actual source without CUDA imports."""
import ast
from pathlib import Path
import unittest
import numpy as np
from scipy.special import comb

ROOT = Path(__file__).resolve().parents[1]


class ConfigurationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse((ROOT / 'eric_rjmcmc.py').read_text())
        cls.scope = dict(np=np, comb=comb)
        nodes = [node for node in cls.tree.body if isinstance(node, ast.ClassDef)
                 and node.name in ('CataloguePrior', 'BirthDistribution')]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'eric_rjmcmc.py', 'exec'), cls.scope)

    def test_signed_linear_prior_support_and_normalization(self):
        encoded_para_bounds = np.array([[0., 2.], [-3., 3.], [0., 2 * np.pi], [-1., 1.]])
        prior = self.scope['CataloguePrior'](encoded_para_bounds=encoded_para_bounds,
            dwd_block_max=64, chirp_minimum=1.)
        default_prior = self.scope['CataloguePrior'](encoded_para_bounds, chirp_minimum=1.)
        self.assertEqual(default_prior.dwd_block_max, 64)
        self.assertEqual(prior.dwd_block_max, 64)
        self.assertEqual(prior.key_order, [0, 1, 2, 3])
        self.assertFalse(hasattr(prior, 'DWD_para_order'))
        self.assertFalse(hasattr(prior, 'count_maximum'))
        self.assertFalse(hasattr(prior, 'bounds'))
        np.testing.assert_array_equal(prior.encoded_para_bounds, encoded_para_bounds)
        self.assertEqual(prior.V, 2 * 4 * 2 * np.pi * 2)
        coords = np.array([[[[1., chirp, 1., 0.]]] for chirp in (-3., -1., -.5, 0., .5, 1., 3.)])
        active = np.ones(coords.shape[:-1], dtype=bool)
        actual = prior.logpdf(encoded_dwd_parameters={'dwd': coords}, inds={'dwd': active}).ravel()
        np.testing.assert_array_equal(default_prior.logpdf({'dwd': coords}, {'dwd': active}),
            prior.logpdf({'dwd': coords}, {'dwd': active}))
        np.testing.assert_array_equal(np.isfinite(actual), [True, True, False, False, False, True, True])
        np.testing.assert_allclose(actual[[0, 1, 5, 6]], -np.log(prior.V) - np.log(64))
        active[:] = False
        np.testing.assert_array_equal(prior.logpdf({'dwd': coords}, {'dwd': active}), 0.)

    def test_births_draw_and_evaluate_the_same_two_intervals(self):
        encoded_para_bounds = np.array([[0., 2.], [-3., 3.], [0., 2 * np.pi], [-1., 1.]])
        np.random.seed(1217)
        for encoded_para_centers in (np.array([[1., 2., 1., 0.]]), np.array([[1., 2., 1., 0.], [.7, -2., 3., .3]])):
            distribution = self.scope['BirthDistribution'](encoded_para_centers=encoded_para_centers,
                encoded_para_bounds=encoded_para_bounds, chirp_minimum=1.)
            self.assertFalse(hasattr(distribution, 'centers'))
            self.assertFalse(hasattr(distribution, 'bounds'))
            self.assertFalse(hasattr(distribution, 'lower_limit'))
            self.assertFalse(hasattr(distribution, 'upper_limit'))
            self.assertEqual(distribution.nearby_lower_limit.shape, encoded_para_centers.shape)
            self.assertEqual(distribution.nearby_upper_limit.shape, encoded_para_centers.shape)
            self.assertFalse(hasattr(distribution, 'logpdf'))
            self.assertFalse(hasattr(distribution, 'rvs'))
            self.assertFalse(hasattr(distribution, 'parameters_proposal_log_pdf'))
            samples = distribution.parameters_proposal_new_dwd(20000)
            self.assertTrue(np.all((abs(samples[:, 1]) >= 1.) & (abs(samples[:, 1]) <= 3.)))
            self.assertTrue(np.all(np.isfinite(distribution.log_psd_bd(samples))))
            self.assertLess(abs(np.mean(samples[:, 1] > 0) - .5), .015)
            self.assertLess(abs(np.mean(abs(samples[:, 1])) - 2.), .02)
            self.assertEqual(distribution.log_psd_bd([[1., 0., 1., 0.]])[0], -np.inf)
            for size in (0, (2, 0)):
                shape = (size,) if isinstance(size, int) else size
                empty = distribution.parameters_proposal_new_dwd(size)
                self.assertEqual(empty.shape, shape + (4,))
                self.assertEqual(distribution.log_psd_bd(empty).shape, (0,))

    def test_default_bounds_use_the_validated_signed_log_magnitude_support(self):
        pipeline = ast.parse((ROOT / 'eric_subtraction_all.py').read_text())
        nodes = [node for node in ast.walk(pipeline) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id in ('f_derivative_pn', 'frequency_derivative_bounds') for t in node.targets)]
        scope = dict(np=np)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'bounds', 'exec'), scope)
        np.testing.assert_allclose(scope['frequency_derivative_bounds'],
                                   [[-4.e-12, -7.60863249122299e-23],
                                    [7.60863249122299e-23, 4.e-12]], rtol=1.e-15, atol=0.)

    def test_initialization_caps_active_sources_without_truncating_bank(self):
        run = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_eryn_rjmcmc')
        body = run.body[1].body
        first = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'para_doppler' for t in node.targets))
        stop = next(i for i, node in enumerate(body[first:], first) if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'state' for t in node.targets))
        for encoded_para_centers in (np.arange(70 * 4).reshape(70, 4).astype(float), np.empty((0, 4))):
            prior = self.scope['CataloguePrior'](np.tile([-1.e6, 1.e6], (4, 1)))
            scope = dict(np=np, prior=prior, encoded_para_centers=encoded_para_centers.copy(), maximum_sources=64)
            exec(compile(ast.Module(body=body[first:stop], type_ignores=[]), 'initialization', 'exec'), scope)
            count = min(len(encoded_para_centers), 64)
            np.testing.assert_array_equal(scope['encoded_para_centers'], encoded_para_centers)
            self.assertEqual(scope['initial_count'], count)
            np.testing.assert_array_equal(scope['active'].sum(axis=-1), np.full((1, 1), count))
            np.testing.assert_array_equal(scope['para_doppler'][:, 0, :count], np.broadcast_to(encoded_para_centers[:count], (1, count, 4)))
        births = next(node for node in ast.walk(run) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Name) and node.func.id == 'BirthDistribution')
        self.assertEqual(ast.unparse(births.args[0]), 'encoded_para_centers')
        self.assertEqual(ast.unparse(births.args[1]), 'encoded_para_bounds')
        prior = next(node for node in ast.walk(run) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == 'CataloguePrior')
        self.assertEqual([ast.unparse(arg) for arg in prior.args],
                         ['encoded_para_bounds', 'maximum_sources', 'chirp_minimum'])

    def test_outside_prior_centers_do_not_create_an_invalid_initial_state(self):
        run = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_eryn_rjmcmc')
        body = run.body[1].body
        first = next(i for i, node in enumerate(body) if isinstance(node, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'para_doppler' for t in node.targets))
        stop = next(i for i, node in enumerate(body[first:], first) if isinstance(node, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'state' for t in node.targets))
        bounds = np.array([[0., 2.], [.01, 2.], [0., 2*np.pi], [-1., 1.]])
        prior = self.scope['CataloguePrior'](bounds, chirp_log_uniform=True)
        mixed = np.array([[1., .1, 1., 0.], [1., -.1, 1., 0.],
                          [1., np.nan, 1., 0.], [1., 3., 1., 0.], [.5, .2, 2., .1]])
        for centers, expected in [(mixed, mixed[[0, 4]]), (mixed[[1, 2, 3]], np.empty((0, 4)))]:
            scope = dict(np=np, prior=prior, encoded_para_centers=centers.copy(), maximum_sources=64)
            exec(compile(ast.Module(body=body[first:stop], type_ignores=[]), 'initialization', 'exec'), scope)
            np.testing.assert_array_equal(scope['encoded_para_centers'], centers)
            np.testing.assert_array_equal(scope['initial_encoded_para_centers'], expected)
            self.assertEqual(scope['initial_count'], len(expected))
            self.assertTrue(np.all(np.isfinite(prior.logpdf({'dwd': scope['para_doppler']}, {'dwd': scope['active']}))))

    def test_production_uses_one_walker_at_beta_one(self):
        run = next(node for node in self.tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == 'run_eryn_rjmcmc')
        defaults = dict(zip([arg.arg for arg in run.args.kwonlyargs], run.args.kw_defaults))
        self.assertNotIn('chain_count', defaults)
        self.assertNotIn('inverse_temperatures', defaults)
        self.assertNotIn('chirp_coordinate', defaults)
        self.assertEqual(ast.literal_eval(defaults['maximum_sources']), 64)
        self.assertIsNone(defaults['iterations'])
        self.assertIsNone(defaults['burn_in'])
        self.assertEqual([arg.arg for arg in run.args.args],
            ['data', 'covariance', 'dwd_template_table', 'frequency_range', 'derivative_bounds',
             'T_obs', 'rate', 'checkpoint_directory'])
        sampler = next(node for node in ast.walk(run) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Name) and node.func.id == 'EnsembleSampler')
        self.assertEqual(ast.literal_eval(sampler.args[0]), 1)
        tempering = next(kw.value for kw in sampler.keywords if kw.arg == 'tempering_kwargs')
        self.assertEqual({kw.arg: ast.unparse(kw.value) for kw in tempering.keywords},
                         dict(betas='np.ones(1)', adaptive='False', permute='False'))
        move = next(node for node in self.tree.body if isinstance(node, ast.ClassDef) and node.name == 'DWDMove')
        self.assertNotIn('temperature_control', {node.attr for node in ast.walk(move) if isinstance(node, ast.Attribute)})
        pipeline = ast.parse((ROOT / 'eric_subtraction_all.py').read_text())
        callers = [node for node in ast.walk(pipeline) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Name) and node.func.id == 'run_eryn_rjmcmc']
        self.assertEqual(len(callers), 1)
        for call in callers:
            self.assertFalse(any(kw.arg is None for kw in call.keywords))
            self.assertFalse({'inverse_temperatures','chain_count','chirp_coordinate'}
                             & {kw.arg for kw in call.keywords})


if __name__ == '__main__':
    unittest.main()

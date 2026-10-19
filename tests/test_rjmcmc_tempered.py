"""Real-Eryn checks for the isolated temperature-swap experiment."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import numpy as np

try:
    import eryn
except ModuleNotFoundError:
    eryn = None


@unittest.skipIf(eryn is None, 'Run with installed Eryn in lisa_pipeline.')
class TemperedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import eric_rjmcmc as production
        import eric_rjmcmc_tempered as experiment
        cls.production, cls.experiment = production, experiment
        cls.directory = TemporaryDirectory()
        class Response:
            def __init__(self, *args):
                pass
            def build(self, doppler):
                return np.zeros((len(doppler), 4, 3, 1))
        class Likelihood:
            def __init__(self, *args, **kwargs):
                pass
            def details(self, parameters):
                count = len(parameters)
                return 3.*count-float(np.square(parameters).sum()), np.zeros(4*count), np.eye(4*count)
            def __call__(self, parameters):
                return self.details(parameters)[0]
        original = experiment.EnsembleSampler
        def capture(*args, **kwargs):
            cls.sampler = original(*args, **kwargs)
            return cls.sampler
        table = np.zeros((2, 11))
        table[:, 0] = [1., 2.]
        table[:, 2] = [1.04, 1.06]
        table[:, 3] = [.001, .003]
        table[:, 6] = 1.
        np.random.seed(713)
        with patch.object(experiment, 'GPUResponse', Response), \
             patch.object(experiment, 'Likelihood', Likelihood), \
             patch.object(experiment, 'EnsembleSampler', capture):
            cls.result = experiment.run_eryn_rjmcmc(np.zeros((3, 1)), np.eye(3)[None], table,
                np.array([1., 1.1]), np.array([1.e-4, .01]), 10., 1., Path(cls.directory.name),
                response_times=np.array([0.]), response_geometry=None, physical_frequency=np.array([1.]),
                iterations=64, burn_in=8, maximum_sources=4)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_math_and_move_classes_are_shared_with_production(self):
        for name in ('CataloguePrior', 'BirthDistribution', 'BirthDeathMove', 'DWDMove',
                     'encode_doppler', 'decode_doppler'):
            self.assertIs(getattr(self.production, name), getattr(self.experiment, name))
        np.testing.assert_array_equal(self.sampler.temperature_control.betas, [1., .8, .6, .4])

    def test_four_replicas_swap_and_recovery_uses_only_cold_samples(self):
        result = self.result
        temperature = result['tempering']
        np.testing.assert_array_equal(result['config']['inverse_temperatures'], [1., .8, .6, .4])
        self.assertFalse(result['config']['adaptive_temperatures'])
        self.assertEqual(temperature['cycle_source_count'].shape, (65, 4))
        self.assertTrue(np.all(temperature['backend_swap_acceptances'] > 0))
        self.assertEqual(temperature['rj_acceptances'].shape, (4, 1))
        cold = result['chains']['0']
        np.testing.assert_array_equal(cold['cycle_state']['source_count'], temperature['cycle_source_count'][:, 0])
        np.testing.assert_array_equal(cold['cycle_state']['log_marginal'], temperature['cycle_log_marginal'][:, 0])
        np.testing.assert_array_equal(cold['cycle_state']['log_prior'], temperature['cycle_log_prior'][:, 0])
        self.assertEqual(cold['sampled_q'].shape, (56, 4, 4))
        np.testing.assert_array_equal(cold['source_count'], temperature['cycle_source_count'][9:, 0])
        modal = np.bincount(cold['source_count'], minlength=5).argmax()
        self.assertEqual(result['modal_count'], modal)
        selected = np.flatnonzero(cold['source_count'] == modal)
        best = selected[np.argmax(cold['log_marginal'][selected])]
        decoded = self.production.decode_doppler(cold['sampled_q'][best][:modal], 1.05, 10.,
                                                  chirp_coordinate='linear')
        np.testing.assert_array_equal(result['doppler'], decoded)
        prior = self.production.CataloguePrior(result['config']['bounds'], 4, chirp_log_uniform=True)
        active = np.isfinite(cold['sampled_q']).all(axis=-1)
        np.testing.assert_allclose(prior.logpdf({'dwd': cold['sampled_q']}, {'dwd': active}),
                                   cold['cycle_state']['log_prior'][9:], atol=1.e-12)

    def test_swaps_move_entire_variable_dimension_models(self):
        control = deepcopy(self.sampler.temperature_control)
        coordinates = np.zeros((4, 1, 4, 4))
        for temperature in range(4):
            coordinates[temperature, 0] = [.01*temperature, .2+.01*temperature, 1., 0.]
        active = np.arange(4)[None, None, :] < np.arange(4)[:, None, None]
        prior = self.production.CataloguePrior(self.result['config']['bounds'], 4, chirp_log_uniform=True)
        log_prior = prior.logpdf({'dwd': coordinates}, {'dwd': active})
        log_like = np.array([[0.], [1.], [2.], [100.]])
        posterior = control.compute_log_posterior_tempered(log_like, log_prior)
        result = control.temperature_swaps({'dwd': coordinates.copy()}, posterior.copy(),
            log_like.copy(), log_prior.copy(), inds={'dwd': active.copy()}, branch_supps={'dwd': None})
        swapped, posterior, likelihood, logp, indices = result[:5]
        order = [3, 0, 1, 2]
        np.testing.assert_array_equal(indices['dwd'], active[order])
        np.testing.assert_array_equal(swapped['dwd'], coordinates[order])
        np.testing.assert_array_equal(likelihood, log_like[order])
        np.testing.assert_array_equal(logp, log_prior[order])
        np.testing.assert_allclose(posterior, control.betas[:, None]*likelihood+logp)

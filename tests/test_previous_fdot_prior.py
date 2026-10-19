"""Physical-prior equivalence to the archived run, with current linear coordinates."""
import ast
from pathlib import Path
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import comb
from scipy.stats import kstest

ROOT = Path(__file__).resolve().parents[1]


def distributions(path):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)
             and node.name in ('CataloguePrior', 'BirthDistribution')]
    scope = dict(np=np, comb=comb)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


class PreviousFdotPriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.current = distributions(ROOT/'eric_rjmcmc.py')
        cls.previous = distributions(ROOT/'server_experiments/ten_blocks_20261006.Txzl9y/rjmcmc_snapshot.py')
        cls.duration = 31104000.
        cls.physical_bounds = np.array([7.60863249122299e-23, 10.**-14.8])
        cls.bounds = np.array([[0., 2.], cls.physical_bounds*cls.duration**2,
                               [0., 2*np.pi], [-1., 1.]])
        cls.old_bounds = cls.bounds.copy()
        cls.old_bounds[1] = np.log10(cls.physical_bounds)

    def test_physical_density_matches_analytic_previous_prior(self):
        prior = self.current['CataloguePrior'](self.bounds, dwd_block_max=1, chirp_log_uniform=True)
        values = np.geomspace(*self.physical_bounds, 101)[1:-1]
        coordinates = np.zeros((len(values), 1, 4))
        coordinates[:, 0, :] = np.column_stack((np.ones(len(values)), values*self.duration**2,
                                               np.ones(len(values)), np.zeros(len(values))))
        active = np.ones(coordinates.shape[:-1], bool)
        actual = prior.logpdf({'dwd': coordinates}, {'dwd': active})
        other_volume = 2.*2*np.pi*2.
        physical_density = np.exp(actual)*self.duration**2*other_volume
        np.testing.assert_allclose(physical_density,
            1./(values*np.log(self.physical_bounds[1]/self.physical_bounds[0])), rtol=1.e-13)

    def test_prior_normalizes_and_preserves_zero_dwd(self):
        prior = self.current['CataloguePrior'](self.bounds, dwd_block_max=1, chirp_log_uniform=True)
        lower, upper = self.bounds[1]
        def integrand(log_chirp):
            chirp = np.exp(log_chirp)
            coordinates = np.array([[[1., chirp, 1., 0.]]])
            return float(np.exp(prior.logpdf({'dwd': coordinates},
                {'dwd': np.ones((1, 1), bool)})[0])*chirp*(2.*2*np.pi*2.))
        integral, error = quad(integrand, np.log(lower)+1.e-12, np.log(upper)-1.e-12)
        self.assertAlmostEqual(integral, 1., places=11)
        self.assertLess(error, 1.e-10)
        inactive = np.full((2, 3, 4), np.nan)
        np.testing.assert_array_equal(prior.logpdf({'dwd': inactive},
            {'dwd': np.zeros((2, 3), bool)}), [0., 0.])

    def test_joint_prior_matches_archived_density_with_jacobian(self):
        prior = self.current['CataloguePrior'](self.bounds, dwd_block_max=4, chirp_log_uniform=True)
        old_prior = self.previous['CataloguePrior'](self.old_bounds, maximum=4)
        coordinates = np.array([[[1., .02, 1., 0.], [.8, .2, 3., .3], [np.nan]*4, [np.nan]*4]])
        active = np.array([[True, True, False, False]])
        old_coordinates = coordinates.copy()
        old_coordinates[..., 1] = np.log10(coordinates[..., 1]/self.duration**2)
        expected = old_prior.logpdf({'dwd': old_coordinates}, {'dwd': active})
        expected -= np.log(coordinates[0, :2, 1]*np.log(10.)).sum()
        np.testing.assert_allclose(prior.logpdf({'dwd': coordinates}, {'dwd': active}), expected, atol=1.e-13)

    def test_birth_density_matches_archived_mixture_with_jacobian(self):
        centers = np.array([[1., .02, .05, 0.], [.8, .2, 3., .3]])
        birth = self.current['BirthDistribution'](centers, self.bounds, chirp_log_uniform=True)
        old_centers = centers.copy()
        old_centers[:, 1] = np.log10(centers[:, 1]/self.duration**2)
        old_birth = self.previous['BirthDistribution'](old_centers, self.old_bounds)
        points = np.array([[1., .01, 2*np.pi-.05, 0.], [.8, .1, 3.05, .3], [1.9, .5, 5., -.9]])
        old_points = points.copy()
        old_points[:, 1] = np.log10(points[:, 1]/self.duration**2)
        expected = old_birth.logpdf(old_points)-np.log(points[:, 1]*np.log(10.))
        np.testing.assert_allclose(birth.log_psd_bd(points), expected, atol=1.e-13)

    def test_birth_draws_follow_log_uniform_cdf(self):
        np.random.seed(1729)
        centers = np.array([[1., .02, .05, 0.], [.8, .2, 3., .3]])
        birth = self.current['BirthDistribution'](centers, self.bounds, chirp_log_uniform=True)
        samples = birth.parameters_proposal_new_dwd((200, 100))
        self.assertEqual(samples.shape, (200, 100, 4))
        chirp = samples[..., 1].ravel()
        lower, upper = self.bounds[1]
        self.assertTrue(np.all((chirp >= lower) & (chirp <= upper)))
        quantiles = np.log(chirp/lower)/np.log(upper/lower)
        self.assertLess(kstest(quantiles, 'uniform').statistic, .012)
        self.assertTrue(np.isfinite(birth.log_psd_bd(samples)).all())
        for size in (0, (2, 0)):
            shape = (size,) if isinstance(size, int) else size
            empty = birth.parameters_proposal_new_dwd(size)
            self.assertEqual(empty.shape, shape+(4,))
            self.assertEqual(birth.log_psd_bd(empty).shape, (0,))

    def test_negative_gap_and_nonfinite_values_are_rejected(self):
        lower, upper = self.bounds[1]
        chirps = np.array([-1., 0., lower*.5, upper*2., np.nan, np.inf, -np.inf])
        points = np.column_stack((np.ones(len(chirps)), chirps, np.ones(len(chirps)), np.zeros(len(chirps))))
        prior = self.current['CataloguePrior'](self.bounds, dwd_block_max=1, chirp_log_uniform=True)
        actual = prior.logpdf({'dwd': points[:, None, :]}, {'dwd': np.ones((len(points), 1), bool)})
        np.testing.assert_array_equal(actual, -np.inf)
        birth = self.current['BirthDistribution'](np.array([[1., .02, 1., 0.]]),
                                                 self.bounds, chirp_log_uniform=True)
        np.testing.assert_array_equal(birth.log_psd_bd(points), -np.inf)


if __name__ == '__main__':
    unittest.main()

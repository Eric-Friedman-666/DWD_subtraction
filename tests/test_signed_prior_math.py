"""CPU-only analytic signed-log-magnitude prior and coordinate checks.

AST extraction avoids scientific runtime imports. These are mathematical/current
API regressions, not an archived-source bitwise-parity or scientific-fit claim.
"""
import ast
from pathlib import Path
import unittest
import numpy as np
from scipy.integrate import quad
from scipy.special import comb
from scipy.stats import kstest

ROOT = Path(__file__).resolve().parents[1]
MINIMUM, DURATION = 7.60863249122299e-23, 31104000.


def definitions():
    path = ROOT/'eric_rjmcmc.py'
    names = {'CataloguePrior', 'BirthDistribution', 'encode_doppler', 'decode_doppler'}
    nodes = [node for node in ast.parse(path.read_text()).body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    scope = dict(np=np, comb=comb)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


class SignedPriorMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = definitions()

    def bounds(self, negative=-4.e-12, positive=4.e-12):
        return np.array([[-1., 1.], [negative*DURATION**2, positive*DURATION**2],
            [0., 2*np.pi], [-1., 1.]])

    def test_signed_log_prior_normalizes_symmetric_and_asymmetric(self):
        for negative, positive in ((-4.e-12, 4.e-12), (-2.e-12, 4.e-12)):
            prior = self.scope['CataloguePrior'](self.bounds(negative, positive), 1,
                MINIMUM*DURATION**2, chirp_log_uniform=True)
            def integrand(value, sign):
                point = np.array([[[0., sign*np.exp(value), np.pi, 0.]]])
                density = np.exp(prior.logpdf({'dwd': point}, {'dwd': np.ones((1, 1), bool)})[0])
                return density*np.exp(value)*(8*np.pi)
            actual = sum(quad(integrand, np.log(MINIMUM*DURATION**2),
                np.log(abs(edge)*DURATION**2), args=(sign,))[0]
                for sign, edge in ((-1, negative), (1, positive)))
            self.assertAlmostEqual(actual, 1., places=11)

    def test_support_gap_edges_nonfinite_and_inactive_nan(self):
        prior = self.scope['CataloguePrior'](self.bounds(), 100,
            MINIMUM*DURATION**2, chirp_log_uniform=True)
        chirps = np.array([-4.e-12, -MINIMUM, -MINIMUM*.5, 0., MINIMUM*.5,
            MINIMUM, 4.e-12, 4.001e-12, np.nan, np.inf])*DURATION**2
        points = np.zeros((len(chirps), 1, 4))
        points[:, 0, 1], points[:, 0, 2] = chirps, np.pi
        finite = np.isfinite(prior.logpdf({'dwd': points},
            {'dwd': np.ones(points.shape[:-1], bool)}))
        np.testing.assert_array_equal(finite, [1, 1, 0, 0, 0, 1, 1, 0, 0, 0])
        blank = np.full((2, 100, 4), np.nan)
        np.testing.assert_array_equal(prior.logpdf({'dwd': blank},
            {'dwd': np.zeros((2, 100), bool)}), [0., 0.])

    def test_positive_support_matches_analytic_log_density_at_max100(self):
        lower, upper = np.array([MINIMUM, 10.**-14.8])*DURATION**2
        bounds = self.bounds()
        bounds[1] = [lower, upper]
        values = np.geomspace(lower, upper, 51)[1:-1]
        points = np.zeros((len(values), 1, 4))
        points[:, 0, 1], points[:, 0, 2] = values, np.pi
        prior = self.scope['CataloguePrior'](bounds, 100, chirp_log_uniform=True)
        actual = prior.logpdf({'dwd': points}, {'dwd': np.ones((len(values), 1), bool)})
        expected = -np.log(values)-np.log(np.log(upper/lower))-np.log(8*np.pi)-np.log(100.)
        np.testing.assert_allclose(actual, expected, rtol=0., atol=1.e-13)

    def test_positive_birth_draws_have_correct_support_and_log_cdf(self):
        bounds = self.bounds()
        bounds[1] = np.array([MINIMUM, 10.**-14.8])*DURATION**2
        birth = self.scope['BirthDistribution'](np.array([[0., .02, np.pi, 0.]]),
            bounds, chirp_log_uniform=True)
        np.random.seed(47721)
        draws = birth.parameters_proposal_new_dwd(20000)
        self.assertTrue(np.isfinite(birth.log_psd_bd(draws)).all())
        self.assertTrue(np.all((draws[:, 1] >= bounds[1, 0]) & (draws[:, 1] <= bounds[1, 1])))
        unit = np.log(draws[:, 1]/bounds[1, 0])/np.log(bounds[1, 1]/bounds[1, 0])
        self.assertLess(kstest(unit, 'uniform').statistic, .015)

    def test_signed_birth_sign_masses_and_log_cdf(self):
        for negative, positive in ((-4.e-12, 4.e-12), (-2.e-12, 4.e-12)):
            birth = self.scope['BirthDistribution'](np.array([[0., .02, np.pi, 0.]]),
                self.bounds(negative, positive), MINIMUM*DURATION**2, chirp_log_uniform=True)
            np.random.seed(47721)
            draws = birth.parameters_proposal_new_dwd(50000)
            chirp = draws[:, 1]/DURATION**2
            self.assertTrue(np.isfinite(birth.log_psd_bd(draws)).all())
            expected = np.log(-negative/MINIMUM)/(np.log(-negative/MINIMUM)+np.log(positive/MINIMUM))
            self.assertLess(abs(np.mean(chirp < 0)-expected), .01)
            for sign, edge in ((-1, negative), (1, positive)):
                values = abs(chirp[chirp*sign > 0])
                unit = np.log(values/MINIMUM)/np.log(abs(edge)/MINIMUM)
                self.assertLess(kstest(unit, 'uniform').statistic, .015)

    def test_signed_birth_density_has_correct_magnitude_jacobian(self):
        birth = self.scope['BirthDistribution'](np.array([[0., .02, np.pi, 0.]]),
            self.bounds(), MINIMUM*DURATION**2, chirp_log_uniform=True)
        points = np.array([[0., -1.e-15*DURATION**2, np.pi, 0.],
            [0., 2.e-15*DURATION**2, np.pi, 0.]])
        logs = birth.log_psd_bd(points)
        self.assertAlmostEqual(float(np.exp(logs[0]-logs[1])), 2., places=12)
        self.assertEqual(birth.parameters_proposal_new_dwd((2, 0)).shape, (2, 0, 4))

    def test_invalid_log_bounds_rejected(self):
        for physical, gap in (((-1.e-12, 1.e-12), 0.), ((-1.e-12, 1.e-12), 2.e-12),
                ((0., 1.e-12), 0.), ((1.e-12, 1.e-12), 0.)):
            with self.assertRaises(ValueError):
                self.scope['CataloguePrior'](self.bounds(*physical), 4,
                    gap*DURATION**2, chirp_log_uniform=True)

    def test_signed_linear_coordinate_roundtrip(self):
        table = np.zeros((4, 11))
        table[:, 2] = [.00300317, .00900317, .01200317, .04296793987485368]
        table[:, 3] = [2.e-17, -2.698721932368033e-15, 3.e-14, 2.758292489484066e-12]
        table[:, 6], table[:, 7] = [.1, 1.2, 3.2, 5.3], [-.8, -.3, .2, .7]
        encoded = self.scope['encode_doppler'](table, .025, DURATION, chirp_coordinate='linear')
        decoded = self.scope['decode_doppler'](encoded, .025, DURATION, chirp_coordinate='linear')
        np.testing.assert_allclose(decoded, table[:, [2, 3, 6, 7]], rtol=2.e-14, atol=1.e-30)


if __name__ == '__main__':
    unittest.main()

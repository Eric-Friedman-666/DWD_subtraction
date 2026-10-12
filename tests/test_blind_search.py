"""Measured-spectrum screening and numerical equivalence of the F projection."""
import ast
from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch
import numpy as np
from scipy.signal import fftconvolve
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from script.dwd_blind_search import amplitude_f_statistic, residual_covariance, template_f_statistic_search


def numerical_reference():
    # Load the unchanged numerical routines without importing CUDA/LISA on macOS.
    scope = dict(np=np, fftconvolve=fftconvolve, dataclass=dataclass)
    for path, names in [('script/eric_preprocess.py', ('periodogram', 'effective_noise_cov', 'inner_product')),
                        ('script/eric_Fstats.py', ('FStatisticResult', 'f_statistic'))]:
        tree = ast.parse((ROOT / path).read_text())
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), scope)
    return scope


class BlindSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = numerical_reference()

    def screen(self, data, edges, duration, rate, **settings):
        module = ModuleType('script.eric_preprocess')
        module.periodogram = self.reference['periodogram']
        module.effective_noise_cov = self.reference['effective_noise_cov']
        with patch.dict(sys.modules, {'script.eric_preprocess': module}):
            return residual_covariance(data, duration, rate,
                neighboring_bins=128, chunk_size=4096, **settings)

    def test_vectorized_projection_matches_existing_f_statistic(self):
        rng = np.random.default_rng(781)
        bases = rng.normal(size=(7, 4, 3, 31)) + 1j * rng.normal(size=(7, 4, 3, 31))
        data = rng.normal(size=(3, 31)) + 1j * rng.normal(size=(3, 31))
        matrix = rng.normal(size=(31, 3, 3)) + 1j * rng.normal(size=(31, 3, 3))
        covariance = matrix @ matrix.conj().swapaxes(-1, -2) + np.eye(3)
        actual = amplitude_f_statistic(data, bases, covariance, 2048., .1)
        expected = self.reference['f_statistic'](data, bases, covariance, 2048., .1)
        np.testing.assert_allclose(actual['two_f'], expected.two_f, rtol=1.e-12, atol=1.e-13)
        np.testing.assert_allclose(actual['coefficients'], expected.coefficients, rtol=1.e-12, atol=1.e-13)
        np.testing.assert_allclose(actual['amplitude_matrix'], expected.matrix, rtol=1.e-12, atol=1.e-13)

    def test_projection_is_bounded_by_data_energy_including_singular_bases(self):
        rng = np.random.default_rng(912)
        bases = rng.normal(size=(1, 4, 3, 19)) + 1j * rng.normal(size=(1, 4, 3, 19))
        bases[:, 3] = bases[:, 0]
        data = rng.normal(size=(3, 19)) + 1j * rng.normal(size=(3, 19))
        covariance = np.broadcast_to(np.eye(3), (19, 3, 3))
        score = amplitude_f_statistic(data, bases, covariance, 2048., .1)['two_f'][0]
        upper_bound = self.reference['inner_product'](data, data, covariance, 2048., .1)
        self.assertTrue(np.isfinite(score) and 0 <= score <= upper_bound + 1.e-12)

    def test_covariance_has_one_positive_definite_matrix_per_bin(self):
        rng = np.random.default_rng(98121)
        N = 16384
        data = np.fft.rfft(rng.normal(size=(3, N)), axis=-1)
        covariance = self.screen(data, None, N, 1.)
        self.assertEqual(covariance.shape, (N // 2 + 1, 3, 3))
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0))

    def test_zero_observation_has_invertible_covariance(self):
        N = 4096
        covariance = self.screen(np.zeros((3, N // 2 + 1), complex), None, N, 1.)
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0))

    def test_bank_templates_are_evaluated_without_a_grid(self):
        duration, rate = 4096., 1.
        frequency = np.arange(290, 370) / duration
        carrier = .081173
        class Builder:
            def build(self, points):
                bases = np.zeros((len(points), 4, 3, len(frequency)), complex)
                for i, point in enumerate(points):
                    shape = np.exp(-((frequency - point[0]) * duration)**2)
                    bases[i, 0, 0], bases[i, 1, 1] = shape, shape
                    bases[i, 2, 0], bases[i, 3, 1] = 1j * shape, 1j * shape
                return bases
        builder = Builder()
        templates = np.array([[carrier, 0., .3, -.2], [.084, 0., .7, .1]])
        data = 20 * builder.build(templates[:1])[0, 0]
        covariance = np.broadcast_to(np.eye(3) / duration, (len(frequency), 3, 3))
        result = template_f_statistic_search(data, covariance, builder, np.arange(len(frequency)),
            duration, rate, templates, batch_size=3)
        expected = self.reference['f_statistic'](data, builder.build(templates), covariance, duration, rate)
        np.testing.assert_allclose(result['template_two_f'], expected.two_f, rtol=1e-12, atol=1e-12)
        np.testing.assert_array_equal(result['doppler'], templates[:1])
        self.assertEqual(result['template_evaluations'], 2)
        table = np.zeros(len(templates), dtype=[('ID', 'i8')] + [(field, 'f8') for field in
            ('Amplitude', 'Frequency', 'FrequencyDerivative', 'InitialPhase', 'Inclination',
             'EclipticLongitude', 'EclipticLatitude', 'Polarization', 'SNR_absolute', 'SNR_relative')])
        table['ID'] = [2111, 29000000]
        for i, field in enumerate(('Frequency', 'FrequencyDerivative', 'EclipticLongitude', 'EclipticLatitude')):
            table[field] = templates[:, i]
        table = table.view(np.recarray)
        unchanged = table.copy()
        from_table = template_f_statistic_search(data, covariance, builder, np.arange(len(frequency)),
            duration, rate, table, batch_size=3)
        np.testing.assert_array_equal(from_table['template_two_f'], result['template_two_f'])
        np.testing.assert_array_equal(from_table['coefficients'], result['coefficients'])
        np.testing.assert_array_equal(from_table['template_dwd'], table[:1])
        self.assertIs(type(from_table['template_dwd']), type(table))
        np.testing.assert_array_equal(from_table['template_parameters'], unchanged)
        np.testing.assert_array_equal(table, unchanged)
        matrix = np.column_stack([table[field] for field in table.dtype.names])
        before = matrix.copy()
        from_matrix = template_f_statistic_search(data, covariance, builder, np.arange(len(frequency)),
            duration, rate, matrix, batch_size=3)
        np.testing.assert_array_equal(from_matrix['template_two_f'], result['template_two_f'])
        np.testing.assert_array_equal(from_matrix['coefficients'], result['coefficients'])
        np.testing.assert_array_equal(from_matrix['template_dwd'], matrix[:1])
        np.testing.assert_array_equal(matrix, before)
        empty = template_f_statistic_search(data, covariance, builder, np.arange(len(frequency)),
            duration, rate, np.empty((0, 4)))
        self.assertEqual(empty['doppler'].shape, (0, 4))
        empty_table = template_f_statistic_search(data, covariance, builder, np.arange(len(frequency)),
            duration, rate, table[:0])
        self.assertEqual(empty_table['template_dwd'].dtype, table.dtype)
        self.assertEqual(empty_table['template_dwd'].shape, (0,))
        empty_matrix = template_f_statistic_search(data, covariance, builder, np.arange(len(frequency)),
            duration, rate, matrix[:0])
        self.assertEqual(empty_matrix['template_dwd'].shape, (0, 11))


if __name__ == '__main__':
    unittest.main()

"""Measured-spectrum screening and numerical equivalence of the F projection."""
import ast
from dataclasses import dataclass
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch
import numpy as np
from scipy.signal import fftconvolve
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def numerical_reference():
    # Load the numerical routines without importing CUDA/LISA on macOS.
    scope = dict(np=np, fftconvolve=fftconvolve, dataclass=dataclass)
    for path, names in [('script/eric_preprocess.py', ('periodogram', 'effective_noise_cov', 'inner_product')),
                        ('script/eric_Fstats.py', ('FStatisticResult', 'f_statistic'))]:
        tree = ast.parse((ROOT / path).read_text())
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), scope)
    return scope


preprocess_reference = ModuleType('script.eric_preprocess')
preprocess_reference.__dict__.update(numerical_reference())
with patch.dict(sys.modules, {'script.eric_preprocess': preprocess_reference}):
    from script.dwd_blind_search import F_statistic
    F_statistic_module = sys.modules['script.dwd_blind_search']


class BlindSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = numerical_reference()
        cls.shared = dict(np=np)
        helper_tree = ast.parse((ROOT / 'script/eric_Fstats.py').read_text())
        nodes = [node for node in helper_tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('complex_noise_overlap', 'noise_inner_product', 'waveform_match', 'gpu_amplitude_bases_fd')]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'script/eric_Fstats.py', 'exec'), cls.shared)
        helpers = ModuleType('script.eric_Fstats')
        helpers.__dict__.update(cls.shared)
        cls.pipeline = {}
        tree = ast.parse((ROOT / 'eric_subtraction_all.py').read_text())
        names = {'effective_noise_cov', 'periodogram', 'F_statistic', 'inner_product', 'tqdm'}
        imports = [ast.ImportFrom(module=node.module, level=node.level,
                    names=[alias for alias in node.names if alias.name in names])
                   for node in tree.body if isinstance(node, ast.ImportFrom)
                   and any(alias.name in names for alias in node.names)]
        preprocess = ModuleType('script.eric_preprocess')
        preprocess.__dict__.update(cls.reference)
        with patch.dict(sys.modules, {'script.eric_Fstats': helpers, 'script.eric_preprocess': preprocess,
                                     'script.dwd_blind_search': F_statistic_module}):
            exec(compile(ast.fix_missing_locations(ast.Module(body=imports, type_ignores=[])),
                         'eric_subtraction_all.py', 'exec'), cls.pipeline)
        nodes = next(node.body for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == 'single_block_process')
        start = next(i for i, node in enumerate(nodes) if isinstance(node, ast.Assign)
                     and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                     and node.value.func.id == 'F_statistic')
        # The empty-bank return belongs to the worker, not this isolated filter.
        # test_filtered_worker covers that gate in the complete function.
        stop = next(i for i, node in enumerate(nodes) if isinstance(node, ast.If)
                    and ast.unparse(node.test) == 'template_number == 0')
        cls.filter_step = compile(ast.Module(body=nodes[start:stop], type_ignores=[]),
                                  'eric_subtraction_all.py', 'exec')

    def screen(self, data, edges, duration, rate, **settings):
        # Exercise the actual Step 7 covariance, without loading GPU/LISA packages.
        tree = ast.parse((ROOT / 'eric_subtraction_all.py').read_text())
        loop = next(node for node in ast.walk(tree) if isinstance(node, ast.While))
        start = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.For))
        stop = next(i for i, node in enumerate(loop.body) if isinstance(node, ast.With))
        nodes = loop.body[start:stop]
        scope = dict(self.pipeline, np=np, lisa_data_fd=np.vstack((np.arange(data.shape[-1]) / duration, data)),
                     effective_noise_covariance=np.empty((data.shape[-1], 3, 3), complex),
                     neighboring_bins=settings.get('neighboring_bins', 128),
                     chunk_size=settings.get('chunk_size', 4096),
                     fft_frequency=np.arange(data.shape[-1]) / duration, T_obs=duration, fs=rate)
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                     'eric_subtraction_all.py', 'exec'), scope)
        return scope['effective_noise_covariance']

    def test_vectorized_projection_matches_existing_f_statistic(self):
        rng = np.random.default_rng(781)
        bases = rng.normal(size=(7, 4, 3, 31)) + 1j * rng.normal(size=(7, 4, 3, 31))
        data = rng.normal(size=(3, 31)) + 1j * rng.normal(size=(3, 31))
        matrix = rng.normal(size=(31, 3, 3)) + 1j * rng.normal(size=(31, 3, 3))
        covariance = matrix @ matrix.conj().swapaxes(-1, -2) + np.eye(3)
        actual = F_statistic(data, bases, covariance=covariance, T_obs=2048., fs=.1)
        expected = self.reference['f_statistic'](data, bases, covariance, 2048., .1)
        self.assertIsInstance(actual, np.ndarray)
        self.assertEqual(actual.shape, (len(bases),))
        np.testing.assert_allclose(actual, expected.two_f / 2, rtol=1.e-12, atol=1.e-13)

    def test_projection_is_bounded_by_data_energy_including_singular_bases(self):
        rng = np.random.default_rng(912)
        bases = rng.normal(size=(1, 4, 3, 19)) + 1j * rng.normal(size=(1, 4, 3, 19))
        bases[:, 3] = bases[:, 0]
        data = rng.normal(size=(3, 19)) + 1j * rng.normal(size=(3, 19))
        covariance = np.broadcast_to(np.eye(3), (19, 3, 3))
        score = F_statistic(data, bases, covariance=covariance, T_obs=2048., fs=.1)[0]
        upper_bound = self.reference['inner_product'](data, data, covariance, 2048., .1) / 2
        self.assertTrue(np.isfinite(score) and 0 <= score <= upper_bound + 1.e-12)

    def test_covariance_has_one_positive_definite_matrix_per_bin(self):
        rng = np.random.default_rng(98121)
        N = 16384
        data = np.fft.rfft(rng.normal(size=(3, N)), axis=-1)
        progress = Mock(side_effect=lambda iterable, **settings: iterable)
        with patch.dict(self.pipeline, {'tqdm': progress}):
            covariance = self.screen(data, None, N, 1.)
        progress.assert_called_once()
        self.assertEqual(progress.call_args.kwargs, dict(desc='Covariance', unit='chunk'))
        np.testing.assert_array_equal(list(progress.call_args.args[0]), np.arange(1, data.shape[-1], 4096))
        self.assertEqual(covariance.shape, (N // 2 + 1, 3, 3))
        self.assertTrue(np.all(np.linalg.eigvalsh(covariance) > 0))

    def test_zero_observation_uses_estimator_without_fallback(self):
        N = 4096
        with self.assertRaises(np.linalg.LinAlgError):
            self.screen(np.zeros((3, N // 2 + 1), complex), None, N, 1.)

    def test_inline_covariance_matches_direct_estimator_for_one_chunk(self):
        rng = np.random.default_rng(20261013)
        duration, rate = 2048., .1
        data = rng.normal(size=(3, 901)) + 1j * rng.normal(size=(3, 901))
        expected = self.reference['effective_noise_cov'](
            self.reference['periodogram'](data[:, 1:], duration, rate),
            neighboring_bins=128, downweight_peaks=True)
        floor = max(float(np.median(np.trace(expected, axis1=-2, axis2=-1).real / 3)) * 1.e-10, 1.e-100)
        expected += floor * np.eye(3)
        actual = self.screen(data, None, duration, rate)
        np.testing.assert_allclose(actual[1:], expected, rtol=1.e-12, atol=1.e-15)
        np.testing.assert_array_equal(actual[0], actual[1])

    def test_pipeline_projection_and_inner_product_match_existing_reference(self):
        rng = np.random.default_rng(20261013)
        bases = rng.normal(size=(7, 4, 3, 31)) + 1j * rng.normal(size=(7, 4, 3, 31))
        data = rng.normal(size=(3, 31)) + 1j * rng.normal(size=(3, 31))
        matrix = rng.normal(size=(31, 3, 3)) + 1j * rng.normal(size=(31, 3, 3))
        covariance = matrix @ matrix.conj().swapaxes(-1, -2) + np.eye(3)
        expected = self.reference['f_statistic'](data, bases, covariance, 2048., .1)
        actual = self.pipeline['F_statistic'](data, bases, covariance=covariance, T_obs=2048., fs=.1)
        np.testing.assert_allclose(actual, expected.two_f / 2, rtol=1.e-12, atol=1.e-13)
        overlaps = F_statistic_module.inner_product(bases[:, :, None], bases[:, None, :],
            covariance, 2048., .1)
        np.testing.assert_allclose(overlaps, expected.matrix, rtol=1.e-12, atol=1.e-13)

    def test_shared_inner_product_matches_scalar_and_batched_overlap(self):
        rng = np.random.default_rng(1331)
        a = rng.normal(size=(5, 3, 46)) + 1j * rng.normal(size=(5, 3, 46))
        b = rng.normal(size=(7, 3, 46)) + 1j * rng.normal(size=(7, 3, 46))
        a, b = a[..., ::2], b[..., ::2]
        matrix = rng.normal(size=(23, 3, 3)) + 1j * rng.normal(size=(23, 3, 3))
        covariance = matrix @ matrix.conj().swapaxes(-1, -2) + np.eye(3)
        inverse = np.linalg.inv(covariance)
        for first, second in [(a[0], b[0]), (a, b[0]), (a[0], b), (a, a),
                              (a[:, None], b[None, :])]:
            before = first.copy(), second.copy(), covariance.copy()
            expected = self.shared['complex_noise_overlap'](first, second, inverse, 2048., .1).real
            actual = F_statistic_module.inner_product(first, second, covariance, 2048., .1)
            np.testing.assert_allclose(actual, expected, rtol=1.e-12, atol=1.e-13)
            for source, unchanged in zip((first, second, covariance), before):
                np.testing.assert_array_equal(source, unchanged)

    def test_shared_inner_product_handles_empty_catalogues(self):
        covariance = np.broadcast_to(np.eye(3), (17, 3, 3))
        empty = np.empty((0, 3, 17), complex)
        other = np.ones((4, 3, 17), complex)
        self.assertEqual(F_statistic_module.inner_product(empty, empty, covariance, 2048., .1).shape, (0,))
        self.assertEqual(F_statistic_module.inner_product(empty[:, None], other[None, :],
                         covariance, 2048., .1).shape, (0, 4))

    def test_phase_maximized_and_signed_waveform_matches_are_preserved(self):
        rng = np.random.default_rng(1332)
        a = rng.normal(size=(3, 17)) + 1j * rng.normal(size=(3, 17))
        b = np.exp(.8j) * a
        inverse = np.broadcast_to(np.eye(3), (17, 3, 3))
        actual = self.shared['waveform_match'](a, b, inverse, 2048., .1, phase_max=True)
        np.testing.assert_allclose(actual, 1., rtol=1.e-12)
        signed = self.shared['waveform_match'](a, b, inverse, 2048., .1, phase_max=False)
        np.testing.assert_allclose(signed, np.cos(.8), rtol=1.e-12)

    def test_pipeline_singular_basis_matches_analytic_projection(self):
        rng = np.random.default_rng(401)
        bases = np.zeros((2, 4, 3, 19), complex)
        bases[:, 0, 0, 0], bases[:, 1, 1, 1], bases[:, 2, 2, 2] = 1., 1., 1.
        bases[:, 3] = bases[:, 0]
        data = rng.normal(size=(3, 19)) + 1j * rng.normal(size=(3, 19))
        covariance = np.broadcast_to(np.eye(3), (19, 3, 3))
        actual = self.pipeline['F_statistic'](data, bases, covariance=covariance, T_obs=2048., fs=.1)
        expected_score = 2 / (2048. * .1**2) * sum(data[c, c].real**2 for c in range(3))
        self.assertEqual(actual.shape, (2,))
        np.testing.assert_allclose(actual, expected_score, rtol=1.e-12, atol=1.e-13)

    def test_pipeline_reuses_existing_science_helpers(self):
        for name, function in [('effective_noise_cov', self.reference['effective_noise_cov']),
                               ('periodogram', self.reference['periodogram']),
                               ('F_statistic', F_statistic),
                               ('inner_product', self.reference['inner_product'])]:
            self.assertIs(self.pipeline[name], function)
        tree = ast.parse((ROOT / 'eric_subtraction_all.py').read_text())
        self.assertEqual([node.name for node in tree.body if isinstance(node, ast.FunctionDef)],
                         ['initialize_worker', 'single_block_process'])

    def test_pipeline_table_selection_and_basis_bins_match_standard_f_statistic(self):
        duration, rate = 4096., 1.
        frequency = np.arange(290, 370) / duration
        core = np.arange(3, len(frequency) - 2, 2)
        class Builder:
            def __init__(self, frequency):
                self.frequency = frequency
            def build(self, points):
                bases = np.zeros((len(points), 4, 3, len(self.frequency)), complex)
                for i, point in enumerate(points):
                    shape = np.exp(-((self.frequency - point[0]) * duration)**2)
                    bases[i, 0, 0], bases[i, 1, 1] = shape, shape
                    bases[i, 2, 0], bases[i, 3, 1] = 1j * shape, 1j * shape
                return bases
        builder = Builder(frequency[core])
        previous_builder = Builder(frequency)
        table = np.zeros((5, 11))
        table[:, 0] = [2111, 29000000, 9, 12, 33]
        table[:, [1, 4, 5, 8, 9, 10]] = np.column_stack((np.full(5, 1.e-22),
            np.arange(5) * .1, np.ones(5), np.arange(5) * .2, np.arange(5) + 20., np.arange(5) + 8.))
        table[:, [2, 3, 6, 7]] = np.array([
            [.081173, 0., .3, -.2], [.084, 0., .7, .1], [.086, 1.e-15, .2, .2],
            [.079, -1.e-15, .8, -.1], [.083, 0., .5, -.2]])
        before = table.copy()
        data = 20 * previous_builder.build(table[:1, [2, 3, 6, 7]])[0, 0, :, core].T
        covariance = np.broadcast_to(np.eye(3) / duration, (len(core), 3, 3))
        for rows, threshold in ((table, 16.), (table[:0], 16.), (table, 1.e100)):
            bases = previous_builder.build(rows[:, [2, 3, 6, 7]])[..., core]
            np.testing.assert_array_equal(builder.build(rows[:, [2, 3, 6, 7]]), bases)
            expected = self.reference['f_statistic'](data, bases, covariance, duration, rate)
            F_values = F_statistic(data, bases, covariance=covariance, T_obs=duration, fs=rate)
            np.testing.assert_allclose(2 * F_values, expected.two_f, rtol=1.e-12, atol=1.e-12)
            retained = expected.two_f >= 2 * threshold
            scope = dict(self.pipeline, np=np, data=data, covariance=covariance,
                T_obs=duration, fs=rate, F_threshold=threshold, dwd_template_table=rows,
                waveform_table_fd=bases, response=builder)
            with patch.object(builder, 'build', side_effect=AssertionError('step 5 rebuilt template bases')):
                exec(self.filter_step, scope)
            actual = scope['filter_result']
            self.assertIsInstance(actual, np.ndarray)
            self.assertEqual(actual.shape, (np.count_nonzero(retained), 11))
            self.assertEqual(scope['template_number'], len(actual))
            np.testing.assert_array_equal(actual, rows[retained])
        np.testing.assert_array_equal(table, before)



if __name__ == '__main__':
    unittest.main()

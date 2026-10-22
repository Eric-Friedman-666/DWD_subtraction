"""Current inclusive F>=16 gate and the four-value checkpoint contract."""
import ast
import os
import pickle
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class FilteredWorkerTests(unittest.TestCase):
    def test_empty_and_boundary_cases_use_four_returns(self):
        source = (ROOT / 'eric_subtraction_all.py').read_text()
        tree = ast.parse(source)
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'single_block_process')
        configured = next(node.value.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'F_threshold' for target in node.targets))
        self.assertEqual(configured, 16.)
        for scores, zero_model in ((np.empty(0), False), (np.full(3, np.nextafter(16., -np.inf)), False),
                                  (np.array([np.nextafter(16., -np.inf), 16., np.nextafter(16., np.inf)]), False),
                                  (np.array([np.nextafter(16., -np.inf), 16., np.nextafter(16., np.inf)]), True)):
            table = np.zeros((len(scores), 11))
            table[:, 0] = np.arange(1, len(scores) + 1)
            table[:, 2] = .15
            calls = []
            class Builder:
                def __init__(self, *args):
                    self.bins = len(args[4])
                def build(self, points):
                    return np.zeros((len(points), 4, 3, self.bins), complex)
            def sampler(data, covariance, filtered, *args, **kwargs):
                calls.append(filtered.copy())
                doppler = filtered[:, [2, 3, 6, 7]]
                if zero_model:
                    doppler = doppler[:0]
                return dict(doppler=doppler, fit_result=dict(coefficients=np.ones((len(doppler), 4))))
            def amplitudes(coefficients):
                n = len(coefficients)
                return {name: np.ones(n) for name in ('A', 'phi0', 'iota', 'psi_bme')}
            frequency = np.array([.05, .1, .15, .2, .25, .3])
            inputs = (np.array([[.1, .2]]), .1, dict(f=2), 100., .1, 16.,
                np.array([[-1.e-10, -1.e-25], [1.e-25, 1.e-10]]), (1, 2), 64,
                np.array([0.]), None, .1, .1)
            scope = dict(np=np, os=os, pickle=pickle, GPUResponse=Builder, F_statistic=lambda *args, **kwargs: scores,
                run_eryn_rjmcmc=sampler, physical_amplitudes=amplitudes, worker_inputs=inputs,
                lisa_data_fd=np.vstack((frequency, np.zeros((3, len(frequency)), complex))),
                effective_noise_covariance=np.broadcast_to(np.eye(3), (len(frequency), 3, 3)))
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / 'eric_subtraction_all.py'), 'exec'), scope)
            with tempfile.TemporaryDirectory() as directory:
                recovered, filtered, bins, posterior = scope['single_block_process'](
                    0, table, work=Path(directory), label='test')
                self.assertTrue((Path(directory) / 'round_test_block_00000/result.pkl').is_file())
            np.testing.assert_array_equal(bins, np.arange(6))
            retained = scores >= 16.
            np.testing.assert_array_equal(filtered, table[retained])
            if not retained.any():
                self.assertEqual(recovered.shape, (0, 8))
                self.assertEqual(filtered.shape, (0, 11))
                self.assertIsNone(posterior)
                self.assertEqual(calls, [])
            else:
                self.assertEqual(len(calls), 1)
                np.testing.assert_array_equal(calls[0][:, 0], [2, 3])
                self.assertIsNotNone(posterior)
                self.assertEqual(recovered.shape, (0 if zero_model else 2, 8))


if __name__ == '__main__':
    unittest.main()

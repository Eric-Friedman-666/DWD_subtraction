"""Current inclusive F>=16 gate and the four-value checkpoint contract."""
import ast
import os
import pickle
from pathlib import Path
import tempfile
import unittest
import numpy as np
from script.dwd_chirp_policy import chirp_bounds_for_bank, bank_padding_values

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
            bound_calls, frequency_calls = [], []
            class Builder:
                def __init__(self, *args):
                    self.bins = len(args[4])
                def build(self, points):
                    return np.zeros((len(points), 4, 3, self.bins), complex)
            def sampler(data, covariance, filtered, *args, **kwargs):
                calls.append(filtered.copy())
                bound_calls.append(np.asarray(args[1]).copy())
                np.testing.assert_array_equal(args[0], frequency_blocks[1])
                frequency_calls.append(np.asarray(kwargs['physical_frequency']).copy())
                self.assertEqual(data.shape, (3, 6))
                self.assertEqual(covariance.shape, (6, 3, 3))
                doppler = filtered[:, [2, 3, 6, 7]]
                if zero_model:
                    doppler = doppler[:0]
                return dict(doppler=doppler, fit_result=dict(coefficients=np.ones((len(doppler), 4))))
            def amplitudes(coefficients):
                n = len(coefficients)
                return {name: np.ones(n) for name in ('A', 'phi0', 'iota', 'psi_bme')}
            frequency = np.array([.005, .05, .1, .15, .2, .25, .3, .4, .5])
            frequency_blocks = np.array([[.01, .11], [.11, .21], [.21, .31]])
            bank_bounds = np.array([[[-3.e-10, -1.e-25], [1.e-25, 3.e-10]],
                                    [[-1.e-10, -1.e-25], [1.e-25, 1.e-10]],
                                    [[-2.e-10, -1.e-25], [1.e-25, 2.e-10]]])
            padding = np.array([.2, .1, .3])
            inputs = (frequency_blocks, .1, dict(f=2), 100., .1, 16.,
                bank_bounds, (1, 2), 64, np.array([0.]), None, .1, padding)
            scope = dict(np=np, os=os, pickle=pickle, GPUResponse=Builder, F_statistic=lambda *args, **kwargs: scores,
                run_eryn_rjmcmc=sampler, physical_amplitudes=amplitudes, worker_inputs=inputs,
                chirp_bounds_for_bank=chirp_bounds_for_bank, bank_padding_values=bank_padding_values,
                lisa_data_fd=np.vstack((frequency, np.zeros((3, len(frequency)), complex))),
                effective_noise_covariance=np.broadcast_to(np.eye(3), (len(frequency), 3, 3)))
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / 'eric_subtraction_all.py'), 'exec'), scope)
            with tempfile.TemporaryDirectory() as directory:
                recovered, filtered, bins, posterior = scope['single_block_process'](
                    1, table, work=Path(directory), label='test')
                self.assertTrue((Path(directory) / 'round_test_block_00001/result.pkl').is_file())
            np.testing.assert_array_equal(bins, np.arange(1, 7))
            retained = scores >= 16.
            np.testing.assert_array_equal(filtered, table[retained])
            if not retained.any():
                self.assertEqual(recovered.shape, (0, 8))
                self.assertEqual(filtered.shape, (0, 11))
                self.assertIsNone(posterior)
                self.assertEqual(calls, [])
                self.assertEqual(bound_calls, [])
                self.assertEqual(frequency_calls, [])
            else:
                self.assertEqual(len(calls), 1)
                np.testing.assert_array_equal(calls[0][:, 0], [2, 3])
                np.testing.assert_array_equal(bound_calls[0], bank_bounds[1])
                np.testing.assert_array_equal(frequency_calls[0], frequency[1:7])
                self.assertIsNotNone(posterior)
                self.assertEqual(recovered.shape, (0 if zero_model else 2, 8))


if __name__ == '__main__':
    unittest.main()

"""Strong, same-owner-block template eligibility for recovered-source matching."""
import ast
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

import test_subtraction_all as fixtures


class SameBlockAssociationTests(unittest.TestCase):
    run_case = fixtures.GlobalRoundTests.run_case

    def test_only_strictly_above_seven_in_the_same_block_can_match(self):
        seen = []
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, output = self.run_case(directory, bins=(10, 11, 12, 23),
                catalogue_relative_snrs=(6.999, 7., 7.001, 20.),
                association_hook=lambda r, b, ids: seen.append((r, b, ids.tolist())))
            self.assertEqual(seen, [(1, 0, [3]), (1, 1, [4])])
            with h5py.File(output, 'r') as saved:
                np.testing.assert_array_equal(saved['final_subtraction/dwd_id'][:], [3, 4])
                self.assertEqual(saved['config/identification_template_scope'][()].decode(),
                                 'remaining_same_block_input_bank')

    def test_reference_bank_is_input_table_not_f_screened_table(self):
        seen = []
        with tempfile.TemporaryDirectory() as directory:
            _, state, _, _ = self.run_case(directory, bins=(10,), catalogue_bins=(10, 11),
                association_hook=lambda r, b, ids: seen.append((r, b, ids.tolist())))
            self.assertEqual(state['sampler'][0][0][:, 0].tolist(), [1])
            self.assertEqual(seen, [(1, 0, [1, 2])])

    def test_previously_subtracted_ids_do_not_reappear_in_matching_bank(self):
        seen = []
        with tempfile.TemporaryDirectory() as directory:
            result, _, _, _ = self.run_case(directory, wait_for_round=(23,),
                association_hook=lambda r, b, ids: seen.append((r, b, ids.tolist())))
            self.assertEqual(result['rounds'], 3)
            self.assertEqual(seen, [(1, 0, [1]), (2, 1, [2])])

    def test_no_global_catalogue_scan_or_drift_window_remains_in_step_seven(self):
        source = (Path(__file__).resolve().parents[1] / 'eric_subtraction_all.py').read_text()
        tree = ast.parse(source)
        main = [node for node in tree.body if isinstance(node, ast.If)][-1]
        assignments = [node for node in ast.walk(main) if isinstance(node, ast.Assign)]
        row_selection = next(node for node in assignments
            if any(isinstance(target, ast.Name) and target.id == 'truth_rows' for target in node.targets))
        self.assertEqual(ast.unparse(row_selection.value), 'table')
        self.assertFalse({'maximum_drift', 'truth_order', 'truth_frequency', 'nearby_indices'}
                         & {node.id for node in ast.walk(main) if isinstance(node, ast.Name)})
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                             and ast.unparse(node.func) == 'h5py.File'
                             and node.args and isinstance(node.args[0], ast.Name)
                             and node.args[0].id == 'snr_file' for node in ast.walk(main)))


if __name__ == '__main__':
    unittest.main()

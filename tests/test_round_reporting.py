"""Round timing and unique-ID reporting without GPU science or a live-run restart."""
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

import test_subtraction_all as fixtures


class RoundReportingTests(unittest.TestCase):
    run_case = fixtures.GlobalRoundTests.run_case

    def test_round_timer_resets_and_counts_unique_ids_not_component_rows(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('time.perf_counter', side_effect=[10., 13.5, 20., 26.25]) as clock:
            result, state, original, output = self.run_case(directory,
                bins=(10, 23, 36, 50), catalogue_ids=(27, 27, 99, 27))
            self.assertEqual(clock.call_count, 4)
            self.assertEqual(result['rounds'], 2)
            self.assertEqual(len(state['tdi']), 1)
            with h5py.File(output) as saved:
                for label, seconds, count in [('01', 3.5, 2), ('02', 6.25, 0)]:
                    self.assertEqual(saved[f'rounds/{label}/elapsed_seconds'][()], seconds)
                    self.assertEqual(saved[f'rounds/{label}/identified_dwd_count'][()], count)
                self.assertEqual(saved['rounds/01/identified_count'][:].sum(), 4)
                np.testing.assert_array_equal(saved['final_subtraction/dwd_id'][:], [27, 27, 99, 27])
                np.testing.assert_allclose(saved['final_subtraction/signal_xyz_td'][:]
                    + saved['final_subtraction/residual_xyz_td'][:], original, atol=1.e-12)

    def test_empty_terminal_round_still_records_time_and_zero_count(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('time.perf_counter', side_effect=[5., 9.]):
            result, state, original, output = self.run_case(directory, empty_fit=True)
            self.assertEqual(result['rounds'], 1)
            self.assertFalse(state['tdi'])
            with h5py.File(output) as saved:
                self.assertEqual(saved['rounds/01/elapsed_seconds'][()], 4.)
                self.assertEqual(saved['rounds/01/identified_dwd_count'][()], 0)
                np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_td'][:], original)

    def test_elapsed_time_is_measured_after_the_round_result_is_written(self):
        writes, events = [], []
        save_tree = fixtures.save_tree
        def observe_write(group, values):
            save_tree(group, values)
            if group.name.startswith('/rounds/'):
                writes.append(group.name)
                events.append(('saved', group.name))
        def clock():
            events.append('clock')
            return float(len(events))
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(fixtures, 'save_tree', side_effect=observe_write), \
             patch('time.perf_counter', side_effect=clock):
            self.run_case(directory, empty_fit=True)
        # The mandatory refit audit is a nested dictionary, saved recursively
        # before its enclosing round. Neither save may occur after the timer.
        self.assertEqual(writes, ['/rounds/01/amplitude_refits', '/rounds/01'])
        self.assertEqual(events, ['clock', ('saved', '/rounds/01/amplitude_refits'),
                                  ('saved', '/rounds/01'), 'clock'])


if __name__ == '__main__':
    unittest.main()

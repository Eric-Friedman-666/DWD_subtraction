"""Only individually identified fitted components enter native subtraction."""
import tempfile
import unittest

import h5py
import numpy as np

import test_subtraction_all as fixtures


class IdentificationGateTests(unittest.TestCase):
    run_case = fixtures.GlobalRoundTests.run_case

    def test_only_identified_rows_are_subtracted_at_any_identified_fraction(self):
        for identified in (0, 1, 15, 16, 17, 25):
            with self.subTest(identified=identified), tempfile.TemporaryDirectory() as directory:
                result, state, original, output = self.run_case(
                    directory, bins=(10,), catalogue_bins=(10,) * 25,
                    association_confirmed_counts={0: identified}, fitted_amplitude=1 / 25,
                    screening_scale=16.)
                passes = identified > 0
                self.assertEqual(result['source_count'], identified)
                self.assertEqual(result['rounds'], 2 if passes else 1)
                self.assertEqual(len(state['waveform']), int(passes))
                self.assertEqual(len(state['tdi']), int(passes))
                with h5py.File(output) as saved:
                    # Current production saves compact round summaries, not
                    # the retired per-block acceptance/recovery HDF5 tree.
                    self.assertEqual(saved['rounds/01/accepted_count'][0], identified)
                    self.assertEqual(saved['rounds/01/recovered_count'][0], 25)
                    self.assertEqual(saved['rounds/01/identified_count'][0], identified)
                    self.assertEqual(saved['rounds/01/ghost_count'][0], 25 - identified)
                    if passes:
                        ids = saved['final_subtraction/dwd_id'][:]
                        self.assertEqual(np.count_nonzero(ids >= 0), identified)
                        self.assertFalse(np.any(ids < 0))
                        self.assertEqual(len(np.unique(ids)), identified)
                        self.assertEqual(len(state['waveform'][0]), identified)
                        np.testing.assert_allclose(saved['final_subtraction/signal_xyz_td'][:],
                            original * identified / 25, atol=1.e-12)
                        np.testing.assert_allclose(saved['final_subtraction/residual_xyz_td'][:],
                            original * (1 - identified / 25), atol=1.e-12)
                    else:
                        np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_td'][:], original)
                        np.testing.assert_array_equal(saved['final_subtraction/signal_xyz_td'][:], 0.)
                    self.assertTrue(saved.attrs['complete'])
                    self.assertIn(saved.attrs['stopping_reason'],
                                  ('no_new_candidates', 'no_identified_components'))

    def test_ghost_only_block_does_not_reject_another_blocks_identified_source(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(
                directory, association_confirmed_counts={0: 0, 1: 1})
            self.assertEqual(result['rounds'], 2)
            self.assertEqual(result['source_count'], 1)
            self.assertEqual(len(state['tdi']), 1)
            with h5py.File(output) as saved:
                np.testing.assert_array_equal(saved['rounds/01/accepted_count'][:], [0, 1, 0, 0])
                np.testing.assert_array_equal(saved['rounds/02/accepted_count'][:], 0)
                residual = saved['final_subtraction/residual_xyz_td'][:]
                signal = saved['final_subtraction/signal_xyz_td'][:]
                self.assertGreater(np.linalg.norm(residual), 0.)
                self.assertGreater(np.linalg.norm(signal), 0.)
                np.testing.assert_allclose(residual + signal, original, atol=1.e-12)
                self.assertEqual(saved['rounds/01/accepted_count'][0], 0)
                self.assertEqual(saved['rounds/01/accepted_count'][1], 1)

    def test_zero_dwd_model_is_not_divided_by_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory, empty_fit=True)
            self.assertEqual(result['source_count'], 0)
            self.assertEqual(result['rounds'], 1)
            self.assertFalse(state['tdi'])
            with h5py.File(output) as saved:
                self.assertEqual(saved['rounds/01/recovered_count'][0], 0)
                self.assertEqual(saved['rounds/01/identified_count'][0], 0)
                self.assertEqual(saved['rounds/01/accepted_count'][0], 0)
                self.assertTrue(np.isfinite(saved['final_subtraction/residual_xyz_td'][:]).all())
                np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_td'][:], original)

    def test_ghost_only_followup_stops_after_subtracting_the_fitted_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory, bins=(10,),
                catalogue_bins=(10, 10),
                fitted_amplitude=.25, association_confirmed_counts={0: (1, 0)})
            self.assertEqual(result['rounds'], 2)
            self.assertEqual(result['source_count'], 1)
            self.assertEqual(len(state['tdi']), 1)
            self.assertEqual(len(state['sampler']), 2)
            np.testing.assert_array_equal(state['sampler'][0][0][:, 0], [1, 2])
            np.testing.assert_array_equal(state['sampler'][1][0][:, 0], [2])
            with h5py.File(output) as saved:
                np.testing.assert_array_equal(saved['final_subtraction/dwd_id'][:], [1])
                self.assertEqual(saved['rounds/02/recovered_count'][0], 1)
                self.assertEqual(saved['rounds/02/identified_count'][0], 0)
                self.assertEqual(saved['rounds/02/ghost_count'][0], 1)
                self.assertEqual(saved['rounds/02/accepted_count'][0], 0)
                self.assertEqual(saved['final_subtraction/parameters'][0, 0], .25)
                np.testing.assert_allclose(saved['final_subtraction/signal_xyz_td'][:], .25 * original, atol=1.e-12)
                np.testing.assert_allclose(saved['final_subtraction/residual_xyz_td'][:], .75 * original, atol=1.e-12)

    def test_old_conditional_f_veto_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, _, _ = self.run_case(directory, bins=(10,), rejected=(10,))
            self.assertEqual(result['source_count'], 1)
            self.assertEqual(len(state['tdi']), 1)


if __name__ == '__main__':
    unittest.main()

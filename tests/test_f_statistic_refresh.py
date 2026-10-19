"""F screening must use the residual produced by the preceding subtraction."""
import tempfile
import unittest

import h5py
import numpy as np

import test_subtraction_all as fixtures


class FStatisticRefreshTests(unittest.TestCase):
    run_case = fixtures.GlobalRoundTests.run_case

    def test_default_threshold_is_16_inclusive(self):
        for initial_f in (15.999, 16., 16.001):
            with self.subTest(initial_f=initial_f), tempfile.TemporaryDirectory() as directory:
                result, state, _, output = self.run_case(
                    directory, bins=(10,), screening_scale=initial_f, fitted_amplitude=.5)
                passed = initial_f >= 16.
                self.assertEqual(len(state['sampler']), int(passed))
                self.assertEqual(result['source_count'], int(passed))
                with h5py.File(output) as saved:
                    self.assertEqual(saved['config/f_threshold'][()], 16.)
                    self.assertEqual(saved['rounds/01/selected_count'][0], int(passed))

    def test_each_subtraction_refreshes_fft_and_f_values(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(
                directory, bins=(10,), catalogue_bins=(10, 10), screening_scale=64.,
                fitted_amplitude=.5, association_confirmed_counts={0: (1, 0)})
            scores = [(ids.tolist(), values.tolist()) for ids, values in state['f_values'] if len(values)]
            self.assertEqual([ids for ids, _ in scores], [[1, 2], [2]])
            np.testing.assert_allclose(scores[0][1], [64., 64.], atol=1.e-12)
            np.testing.assert_allclose(scores[1][1], [16.], atol=1.e-12)
            self.assertEqual(result['rounds'], 2)
            self.assertEqual(len(state['sampler']), 2)
            self.assertEqual(len(state['tdi']), 1)
            with h5py.File(output) as saved:
                bins = saved['blocks/00000/retained_bins'][:]
                before = np.fft.rfft(original, axis=-1)[:, bins]
                for label, fraction, selected in (('01', 1., 2), ('02', .5, 1)):
                    block = saved[f'rounds/{label}/blocks/00000']
                    np.testing.assert_allclose(block['screening/residual_input_xyz_fd'][:], fraction * before, atol=1.e-12)
                    self.assertEqual(saved[f'rounds/{label}/selected_count'][0], selected)
                np.testing.assert_allclose(saved['final_subtraction/residual_xyz_td'][:], .5 * original, atol=1.e-12)


if __name__ == '__main__':
    unittest.main()

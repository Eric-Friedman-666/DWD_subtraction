"""Current compact-writer contracts, separate from the legacy rich-output suite."""
import json
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
import test_subtraction_all as fixtures
from script.dwd_chirp_policy import (
    bank_configuration_record, frequency_dependent_chirp_bounds,
    frequency_dependent_padding,
)


class ReleaseControllerTests(unittest.TestCase):
    run_case = fixtures.GlobalRoundTests.run_case

    def test_current_settings_and_signed_prior_are_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            result, state, original, output = self.run_case(folder, bins=(10, 36))
            with h5py.File(output) as saved:
                np.testing.assert_array_equal(saved['config/derivative_bounds'][:],
                    [[-4.e-12, -7.60863249122299e-23], [7.60863249122299e-23, 4.e-12]])
                self.assertEqual(saved['config/chirp_prior'][()].decode(), 'signed_log_uniform')
                self.assertEqual(saved['config/response_step_seconds'][()], 900.)
                self.assertEqual(saved['config/maximum_sources'][()], 100)
                self.assertEqual(saved['config/relative_snr_threshold'][()], 7.)

    def test_finalized_status_and_counts_follow_closed_round_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            result, state, original, output = self.run_case(folder, bins=(10, 36))
            status = json.loads(output.with_suffix('.status.json').read_text())
            self.assertTrue(status['complete'])
            self.assertEqual(status['phase'], 'complete')
            self.assertEqual(status['finalized_round_counts'], {'01': 2, '02': 0})
            self.assertEqual(status['finalized_identified_dwd_count'], 2)
            self.assertEqual(status['last_completed_round'], 2)
            with h5py.File(output) as saved:
                self.assertTrue(saved.attrs['complete'])
                self.assertEqual(saved['rounds/01/identified_dwd_count'][()], 2)
                self.assertEqual(saved['rounds/02/identified_dwd_count'][()], 0)

    def test_combined_native_subtraction_occurs_once_after_all_fits(self):
        with tempfile.TemporaryDirectory() as folder:
            result, state, original, output = self.run_case(folder, bins=(10, 36))
            self.assertEqual(len(state['tdi']), 1)
            self.assertEqual(len(state['waveform']), 1)
            self.assertEqual(len(state['sampler']), 2)
            with h5py.File(output) as saved:
                np.testing.assert_allclose(saved['final_subtraction/signal_xyz_td'][:], original, atol=1.e-12)
                np.testing.assert_allclose(saved['final_subtraction/residual_xyz_td'][:], 0., atol=1.e-12)

    def test_finalized_earlier_ids_never_reenter_the_next_bank(self):
        with tempfile.TemporaryDirectory() as folder:
            result, state, original, output = self.run_case(folder, bins=(10, 36))
            directories = sorted(p.name for p in output.with_suffix('.checkpoints').glob('round_*_block_*'))
            self.assertEqual(directories, ['round_01_block_00000', 'round_01_block_00002'])
            self.assertEqual(result['rounds'], 2)

    def test_strict_saved_snr_seven_filter_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            result, state, original, output = self.run_case(folder, bins=(10, 36),
                catalogue_relative_snrs=(7., 7.01))
            with h5py.File(output) as saved:
                np.testing.assert_array_equal(saved['final_subtraction/dwd_id'][:], [2])
            self.assertEqual(result['source_count'], 1)

    def test_input_empty_round_never_launches_a_fit_or_creates_block_results(self):
        with tempfile.TemporaryDirectory() as folder:
            result, state, original, output = self.run_case(folder, bins=(10, 36),
                catalogue_relative_snrs=(7., 1.))
            self.assertEqual(state['search'], [])
            self.assertEqual(state['sampler'], [])
            self.assertEqual(state['tdi'], [])
            self.assertFalse(list(output.with_suffix('.checkpoints').glob('round_*_block_*')))

    def test_f_screened_empty_bank_is_saved_but_not_subtracted(self):
        with tempfile.TemporaryDirectory() as folder:
            result, state, original, output = self.run_case(folder, bins=(), catalogue_bins=(10,))
            self.assertEqual(len(state['search']), 1)
            self.assertEqual(state['sampler'], [])
            self.assertEqual(state['tdi'], [])
            directory = output.with_suffix('.checkpoints')/'round_01_block_00000'
            self.assertTrue((directory/'result.pkl').exists())
            self.assertFalse((directory/'eryn_chain_00.h5').exists())

    def test_reference_and_fit_frequency_grids_are_identical(self):
        with tempfile.TemporaryDirectory() as folder:
            result, state, original, output = self.run_case(folder, bins=(10, 36))
            # The actual controller rejects any mismatch at matching time.
            self.assertEqual(result['source_count'], 2)
            with h5py.File(output) as saved:
                padding = saved['config/likelihood_frequency_padding_hz'][:]
                bounds = saved['config/derivative_bounds'][:]
                bank_bounds = saved['config/derivative_bounds_by_block'][:]
                blocks = saved['config/frequency_blocks_hz'][:]
                duration = float(saved['config/duration'][()])
                rate = float(saved['config/rate'][()])
                width = float(saved['config/frequency_block_width'][()])
                policy_hash = saved['config/chirp_bounds_policy_sha256'][()].decode()
            expected_bounds = frequency_dependent_chirp_bounds(blocks, bounds)
            expected_padding = frequency_dependent_padding(duration, width, blocks, expected_bounds)
            np.testing.assert_array_equal(bank_bounds, expected_bounds)
            np.testing.assert_array_equal(padding, expected_padding)
            record = bank_configuration_record(duration, width, blocks, bounds,
                expected_bounds, expected_padding)
            atomic = json.loads((output.parent/'frequency_banks.settings.json').read_text())
            self.assertEqual(policy_hash, record['policy_sha256'])
            for name, value in record.items():
                self.assertEqual(atomic[name], value)
            frequency = np.fft.rfftfreq(round(duration*rate), d=1./rate)
            for _, data, settings, worker_bounds, block in state['sampler']:
                index, = np.flatnonzero(np.all(blocks == block, axis=1))
                np.testing.assert_array_equal(worker_bounds, expected_bounds[index])
                keep = ((frequency >= block[0]-padding[index])
                    & (frequency <= block[1]+padding[index]) & (frequency > 0.))
                np.testing.assert_array_equal(settings['physical_frequency'], frequency[keep])
                self.assertEqual(data.shape[-1], np.count_nonzero(keep))

    def test_final_native_fft_and_residual_are_consistent(self):
        with tempfile.TemporaryDirectory() as folder:
            result, state, original, output = self.run_case(folder, bins=(10, 36))
            with h5py.File(output) as saved:
                residual = saved['final_subtraction/residual_xyz_td'][:]
                model = saved['final_subtraction/signal_xyz_td'][:]
                np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_fd'][:], np.fft.rfft(residual, axis=-1))
                np.testing.assert_allclose(model+residual, original, atol=1.e-12)


if __name__ == '__main__':
    unittest.main()

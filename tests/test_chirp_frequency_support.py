import ast
from pathlib import Path
import unittest
import numpy as np
from script.dwd_frequency_support import (
    chirp_frequency_padding, response_grid, response_sample_block, RESPONSE_STEP_SECONDS)

ROOT = Path(__file__).resolve().parents[1]
T, RATE = 31104000., .1
BOUNDS = np.array([[-4.e-12, -7.60863249122299e-23],
                   [7.60863249122299e-23, 4.e-12]])


class ChirpSupportTests(unittest.TestCase):
    def test_padding_contains_both_signs_full_prior_drift(self):
        pad = chirp_frequency_padding(T, 1.e-5, .05, BOUNDS)
        self.assertGreater(pad, 4.e-12*T + .05*1.e-4)
        self.assertAlmostEqual(pad, 4.e-12*T + .05*1.e-4 + 8./T)

    def test_padding_has_a_block_width_floor(self):
        self.assertEqual(chirp_frequency_padding(1000., .1, .05, BOUNDS), .1)

    def test_padding_does_not_consume_catalogue_values(self):
        node = next(n for n in ast.parse((ROOT/'script/dwd_frequency_support.py').read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'chirp_frequency_padding')
        self.assertEqual([a.arg for a in node.args.args],
                         ['duration', 'block_width', 'maximum_frequency', 'derivative_bounds'])

    def test_signed_and_positive_bounds_choose_same_conservative_padding(self):
        self.assertEqual(chirp_frequency_padding(T, 1.e-5, .05, BOUNDS),
                         chirp_frequency_padding(T, 1.e-5, .05, BOUNDS[1]))

    def test_bad_padding_inputs_fail(self):
        for duration, bounds in ((0., BOUNDS), (-1., BOUNDS), (T, [np.nan, 1.])):
            with self.assertRaises(ValueError):
                chirp_frequency_padding(duration, 1.e-5, .05, bounds)

    def test_full_native_grid_is_covered_without_omitted_tail(self):
        times = response_grid(T, RATE)
        self.assertEqual(len(times), T/RESPONSE_STEP_SECONDS)
        block = response_sample_block(times, T, RATE)
        self.assertEqual(block*len(times), round(T*RATE))
        self.assertAlmostEqual(times[0], (block-1)/(2*RATE))
        self.assertAlmostEqual(times[-1]+(block-1)/(2*RATE), T-1/RATE)

    def test_grid_integration_factor_follows_resolution(self):
        for step in (3600., 900., 600., 300., 150., 100.):
            times = response_grid(T, RATE, step)
            self.assertEqual(response_sample_block(times, T, RATE), round(step*RATE))

    def test_nonpartitioning_or_aliased_midpoints_fail_closed(self):
        for args in ((T, RATE, 75.), (T+10., RATE, 300.)):
            with self.assertRaises(ValueError):
                response_grid(*args)
        with self.assertRaises(ValueError):
            response_sample_block(response_grid(T, RATE)+1., T, RATE)

    def test_workers_and_references_use_the_identical_padding(self):
        text = (ROOT/'eric_subtraction_all.py').read_text()
        self.assertIn('padding = frequency_padding_hz[selected_block_indices[0]]', text)
        self.assertIn('padding = bank_padding_values(frequency_padding_hz, freq_block_width,', text)
        self.assertIn('frequency_padding_hz=frequency_padding_hz,', text)
        self.assertIn('chirp_bounds_by_block=frequency_derivative_bounds_by_block)', text)
        self.assertIn('response_times, response_geometry, amplitude_prior_sigma, frequency_padding_hz)', text)

    def test_ownership_and_scientific_settings_are_not_widened(self):
        text = (ROOT/'eric_subtraction_all.py').read_text()
        for value in ('freq_block_width = 1.e-5', 'SNR_r_threshold = 7.',
                      'maximum_sources = 100', 'F_threshold = 16.',
                      'sample_number = (500, 1500)', 'amplitude_prior_sigma = .1',
                      'refit_identified_amplitudes = True'):
            self.assertIn(value, text)
        self.assertIn("dwd_template_table[:, fields['f']] < frequency_blocks[index, 1]", text)

    def test_window_and_grid_are_persisted_with_correct_prior_label(self):
        text = (ROOT/'eric_subtraction_all.py').read_text()
        for value in ("else 'signed_log_uniform'", 'likelihood_frequency_padding_hz=frequency_padding_hz',
                      'response_step_seconds=RESPONSE_STEP_SECONDS'):
            self.assertIn(value, text)


if __name__ == '__main__':
    unittest.main()

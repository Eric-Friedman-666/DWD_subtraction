"""Per-bank prior/support contracts; any HDF5 here is a NEW temporary fixture."""
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from script.dwd_chirp_policy import (
    MAX_DETACHED_CHIRP_MASS_SOLAR, GR_POSITIVE_COEFFICIENT,
    LEGACY_LOW_FREQUENCY_FLOOR_HZ_PER_S,
    bank_configuration_record, bank_padding_values, chirp_bounds_for_bank,
    chirp_policy_metadata, frequency_dependent_chirp_bounds,
    frequency_dependent_padding,
)

ROOT = Path(__file__).resolve().parents[1]
GAP = 7.60863249122299e-23
GLOBAL = np.array([[-4.e-12, -GAP], [GAP, 4.e-12]])
T = 360. * 86400.


class FrequencyDependentChirpTests(unittest.TestCase):
    def blocks(self):
        return np.array([[.003, .00301], [.009, .00901], [.012, .01201],
                         [.02, .02001], [.04296, .04297], [.04999, .05]])

    def test_gr_coefficient_is_derived_from_mass_not_printed_positive_typo(self):
        self.assertAlmostEqual(MAX_DETACHED_CHIRP_MASS_SOLAR, 1.2187707886145736)
        self.assertAlmostEqual(GR_POSITIVE_COEFFICIENT, 8.05972379449993e-7, places=18)

    def test_low_frequency_floor_preserves_old_ceiling_and_both_signs(self):
        bounds = frequency_dependent_chirp_bounds(self.blocks(), GLOBAL)
        self.assertEqual(bounds[0, 1, 1], LEGACY_LOW_FREQUENCY_FLOOR_HZ_PER_S)
        self.assertEqual(bounds[0, 0, 0], -LEGACY_LOW_FREQUENCY_FLOOR_HZ_PER_S)
        np.testing.assert_array_equal(bounds[:, 0, 1], -GAP)
        np.testing.assert_array_equal(bounds[:, 1, 0], GAP)

    def test_caps_are_monotone_symmetric_and_never_expand_global_support(self):
        bounds = frequency_dependent_chirp_bounds(self.blocks(), GLOBAL)
        cap = bounds[:, 1, 1]
        self.assertTrue((np.diff(cap) >= 0.).all())
        self.assertTrue((cap <= GLOBAL[1, 1]).all())
        np.testing.assert_array_equal(bounds[:, 0, 0], -cap)
        self.assertEqual(cap[-1], GLOBAL[1, 1])
        self.assertEqual(cap[-2], GLOBAL[1, 1])

    def test_upper_edge_envelope_covers_every_frequency_inside_the_bank(self):
        blocks = self.blocks()
        bounds = frequency_dependent_chirp_bounds(blocks, GLOBAL)
        for block, local in zip(blocks, bounds):
            for f in np.linspace(*block, 17):
                envelope = max(LEGACY_LOW_FREQUENCY_FLOOR_HZ_PER_S,
                    GR_POSITIVE_COEFFICIENT*f**(11./3.), 5.e-6*f**(13./3.))
                self.assertGreaterEqual(local[1, 1], min(GLOBAL[1, 1], envelope))

    def test_prior_and_padding_arrays_do_not_change_ownership_edges(self):
        blocks = self.blocks()
        original = blocks.copy()
        bounds = frequency_dependent_chirp_bounds(blocks, GLOBAL)
        padding = frequency_dependent_padding(T, 1.e-5, blocks, bounds)
        np.testing.assert_array_equal(blocks, original)
        with self.assertRaises(ValueError):
            bounds[0, 0, 0] = 0.
        with self.assertRaises(ValueError):
            padding[0] = 0.

    def test_padding_contains_declared_both_sign_drifts_and_doppler_cross_term(self):
        blocks = self.blocks()
        bounds = frequency_dependent_chirp_bounds(blocks, GLOBAL)
        pad = frequency_dependent_padding(T, 1.e-5, blocks, bounds)
        required = bounds[:, 1, 1]*T + 1.e-4*(blocks[:, 1]+bounds[:, 1, 1]*T)+8./T
        np.testing.assert_array_equal(pad, np.maximum(1.e-5, required))
        np.testing.assert_array_equal(pad[:3], 1.e-5)
        self.assertLess(pad[3], 3.e-5)

    def test_gap_endpoints_and_zero_are_not_moved_by_policy(self):
        bounds = frequency_dependent_chirp_bounds(self.blocks(), GLOBAL)
        for local in bounds:
            self.assertEqual(local[0, 1], -GAP)
            self.assertEqual(local[1, 0], GAP)
            self.assertLess(local[0, 0], -GAP)
            self.assertGreater(local[1, 1], GAP)

    def test_all_global_indices_and_final_endpoint_are_retained(self):
        blocks = np.arange(.0001, .05-1.e-15, 1.e-5)[:, None] + [0., 1.e-5]
        blocks[-1, 1] = .05
        bounds = frequency_dependent_chirp_bounds(blocks, GLOBAL)
        self.assertEqual(len(bounds), 4990)
        np.testing.assert_array_equal(chirp_bounds_for_bank(bounds, 4989, 4990), bounds[-1])
        self.assertEqual(blocks[-1, 1], .05)

    def test_invalid_blocks_caps_and_zero_gap_fail_closed(self):
        for blocks in ([], [[0., .1]], [[.1, .1]], [[.2, .1]], [[.1, np.nan]]):
            with self.subTest(blocks=blocks), self.assertRaises(ValueError):
                frequency_dependent_chirp_bounds(blocks, GLOBAL)
        for bounds in ([1.e-20, 4.e-12], [[-4.e-12, 0.], [0., 4.e-12]],
                       [[-4.e-12, -GAP], [GAP, 3.e-12]]):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                frequency_dependent_chirp_bounds(self.blocks(), bounds)

    def test_padding_shape_and_nonfinite_values_fail_closed(self):
        blocks = self.blocks()
        bounds = frequency_dependent_chirp_bounds(blocks, GLOBAL)
        for duration in (0., np.inf, np.nan):
            with self.assertRaises(ValueError):
                frequency_dependent_padding(duration, 1.e-5, blocks, bounds)
        with self.assertRaises(ValueError):
            frequency_dependent_padding(T, 1.e-5, blocks, GLOBAL)

    def test_global_index_selection_not_selected_bank_position(self):
        bounds = frequency_dependent_chirp_bounds(self.blocks(), GLOBAL)
        for i in (5, 1, 0):
            np.testing.assert_array_equal(chirp_bounds_for_bank(bounds, i, 6), bounds[i])
        for i in (-1, 6):
            with self.assertRaises(ValueError):
                chirp_bounds_for_bank(bounds, i, 6)
        with self.assertRaises(ValueError):
            chirp_bounds_for_bank(bounds[:2], 1, 6)

    def test_scalar_legacy_padding_and_global_vectors_are_unambiguous(self):
        np.testing.assert_array_equal(bank_padding_values(None, .1, 3), [.1]*3)
        np.testing.assert_array_equal(bank_padding_values(.2, .1, 3), [.2]*3)
        np.testing.assert_array_equal(bank_padding_values([.1,.2,.3], .1, 3), [.1,.2,.3])
        for values in ([.1,.2], [.1,np.nan,.2], [.1,.01,.2]):
            with self.assertRaises(ValueError):
                bank_padding_values(values, .1, 3)

    def test_policy_hash_is_deterministic_and_sensitive_to_all_declared_support(self):
        blocks = self.blocks()
        bounds = frequency_dependent_chirp_bounds(blocks, GLOBAL)
        pads = frequency_dependent_padding(T, 1.e-5, blocks, bounds)
        first = bank_configuration_record(T, 1.e-5, blocks, GLOBAL, bounds, pads)
        second = bank_configuration_record(T, 1.e-5, blocks.copy(), GLOBAL.copy(),
                                            bounds.copy(), pads.copy())
        self.assertEqual(first, second)
        json.dumps(first, allow_nan=False)
        changed = pads.copy(); changed[-1] *= 1.01
        other = bank_configuration_record(T, 1.e-5, blocks, GLOBAL, bounds, changed)
        self.assertNotEqual(first['policy_sha256'], other['policy_sha256'])
        self.assertFalse(first['policy']['catalogue_truth_used'])

    def test_policy_has_no_catalogue_residual_or_recovery_arguments(self):
        tree = ast.parse((ROOT/'script/dwd_chirp_policy.py').read_text())
        funcs = {x.name:x for x in tree.body if isinstance(x, ast.FunctionDef)}
        self.assertEqual([x.arg for x in funcs['frequency_dependent_chirp_bounds'].args.args],
                         ['frequency_blocks','global_bounds'])
        self.assertEqual([x.arg for x in funcs['frequency_dependent_padding'].args.args],
                         ['duration','block_width','frequency_blocks','bank_bounds'])

    def test_actual_controller_passes_global_indexed_support_everywhere(self):
        source = (ROOT/'eric_subtraction_all.py').read_text()
        self.assertIn('frequency_derivative_bounds_by_block = frequency_dependent_chirp_bounds(', source)
        self.assertIn('frequency_derivative_bounds = chirp_bounds_for_bank(', source)
        self.assertIn('chirp_bounds_by_block=frequency_derivative_bounds_by_block)', source)
        self.assertIn("output_file.parent / 'frequency_banks.settings.json'", source)
        self.assertIn('frequency_derivative_bounds_by_block, sample_number, maximum_sources,', source)
        self.assertIn("raise ValueError(f'Reference/checkpoint mismatch: block {block_index}')", source)


class PerBankReferenceCacheTests(unittest.TestCase):
    def test_cold_warm_subset_and_policy_identity_use_exact_global_bank_bins(self):
        from test_reference_waveform_cache import Response, coefficients, reconstruct
        from script.dwd_reference_cache import precompute_reference_waveforms
        Response.calls = []
        Response.files_open = 0
        blocks = np.array([[.1,.2],[.2,.3],[.3,.4]])
        frequency = np.arange(0., .61, .01)
        row = np.array([[91.,1.e-22,.15,1.e-17,.4,.7,.2,-.1,.3,20.,8.]])
        pads = np.array([.1,.12,.15])
        bounds = np.broadcast_to(GLOBAL, (3,2,2)).copy()
        def prepare(path, local_bounds=bounds):
            return precompute_reference_waveforms(path, np.array([2,0]),
                [row.copy(),row.copy()], blocks, .1, frequency, 100., 1.,
                np.array([0.,1.]), np.array([2.,3.]), Response, coefficients, reconstruct,
                frequency_padding_hz=pads, chirp_bounds_by_block=local_bounds)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'new_fixture_only.h5'
            cold = prepare(path)
            for i in (2,0):
                expected = np.flatnonzero((frequency>=blocks[i,0]-pads[i]) &
                    (frequency<=blocks[i,1]+pads[i]) & (frequency>0))
                np.testing.assert_array_equal(cold[i]['retained_bins'], expected)
            calls = len(Response.calls)
            with patch.object(Response,'__init__',side_effect=AssertionError('Cache hit launched GPU')):
                warm = prepare(path)
            self.assertEqual(len(Response.calls), calls)
            np.testing.assert_array_equal(cold[2]['waveforms'], warm[2]['waveforms'])
            changed = bounds.copy(); changed[0,1,1] *= .9
            prepare(path, changed)
            self.assertEqual(len(Response.calls), calls+2)

    def test_invalid_selected_only_padding_is_rejected_before_any_cache_open(self):
        from script.dwd_reference_cache import bank_padding_values
        with self.assertRaises(ValueError):
            bank_padding_values([.1,.2], .1, 3)


if __name__ == '__main__':
    unittest.main()

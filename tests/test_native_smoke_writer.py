"""CPU-only actual-writer regressions using new temporary synthetic HDF files."""
import json
from pathlib import Path
import tempfile
import unittest
import h5py
import numpy as np
from script.tool import _save_tree
from script.dwd_amplitude_refit import joint_amplitude_only_refit
from native_smoke_audit import json_attribute, checked_winner_refit, strict_native_quality_passed


class NativeSmokeWriterTests(unittest.TestCase):
    def test_actual_writer_stores_strings_as_json_attributes_not_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'own_synthetic_audit.h5'
            with h5py.File(path, 'w') as saved:
                _save_tree(saved.create_group('audit'), dict(
                    prior_penalty='none_in_post_identification_refit', numeric=np.array([1., 2.])))
            with h5py.File(path, 'r') as saved:
                audit = saved['audit']
                self.assertNotIn('prior_penalty', audit)
                self.assertIn('prior_penalty', audit.attrs)
                self.assertEqual(audit.attrs['prior_penalty'], json.dumps('none_in_post_identification_refit'))
                self.assertEqual(json_attribute(audit, 'prior_penalty'), 'none_in_post_identification_refit')
                np.testing.assert_array_equal(audit['numeric'][:], [1., 2.])

    def test_real_winner_refit_audit_roundtrips_through_real_writer(self):
        rng = np.random.default_rng(20261019)
        sources = rng.normal(size=(3, 3, 32))+1j*rng.normal(size=(3, 3, 32))
        sources[1] = .7*sources[0]  # Duplicate shape must remain excluded.
        data = 2.*sources[0]+.5*sources[2]
        covariance = np.broadcast_to(np.eye(3), (32, 3, 3)).copy()
        association = dict(matched_dwd_id=np.array([41, -1, 42]), best_template_id=np.array([41, 41, 42]),
            best_overlap=np.array([.99, .98, .97]), minimum_overlap=.9)
        scales, audit = joint_amplitude_only_refit(data, sources, association, covariance, 4., 1.,
            inner_product=lambda first, second, *args: float(np.vdot(first, second).real))
        np.testing.assert_allclose(scales, [2., 1., .5], atol=1.e-11)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'own_real_refit_audit.h5'
            with h5py.File(path, 'w') as saved:
                _save_tree(saved.create_group('audit'), audit)
            with h5py.File(path, 'r') as saved:
                result = checked_winner_refit(saved['audit'])
                self.assertEqual(result['selected_winners'], 2)
                self.assertEqual(result['excluded_components'], 1)
                self.assertEqual(result['serialization'], 'actual_writer_json_attributes')

    def test_dataset_only_string_is_not_a_silent_schema_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'own_wrong_schema.h5'
            with h5py.File(path, 'w') as saved:
                saved.create_dataset('prior_penalty', data='none_in_post_identification_refit')
            with h5py.File(path, 'r') as saved:
                with self.assertRaisesRegex(AssertionError, 'Missing actual-writer'):
                    json_attribute(saved, 'prior_penalty')

    def test_strict_native_gate_rejects_old_failure_boundary_and_nonfinite_values(self):
        def rows(value):
            return [dict(native_foreground_power_removed_percent=value)]
        for value in (94.06140412376406, 95., np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                self.assertFalse(strict_native_quality_passed(rows(value)))
        self.assertTrue(strict_native_quality_passed(rows(np.nextafter(95., np.inf))))
        self.assertFalse(strict_native_quality_passed([]))


if __name__ == '__main__':
    unittest.main()

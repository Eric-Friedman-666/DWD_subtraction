"""Strict readers for the actual compact-writer native-smoke audit schema.

The real script.tool._save_tree stores Unicode/object values as JSON-encoded
HDF attributes, not datasets. These helpers read only a test's own CLOSED
synthetic outputs; they are never a production-monitoring fallback.
"""
import json
import math
import numpy as np


def json_attribute(group, name):
    if name not in group.attrs:
        raise AssertionError(f'Missing actual-writer JSON attribute: {name}')
    value = group.attrs[name]
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    if not isinstance(value, str):
        raise AssertionError(f'Expected JSON text attribute: {name}')
    return json.loads(value)


def checked_winner_refit(audit):
    assert json_attribute(audit, 'method') == 'joint_nonnegative_winner_only_fixed_shape_scale'
    assert json_attribute(audit, 'prior_penalty') == 'none_in_post_identification_refit'
    selected = audit['selected_component_indices'][:]
    fitted = audit['fitted_component_indices'][:]
    excluded = audit['excluded_component_indices'][:]
    scales = audit['scale_factors'][:]
    matched = audit['matched_dwd_id'][:]
    np.testing.assert_array_equal(selected, fitted)
    np.testing.assert_array_equal(audit['prior_precision_diagonal'][:], 0.)
    assert len(audit['nuisance_component_indices']) == 0
    assert len(np.unique(matched)) == len(selected) == int(audit['fit_source_count'][()])
    assert bool(audit['solved'][()])
    assert np.isfinite(scales).all() and np.all(scales[selected] >= 0.)
    np.testing.assert_array_equal(scales[excluded], 1.)
    before, after = float(audit['objective_before'][()]), float(audit['objective_after'][()])
    kkt = float(audit['relative_kkt_residual'][()])
    assert math.isfinite(before) and math.isfinite(after) and math.isfinite(kkt)
    assert after <= before+1.e-8
    assert kkt <= 1.e-8
    return dict(bank=audit.name, selected_winners=len(selected),
        excluded_components=len(excluded), objective_before=before,
        objective_after=after, relative_kkt_residual=kkt,
        prior_penalty='none_in_post_identification_refit',
        serialization='actual_writer_json_attributes')


def strict_native_quality_passed(rows):
    # Exactly the retained strictly-greater-than-95 gate, never >= or a waiver.
    return bool(rows) and all(
        math.isfinite(float(row['native_foreground_power_removed_percent']))
        and float(row['native_foreground_power_removed_percent']) > 95.
        for row in rows)

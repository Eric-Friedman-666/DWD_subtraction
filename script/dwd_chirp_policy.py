"""Truth-free per-ownership-bank signed chirp prior and likelihood support.

The positive envelope is independently derived from detached, GW-driven point
masses with m1=m2=1.4 solar masses, NOT transcribed from the anomalous positive
coefficient printed in arXiv:2004.08464v1 section IV.B.  Its empirical negative
envelope (5e-6*f**(13/3)) is included conservatively and then symmetrized.
Neither law is a universal bound for interacting or accelerated binaries.
The old positive ceiling is retained as a symmetric low-frequency floor.

This module consumes only declared settings and fixed ownership edges, never
catalogue true derivatives, residuals, SNR, fitted amplitudes or recovered IDs.
"""
import hashlib
import json
import math

import numpy as np

from script.dwd_frequency_support import RESPONSE_STEP_SECONDS

POLICY_NAME = 'signed_log_magnitude_bank_gr_negative_envelope_floor_v1'
GLOBAL_CHIRP_CAP_HZ_PER_S = 4.e-12
LEGACY_LOW_FREQUENCY_FLOOR_HZ_PER_S = 1.584893192461111e-15
MAX_COMPONENT_MASS_SOLAR = 1.4
GRAVITATIONAL_CONSTANT_SI = 6.67430e-11
LIGHT_SPEED_M_PER_S = 299792458.
SOLAR_MASS_KG = 1.988409870698051e30
MAX_DETACHED_CHIRP_MASS_SOLAR = MAX_COMPONENT_MASS_SOLAR / 2.**.2
GR_POSITIVE_COEFFICIENT = (96./5. * math.pi**(8./3.) *
    (GRAVITATIONAL_CONSTANT_SI * SOLAR_MASS_KG *
     MAX_DETACHED_CHIRP_MASS_SOLAR / LIGHT_SPEED_M_PER_S**3)**(5./3.))
EMPIRICAL_NEGATIVE_COEFFICIENT = 5.e-6


def chirp_policy_metadata():
    return dict(name=POLICY_NAME, coordinate='signed_log_uniform_magnitude',
        sign_mass='equal_for_symmetric_support', ownership_edge='bank_upper',
        low_frequency_floor_hz_per_s=LEGACY_LOW_FREQUENCY_FLOOR_HZ_PER_S,
        maximum_component_mass_solar=MAX_COMPONENT_MASS_SOLAR,
        maximum_detached_chirp_mass_solar=MAX_DETACHED_CHIRP_MASS_SOLAR,
        gr_positive_coefficient=GR_POSITIVE_COEFFICIENT,
        gr_frequency_exponent=11./3.,
        empirical_negative_coefficient=EMPIRICAL_NEGATIVE_COEFFICIENT,
        empirical_negative_frequency_exponent=13./3.,
        negative_support='mirrored_max_of_both_envelopes_and_old_floor',
        global_ceiling_policy='never_expand_supplied_global_signed_bounds',
        doppler_fraction=1.e-4, doppler_frequency='bank_upper_plus_maximum_drift',
        fourier_guard_bins=8.,
        physical_assumptions='detached_GR_mass_envelope_and_empirical_negative_envelope_not_universal',
        catalogue_truth_used=False,
        reference='https://arxiv.org/abs/2004.08464')


def _frequency_blocks(values):
    blocks = np.asarray(values, dtype=float)
    return blocks


def frequency_dependent_chirp_bounds(frequency_blocks, global_bounds):
    """Return immutable [global_bank_index, sign, endpoint] signed support.

    Each bank uses its upper frequency edge so the bound never tightens within
    that ownership bank.  A fixed sign gap and the existing global cap remain.
    """
    blocks = _frequency_blocks(frequency_blocks)
    bounds = np.asarray(global_bounds, dtype=float)
    cap, minimum = -float(bounds[0, 0]), float(bounds[1, 0])
    upper_frequency = blocks[:, 1]
    envelope = np.maximum.reduce((
        np.full(len(blocks), LEGACY_LOW_FREQUENCY_FLOOR_HZ_PER_S),
        GR_POSITIVE_COEFFICIENT * upper_frequency**(11./3.),
        EMPIRICAL_NEGATIVE_COEFFICIENT * upper_frequency**(13./3.)))
    local_cap = np.minimum(cap, envelope)
    result = np.empty((len(blocks), 2, 2), dtype=float)
    result[:, 0, 0], result[:, 0, 1] = -local_cap, -minimum
    result[:, 1, 0], result[:, 1, 1] = minimum, local_cap
    result.setflags(write=False)
    return result


def frequency_dependent_padding(duration, block_width, frequency_blocks, bank_bounds):
    """Apply the declared local prior's drift/Doppler/8-bin guard per bank."""
    duration, block_width = float(duration), float(block_width)
    blocks = _frequency_blocks(frequency_blocks)
    bounds = np.asarray(bank_bounds, dtype=float)
    maximum_derivative = np.max(np.abs(bounds), axis=(1, 2))
    maximum_drift = maximum_derivative * duration
    padding = np.maximum(block_width,
        maximum_drift + 1.e-4 * (blocks[:, 1] + maximum_drift) + 8. / duration)
    padding.setflags(write=False)
    return padding


def chirp_bounds_for_bank(values, block_index, block_count):
    """Select by GLOBAL index; retain legacy scalar-bank support in test callers."""
    values = np.asarray(values, dtype=float)
    index = int(block_index)
    if values.shape in ((2,), (2, 2)):
        return values
    return values[index]


def bank_padding_values(values, block_width, block_count):
    """Normalize scalar legacy padding or a GLOBAL-indexed per-bank vector."""
    width, count = float(block_width), int(block_count)
    values = np.asarray(width if values is None else values, dtype=float)
    if values.ndim == 0:
        values = np.full(count, float(values))
    return values


def bank_configuration_record(duration, block_width, frequency_blocks,
                              global_bounds, bank_bounds, padding):
    """Small atomic-JSON-compatible audit, independent of live HDF5 inspection."""
    blocks = _frequency_blocks(frequency_blocks)
    bounds = np.asarray(bank_bounds, dtype=float)
    pads = bank_padding_values(padding, block_width, len(blocks))
    record = dict(schema_version=1, policy=chirp_policy_metadata(),
        duration_seconds=float(duration), response_step_seconds=float(RESPONSE_STEP_SECONDS),
        ownership_block_width_hz=float(block_width),
        global_derivative_bounds_hz_per_s=np.asarray(global_bounds, dtype=float).tolist(),
        frequency_blocks_hz=blocks.tolist(),
        derivative_bounds_hz_per_s_by_block=bounds.tolist(),
        likelihood_frequency_padding_hz_by_block=pads.tolist())
    encoded = json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    record['policy_sha256'] = hashlib.sha256(encoded).hexdigest()
    return record

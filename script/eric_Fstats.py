"""Fixed-dimensional multi-channel F-statistic utilities for DWDs.

The routines in this module follow the raw, unnormalised DFT convention used
by :mod:`script.spectrum` and retain the complete correlated X2/Y2/Z2 noise
covariance.  They deliberately contain no source-count logic; an RJMCMC layer
can use the same likelihood primitives later.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

import lisagwresponse
import numpy as np
from scipy.optimize import least_squares

from script.eric_preprocess import inner_product
from script.native_lisa_filter import native_signal_filter
from script.spectrum import dwd_xyz_response, dwd_xyz_response_fd, prepare_xyz_geometry


@dataclass(frozen=True)
class FStatisticResult:
    """Analytically maximised amplitude result for one or more templates."""

    two_f: np.ndarray
    coefficients: np.ndarray
    matrix: np.ndarray


# 1. Four amplitude coefficients
def physical_amplitude_coordinates(A_plus, A_cross, psi, psi_0):
    A_plus = np.asarray(A_plus, dtype=np.float64)
    A_cross = np.asarray(A_cross, dtype=np.float64)
    psi = np.asarray(psi, dtype=np.float64)
    psi_0 = np.asarray(psi_0, dtype=np.float64)

    cpsi = np.cos(2 * psi)
    spsi = np.sin(2 * psi)
    cphase = np.cos(psi_0)
    sphase = np.sin(psi_0)

    return np.stack([
        A_plus*cpsi*cphase + A_cross*spsi*sphase,
        A_plus*spsi*cphase - A_cross*cpsi*sphase,
        -A_plus*cpsi*sphase + A_cross*spsi*cphase,
        -A_plus*spsi*sphase - A_cross*cpsi*cphase,
    ], axis=-1)


# 2. Four basis waveforms
def amplitude_basis_td(a_I, b_I, Phi_0):
    cos_phase = np.cos(Phi_0)
    sin_phase = np.sin(Phi_0)

    return np.stack([
        a_I * cos_phase,
        b_I * cos_phase,
        a_I * sin_phase,
        b_I * sin_phase,
    ], axis=-3)


def amplitude_basis_fd(a_cos, b_cos, *, xp=np):
    # The existing analytic FD response already includes the cosine phase.
    return xp.stack([
        a_cos,
        b_cos,
        1j * a_cos,
        1j * b_cos,
    ], axis=-3)


# 3. Combine amplitudes and bases, keeping each DWD separate
def reconstruct_waveform(A_mu, h_mu):
    return np.einsum('...m,...mct->...ct', np.asarray(A_mu), np.asarray(h_mu))


def DWD_sum(source_bases, coefficients):
    # Sum all DWDs into one XYZ waveform.
    return np.einsum(
        "sm,smcf->cf", np.asarray(coefficients), np.asarray(source_bases), optimize=True
    )


def from_A_to_DWD_para(coordinates):
    """Return one physical ``(A, iota, psi, phi0)`` representative.

    The map has the familiar discrete DWD amplitude degeneracies.  A small
    deterministic multistart least-squares solve therefore returns one
    equivalent representative rather than claiming a unique inverse.
    """

    target = np.asarray(coordinates, dtype=np.float64).reshape(4)
    scale = max(float(np.linalg.norm(target)), np.finfo(float).tiny)

    def residual(values):
        log_amplitude, cosine_inclination, polarization, phase = values
        amplitude = np.exp(log_amplitude)
        inclination = np.arccos(np.clip(cosine_inclination, -1.0, 1.0))
        model = physical_amplitude_coordinates(
            amplitude * (1.0 + np.cos(inclination) ** 2),
            2.0 * amplitude * np.cos(inclination),
            polarization,
            phase,
        )
        return (model - target) / scale

    amplitude_guess = max(scale / 2.0, np.finfo(float).tiny)
    bounds = (
        np.array([np.log(amplitude_guess) - 12.0, -1.0, 0.0, 0.0]),
        np.array([np.log(amplitude_guess) + 12.0, 1.0, np.pi, 2.0 * np.pi]),
    )
    best = None
    for cosine_inclination in (-0.8, 0.0, 0.8):
        for polarization in (0.0, 0.25 * np.pi, 0.5 * np.pi):
            for phase in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi):
                start = np.array(
                    [
                        np.log(amplitude_guess),
                        cosine_inclination,
                        min(polarization, np.nextafter(np.pi, 0.0)),
                        min(phase, np.nextafter(2.0 * np.pi, 0.0)),
                    ]
                )
                fit = least_squares(
                    residual,
                    start,
                    bounds=bounds,
                    xtol=1.0e-13,
                    ftol=1.0e-13,
                    gtol=1.0e-13,
                    max_nfev=2000,
                )
                if best is None or np.linalg.norm(fit.fun) < np.linalg.norm(best.fun):
                    best = fit
    log_amplitude, cosine_inclination, polarization, phase = best.x
    return {
        "A": float(np.exp(log_amplitude)),
        "iota": float(np.arccos(np.clip(cosine_inclination, -1.0, 1.0))),
        "psi_bme": float(np.mod(polarization, np.pi)),
        "phi0": float(np.mod(phase, 2.0 * np.pi)),
        "relative_coordinate_residual": float(np.linalg.norm(best.fun)),
    }


def dwd_amplitude_basis_fd(
    doppler_parameters,
    sample_times,
    orbit_model,
    *,
    carrier_frequency,
    prepared_geometry,
    signal_filter_amplitude=None,
    amplitude_scale=1.0e-22,
):
    """Generate the four Method-1 XYZ basis responses for trial points.

    Parameters in ``doppler_parameters`` may be scalars or equally sized
    arrays and use the LDC fields ``f``, ``df``, ``longitude`` and
    ``latitude``.  The result has shape ``(trial, 4, 3, Fourier-bin)``.

    Two direct LISA-package responses at edge-on reference orientations span
    the cosine quadratures.  The sine quadratures are exactly ``1j`` times
    those responses because the existing analytic response uses
    ``exp(-1j * phi0)`` and the Method-1 coordinates carry ``-sin(phi0)``.
    This halves the expensive moving-orbit response calls while retaining
    every response, coordinate, phase and polarization convention already
    validated in :func:`script.spectrum.dwd_xyz_response_fd`.
    """

    names = ("f", "df", "longitude", "latitude")
    values = {
        name: np.atleast_1d(
            np.asarray(doppler_parameters[name], dtype=np.float64)
        )
        for name in names
    }
    trial_count = max(len(value) for value in values.values())
    values = {
        name: np.broadcast_to(value, (trial_count,))
        for name, value in values.items()
    }
    reference_polarization = np.array(
        [0.0, 0.25 * np.pi], dtype=np.float64
    )
    response_parameters = {
        name: np.repeat(value, 2) for name, value in values.items()
    }
    response_parameters.update(
        {
            "A": np.full(2 * trial_count, amplitude_scale),
            "iota": np.full(2 * trial_count, 0.5 * np.pi),
            "psi_bme": np.tile(reference_polarization, trial_count),
            "phi0": np.zeros(2 * trial_count, dtype=np.float64),
        }
    )
    carrier = np.broadcast_to(
        np.atleast_1d(np.asarray(carrier_frequency, dtype=np.float64)),
        (trial_count,),
    )
    response = dwd_xyz_response_fd(
        response_parameters,
        sample_times,
        orbit_model,
        carrier_frequency=np.repeat(carrier, 2),
        prepared_geometry=prepared_geometry,
    )
    cosine_response = response.transpose(1, 0, 2).reshape(
        trial_count, 2, 3, response.shape[-1]
    )
    response = amplitude_basis_fd(cosine_response[:, 0], cosine_response[:, 1])
    if signal_filter_amplitude is not None:
        response *= np.asarray(signal_filter_amplitude)[None, None, None, :]
    return response


def generate_correlated_noise_fd(covariance, duration, sampling_rate, rng):
    """Draw proper-complex XYZ raw-DFT noise with the requested one-sided CSD."""

    covariance = np.asarray(covariance, dtype=np.complex128)
    covariance = 0.5 * (
        covariance + np.conjugate(np.swapaxes(covariance, 1, 2))
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    square_root = np.einsum(
        "fij,fj,fkj->fik",
        eigenvectors,
        np.sqrt(np.maximum(eigenvalues, 0.0)),
        np.conjugate(eigenvectors),
        optimize=True,
    )
    standard = (
        rng.standard_normal((len(covariance), 3))
        + 1.0j * rng.standard_normal((len(covariance), 3))
    ) / np.sqrt(2.0)
    scale = np.sqrt(duration * sampling_rate**2 / 2.0)
    return (
        scale * np.einsum("fij,fj->fi", square_root, standard, optimize=True)
    ).T


def complex_noise_overlap(a, b, inverse_covariance, duration, sampling_rate):
    """Return the complex full-XYZ overlap before taking its real part."""

    a = np.asarray(a)
    b = np.asarray(b)
    inverse_covariance = np.asarray(inverse_covariance)
    contraction = np.einsum(
        "...cf,fcd,...df->...",
        np.conjugate(a),
        inverse_covariance,
        b,
        optimize=True,
    )
    return 4.0 * contraction / (duration * sampling_rate**2)


def noise_inner_product(a, b, inverse_covariance, duration, sampling_rate):
    """Return the real Method-1 full-XYZ noise-weighted inner product."""

    return np.real(
        complex_noise_overlap(a, b, inverse_covariance, duration, sampling_rate)
    )


def waveform_match(a, b, inverse_covariance, duration, sampling_rate, phase_max=True):
    """Return the normalized full-covariance waveform match."""

    aa = noise_inner_product(a, a, inverse_covariance, duration, sampling_rate)
    bb = noise_inner_product(b, b, inverse_covariance, duration, sampling_rate)
    overlap = complex_noise_overlap(
        a, b, inverse_covariance, duration, sampling_rate
    )
    numerator = np.abs(overlap) if phase_max else np.real(overlap)
    denominator = np.sqrt(np.maximum(aa * bb, np.finfo(float).tiny))
    return np.asarray(numerator / denominator)


def f_statistic(data, bases, noise_covariance, duration, sampling_rate):
    """Evaluate one or a batch of four-amplitude F-statistics."""

    data = np.asarray(data)
    bases = np.asarray(bases)
    noise_covariance = np.asarray(noise_covariance)
    if bases.ndim == 3:
        bases = bases[None, ...]
    x = np.asarray([
        [inner_product(h, data, noise_covariance, duration, sampling_rate)
         for h in candidate]
        for candidate in bases
    ]).reshape(len(bases), 4)
    M = np.asarray([
        [[inner_product(a, b, noise_covariance, duration, sampling_rate)
          for b in candidate] for a in candidate]
        for candidate in bases
    ]).reshape(len(bases), 4, 4)
    M_inv = np.linalg.inv(M)
    A = np.einsum("...ij,...j->...i", M_inv, x)
    two_f = np.maximum(np.einsum("...i,...i->...", x, A), 0.0)
    return FStatisticResult(two_f, A, M)


def physical_amplitudes(coefficients):
    a1,a2,a3,a4 = np.moveaxis(np.asarray(coefficients)*1e-22,-1,0)
    plus,minus = a1+1j*a3+1j*(a2+1j*a4),a1+1j*a3-1j*(a2+1j*a4)
    p,m = np.sqrt(abs(plus)),np.sqrt(abs(minus))
    denominator = np.maximum(p+m,np.finfo(float).tiny)
    return dict(A=.25*(p+m)**2,iota=np.arccos(np.clip((p-m)/denominator,-1,1)),
                psi_bme=(.25*(np.angle(plus)-np.angle(minus)))%np.pi,
                phi0=(-.5*(np.angle(plus)+np.angle(minus)))%(2*np.pi))


def build_response(output,wide_frequency,duration,rate,*,orbit_file=None):
    model = lisagwresponse.GalacticBinary(A=1.,f=.001,ra=0.,dec=0.,orbits=str(output/'orbit.h5' if orbit_file is None else orbit_file),
                                         orbit_interp_order=5,shift_sun='never')
    times = np.arange(int(round(duration/3600)))*3600+.5*(int(round(3600*rate))-1)/rate
    geometry = prepare_xyz_geometry(times,model)
    # This instance builds waveforms only. Each posterior uses its own measured covariance.
    builder = import_module('script.eric_generate_waveform').GPUResponse(np.broadcast_to(np.eye(3),(len(wide_frequency),3,3)),
                          np.zeros((3,len(wide_frequency)),complex),times,geometry,wide_frequency,duration,rate)
    return builder,times,geometry


def select_f_statistic_candidates(two_f, threshold=64.0):
    """Apply the fixed F threshold to code values stored as 2F."""
    two_f = np.asarray(two_f, dtype=float)
    two_f_threshold = 2.0*float(threshold)
    return np.flatnonzero(two_f >= two_f_threshold), two_f_threshold


def response_to_frequency_domain(envelope, source_frequency, *, physical_frequency,
                                 detector_sampling_rate, observation_duration,
                                 detector_sample_interval, response_block_count,
                                 response_times):
    signal_filter_amplitude, _ = native_signal_filter(physical_frequency, detector_sampling_rate)
    envelope = np.asarray(envelope)
    source_frequency = np.asarray(source_frequency, dtype=np.float64)
    transformed = np.zeros(
        (len(source_frequency), 3, len(physical_frequency)),
        dtype=np.complex128,
    )
    for source_index, frequency_value in enumerate(source_frequency):
        frequency_offset = physical_frequency - frequency_value
        carrier_bin = int(round(frequency_value * observation_duration))
        fractional_bin = frequency_value * observation_duration - carrier_bin
        native_bins = np.rint(physical_frequency * observation_duration).astype(np.int64)
        block_factor = (
            int(round(3600.0 * detector_sampling_rate))
            * np.sinc(
                int(round(3600.0 * detector_sampling_rate))
                * frequency_offset
                * detector_sample_interval
            )
            / np.sinc(frequency_offset * detector_sample_interval)
        )
        transformed_envelope = np.fft.fft(
            envelope[:, source_index]
            * np.exp(2.0j * np.pi * fractional_bin
                     * np.arange(response_block_count) / response_block_count)[None, :],
            axis=-1,
        )
        transformed[source_index] = (
            0.5
            * transformed_envelope[:, (native_bins - carrier_bin) % response_block_count]
            * np.exp(-2.0j * np.pi * response_times[0] * frequency_offset)[None, :]
            * block_factor[None, :]
            * signal_filter_amplitude[None, :]
        )
    return transformed


def physical_parameters_from_coordinates(coordinates):
    """Invert the four fitted amplitudes without an injected-parameter guess."""
    a1, a2, a3, a4 = np.asarray(coordinates, dtype=np.float64)
    plus_helicity = a1 + 1.0j * a3 + 1.0j * (a2 + 1.0j * a4)
    minus_helicity = a1 + 1.0j * a3 - 1.0j * (a2 + 1.0j * a4)
    plus_root = np.sqrt(abs(plus_helicity))
    minus_root = np.sqrt(abs(minus_helicity))
    root_sum = max(plus_root + minus_root, np.finfo(float).tiny)
    return {
        "A": float(0.25 * root_sum**2),
        "iota": float(np.arccos(np.clip((plus_root - minus_root) / root_sum, -1.0, 1.0))),
        "psi_bme": float(np.mod(0.25 * (np.angle(plus_helicity) - np.angle(minus_helicity)), np.pi)),
        "phi0": float(np.mod(-0.5 * (np.angle(plus_helicity) + np.angle(minus_helicity)), 2.0 * np.pi)),
    }


def build_amplitude_bases(doppler_parameters, *, response_times, orbit_model,
                          response_geometry, physical_frequency,
                          response_to_frequency_domain):
    doppler_parameters = np.asarray(doppler_parameters, dtype=np.float64)
    local_source_count = len(doppler_parameters)
    reference_parameters = {
        "f": np.repeat(doppler_parameters[:, 0], 2),
        "df": np.repeat(doppler_parameters[:, 1], 2),
        "longitude": np.repeat(doppler_parameters[:, 2], 2),
        "latitude": np.repeat(doppler_parameters[:, 3], 2),
        "A": np.full(2 * local_source_count, 1.0e-22),
        "iota": np.full(2 * local_source_count, 0.5 * np.pi),
        "psi_bme": np.tile(
            np.array([0.0, 0.25 * np.pi]), local_source_count
        ),
        "phi0": np.zeros(2 * local_source_count),
    }
    reference_frequency = np.repeat(doppler_parameters[:, 0], 2)
    envelope = dwd_xyz_response(
        reference_parameters,
        response_times,
        orbit_model,
        carrier_frequency=reference_frequency,
        prepared_geometry=response_geometry,
    )
    cosine_bases = response_to_frequency_domain(
        envelope, reference_frequency
    ).reshape(local_source_count,2,3,len(physical_frequency),)
    return amplitude_basis_fd(cosine_bases[:, 0], cosine_bases[:, 1])


def candidate_amplitude_bases(doppler_parameters, *, build_amplitude_bases,
                              physical_frequency):
    points = np.asarray(doppler_parameters).reshape(-1, 4)
    if not len(points):
        return np.empty((0, 4, 3, len(physical_frequency)), dtype=complex)
    return build_amplitude_bases(points)


def gpu_amplitude_bases_fd(doppler_parameters, *, builder, core_indices):
    return builder.build(doppler_parameters)[:,:,:,core_indices]

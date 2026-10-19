from __future__ import annotations
from functools import partial
import csv
import json
import os
from pathlib import Path
for thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_variable, "1")
import h5py
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import time
from tempfile import TemporaryDirectory
import lisagwresponse
import lisaorbits
import scipy.optimize
from tqdm.auto import tqdm
from script.eric_Fstats import (
    noise_inner_product,
    physical_amplitude_coordinates,
    DWD_sum,
    reconstruct_waveform,
    response_to_frequency_domain,
    physical_parameters_from_coordinates,
    build_amplitude_bases,
    candidate_amplitude_bases,
)
from script.spectrum import prepare_xyz_geometry
import lisainstrument
import pytdi
import pytdi.michelson
from script.tool import _bme_to_icrs
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from script.eric_Fstats import f_statistic
from script.eric_preprocess import effective_noise_cov
import shutil
from tempfile import mkdtemp
from script.tool import _save_tree
from script.dwd_bank_association import assign_bank_ids, classify_injection_recovery
from script.lisa_daily_simulation import simulate_lisa_daily

input_file = Path("./data/10dwd_ids.h5")
LDC_file = Path("./data/LDC2_sangria_training_v2.h5")
output_file = Path("./data/10dwd_subtraction.h5")

segment_count = 1
segment_duration = 360.0 * 86400.0
observation_duration = segment_count * segment_duration
detector_sampling_rate = 0.1
detector_sample_interval = 1.0 / detector_sampling_rate
frequency_resolution = 1.0 / segment_duration
tdi_channels = ("X2", "Y2", "Z2")
orbit = "unequal"
reuse_saved_data = True

freq_block_center = 0.0032922542578130718
freq_block_width = 10.0e-6
freq_block_range = (
    freq_block_center - 0.5 * freq_block_width,
    freq_block_center + 0.5 * freq_block_width,
)
catalogue_frequency_derivative_range = (
    7.60863249122299e-23,
    2.758292489484066e-12,
)

log10_frequency_derivative_upper_limits = (
    (0.0,     1.0e-3, -16.9),
    (1.0e-3,  2.0e-3, -15.8),
    (2.0e-3,  3.0e-3, -15.2),
    (3.0e-3,  4.0e-3, -14.8),
    (4.0e-3,  5.0e-3, -14.5),
    (5.0e-3,  7.0e-3, -14.1),
    (7.0e-3, 10.0e-3, -13.5),
    (10.0e-3, 15.0e-3, -12.7),
    (15.0e-3, 25.0e-3, -12.5),
    (25.0e-3, 50.0e-3, -11.2),
)

log10_frequency_derivative_bounds = (
    float(np.log10(catalogue_frequency_derivative_range[0])),
    max(
        log10_upper
        for low, high, log10_upper in log10_frequency_derivative_upper_limits
        if low <= freq_block_range[1] and high > freq_block_range[0]
    ),
)
frequency_derivative_bounds = (
    catalogue_frequency_derivative_range[0],
    10.0 ** log10_frequency_derivative_bounds[1],
)
log_frequency_scale = freq_block_center * observation_duration * np.log(10.0)

template_file = Path("./data/dwd_template_1block.h5")

arm_length = 2.5e9
N_acc = 3.0e-15
delta_x = 15.0e-12
initial_n_acc = 4.0e-15
initial_delta_x = 12.0e-12
noise_ratio_bounds = (0.2, 5.0)

parameter_fields = {
    "A": "Amplitude",
    "f": "Frequency",
    "df": "FrequencyDerivative",
    "phi0": "InitialPhase",
    "iota": "Inclination",
    "longitude": "EclipticLongitude",
    "latitude": "EclipticLatitude",
    "psi_bme": "Polarization",
}

started = time.perf_counter()

print("1/6  Reading and verifying the ten-DWD input", flush=True)

with h5py.File(input_file, "r") as hdf5:
    requested_source_id = np.asarray(hdf5["dwd_id"], dtype=np.int64)

catalogue_index_by_id = {}
with h5py.File(LDC_file, "r") as hdf5:  # read from LDC file
    catalogue = hdf5["sky/dgb/cat"]
    catalogue_chunk_size = 1000000
    progress = tqdm(
        total=len(catalogue),
        desc="Resolving DWD IDs in LDC",
        unit="row",
        unit_scale=True,
    )
    for chunk_start in range(0, len(catalogue), catalogue_chunk_size):
        chunk_stop = min(chunk_start + catalogue_chunk_size, len(catalogue))
        names = np.asarray(
            catalogue.fields("Name")[chunk_start:chunk_stop, 0],
            dtype=np.int64,
        )
        selected = np.flatnonzero(np.isin(names, requested_source_id))
        for local_index in selected:
            catalogue_index_by_id[int(names[local_index])] = int(
                chunk_start + local_index
            )
        progress.update(chunk_stop - chunk_start)
    progress.close()
    catalogue_index = np.asarray(
        [catalogue_index_by_id[int(source_id)] for source_id in requested_source_id],
        dtype=np.int64,
    )
    records = np.asarray(
        [catalogue[int(index), 0] for index in catalogue_index]
    )

source_id = np.asarray(records["Name"], dtype=np.int64)
injected = {
    name: np.asarray(records[field], dtype=np.float64)
    for name, field in parameter_fields.items()
}
input_id_order_matches = bool(np.array_equal(source_id, requested_source_id))
input_unique_id_count = int(len(np.unique(requested_source_id)))

# ---------------------------------------------------------------------------
# 2. Read the saved observation, or simulate DWDs and ACC+OMS noise
# ---------------------------------------------------------------------------

print("2/6  Preparing the DWD+ACC+OMS observation", flush=True)

with TemporaryDirectory(prefix="10dwd_orbit_") as temporary_directory:
    orbit_file = Path(temporary_directory) / "unequal_arm_orbit.h5"
    orbit_start = -20.0e6
    orbit_stop = observation_duration + 20.0e6
    orbit_count = int(math.ceil((orbit_stop - orbit_start) / 86400.0)) + 1
    lisaorbits.KeplerianOrbits(L=arm_length).write(
        str(orbit_file), t0=orbit_start, dt=86400.0, size=orbit_count, mode="w",
    )

    sample_count = int(round(observation_duration * detector_sampling_rate))
    simulation_time = np.arange(sample_count, dtype=np.float64) * detector_sample_interval
    simulation_reused = reuse_saved_data and output_file.exists()
    if simulation_reused:
        print(f"     Reading saved DWD signals and noise from {output_file}; simulation skipped", flush=True)
        with h5py.File(output_file, "r") as previous:
            expected = dict(observation_duration_s=observation_duration,
                detector_sampling_rate_hz=detector_sampling_rate, arm_length_m=arm_length,
                orbit_start_s=orbit_start, orbit_sample_interval_s=86400.0, orbit_sample_count=orbit_count)
            compatible = all(f"config/{name}" in previous and previous[f"config/{name}"][()] == value
                             for name, value in expected.items())
            compatible = compatible and "injection/dwd_id" in previous and np.array_equal(
                previous["injection/dwd_id"][:], source_id)
            compatible = compatible and all(f"injection/parameters/{name}" in previous
                and np.array_equal(previous[f"injection/parameters/{name}"][:], values)
                for name, values in injected.items())
            series = ("simulation/total_xyz_td", "simulation/signal_xyz_td")
            available = all(name in previous and previous[name].shape == (3, sample_count) for name in series)
            available = available and all(name in previous for name in (
                "simulation/instrument_random_seed", "config/retained_frequency_bins", "injection/source_xyz_fd"))
            if not (compatible and available):
                raise ValueError("Saved observation is missing required data or does not match the current "
                                 "DWDs, duration, sampling or orbit. Set reuse_saved_data=False to simulate afresh.")
            total_xyz_td = previous[series[0]][:]
            injected_signal_xyz_td = previous[series[1]][:]
            pure_noise_xyz_td = (previous["simulation/noise_xyz_td"][:]
                if "simulation/noise_xyz_td" in previous and previous["simulation/noise_xyz_td"].shape == (3, sample_count)
                else total_xyz_td - injected_signal_xyz_td)
            noise_internal_seed = int(previous["simulation/instrument_random_seed"][()])
            saved_source_bins = previous["config/retained_frequency_bins"][:]
            injected_source_xyz_fd = previous["injection/source_xyz_fd"][:]
            if injected_source_xyz_fd.shape != (len(source_id), 3, len(saved_source_bins)):
                raise ValueError("Saved per-source FFT shape is incompatible; set reuse_saved_data=False to simulate afresh.")
    else:
        print("     Simulating DWD signals and fresh ACC+OMS noise", flush=True)
        injected_source_xyz_td, pure_noise_xyz_td, noise_internal_seed = simulate_lisa_daily(
            injected, orbit_file, observation_duration, detector_sampling_rate,
            N_acc, delta_x, workers=30,
        )
        injected_signal_xyz_td = np.sum(injected_source_xyz_td, axis=0)
        total_xyz_td = injected_signal_xyz_td + pure_noise_xyz_td
    simulation_seconds = time.perf_counter() - started

    # -----------------------------------------------------------------------
    # 3. FFT the simulated data; construct the analysis response and covariance
    # -----------------------------------------------------------------------

    print("3/6  Computing the native FFT grid and analysis covariance", flush=True)
    padding = float(freq_block_range[1] * 1.0e-4 + frequency_derivative_bounds[1] * observation_duration + 8.0 / observation_duration)
    padded_frequency_bounds = (freq_block_range[0] - padding, freq_block_range[1] + padding,)
    fft_frequency = np.fft.rfftfreq(sample_count, d=detector_sample_interval)
    retained_frequency_bins = np.flatnonzero(
        (fft_frequency >= padded_frequency_bounds[0])
        & (fft_frequency <= padded_frequency_bounds[1])
    )
    physical_frequency = fft_frequency[retained_frequency_bins]
    if simulation_reused:
        saved_indices = np.searchsorted(saved_source_bins, retained_frequency_bins)
        if np.any(saved_indices >= len(saved_source_bins)) or not np.array_equal(
                saved_source_bins[saved_indices], retained_frequency_bins):
            raise ValueError("Saved per-source FFTs do not cover this frequency block; "
                             "set reuse_saved_data=False to simulate afresh.")
        injected_source_xyz_fd = injected_source_xyz_fd[:, :, saved_indices]
    else:
        injected_source_xyz_fd = np.stack([
            np.fft.rfft(values, axis=-1)[:, retained_frequency_bins]
            for values in tqdm(injected_source_xyz_td, desc="DWD FFTs", unit="DWD")
        ])
        del injected_source_xyz_td
    injected_signal_xyz_fd = np.fft.rfft(injected_signal_xyz_td, axis=-1)[:, retained_frequency_bins]
    total_xyz_fd = np.fft.rfft(total_xyz_td, axis=-1)[:, retained_frequency_bins]


    orbit_model = lisagwresponse.GalacticBinary(
        A=1.0, f=1.0e-3, df=0.0, phi0=0.0, iota=0.0, psi=0.0,
        t_init=0.0, ra=0.0, dec=0.0, orbits=str(orbit_file),
        orbit_interp_order=5, shift_sun="never",
    )
    response_block_count = int(round(observation_duration / 3600.0))
    response_times = (
        np.arange(response_block_count, dtype=np.float64) * 3600.0
        + 0.5 * (int(round(3600.0 * detector_sampling_rate)) - 1) * detector_sample_interval
    )
    response_geometry = prepare_xyz_geometry(response_times, orbit_model)

    response_to_frequency_domain = partial(
        response_to_frequency_domain,
        physical_frequency=physical_frequency,
        detector_sampling_rate=detector_sampling_rate,
        observation_duration=observation_duration,
        detector_sample_interval=detector_sample_interval,
        response_block_count=response_block_count,
        response_times=response_times,
    )


    # -----------------------------------------------------------------------
    # 4. Fit the catalogue template bank repeatedly, then refine selected sources
    # -----------------------------------------------------------------------


    build_amplitude_bases = partial(
        build_amplitude_bases,
        response_times=response_times,
        orbit_model=orbit_model,
        response_geometry=response_geometry,
        physical_frequency=physical_frequency,
        response_to_frequency_domain=response_to_frequency_domain,
    )


    with h5py.File(template_file, "r") as hdf5:
        bank_source_id = np.asarray(hdf5["templates/dwd_id"], dtype=np.int64)
        bank_catalogue_index = np.asarray(hdf5["templates/catalogue_index"], dtype=np.int64)
        bank_catalogue_snr = np.asarray(hdf5["templates/catalogue_optimal_snr_xyz"])
        bank_parameters = {name: np.asarray(hdf5[f"templates/{name}"]) for name in parameter_fields}
        bank_snr_duration = float(hdf5["snr_reference"].attrs["observation_duration_s"])
        bank_frequency_range = np.asarray(hdf5["config/freq_block_range_hz"])
    bank_doppler = np.column_stack([
        bank_parameters[name] for name in ("f", "df", "longitude", "latitude")
    ])

    print("4/6  Prepare fixed XYZ covariance and match templates; select F >= 64", flush=True)
    wide_center_bin = int(round(freq_block_center * observation_duration))
    wide_bins = np.arange(wide_center_bin - 2048, wide_center_bin + 2049)
    wide_frequency = fft_frequency[wide_bins]
    wide_total_xyz_fd = np.fft.rfft(total_xyz_td, axis=-1)[:, wide_bins]
    psd = 2.0 / (segment_duration * detector_sampling_rate**2) * np.einsum(
        "ik,jk->kij", wide_total_xyz_fd, wide_total_xyz_fd.conj(),
    )
    covariance_estimate = effective_noise_cov(psd, neighboring_bins=256, downweight_peaks=True)
    core_indices = retained_frequency_bins - wide_bins[0]
    fixed_covariance = np.array(covariance_estimate[core_indices], copy=True)
    fixed_inverse = np.linalg.inv(fixed_covariance)

    candidate_amplitude_bases = partial(
        candidate_amplitude_bases,
        build_amplitude_bases=build_amplitude_bases,
        physical_frequency=physical_frequency,
    )

    bank_bases = np.stack([
        candidate_amplitude_bases(point[None])[0]
        for point in tqdm(bank_doppler, desc="All-bank F-statistics", unit="template")
    ])
    bank_two_f = f_statistic(
        total_xyz_fd, bank_bases, fixed_covariance, segment_duration, detector_sampling_rate,
    ).two_f
    selected_candidates = np.flatnonzero(bank_two_f >= 128.0)
    print(f"     {len(selected_candidates)} of {len(bank_source_id)} templates pass F >= 64", flush=True)

    print("5/6  Eryn RJMCMC: GPU waveform/likelihood, one host CPU, four replicas, all selected candidates initially active", flush=True)
    analysis_directory = Path(mkdtemp(prefix="eryn_", dir=output_file.parent))
    checkpoint_directory = analysis_directory / "checkpoints"

    from script.eric_rjmcmc import run_eryn_rjmcmc

    posterior = run_eryn_rjmcmc(
        total_xyz_fd, fixed_covariance, bank_doppler[selected_candidates],
        freq_block_range, frequency_derivative_bounds, freq_block_center,
        segment_duration, detector_sampling_rate, checkpoint_directory,
        chain_count=1, inverse_temperatures=(1.0, 0.8, 0.6, 0.4),
        iterations=2500, burn_in=500, maximum_sources=len(bank_source_id),
        response_times=response_times,
        response_geometry=response_geometry, physical_frequency=physical_frequency,
    )
    recovered_source_xyz_fd = reconstruct_waveform(
        posterior["fit_result"]["coefficients"], posterior["bases"],
    )
    recovered_signal_xyz_fd = recovered_source_xyz_fd.sum(axis=0)
    residual_xyz_fd = total_xyz_fd - recovered_signal_xyz_fd
    recovered = {
        name: posterior["doppler"][:, column]
        for column, name in enumerate(("f", "df", "longitude", "latitude"))
    }
    amplitude_parameters = [
        physical_parameters_from_coordinates(row * 1.0e-22)
        for row in posterior["fit_result"]["coefficients"]
    ]
    recovered.update({name: np.array([row[name] for row in amplitude_parameters])
                      for name in ("A", "iota", "psi_bme", "phi0")})

    print("6/6  Identify recovered waveforms against the complete bank, then validate injections", flush=True)
    bank_amplitude = np.asarray(bank_parameters["A"], dtype=np.float64)
    bank_cos_iota = np.cos(np.asarray(bank_parameters["iota"], dtype=np.float64))
    bank_amplitudes = physical_amplitude_coordinates(
        bank_amplitude * (1.0 + bank_cos_iota**2),
        2.0 * bank_amplitude * bank_cos_iota,
        bank_parameters["psi_bme"], bank_parameters["phi0"],
    ) / 1.0e-22
    bank_source_xyz_fd = reconstruct_waveform(bank_amplitudes, bank_bases)
    identification = assign_bank_ids(
        recovered_source_xyz_fd, bank_source_xyz_fd, bank_source_id,
        fixed_inverse, segment_duration, detector_sampling_rate, minimum_overlap=0.9,
    )
    bank_confirmed_signal = recovered_source_xyz_fd[identification["bank_confirmed"]].sum(axis=0)
    bank_confirmed_residual = total_xyz_fd - bank_confirmed_signal
    # Injection labels enter only this validation stage, never the ID assignment.
    injection_validation = classify_injection_recovery(identification, source_id)
    injection_validation["recovered_status"] = injection_validation["recovered_status"].astype("S24")
    injected_indices = np.flatnonzero(injection_validation["injection_to_recovery"] >= 0)
    recovered_indices = injection_validation["injection_to_recovery"][injected_indices]
    injected_norm = noise_inner_product(
        injected_source_xyz_fd[injected_indices], injected_source_xyz_fd[injected_indices],
        fixed_inverse, segment_duration, detector_sampling_rate,
    )
    recovered_norm = noise_inner_product(
        recovered_source_xyz_fd[recovered_indices], recovered_source_xyz_fd[recovered_indices],
        fixed_inverse, segment_duration, detector_sampling_rate,
    )
    cross = noise_inner_product(
        recovered_source_xyz_fd[recovered_indices], injected_source_xyz_fd[injected_indices],
        fixed_inverse, segment_duration, detector_sampling_rate,
    )
    waveform_error = recovered_source_xyz_fd[recovered_indices] - injected_source_xyz_fd[injected_indices]
    normalized_error = noise_inner_product(
        waveform_error, waveform_error, fixed_inverse, segment_duration, detector_sampling_rate,
    ) / np.sqrt(injected_norm * recovered_norm)
    comparison_recovered = {name: recovered[name][recovered_indices].copy() for name in parameter_fields}
    comparison_injected = {name: injected[name][injected_indices].copy() for name in parameter_fields}
    phase_choices = []
    for branch in (0, 1):
        phase = comparison_injected["phi0"] + (
            comparison_recovered["phi0"] + branch*np.pi - comparison_injected["phi0"] + np.pi
        ) % (2*np.pi) - np.pi
        psi = comparison_injected["psi_bme"] + (
            comparison_recovered["psi_bme"] + branch*np.pi/2 - comparison_injected["psi_bme"] + np.pi/2
        ) % np.pi - np.pi/2
        cost = (phase-comparison_injected["phi0"])**2 + 4*(psi-comparison_injected["psi_bme"])**2
        phase_choices.append((cost, phase, psi))
    alternate = phase_choices[1][0] < phase_choices[0][0]
    comparison_recovered["phi0"] = np.where(alternate, phase_choices[1][1], phase_choices[0][1])
    comparison_recovered["psi_bme"] = np.where(alternate, phase_choices[1][2], phase_choices[0][2])
    comparison_recovered["longitude"] = comparison_injected["longitude"] + (
        comparison_recovered["longitude"] - comparison_injected["longitude"] + np.pi
    ) % (2*np.pi) - np.pi
    for name in ("A", "f", "df"):
        comparison_injected["log10_" + name] = np.log10(comparison_injected[name])
        comparison_recovered["log10_" + name] = np.log10(comparison_recovered[name])
    residual_bank_two_f = f_statistic(
        residual_xyz_fd, bank_bases, fixed_covariance, segment_duration, detector_sampling_rate,
    ).two_f
    injection_validation.update(
        injected_index=injected_indices, recovered_index=recovered_indices,
        dwd_id=source_id[injected_indices], overlap=cross / np.sqrt(injected_norm * recovered_norm),
        delta=normalized_error, injected_parameters=comparison_injected,
        recovered_parameters_same_branch=comparison_recovered,
        residual_bank_two_f=residual_bank_two_f,
        residual_norm=float(noise_inner_product(residual_xyz_fd, residual_xyz_fd,
            fixed_inverse, segment_duration, detector_sampling_rate)),
    )
    tree = dict(
        config=dict(observation_duration_s=observation_duration, segment_count=segment_count,
            segment_duration_s=segment_duration, detector_sampling_rate_hz=detector_sampling_rate,
            physical_frequency_hz=physical_frequency, retained_frequency_bins=retained_frequency_bins,
            wide_frequency_hz=wide_frequency, core_indices=core_indices,
            freq_block_center_hz=freq_block_center, freq_block_range_hz=freq_block_range,
            padded_frequency_bounds_hz=padded_frequency_bounds,
            frequency_derivative_bounds_hz_per_s=frequency_derivative_bounds,
            tdi_channels=list(tdi_channels), arm_length_m=arm_length,
            orbit_start_s=orbit_start, orbit_sample_interval_s=86400.0, orbit_sample_count=orbit_count,
            orbit_interpolation_order=5, response_interval_s=3600.0,
            input_file=str(input_file), template_file=str(template_file)),
        data=dict(total_xyz_fd=total_xyz_fd, signal_xyz_fd=injected_signal_xyz_fd,
            total_xyz_fd_wide=wide_total_xyz_fd),
        simulation=dict(time_s=simulation_time, total_xyz_td=total_xyz_td,
            signal_xyz_td=injected_signal_xyz_td, noise_xyz_td=pure_noise_xyz_td,
            noise_reused=simulation_reused, observation_reused=simulation_reused,
            instrument_random_seed=noise_internal_seed),
        injection=dict(dwd_id=source_id, catalogue_index=catalogue_index,
            parameters=injected, source_xyz_fd=injected_source_xyz_fd),
        estimator=dict(covariance=covariance_estimate, half_width=256,
            ridge=1.0e-5, cutoff=15.0),
        fixed_covariance=fixed_covariance,
        templates=dict(dwd_id=bank_source_id, catalogue_index=bank_catalogue_index,
            catalogue_optimal_snr_xyz=bank_catalogue_snr, parameters=bank_parameters,
            doppler=bank_doppler, bases=bank_bases, source_xyz_fd=bank_source_xyz_fd,
            f=bank_two_f/2, two_f=bank_two_f, selected=bank_two_f >= 128.0,
            selected_indices=selected_candidates),
        rjmcmc={key: value for key, value in posterior.items() if key not in ("bases", "fit_result")},
        recovery=dict(parameters=recovered, doppler=posterior["doppler"], bases=posterior["bases"],
            coefficients=posterior["fit_result"]["coefficients"], source_xyz_fd=recovered_source_xyz_fd,
            signal_xyz_fd=recovered_signal_xyz_fd, residual_xyz_fd=residual_xyz_fd,
            amplitude_matrix=posterior["fit_result"]["amplitude_matrix"],
            amplitude_covariance=posterior["fit_result"]["amplitude_covariance"]),
        bank_identification=identification,
        bank_confirmed_subtraction=dict(signal_xyz_fd=bank_confirmed_signal,
            residual_xyz_fd=bank_confirmed_residual),
        injection_id_validation=injection_validation,
        runtime=dict(total_seconds=time.perf_counter()-started, simulation_seconds=simulation_seconds),
    )
    temporary_output = analysis_directory / "result.h5.tmp"
    with h5py.File(temporary_output, "w") as hdf5:
        _save_tree(hdf5, tree)
        with h5py.File(posterior["checkpoint_file"], "r") as checkpoint:
            checkpoint.copy("mcmc", hdf5["rjmcmc"], name="eryn_backend")
        hdf5.attrs.update(
            complete=True,
            sampler=posterior["engine"], cpu_core_limit=1,
            result_description="DWD recovery conditional on a fixed, data-estimated effective covariance.",
            id_rule="All bank waveforms; signed O > 0.9; higher-overlap pair wins, loser tries next.",
            injection_ids_used_for_identification=False, noise_parameters_inferred=False,
            independent_chain_convergence_assessed=False, checkpoint_directory=str(checkpoint_directory),
            screening="Catalogue-informed bank, not a blind all-sky search.",
            uncertainty="Raw conditional posterior draws saved; assess within-run mixing before quoting intervals.",
            ghost_rule="Unmatched bank ID or, only during validation, assigned ID absent from injections.",
            subtraction_rule="Full posterior and bank-confirmed residuals saved separately; no truth-based pruning.",
        )
    if output_file.exists():
        shutil.copy2(output_file, analysis_directory / "previous_output.h5")
    os.replace(temporary_output, output_file)
    print("Saved:", output_file, "Checkpoints:", checkpoint_directory, flush=True)
    print("Bank-confirmed:", int(identification["bank_confirmed"].sum()),
          "Ghost:", int(injection_validation["ghost"].sum()),
          "Lurking:", len(injection_validation["lurking_dwd_id"]), flush=True)

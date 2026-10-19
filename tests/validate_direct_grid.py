"""Bounded CUDA comparison of direct bins against the previous cropped-grid route.

Usage: python tests/validate_direct_grid.py PROJECT OUTPUT
OUTPUT/before/dwd_blind_search.py is the frozen pre-change search helper.
The script's checkout supplies the new helpers; PROJECT supplies native LISA
dependencies and the existing orbit. Production observations are never changed.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
from pathlib import Path
from tempfile import mkdtemp
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT, OUTPUT = map(Path, sys.argv[1:3])
sys.path.insert(0, str(PROJECT))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


science = load('direct_grid_science', ROOT / 'script/eric_Fstats.py')
search = load('direct_grid_search', ROOT / 'script/dwd_blind_search.py')
previous_search = load('previous_grid_search', OUTPUT / 'before/dwd_blind_search.py')
from script.eric_generate_waveform import GPUResponse
from script.dwd_bank_association import assign_bank_ids
from eric_rjmcmc import marginal_likelihood, run_eryn_rjmcmc


def relative_rows(actual, reference):
    delta = (actual - reference).reshape(len(reference), -1)
    norm = np.linalg.norm(reference.reshape(len(reference), -1), axis=1)
    return float(np.max(np.linalg.norm(delta, axis=1) / np.maximum(norm, 1.e-300)))


OUTPUT.mkdir(parents=True, exist_ok=True)
started = time.perf_counter()
report = dict(scope='Moving-orbit GPUResponse, 30 and 360 days, five blocks per duration, eight synthetic Doppler points per block; same retained bins and full complex XYZ covariance.',
    source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ('eric_subtraction_all.py', 'script/eric_Fstats.py', 'script/dwd_blind_search.py')}, cases=[])
orbit = PROJECT / 'data/dwd_observation_full_catalogue.orbits.h5'
width, rate = 1.e-5, .1
rng = np.random.default_rng(20261013)
smoke = None
for days in (30, 360):
    duration = days * 86400.
    frequency = np.fft.rfftfreq(round(duration * rate), d=1 / rate)
    for lower in (.0001, .001, .003, .01, .04999):
        bounds = np.array([lower, lower + width])
        center = bounds.mean()
        retained = np.flatnonzero((frequency >= bounds[0] - width)
            & (frequency <= bounds[1] + width) & (frequency > 0))
        center_bin = int(round(center * duration))
        wide = np.arange(max(1, min(center_bin - 2048, retained[0] - 256)),
            min(len(frequency), max(center_bin + 2049, retained[-1] + 257)))
        crop = retained - wide[0]
        old_builder, times, geometry = science.build_response(OUTPUT, frequency[wide], duration, rate, orbit_file=orbit)
        builder = GPUResponse(np.broadcast_to(np.eye(3), (len(retained), 3, 3)),
            np.zeros((3, len(retained)), complex), times, geometry, frequency[retained], duration, rate)
        points = np.column_stack((lower + width * np.array([0., .0001, .1, .3, .6, .9, .9999, 1.]),
            np.array([-1.e-10, -1.e-15, 0., 2.e-17, -2.e-17, 1.e-14, 1.e-10, 0.]),
            np.linspace(.1, 6.1, 8), np.linspace(-1.2, 1.2, 8)))
        reference = old_builder.build(points)[..., crop]
        direct = builder.build(points)
        np.testing.assert_allclose(direct, reference, rtol=2.e-11, atol=1.e-35)
        amplitudes = rng.normal(size=(len(points), 4))
        old_wave = science.reconstruct_waveform(amplitudes, reference)
        new_wave = science.reconstruct_waveform(amplitudes, direct)
        scale = float(np.sqrt(np.mean(abs(old_wave)**2)))
        matrix = np.array([[1., .12 + .04j, -.07j], [.12 - .04j, 1.3, .08], [.07j, .08, .8]])
        covariance = np.broadcast_to(matrix * scale**2 * 4 / (duration * rate**2), (len(retained), 3, 3)).copy()
        data = old_wave[2:5].sum(axis=0) + scale * (rng.normal(size=old_wave.shape[1:]) + 1j * rng.normal(size=old_wave.shape[1:]))
        table = np.zeros((len(points), 11))
        table[:, 0] = np.arange(20001, 20001 + len(points))
        table[:, [2, 3, 6, 7]] = points
        old_F_values = previous_search.amplitude_f_statistic(data, reference, covariance, duration, rate)['two_f'] / 2
        F_values = search.F_statistic(data, direct, covariance=covariance, T_obs=duration, fs=rate)
        np.testing.assert_allclose(F_values, old_F_values, rtol=2.e-10, atol=1.e-10)
        selected, old_selected = F_values >= 16., old_F_values >= 16.
        np.testing.assert_array_equal(np.flatnonzero(selected), np.flatnonzero(old_selected))
        np.testing.assert_array_equal(table[selected, 0], table[old_selected, 0])
        old_labels = assign_bank_ids(old_wave, old_wave, table[:, 0].astype(int), np.linalg.inv(covariance), duration, rate)
        labels = assign_bank_ids(new_wave, old_wave, table[:, 0].astype(int), np.linalg.inv(covariance), duration, rate)
        np.testing.assert_array_equal(labels['matched_dwd_id'], old_labels['matched_dwd_id'])
        old_logm = marginal_likelihood(data, reference[2:5], covariance, 0., duration, rate)
        logm = marginal_likelihood(data, direct[2:5], covariance, 0., duration, rate)
        np.testing.assert_allclose(logm, old_logm, rtol=2.e-10, atol=1.e-9)
        case = dict(days=days, bounds_hz=bounds.tolist(), retained_bins=len(retained), old_grid_bins=len(wide),
            sources=len(points), basis_max_relative_l2=relative_rows(direct, reference),
            waveform_max_relative_l2=relative_rows(new_wave, old_wave),
            waveform_max_absolute=float(np.max(abs(new_wave - old_wave))),
            two_f_max_absolute=float(2 * np.max(abs(F_values - old_F_values))),
            two_f_max_relative=float(np.max(abs(F_values - old_F_values) / np.maximum(abs(old_F_values), 5.e-301))),
            log_marginal_absolute_difference=float(abs(logm - old_logm)), selected_ids_identical=True, association_ids_identical=True)
        report['cases'].append(case)
        print(json.dumps(case), flush=True)
        if days == 30 and lower == .003:
            smoke = data, covariance, table[3:5], bounds, center, duration, times, geometry, frequency[retained]
data, covariance, table, bounds, center, duration, times, geometry, frequency = smoke
np.random.seed(20261013)
checkpoints = Path(mkdtemp(prefix='rjmcmc_', dir=OUTPUT))
posterior = run_eryn_rjmcmc(data, covariance, table, bounds,
    np.array([[-1.e-10, -1.e-25], [1.e-25, 1.e-10]]), duration, rate, checkpoints,
    response_times=times, response_geometry=geometry,
    physical_frequency=frequency, iterations=32, burn_in=16, maximum_sources=4)
builder = GPUResponse(covariance, data, times, geometry, frequency, duration, rate)
np.testing.assert_array_equal(posterior['bases'], builder.build(posterior['doppler']))
assert np.isfinite(posterior['fit_result']['log_marginal'])
report['rjmcmc_smoke'] = dict(cycles=32, burn_in=16, days=30, initial_sources=2, maximum_sources=4,
    checkpoint_directory=str(checkpoints),
    modal_count=int(posterior['modal_count']), posterior_bases_on_direct_bins=True,
    returned_bases_identical=True, convergence_assessed=False)
report.update(status='passed', seconds=time.perf_counter() - started, gpu=str(builder.device.device_kind))
(OUTPUT / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report['rjmcmc_smoke']), flush=True)

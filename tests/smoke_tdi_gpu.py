"""Native-PyTDI parity and real caller smoke tests; run on a CUDA host.

Usage: python tests/smoke_tdi_gpu.py tmp/tdi_api_20261011
The optional full-year comparison uses the saved native reference in
tmp/TDI_gpu_20261011. No production observation or sampler is run.
"""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

import h5py
import lisaorbits
import numpy as np
import pytdi
import pytdi.michelson

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from eric_generate_waveform import tdi_gpu
from script.oneblock_observation import validate_observation

OUT = ROOT / (sys.argv[1] if len(sys.argv) > 1 else 'tmp/tdi_api_20261011')
OUT.mkdir(parents=True, exist_ok=True)
host = os.uname().nodename
run = OUT / host
run.mkdir(exist_ok=True)
report = dict(status='running', host=host, fs=.1, comparisons={},
    source_sha256={name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
        ('eric_generate_waveform.py', 'eric_simulation_DWD.py', 'eric_subtraction_1block.py')})
original_build, original_shift = pytdi.TDICombination.build, pytdi.dsp.timeshift


def compare(label, actual, reference):
    delta = actual - reference
    report['comparisons'][label] = dict(shape=list(actual.shape),
        max_absolute=float(np.abs(delta).max()),
        relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(reference), np.finfo(float).tiny)),
        bitwise_identical=bool(np.array_equal(np.ascontiguousarray(actual).view('u8'),
                                              np.ascontiguousarray(reference).view('u8'))))
    assert report['comparisons'][label]['bitwise_identical'], (label, report['comparisons'][label])


def native(measurements, args, combination='XYZ2', keep=slice(None)):
    return np.stack([getattr(pytdi.michelson, axis + combination[-1]).build(**args)(measurements)[..., keep]
                     for axis in 'XYZ'], axis=-2)


try:
    orbit = run / 'orbit.h5'
    lisaorbits.KeplerianOrbits(L=2.5e9).write(str(orbit), t0=-20e6, dt=86400., size=825, mode='w')
    config = dict(dt=10., orbits=str(orbit), orbit_dataset='tcb/ltt')
    size, t0 = 38880, -86400.
    positions = np.arange(size, dtype=float)
    links = np.column_stack([np.sin(positions * (.021 + column * .0027))
                            + .02 * np.cos(positions * .073) for column in range(6)])
    science = {'sci_' + mosa: links[:, column] for column, mosa in enumerate(pytdi.Data.MOSAS)}
    data = pytdi.Data.from_orbits(str(orbit), fs=.1, t0=t0, dataset='tcb/ltt', **science)
    for combination in ('XYZ1', 'XYZ2'):
        reference = native(data.measurements, data.args, combination)
        compare('six_links_' + combination, tdi_gpu(links, config, combination, t0=t0), reference)
        compare('six_links_crop_' + combination,
            tdi_gpu(links, config, combination, t0=t0, keep=slice(100, -100)), reference[:, 100:-100])
    batch = np.stack((links, links * .37))
    compare('batched_six_links', tdi_gpu(batch, config, t0=t0),
            np.stack((native(data.measurements, data.args), native(
                {name: value * .37 for name, value in data.measurements.items()}, data.args))))
    shorthand = {mosa: links[:, column] for column, mosa in enumerate(pytdi.Data.MOSAS)}
    compare('shorthand', tdi_gpu(shorthand, config, t0=t0), native(data.measurements, data.args))
    zeros = {name: np.zeros(size) for name in science}
    zero_data = pytdi.Data.from_orbits(str(orbit), fs=.1, t0=t0, dataset='tcb/ltt', **zeros)
    compare('zero_streams', tdi_gpu(np.zeros_like(links), config, t0=t0), native(zero_data.measurements, zero_data.args))

    # Run the real generator on ten independent parameter rows, with native noise.
    fields = ('Amplitude', 'Frequency', 'FrequencyDerivative', 'InitialPhase',
              'Inclination', 'EclipticLongitude', 'EclipticLatitude', 'Polarization')
    parameters = np.column_stack((np.linspace(1e-23, 4e-22, 10), np.linspace(.000817123, .01234789, 10),
        np.linspace(-2e-17, 3e-17, 10), np.linspace(.12, 5.87, 10), np.linspace(.2, 2.9, 10),
        np.linspace(.03, 6.1, 10), np.linspace(-1.1, 1.2, 10), np.linspace(.2, 2.6, 10)))
    table = np.zeros(10, dtype=[(field, 'f8', (1,)) for field in fields])
    for column, field in enumerate(fields):
        table[field][:, 0] = parameters[:, column]
    with h5py.File(run / 'parameters.h5', 'w') as saved:
        saved.create_dataset('sky/dgb/cat', data=table[:5])
        saved.create_dataset('sky/igb/cat', data=table[5:])
    tree = ast.parse((ROOT / 'eric_simulation_DWD.py').read_text())
    replacements = dict(catalogue_file=run / 'parameters.h5', observation_file=run / 'observation.h5', T_obs=2.5 * 86400.)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in replacements:
            value = replacements[node.targets[0].id]
            node.value = ast.parse(f'Path({str(value)!r})' if isinstance(value, Path) else repr(value), mode='eval').body
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'tdi_gpu':
            node.func.id = 'compare_simulation'
    simulation_scope = {}
    def compare_simulation(measurements, args, *, keep):
        actual = tdi_gpu(measurements, args, keep=keep)
        compare('simulation_' + simulation_scope['name'], actual, native(measurements, args, keep=keep))
        return actual
    simulation_scope.update(__name__='__main__', compare_simulation=compare_simulation)
    replacements['observation_file'].write_bytes(b'previous test output')
    exec(compile(ast.fix_missing_locations(tree), 'eric_simulation_DWD.py', 'exec'), simulation_scope)
    with h5py.File(replacements['observation_file']) as saved:
        validate_observation(saved)
        observation = saved['observation/total_xyz_td'][:]
        injected_signal = saved['observation/signal_xyz_td'][:]
        assert len(saved['population/catalogues/all']) == 10
        assert set(saved['observation']) == {'total_xyz_td', 'signal_xyz_td', 'noise_xyz_td'}
        for name, values in simulation_scope['tdi_results'].items():
            assert np.array_equal(values.view('u8'), saved['observation/' + name + '_xyz_td'][:].view('u8'))
    assert not simulation_scope['workspace'].exists()
    orbit = replacements['observation_file'].with_suffix('.orbits.h5')
    with h5py.File(replacements['observation_file']) as saved, h5py.File(orbit) as exported:
        assert dict(saved['orbit'].attrs) == dict(exported.attrs)
        for name in ('tcb/x', 'tcb/ltt', 'tps/ppr'):
            assert np.array_equal(saved['orbit/' + name][:], exported[name][:])
    report['simulation_pipeline'] = dict(synthetic_sources=10, days=2.5, shape=list(observation.shape),
        output_replaced=True, workspace_removed=True, stored_gpu_streams_verified=True,
        companion_orbit_matches_observation=True)

    # Execute the real accepted post-fit subtraction block, without running inference.
    import lisagwresponse
    import lisainstrument
    from eric_generate_waveform import DWD_simulation_gpu
    from script.eric_Fstats import physical_amplitudes, physical_amplitude_coordinates
    from script.tool import _save_tree
    controller = ast.parse((ROOT / 'eric_subtraction_1block.py').read_text())
    block = next(node for node in ast.walk(controller) if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name) and node.test.id == 'accepted'
        and any(isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Name)
                and item.targets[0].id == 'identified_dwd_parameters' for item in node.body))
    nday, n = 8640, observation.shape[-1]
    subtraction_config = dict(simulation_scope['detector_config'], orbits=str(orbit))
    cosine = np.cos(parameters[:, 4])
    coefficients = physical_amplitude_coordinates(parameters[:, 0] * (1 + cosine**2),
        2 * parameters[:, 0] * cosine, parameters[:, 7], parameters[:, 3]) / 1e-22
    scope = dict(accepted=True, np=np, h5py=h5py, work=run, orbit_file=orbit,
        N=n, N_day=nday, fs=.1, T_obs=2.5 * 86400., detector_config=subtraction_config,
        posterior={'doppler': parameters[:, [1, 2, 5, 6]]}, new_coefficients=coefficients,
        physical_amplitudes=physical_amplitudes, DWD_simulation_gpu=DWD_simulation_gpu,
        lisagwresponse=lisagwresponse, lisainstrument=lisainstrument, pytdi=pytdi,
        accumulated_xyz_td=np.zeros_like(observation), total_xyz_td=observation.copy(),
        wide_bins=np.arange(1, 129), retained_bins=np.arange(33, 65), core_indices=np.arange(32, 64),
        new_recovery={}, bank_bases=None, fixed_covariance=None,
        f_statistic=lambda *args: type('Statistic', (), {'two_f': np.zeros(10)})())
    calls = []
    def compare_subtraction(measurements, args, *, keep):
        calls.append('tdi')
        actual = tdi_gpu(measurements, args, keep=keep)
        compare('subtraction', actual, native(measurements, args, keep=keep))
        return actual
    scope['tdi_gpu'] = compare_subtraction
    exec(compile(ast.Module(body=[block], type_ignores=[]), 'eric_subtraction_1block.py', 'exec'), scope)
    assert calls == ['tdi']
    compare('subtraction_residual_identity', scope['residual_xyz_td'], observation - scope['signal_xyz_td'])
    compare('subtraction_fft_identity', scope['residual_xyz_fd_wide'],
            np.fft.rfft(scope['residual_xyz_td'], axis=-1)[:, scope['wide_bins']])
    compare('observation_preserved', scope['total_xyz_td'], observation)
    report['subtraction_pipeline'] = dict(fitted_sources=10, days=2.5, shape=list(scope['signal_xyz_td'].shape),
        sampler_run=False, fit_parameters_outside_catalogue_supported=True,
        recovered_vs_injected_relative_l2=float(np.linalg.norm(scope['signal_xyz_td'] - injected_signal) /
                                              np.linalg.norm(injected_signal)))
    assert report['subtraction_pipeline']['recovered_vs_injected_relative_l2'] < 1e-12
    scope.update(accumulated_xyz_td=scope['signal_xyz_td'], accumulated_xyz_fd_wide=scope['signal_xyz_fd_wide'],
        retained_residual_xyz_fd_wide=scope['residual_xyz_fd_wide'], previous_doppler=scope['posterior']['doppler'])
    scope['accepted'] = False
    exec(compile(ast.Module(body=[block], type_ignores=[]), 'eric_subtraction_1block.py', 'exec'), scope)
    assert calls == ['tdi']
    final = next(node for node in ast.walk(controller) if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'final_data')
    exec(compile(ast.Module(body=[final], type_ignores=[]), 'eric_subtraction_1block.py', 'exec'), scope)
    with h5py.File(run / 'subtraction.h5', 'w') as saved:
        _save_tree(saved.create_group('final_subtraction'), scope['final_data'])
    with h5py.File(run / 'subtraction.h5') as saved:
        for name, values in scope['final_data'].items():
            assert np.array_equal(np.ascontiguousarray(values).view('u8'),
                                  np.ascontiguousarray(saved['final_subtraction/' + name][:]).view('u8'))
    report['subtraction_pipeline'].update(rejected_round_skips_tdi=True, final_time_domain_streams_saved=True)

    # Recheck all nine full-year streams against the previously computed native reference.
    baseline = ROOT / 'tmp/TDI_gpu_20261011'
    if (baseline / 'native_chunked_360d.h5').exists():
        with h5py.File(baseline / 'input_360d.h5') as source:
            attrs = dict(source.attrs)
            args = dict(fs=float(attrs['fs']), delays={key: value[:] for key, value in source['delays'].items()},
                        delay_derivatives={key: value[:] for key, value in source['delay_derivatives'].items()})
            signal = {key: value[:] for key, value in source['signal'].items()}
            noise = {key: value[:] for key, value in source['noise'].items()}
        total = {key: signal[key] + noise[key] for key in signal}
        full = {key: np.stack((total[key], signal[key], noise[key])) for key in signal}
        started = time.perf_counter()
        xyz = tdi_gpu(full, args) / float(attrs['central_freq'])
        seconds = time.perf_counter() - started
        with h5py.File(baseline / 'native_chunked_360d.h5') as saved:
            for index, name in enumerate(('total', 'signal', 'noise')):
                compare('360d_' + name, xyz[index], saved[name][:])
        report['full_year'] = dict(days=360, streams=9, samples_per_stream=xyz.shape[-1],
            gpu_seconds=seconds, input_geometry_prepared=True, existing_native_reference_used=True)
    assert pytdi.TDICombination.build is original_build and pytdi.dsp.timeshift is original_shift
    import cupy as cp
    report.update(status='passed', gpu=cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
                  native_package_unmodified=True)
except Exception as error:
    report.update(status='failed', error=repr(error))
    raise
finally:
    (OUT / (host + '.json')).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

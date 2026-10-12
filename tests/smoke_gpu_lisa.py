"""Compare population link synthesis with the installed native LISA response.

Default execution requires a real CUDA device. --backend cpu is an explicit
secondary check of the same JAX algorithm and cannot establish GPU execution.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name, '1')
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIELDS = ('Amplitude', 'Frequency', 'FrequencyDerivative', 'InitialPhase',
          'Inclination', 'EclipticLongitude', 'EclipticLatitude', 'Polarization')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('gpu', 'cpu'), default='gpu')
    parser.add_argument('--sources-per-table', type=int, default=512)
    parser.add_argument('--batch-size', type=int, default=2048)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(requested_backend=args.backend, status='running',
                  reference='lisagwresponse.GalacticBinary.compute_gw_response',
                  threshold_relative_l2=1e-6, duration=360 * 86400., fs=.1, batch_size=args.batch_size, cases=[],
                  helper_sha256=hashlib.sha256((ROOT / 'script/dwd_population_gpu.py').read_bytes()).hexdigest(),
                  versions={name: importlib.metadata.version(name) for name in
                            ('lisagwresponse', 'lisaorbits', 'jax', 'jaxlib', 'numpy', 'h5py')})
    path = args.output / 'report.json'
    try:
        import lisaorbits
        import lisagwresponse
        from lisagwresponse.response import LINKS
        from script.tool import _bme_to_icrs
        import script.dwd_population_gpu as population
        if args.backend == 'cpu':
            import jax
            import jax.numpy as jnp
            jax.config.update('jax_enable_x64', True)
            device = jax.devices('cpu')[0]
            population.gpu_runtime = lambda: (jax, jnp, device)
        else:
            _, _, device = population.gpu_runtime()
        report.update(device=str(device), device_kind=device.device_kind,
                      gpu_executed=args.backend == 'gpu')
        orbit = args.output / 'orbit.h5'
        lisaorbits.KeplerianOrbits(L=2.5e9).write(str(orbit), t0=-20e6, dt=86400., size=825, mode='w')
        tables = []
        with h5py.File(ROOT / 'data/LDC2_sangria_training_v2.h5') as source:
            for kind in ('dgb', 'igb'):
                dataset = source[f'sky/{kind}/cat']
                indices = np.unique(np.linspace(0, len(dataset)-1, args.sources_per_table, dtype=int))
                tables.append(dataset[indices, 0])
            # A full scan selects the real largest absolute chirp from both tables.
            extreme = None
            for kind in ('dgb', 'igb'):
                dataset = source[f'sky/{kind}/cat']
                for first in range(0, len(dataset), 1_000_000):
                    derivative = dataset.fields('FrequencyDerivative')[first:first+1_000_000, 0]
                    index = int(np.argmax(np.abs(derivative)))
                    if extreme is None or abs(derivative[index]) > abs(extreme['FrequencyDerivative']):
                        extreme = dataset[first+index, 0]
        physical = np.concatenate([np.column_stack([rows[field] for field in FIELDS]) for rows in tables])
        extreme = np.array([[float(extreme[field]) for field in FIELDS]])
        N = int(report['duration'] * report['fs'])
        positions = np.unique(np.concatenate((np.arange(32), np.linspace(32, N-33, 129, dtype=int), N-1-np.arange(32))))
        times = positions / report['fs']
        for label, values in [('balanced_population', physical), ('largest_absolute_chirp', extreme)]:
            ra, dec, psi = _bme_to_icrs(values[:, 5], values[:, 6], values[:, 7])
            packed = np.column_stack((values[:, :5], ra, dec, psi))
            started = time.perf_counter()
            engine = population.PopulationLinksGPU(orbit, report['duration'], report['fs'], batch_size=args.batch_size, minimum=512)
            widths = engine.add(packed)
            actual = engine.finish()
            reference = np.zeros((len(times), 6))
            for row in packed:
                native = lisagwresponse.GalacticBinary(A=float(row[0]), f=float(row[1]), df=float(row[2]),
                    phi0=float(row[3]), iota=float(row[4]), ra=float(row[5]), dec=float(row[6]), psi=float(row[7]),
                    t_init=0., orbits=str(orbit), orbit_interp_order=5, shift_sun='never')
                reference += native.compute_gw_response(times, LINKS)
            selected = actual[engine.padding+positions]
            delta = selected-reference
            per_link = np.linalg.norm(delta, axis=0) / np.linalg.norm(reference, axis=0)
            case = dict(label=label, sources=len(packed), sample_times=len(times), link_order=[int(v) for v in LINKS],
                        per_link_relative_l2=per_link.tolist(), relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(reference)),
                        max_absolute_error=float(np.max(np.abs(delta))), bitwise_equal=bool(np.array_equal(selected, reference)),
                        finite=bool(np.isfinite(actual).all()), envelope_sizes=widths,
                        actual_batches={f'{size}/{batch}': batch for size, batch in engine.prepared})
            if label == 'largest_absolute_chirp':
                error_power = np.zeros(6); reference_power = np.zeros(6); largest_error = 0.
                for first in range(0, engine.count, 65536):
                    last = min(first+65536, engine.count)
                    native_values = native.compute_gw_response((np.arange(first, last)-engine.padding)/report['fs'], LINKS)
                    difference = actual[first:last]-native_values
                    error_power += np.sum(difference**2, axis=0)
                    reference_power += np.sum(native_values**2, axis=0)
                    largest_error = max(largest_error, float(np.max(np.abs(difference))))
                case.update(full_padded_samples=engine.count,
                            full_interval_per_link_relative_l2=np.sqrt(error_power/reference_power).tolist(),
                            full_interval_relative_l2=float(np.sqrt(error_power.sum()/reference_power.sum())),
                            full_interval_max_absolute_error=largest_error)
            case['passed'] = case['finite'] and max(case['per_link_relative_l2']) < report['threshold_relative_l2']
            if 'full_interval_per_link_relative_l2' in case:
                case['passed'] &= max(case['full_interval_per_link_relative_l2']) < report['threshold_relative_l2']
            case['seconds'] = time.perf_counter()-started
            report['cases'].append(case)
            path.write_text(json.dumps(report, indent=2)+'\n')
            print(json.dumps(case), flush=True)
            del engine, actual
        report['status'] = 'passed' if all(case['passed'] for case in report['cases']) else 'failed'
    except Exception as error:
        report.update(status='blocked', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        path.write_text(json.dumps(report, indent=2)+'\n')
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

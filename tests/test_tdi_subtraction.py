"""Exercise the controller's accepted/rejected time-domain subtraction paths."""
import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / 'eric_subtraction_1block.py').read_text())
ACCEPTED = next(node for node in ast.walk(TREE) if isinstance(node, ast.If)
                and isinstance(node.test, ast.Name) and node.test.id == 'accepted'
                and any(isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Name)
                        and item.targets[0].id == 'identified_dwd_parameters' for item in node.body))
FINAL = next(node for node in ast.walk(TREE) if isinstance(node, ast.Assign)
             and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'final_data')


class TimeDomainSubtractionTests(unittest.TestCase):
    def fixture(self, directory, accepted):
        calls = []
        N, N_day, fs = 8, 2, 2 / 86400.
        stream = np.arange(3 * N, dtype=float).reshape(3, N) / 32
        previous = np.full_like(stream, .125)
        original = stream + previous + 1.
        doppler = np.array([[.003123, -2e-17, 1.2, -.3], [.004567, 3e-17, 4., .5]])
        physical = dict(A=np.array([1e-22, 2e-22]), phi0=np.array([.2, .7]),
                        iota=np.array([.9, 1.3]), psi_bme=np.array([.4, 1.1]))

        def synthesize(parameters, config):
            calls.append(('waveform', parameters.copy(), config.copy()))
            return np.zeros((N + 4 * N_day, 6))

        class Response:
            def __init__(self, **settings):
                pass
            def init_response_file(self, saved, dt, size, t0, frame):
                calls.append(('file', dt, size, t0, frame))
                saved.create_dataset('tcb/y', shape=(size, 6))

        class Instrument:
            central_freq = 4.
            def __init__(self, **settings):
                calls.append(('instrument', settings))
            def disable_all_noises(self):
                calls.append(('noises_off',))
            def disable_dopplers(self):
                calls.append(('dopplers_off',))
            def export_numpy_full(self):
                return SimpleNamespace(measurements={'sci_12': np.ones(N + 2 * N_day),
                    'tmi_12': np.ones(N + 2 * N_day), 'ref_12': np.ones(N + 2 * N_day)},
                    args={'fs': fs})

        def tdi(measurements, config, *, keep):
            calls.append(('tdi', tuple(measurements), config, keep))
            return 4 * stream

        scope = dict(accepted=accepted, np=np, h5py=h5py, work=Path(directory),
            N=N, N_day=N_day, fs=fs, T_obs=N / fs, orbit_file=Path('moving_orbit.h5'),
            detector_config=dict(T_obs=N / fs, dt=1 / fs, orbits='moving_orbit.h5',
                                 include_padding=True, gpu_batch_size=10000),
            posterior={'doppler': doppler}, new_coefficients=np.ones((2, 4)),
            physical_amplitudes=lambda coefficients: physical,
            DWD_simulation_gpu=synthesize, tdi_gpu=tdi,
            lisagwresponse=SimpleNamespace(GalacticBinary=Response),
            lisainstrument=SimpleNamespace(Instrument=Instrument),
            pytdi=SimpleNamespace(Data=SimpleNamespace(from_instrument=lambda values: values)),
            accumulated_xyz_td=previous.copy(), total_xyz_td=original.copy(),
            wide_bins=np.array([1, 2, 3]), retained_bins=np.array([2, 3]), core_indices=np.array([1, 2]),
            new_recovery={}, bank_bases=object(), fixed_covariance=object(),
            f_statistic=lambda *args: SimpleNamespace(two_f=np.array([1., 2.])),
            previous_doppler=np.empty((0, 4)), accumulated_xyz_fd_wide=np.zeros((3, 3), complex),
            retained_residual_xyz_fd_wide=np.fft.rfft(original - previous, axis=-1)[:, [1, 2, 3]])
        return scope, calls, stream, previous, original, physical

    def test_accepted_fit_uses_physical_rows_and_subtracts_in_time_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            scope, calls, stream, previous, original, physical = self.fixture(directory, True)
            exec(compile(ast.Module(body=[ACCEPTED], type_ignores=[]), 'controller', 'exec'), scope)
            rows = calls[0][1]
            np.testing.assert_array_equal(rows[:, 0], physical['A'])
            np.testing.assert_array_equal(rows[:, 1:3], scope['posterior']['doppler'][:, :2])
            np.testing.assert_array_equal(rows[:, 5:7], scope['posterior']['doppler'][:, 2:])
            np.testing.assert_array_equal(rows[:, [3, 4, 7]], np.column_stack(
                (physical['phi0'], physical['iota'], physical['psi_bme'])))
            self.assertEqual(rows.shape, (2, 8))
            self.assertEqual(calls[0][2]['orbits'], 'moving_orbit.h5')
            self.assertIn(('noises_off',), calls)
            self.assertIn(('dopplers_off',), calls)
            native = next(item[1] for item in calls if item[0] == 'instrument')
            self.assertEqual(native['size'], 12)
            self.assertEqual(native['t0'], -86400.)
            self.assertNotIn('include_padding', native)
            tdi_call = next(item for item in calls if item[0] == 'tdi')
            self.assertEqual(tdi_call[1], ('sci_12', 'tmi_12', 'ref_12'))
            self.assertEqual(tdi_call[3], slice(2, 10))
            np.testing.assert_array_equal(scope['signal_xyz_td'], previous + stream)
            np.testing.assert_array_equal(scope['residual_xyz_td'], original - previous - stream)
            expected = np.fft.rfft(scope['residual_xyz_td'], axis=-1)[:, [1, 2, 3]]
            np.testing.assert_array_equal(scope['residual_xyz_fd_wide'], expected)
            np.testing.assert_array_equal(scope['total_xyz_td'], original)
            np.testing.assert_array_equal(scope['accumulated_xyz_td'], previous)

    def test_rejected_fit_never_generates_waveforms_or_calls_tdi(self):
        with tempfile.TemporaryDirectory() as directory:
            scope, calls, _, previous, original, _ = self.fixture(directory, False)
            exec(compile(ast.Module(body=[ACCEPTED], type_ignores=[]), 'controller', 'exec'), scope)
            self.assertEqual(calls, [])
            self.assertFalse((Path(directory) / 'recovered_links.h5').exists())
            np.testing.assert_array_equal(scope['accumulated_xyz_td'], previous)
            exec(compile(ast.Module(body=[FINAL], type_ignores=[]), 'controller', 'exec'), scope)
            path = Path(directory) / 'final.h5'
            with h5py.File(path, 'w') as saved:
                for name, values in scope['final_data'].items():
                    saved.create_dataset(name, data=values)
            with h5py.File(path) as saved:
                np.testing.assert_array_equal(saved['signal_xyz_td'][:], previous)
                np.testing.assert_array_equal(saved['residual_xyz_td'][:], original - previous)
                np.testing.assert_array_equal(saved['residual_xyz_fd_wide'][:],
                    np.fft.rfft(original - previous, axis=-1)[:, [1, 2, 3]])


if __name__ == '__main__':
    unittest.main()

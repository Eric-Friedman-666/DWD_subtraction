"""Round coordination and catalogue independence; numerical screening has separate tests."""
import ast
import json
from contextlib import nullcontext
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import h5py
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run_pipeline(orbit_file, snr_file, observation_file, output_file, **settings):
    """Execute the actual sequential script with fixture parameter assignments."""
    source = ROOT / 'eric_subtraction_all.py'
    tree = ast.parse(source.read_text())
    scope = dict(orbit_file=orbit_file, snr_file=snr_file,
                 observation_file=observation_file, output_file=output_file, **settings)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Assign) and 25 <= node.lineno < 39
        and any(isinstance(target, ast.Name) and target.id in scope for target in node.targets))]
    exec(compile(tree, str(source), 'exec'), scope)
    return {**scope['pipeline_result'], 'temporary_directory': str(scope['work'])}


def save_tree(group, values):
    for name, value in values.items():
        if isinstance(value, dict):
            save_tree(group.create_group(name), value)
        else:
            group.create_dataset(name, data=value)


class GlobalRoundTests(unittest.TestCase):
    def run_case(self, directory, bins=(10, 23), *, catalogue_bins=None, rejected=(),
                 wait_for_round=(), neighbor_alias=False, empty_fit=False,
                 subtraction_frequency_range=None, catalogue_relative_snrs=None, catalogue_ids=None):
        directory = Path(directory)
        duration, rate, N = 128000., .001, 128
        catalogue_bins = bins if catalogue_bins is None else catalogue_bins
        fields = ('Amplitude', 'Frequency', 'FrequencyDerivative', 'InitialPhase', 'Inclination',
            'EclipticLongitude', 'EclipticLatitude', 'Polarization', 'SNR_absolute', 'SNR_relative')
        rows = np.zeros((len(catalogue_bins), 1), dtype=[(name, 'f8') for name in fields])
        for i, carrier in enumerate(catalogue_bins):
            rows[i, 0]['Amplitude'], rows[i, 0]['Frequency'] = 1., carrier / duration
            rows[i, 0]['Inclination'], rows[i, 0]['SNR_absolute'] = np.pi / 2, 20.
            rows[i, 0]['SNR_relative'] = 20.
        if catalogue_relative_snrs is not None:
            rows['SNR_relative'][:, 0] = catalogue_relative_snrs
        if catalogue_ids is None:
            catalogue_ids = np.arange(1, len(rows) + 1, dtype=np.int64)
        snr, observation, output = [directory / name for name in ('snr.h5', 'observation.h5', 'result.h5')]
        with h5py.File(snr, 'w') as saved:
            saved.create_dataset('catalogue', data=np.column_stack((catalogue_ids,
                *[rows[field][:, 0] for field in fields])))
            saved.create_dataset('settings/observation_duration_s', data=duration)
            saved.create_dataset('settings/detector_sampling_rate_hz', data=rate)
            saved.create_dataset('status/complete', data=True)
        def wave(carrier):
            if neighbor_alias and carrier in (23, 24):
                original_wave = np.tile(np.sin(2 * np.pi * 10 * np.arange(N) / N), (3, 1))
                original_wave[[1, 2] if carrier == 23 else [0]] = 0
                return original_wave
            return np.tile(np.sin(2 * np.pi * carrier * np.arange(N) / N), (3, 1))
        original = sum((wave(carrier) for carrier in bins), np.zeros((3, N)))
        with h5py.File(observation, 'w') as saved:
            saved.create_dataset('observation/total_xyz_td', data=original)
            saved['observation'].attrs.update(duration=duration, rate=rate)
            saved['observation'].attrs['detector_config'] = json.dumps(dict(
                T_obs=duration, dt=1 / rate, orbits='stored_orbit_location.h5', orbit_dataset='tcb/ltt',
                gpu_batch_size=3, minimum_envelope_samples=1024, include_padding=True,
                aafilter=('kaiser', 222., .2 * rate, .7 * rate),
                clock_offsets=.02, clock_freqoffsets=.03, clock_freqlindrifts=.04,
                clock_freqquaddrifts=.05))
        state = dict(sampler=[], waveform=[], tdi=[], geometry=0, screen=[], search=[], new_signal=None, waveform_config=[], instrument_config=[])
        class Builder:
            def __init__(self, covariance, data, times, geometry, frequency, T_obs, fs):
                self.frequency = frequency
            def build(self, doppler):
                basis = np.zeros((len(doppler), 4, 3, len(self.frequency)), complex)
                indices = np.rint(self.frequency * duration).astype(int)
                for i, point in enumerate(doppler):
                    carrier = int(round(point[0] * duration))
                    basis[i, 0] = np.fft.rfft(wave(carrier), axis=-1)[:, indices]
                    basis[i, 3, 0] = carrier
                return basis
        def geometry(work, frequency, T_obs, fs, **settings):
            state['geometry'] += 1
            return Builder(None, None, None, None, frequency, T_obs, fs), np.array([0.]), None
        def reconstruct(coefficients, bases):
            return np.einsum('sm,smck->sck', coefficients, bases)
        def inner(a, b, *args):
            return np.sum(np.asarray(a).conj() * b, axis=(-2, -1)).real
        def covariance(data, T_obs, fs):
            state['screen'].append(data.copy())
            return np.broadcast_to(np.eye(3), (data.shape[-1], 3, 3)).copy()
        def search(data, covariance, builder, core, T_obs, fs, templates, **settings):
            state['search'].append((templates.copy(), data.copy()))
            self.assertEqual(templates.shape[1], 11)
            indices = np.rint(builder.frequency[core] * duration).astype(int)
            active = indices[np.sum(abs(data)**2, axis=0) > 1.e-6]
            points = templates[np.isin(np.rint(templates[:, 2] * duration).astype(int), active)]
            if any(int(round(f * duration)) in wait_for_round for f in points[:, 2]) and not state['waveform']:
                points = points[:0]
            return dict(template_dwd=points, two_f=np.full(len(points), 100.))
        def statistics(data, bases, covariance, T_obs, fs):
            scores = []
            for basis in bases:
                carrier = int(round(basis[3, 0, 0].real))
                scores.append(0. if carrier in rejected else (100. if np.vdot(basis[0], data).real > 1.e-6 else 0.))
            return dict(two_f=np.asarray(scores))
        def sampler(data, covariance, table, block_range, bounds, center, T_obs, fs, checkpoints, **settings):
            state['sampler'].append((table.copy(), data.copy(), settings, bounds.copy(), block_range.copy()))
            self.assertEqual(table.shape[1], 11)
            doppler = table[:, [2, 3, 6, 7]]
            self.assertFalse(covariance.flags.writeable)
            self.assertEqual(settings['chirp_coordinate'], 'linear')
            if empty_fit:
                doppler = doppler[:0]
            return dict(doppler=doppler.copy(), fit_result=dict(coefficients=np.tile([1., 0., 0., 0.], (len(doppler), 1))), config={})
        def associate(recovered, templates, ids, inverse, T_obs, fs, **settings):
            matched = np.full(len(recovered), -1, dtype=np.int64)
            for i, source in enumerate(recovered):
                if len(templates):
                    overlap = np.array([np.vdot(candidate, source).real / max(np.linalg.norm(candidate) * np.linalg.norm(source), 1.e-300) for candidate in templates])
                    if np.max(overlap) > .9:
                        matched[i] = ids[int(np.argmax(overlap))]
            return dict(matched_dwd_id=matched, best_overlap=np.ones(len(matched)))
        def waveform(parameters, config):
            state['waveform'].append(parameters.copy())
            state['waveform_config'].append(config.copy())
            state['new_signal'] = sum((row[0] * wave(int(round(row[1] * duration))) for row in parameters), np.zeros_like(original))
            return np.zeros((N + 4 * round(86400 * rate), 6))
        class Response:
            def __init__(self, **settings):
                pass
            def init_response_file(self, saved, dt, size, t0, frame):
                saved.create_dataset('tcb/y', shape=(size, 6))
        class Instrument:
            central_freq = 1.
            def __init__(self, **settings):
                state['instrument_config'].append(settings.copy())
            def disable_all_noises(self):
                pass
            def disable_dopplers(self):
                pass
            def export_numpy_full(self):
                return SimpleNamespace(measurements={}, args={})
        def tdi(measurements, config, **settings):
            state['tdi'].append(settings)
            return state['new_signal']
        replacements = {
            'lisagwresponse': dict(GalacticBinary=Response), 'lisainstrument': dict(Instrument=Instrument),
            'pytdi': dict(Data=SimpleNamespace(from_instrument=lambda value: value)),
            'eric_generate_waveform': dict(DWD_simulation_gpu=waveform, tdi_gpu=tdi),
            'eric_rjmcmc': dict(run_eryn_rjmcmc=sampler),
            'script.eric_Fstats': dict(build_response=geometry, noise_inner_product=inner,
                physical_amplitude_coordinates=lambda plus, cross, psi, phi: np.column_stack((plus, np.zeros((len(plus), 3)))),
                physical_amplitudes=lambda coeff: dict(A=coeff[:, 0], phi0=np.zeros(len(coeff)), iota=np.zeros(len(coeff)), psi_bme=np.zeros(len(coeff))), reconstruct_waveform=reconstruct),
            'script.dwd_blind_search': dict(residual_covariance=covariance, template_f_statistic_search=search, amplitude_f_statistic=statistics),
            'script.dwd_bank_association': dict(assign_bank_ids=associate),
            'script.oneblock_observation': dict(read_checkpoint=lambda saved: {}, write_checkpoint=lambda saved, values: None),
            'script.tool': dict(_save_tree=save_tree, single_cpu=nullcontext)}
        modules = {}
        for name, values in replacements.items():
            module = ModuleType(name)
            module.__dict__.update(values)
            modules[name] = module
        with patch.dict(sys.modules, modules):
            result = run_pipeline(directory / 'orbit.h5', snr, observation, output,
                frequency_range=(.00005, .00045), freq_block_width=.0001, sample_number=(1, 2),
                subtraction_frequency_range=subtraction_frequency_range)
        return result, state, original, output

    def test_saved_relative_snr_filters_templates_and_preserves_original_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory,
                bins=(10, 12, 23, 24), catalogue_relative_snrs=(6.9, 7., 7.1, 20.))
            self.assertEqual(result['source_count'], 2)
            first_round_templates = np.concatenate([templates for templates, _ in state['search'][:4]])
            np.testing.assert_array_equal(np.rint(first_round_templates[:, 2] * 128000.), [23, 24])
            tree = ast.parse((ROOT / 'eric_subtraction_all.py').read_text())
            start = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.With)
                and isinstance(node.items[0].context_expr.args[0], ast.Name)
                and node.items[0].context_expr.args[0].id == 'snr_file')
            field_definition = next(node for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'fields' for t in node.targets))
            scope = dict(np=np, h5py=h5py, snr_file=Path(directory) / 'snr.h5', SNR_r_threshold=7.)
            exec(compile(ast.Module(body=[field_definition, tree.body[start]], type_ignores=[]),
                str(ROOT / 'eric_subtraction_all.py'), 'exec'), scope)
            table = scope['template_dwd']
            self.assertEqual(table.shape, (2, 11))
            np.testing.assert_array_equal(table[:, 0], [3, 4])
            with h5py.File(Path(directory) / 'snr.h5') as saved:
                np.testing.assert_array_equal(table, saved['catalogue'][2:])
            with h5py.File(output) as saved:
                np.testing.assert_array_equal(saved['final_subtraction/dwd_id'][:], [3, 4])
                np.testing.assert_array_equal(
                    saved['rounds/01/blocks/00001/screening/search/template_dwd_id'][:], [3, 4])
                np.testing.assert_allclose(saved['final_subtraction/residual_xyz_td'][:],
                    original - saved['final_subtraction/signal_xyz_td'][:], atol=1.e-14)

    def test_no_saved_relative_snr_above_threshold_keeps_original_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory,
                catalogue_relative_snrs=(7., 6.))
            self.assertEqual(result['source_count'], 0)
            self.assertTrue(all(len(templates) == 0 for templates, _ in state['search']))
            self.assertFalse(state['sampler'] or state['waveform'] or state['tdi'])
            with h5py.File(output) as saved:
                np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_td'][:], original)

    def test_ids_are_read_from_the_single_table_without_renumbering(self):
        with tempfile.TemporaryDirectory() as directory:
            _, state, _, output = self.run_case(directory, catalogue_ids=(2111, 29000000))
            np.testing.assert_array_equal(state['search'][0][0][:, 0], [2111])
            with h5py.File(output) as saved:
                np.testing.assert_array_equal(saved['final_subtraction/dwd_id'][:], [2111, 29000000])

    def test_snr_writer_saves_the_numerical_eleven_column_table(self):
        tree = ast.parse((ROOT / 'eric_snr.py').read_text())
        assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in ('catalogue_columns', 'merged_rows') for t in node.targets)]
        create = next(node for node in ast.walk(tree) if isinstance(node, ast.With)
            and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and call.func.attr == 'create_dataset' and call.args
                and isinstance(call.args[0], ast.Constant) and call.args[0].value == 'catalogue'
                for call in ast.walk(node)))
        write = next(node for node in ast.walk(tree) if isinstance(node, ast.With)
            and isinstance(node.items[0].context_expr, ast.Call)
            and isinstance(node.items[0].context_expr.func, ast.Attribute)
            and node.items[0].context_expr.func.attr == 'File'
            and any(isinstance(n, ast.Name) and n.id == 'merged_rows' for n in ast.walk(node)))
        physical_fields = ('Amplitude', 'Frequency', 'FrequencyDerivative', 'InitialPhase',
            'Inclination', 'EclipticLongitude', 'EclipticLatitude', 'Polarization')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'snr.h5'
            rows = np.zeros(3, dtype=[('Name', 'S7')] + [(name, 'f8') for name in
                (*physical_fields, 'SNR_absolute', 'SNR_relative')])
            for i, field in enumerate(rows.dtype.names[1:]):
                rows[field] = np.arange(3) + i / 10.
            with h5py.File(path, 'w') as saved:
                saved.create_group('settings')
                saved.create_dataset('sky/igb/cat', data=rows[:, None])
            scope = dict(np=np, h5py=h5py, output_file=path, source_count=3,
                PARAMETER_FIELDS=physical_fields, dwd_ids=np.arange(3), rows=rows,
                dataset_path='sky/igb/cat', catalogue_offsets={'igb': 0}, group='igb', first=0, last=3)
            nodes = [next(node for node in assignments if node.targets[0].id == 'catalogue_columns'),
                create, next(node for node in assignments if node.targets[0].id == 'merged_rows'), write]
            exec(compile(ast.Module(body=nodes, type_ignores=[]), 'snr_export', 'exec'), scope)
            with h5py.File(path) as saved:
                self.assertEqual(saved['catalogue'].shape, (3, 11))
                np.testing.assert_array_equal(saved['catalogue'][:, 0], [1, 2, 3])
                expected = np.column_stack([rows[name] for name in rows.dtype.names[1:]])
                np.testing.assert_array_equal(saved['catalogue'][:, 1:], expected)
                self.assertEqual(tuple(saved['settings/catalogue_columns'].asstr()[:]),
                    ('ID', *physical_fields, 'SNR_absolute', 'SNR_relative'))

    def test_grid_is_saved_as_contiguous_start_end_pairs(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, _, output = self.run_case(directory, bins=(), catalogue_bins=())
            with h5py.File(output) as saved:
                blocks = saved['config/frequency_blocks_hz'][:]
                self.assertEqual(blocks.shape, (4, 2))
                np.testing.assert_allclose(blocks, [[.00005, .00015], [.00015, .00025],
                    [.00025, .00035], [.00035, .00045]], rtol=0, atol=1e-18)
                np.testing.assert_allclose(blocks[:-1, 1], blocks[1:, 0], rtol=0, atol=1e-18)

    def test_subtraction_reuses_saved_detector_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            _, state, _, _ = self.run_case(directory)
            config = state['waveform_config'][0]
            self.assertEqual(config['gpu_batch_size'], 3)
            self.assertEqual(config['minimum_envelope_samples'], 1024)
            self.assertEqual(config['T_obs'], 128000.)
            self.assertEqual(config['dt'], 1000.)
            self.assertEqual(config['orbits'], str(Path(directory) / 'orbit.h5'))
            instrument = state['instrument_config'][0]
            self.assertEqual(instrument['aafilter'], ['kaiser', 222., .0002, .0007])
            self.assertEqual(instrument['clock_offsets'], .02)
            self.assertEqual(instrument['clock_freqquaddrifts'], .05)

    def test_one_snapshot_and_one_tdi_sum_for_multiple_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory)
            self.assertEqual(result['rounds'], 2)
            self.assertEqual(len(state['sampler']), 2)
            self.assertEqual(len(state['waveform']), 1)
            self.assertEqual(len(state['waveform'][0]), 2)
            self.assertEqual(len(state['tdi']), 1)
            self.assertEqual(state['geometry'], 1)
            with h5py.File(output) as saved:
                first = saved['rounds/01']
                self.assertEqual(first.attrs['visited_block_count'], 4)
                self.assertEqual(len(state['search']), 8)
                for group in first['blocks'].values():
                    self.assertEqual(group.attrs['residual_input_sha256'], first.attrs['residual_input_sha256'])
                np.testing.assert_allclose(saved['final_subtraction/signal_xyz_td'][:], original, atol=1.e-14)
                np.testing.assert_array_equal(saved['final_subtraction/dwd_id'][:], [1, 2])
                np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_fd'][:], np.fft.rfft(saved['final_subtraction/residual_xyz_td'][:], axis=-1))
    def test_sources_without_bank_templates_are_not_proposed(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory, catalogue_bins=())
            self.assertEqual(result['source_count'], 0)
            self.assertFalse(state['sampler'] or state['waveform'] or state['tdi'])
            with h5py.File(output) as saved:
                self.assertNotIn('catalogue_informed_bank', saved.attrs)
                np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_td'][:], original)

    def test_catalogue_predictions_cannot_trigger_an_empty_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory, bins=(), catalogue_bins=(10, 23))
            self.assertEqual(result['source_count'], 0)
            self.assertEqual(len(state['screen']), 1)
            self.assertFalse(state['sampler'] or state['waveform'] or state['tdi'])
            self.assertEqual(len(state['search']), 4)
            with h5py.File(output) as saved:
                self.assertEqual(saved['rounds/01'].attrs['visited_block_count'], 4)
                np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_td'][:], original)
    def test_rejected_fit_does_not_stop_other_blocks(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory, rejected=(10,))
            self.assertEqual(result['source_count'], 1)
            self.assertEqual(len(state['tdi']), 1)
            with h5py.File(output) as saved:
                self.assertFalse(saved['rounds/01/blocks/00000'].attrs['accepted_for_subtraction'])
                self.assertTrue(saved['rounds/01/blocks/00001'].attrs['accepted_for_subtraction'])
                np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_td'][:], original - saved['final_subtraction/signal_xyz_td'][:])
    def test_all_rejected_fits_preserve_original(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory, rejected=(10, 23))
            self.assertEqual(result['stopping_reason'], 'no_accepted_blocks')
            self.assertFalse(state['waveform'] or state['tdi'])
            with h5py.File(output) as saved:
                np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_td'][:], original)
    def test_sampler_zero_source_model_preserves_the_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory, empty_fit=True)
            self.assertEqual(result['source_count'], 0)
            self.assertFalse(state['tdi'] or state['waveform'])
            with h5py.File(output) as saved:
                np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_td'][:], original)
    def test_blocks_are_rechecked_after_the_global_residual_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, _, output = self.run_case(directory, wait_for_round=(23,))
            self.assertEqual(result['rounds'], 3)
            self.assertEqual(len(state['tdi']), 2)
            self.assertEqual(len(state['screen']), 3)
            with h5py.File(output) as saved:
                self.assertEqual(saved['rounds/01/block_status'][1], b'no_new_candidates')
                self.assertEqual(saved['rounds/02/block_status'][1], b'accepted')
                self.assertIn('00001', saved['blocks'])
    def test_requested_range_keeps_global_indices_and_skips_other_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, original, output = self.run_case(directory,
                subtraction_frequency_range=(.00015, .00025))
            self.assertEqual(result['selected_block_indices'], [1])
            self.assertEqual(result['total_block_count'], 4)
            self.assertEqual(result['source_count'], 1)
            self.assertEqual(len(state['sampler']), 1)
            np.testing.assert_array_equal(state['sampler'][0][3], [-1.e-10, 1.e-10])
            np.testing.assert_allclose(state['sampler'][0][4], [.00015, .00025], rtol=0, atol=1e-18)
            np.testing.assert_array_equal(state['search'][0][0][:, 2], [23 / 128000.])
            np.testing.assert_array_equal(state['search'][0][0][:, 0], [2])
            with h5py.File(output) as saved:
                np.testing.assert_array_equal(saved['config/block_indices'][:], np.arange(4))
                np.testing.assert_array_equal(saved['config/selected_block_indices'][:], [1])
                np.testing.assert_array_equal(saved['config/subtraction_frequency_range'][:], [.00015, .00025])
                np.testing.assert_array_equal(saved['final_subtraction/block_index'][:], [1])
                self.assertEqual(list(saved['blocks']), ['00001'])
                self.assertEqual(saved['blocks/00001/block_index'][()], 1)
                np.testing.assert_array_equal(saved['blocks/00001/frequency_range'][:], saved['config/frequency_blocks_hz'][1])
                np.testing.assert_array_equal(saved['rounds/01/block_status'][:],
                    [b'outside_subtraction_range', b'accepted', b'outside_subtraction_range', b'outside_subtraction_range'])
                self.assertEqual(saved['rounds/01'].attrs['visited_block_count'], 1)
                self.assertEqual(saved['rounds/01'].attrs['selected_block_count'], 1)
                # The source in block 0 remains in the shared residual.
                outside = np.tile(np.sin(2 * np.pi * 10 * np.arange(128) / 128), (3, 1))
                np.testing.assert_allclose(saved['final_subtraction/residual_xyz_td'][:], outside, atol=1.e-14)
                np.testing.assert_array_equal(saved['final_subtraction/residual_xyz_td'][:],
                    original - saved['final_subtraction/signal_xyz_td'][:])
    def test_exact_block_edges_do_not_select_floating_point_slivers(self):
        with tempfile.TemporaryDirectory() as directory:
            result, state, _, _ = self.run_case(directory, subtraction_frequency_range=(.00015, .00025))
            self.assertEqual(result['selected_block_indices'], [1])
            self.assertEqual(len(state['sampler']), 1)
    def test_source_range_is_lower_inclusive_and_upper_exclusive(self):
        bounds = (.00005, .00015)
        with tempfile.TemporaryDirectory() as directory:
            result, _, _, output = self.run_case(directory, subtraction_frequency_range=bounds)
            self.assertEqual(result['selected_block_indices'], [0])
            self.assertEqual(result['source_count'], 1)
            with h5py.File(output) as saved:
                frequencies = saved['final_subtraction/parameters'][:, 1]
                np.testing.assert_array_equal(frequencies, [10 / 128000.])
                np.testing.assert_array_equal(saved['final_subtraction/block_index'][:], [0])
                for group in saved['blocks'].values():
                    limits = group['frequency_range'][:]
                    self.assertTrue(limits[0] >= bounds[0] - 1e-18 and limits[1] <= bounds[1] + 1e-18)
    def test_changing_range_does_not_renumber_the_grid(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first, _, _, output_a = self.run_case(a, subtraction_frequency_range=(.00015, .00025))
            second, state, original, output_b = self.run_case(b, subtraction_frequency_range=(.00025, .00045))
            self.assertEqual(first['selected_block_indices'], [1])
            self.assertEqual(second['selected_block_indices'], [2, 3])
            self.assertFalse(state['sampler'] or state['waveform'] or state['tdi'])
            with h5py.File(output_a) as left, h5py.File(output_b) as right:
                np.testing.assert_array_equal(left['config/frequency_blocks_hz'][:], right['config/frequency_blocks_hz'][:])
                np.testing.assert_array_equal(left['config/block_indices'][:], right['config/block_indices'][:])
                np.testing.assert_array_equal(right['final_subtraction/residual_xyz_td'][:], original)
    def test_default_and_explicit_full_range_select_all_blocks(self):
        for bounds in (None, (0., .0005)):
            with self.subTest(bounds=bounds), tempfile.TemporaryDirectory() as directory:
                result, _, _, output = self.run_case(directory, subtraction_frequency_range=bounds)
                self.assertEqual(result['selected_block_indices'], [0, 1, 2, 3])
                with h5py.File(output) as saved:
                    np.testing.assert_array_equal(saved['config/block_indices'][:], np.arange(4))
                    np.testing.assert_array_equal(saved['config/selected_block_indices'][:], np.arange(4))
    def test_signed_chirp_and_legacy_coordinate_round_trips(self):
        tree = ast.parse((ROOT / 'eric_rjmcmc.py').read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('encode_doppler', 'decode_doppler')]
        scope = dict(np=np)
        exec(compile(ast.Module(body=functions, type_ignores=[]), 'coordinates', 'exec'), scope)
        points = np.array([[.003, -2.e-15, 1., -.2], [.004, 0., 2., .1], [.005, 2.e-12, 3., .4]])
        value = scope['encode_doppler'](points, .004, 86400., chirp_coordinate='linear')
        np.testing.assert_allclose(scope['decode_doppler'](value, .004, 86400., chirp_coordinate='linear'), points, rtol=1.e-14, atol=1.e-28)
        table = np.zeros(len(points), dtype=[('ID', 'i8')] + [(field, 'f8') for field in
            ('Amplitude', 'Frequency', 'FrequencyDerivative', 'InitialPhase', 'Inclination',
             'EclipticLongitude', 'EclipticLatitude', 'Polarization', 'SNR_absolute', 'SNR_relative')])
        table['ID'] = [2111, 3001, 29000000]
        for i, field in enumerate(('Frequency', 'FrequencyDerivative', 'EclipticLongitude', 'EclipticLatitude')):
            table[field] = points[:, i]
        unchanged = table.copy()
        np.testing.assert_array_equal(scope['encode_doppler'](table, .004, 86400., chirp_coordinate='linear'), value)
        np.testing.assert_array_equal(table, unchanged)
        matrix = np.column_stack([table[field] for field in table.dtype.names])
        before = matrix.copy()
        np.testing.assert_array_equal(scope['encode_doppler'](matrix, .004, 86400., chirp_coordinate='linear'), value)
        np.testing.assert_array_equal(matrix, before)
        positive = points[2:]
        old = np.column_stack((np.log10(positive[:, 0] / .004) * .004 * 86400. * np.log(10.), np.log10(positive[:, 1]), positive[:, 2] % (2 * np.pi), np.sin(positive[:, 3])))
        np.testing.assert_array_equal(scope['encode_doppler'](positive, .004, 86400.), old)


if __name__ == '__main__':
    unittest.main()

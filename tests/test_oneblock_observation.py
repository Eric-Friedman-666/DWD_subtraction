"""HDF5 format and legacy-reader integration checks; no simulation or sampler."""
import ast
import os
from pathlib import Path
import shutil
import tempfile
import unittest

import h5py
import numpy as np

from script.oneblock_observation import (
    CACHED_FILES, PARAMETERS, PARAMETER_FIELDS, pack_observation, prepare_cached_inputs,
)


class ObservationBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.legacy = self.root / 'legacy'
        self.legacy.mkdir()
        self.output = self.root / 'dwd_observation.h5'
        self.total = np.arange(36, dtype=np.float64).reshape(3, 12)
        for name in CACHED_FILES:
            with h5py.File(self.legacy / name, 'x') as saved:
                saved.attrs['complete'] = True
                saved.attrs['source'] = name
        with h5py.File(self.legacy / 'observation.h5', 'a') as saved:
            saved.attrs.update(duration=120., rate=.1)
            saved.create_dataset('total_xyz_td', data=self.total, compression='gzip', chunks=(3, 4))
            saved.create_dataset('noise_xyz_td', data=np.ones_like(self.total))
        with h5py.File(self.legacy / 'population.h5', 'a') as saved:
            saved.create_dataset('dwd_id', data=[101, 202])
            for name in PARAMETERS:
                saved.create_dataset('parameters/' + name, data=[.1, .2])
        with h5py.File(self.legacy / 'orbit.h5', 'a') as saved:
            saved.attrs.update(dt=86400., t0=-2e7, size=4)
            saved.create_dataset('tcb/ltt', data=np.arange(24.).reshape(4, 6))
        with h5py.File(self.legacy / 'population_links.h5', 'a') as saved:
            saved.create_dataset('tcb/y', data=np.arange(72.).reshape(12, 6))
        with h5py.File(self.legacy / 'population_tdi.h5', 'a') as saved:
            saved.create_dataset('signal_xyz_td', data=self.total - 1)

    def test_pack_preserves_datasets_attributes_and_filters(self):
        pack_observation(self.legacy, self.output)
        with h5py.File(self.output, 'r') as saved:
            self.assertEqual(set(saved), {Path(name).stem for name in CACHED_FILES})
            self.assertTrue(saved.attrs['complete'])
            self.assertEqual(saved['orbit'].attrs['dt'], 86400.)
            self.assertEqual(saved['observation'].attrs['source'], 'observation.h5')
            np.testing.assert_array_equal(saved['observation/total_xyz_td'], self.total)
            self.assertEqual(saved['observation/total_xyz_td'].compression, 'gzip')
            self.assertEqual(saved['observation/total_xyz_td'].chunks, (3, 4))

    def test_relocated_bundle_is_independent_of_original_files(self):
        with h5py.File(self.legacy / 'external.h5', 'x') as other:
            other.create_dataset('data', data=[9., 8., 7.])
        with h5py.File(self.legacy / 'population_tdi.h5', 'a') as saved:
            saved['external'] = h5py.ExternalLink('external.h5', '/data')
            saved['signal_alias'] = h5py.SoftLink('/signal_xyz_td')
            refs = saved.create_dataset('refs', (1,), dtype=h5py.ref_dtype)
            refs[0] = saved['signal_xyz_td'].ref
        pack_observation(self.legacy, self.output)
        shutil.rmtree(self.legacy)
        moved = self.root / 'renamed' / 'measured_xyz.h5'
        moved.parent.mkdir()
        self.output.rename(moved)
        with h5py.File(moved, 'r') as saved:
            np.testing.assert_array_equal(saved['population_tdi/external'], [9., 8., 7.])
            np.testing.assert_array_equal(saved['population_tdi/signal_alias'], self.total - 1)
            np.testing.assert_array_equal(saved[saved['population_tdi/refs'][0]], self.total - 1)
        work = self.root / 'work'
        work.mkdir()
        prepare_cached_inputs(moved, work)
        with h5py.File(work / 'observation.h5', 'r') as observation:
            np.testing.assert_array_equal(observation['total_xyz_td'], self.total)
            self.assertTrue(observation.attrs['complete'])
        # These are the exact root paths used by the response and truth readers.
        with h5py.File(work / 'orbit.h5', 'r') as orbit, h5py.File(work / 'population.h5', 'r') as population:
            self.assertEqual(orbit['tcb/ltt'].shape, (4, 6))
            self.assertEqual(orbit.attrs['size'], 4)
            np.testing.assert_array_equal(population['dwd_id'], [101, 202])
            self.assertEqual(population['parameters/f'].shape, (2,))
        self.assertLess(sum(path.stat().st_size for path in work.iterdir()), 100_000)

    def test_exclusive_publication_and_broken_symlink(self):
        pack_observation(self.legacy, self.output)
        original = self.output.read_bytes()
        with self.assertRaises(FileExistsError):
            pack_observation(self.legacy, self.output)
        self.assertEqual(self.output.read_bytes(), original)
        broken = self.root / 'broken.h5'
        broken.symlink_to(self.root / 'missing.h5')
        with self.assertRaises(FileExistsError):
            pack_observation(self.legacy, broken)
        self.assertTrue(broken.is_symlink())



    def test_controller_reads_the_configured_bundle_without_a_parent_cache(self):
        pack_observation(self.legacy, self.output)
        shutil.rmtree(self.legacy)
        primary = Path(__file__).resolve().parents[1] / 'eric_subtraction_1block.py'
        tree = ast.parse(primary.read_text())
        nodes = [node for node in ast.walk(tree)
                 if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                     and isinstance(node.value.func, ast.Name)
                     and node.value.func.id == 'prepare_cached_inputs')
                 or (isinstance(node, ast.With) and any(
                     isinstance(child, ast.Assign) and any(
                         isinstance(target, ast.Name) and target.id == 'total_xyz_td'
                         for target in child.targets) for child in node.body))]
        nodes.sort(key=lambda node: node.lineno)
        work = self.root / 'workspace'
        work.mkdir()
        scope = dict(observation_file=self.output, work=work, cached_files=CACHED_FILES,
                     prepare_cached_inputs=prepare_cached_inputs, h5py=h5py,
                     freq_block_range=np.array([.003,.004]),T_obs=120.)
        for node in nodes:
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(primary), 'exec'), scope)
        np.testing.assert_array_equal(scope['total_xyz_td'], self.total)

    def test_controller_selects_saved_snr_and_preserves_observation_row_ids(self):
        fields = dict(A='Amplitude',f='Frequency',df='FrequencyDerivative',phi0='InitialPhase',
                      iota='Inclination',longitude='EclipticLongitude',latitude='EclipticLatitude',psi_bme='Polarization')
        dtype = [(field,'f8') for field in fields.values()] + [('SNR_absolute','f8'),('SNR_relative','f8')]
        records = np.zeros(4,dtype=dtype)
        records['Frequency'] = [.00315,.00315,.0032,.035]
        records['SNR_absolute'] = [8,7,9,10]
        records['SNR_relative'] = [1,9,8,9]
        snr_file = self.root/'snr.h5'
        with h5py.File(snr_file,'w') as saved:
            saved.create_dataset('sky/dgb/cat',data=records[:3,None])
            saved.create_dataset('sky/igb/cat',data=records[3:,None])
            saved.create_dataset('settings/observation_duration_s',data=120.)
            saved.create_dataset('settings/detector_sampling_rate_hz',data=.1)
            saved.create_dataset('status/complete',data=True)
        primary = Path(__file__).resolve().parents[1]/'eric_subtraction_1block.py'
        body = ast.parse(primary.read_text()).body
        start = next(i for i,node in enumerate(body) if isinstance(node,ast.With)
                     and 'sky/dgb/cat' in ast.dump(node))
        end = next(i for i,node in enumerate(body) if isinstance(node,ast.Assign)
                   and any(isinstance(target,ast.Name) and target.id=='bank_doppler' for target in node.targets))
        scope = dict(snr_file=snr_file,h5py=h5py,np=np,T_obs=120.,fs=.1,SNR_a_threshold=7.,
                     freq_block_range=np.array([.0031,.0032]),parameter_fields=fields)
        code = compile(ast.Module(body=body[start:end+1],type_ignores=[]),str(primary),'exec')
        exec(code,scope)
        np.testing.assert_array_equal(scope['bank_ids'],[1,3])
        np.testing.assert_array_equal(scope['target_dwd_index'],[0,2])
        np.testing.assert_array_equal(scope['target_dwd']['SNR_relative'],[1,8])
        with h5py.File(snr_file,'a') as saved:
            saved['status/complete'][()] = False
        with self.assertRaisesRegex(ValueError,'complete and match'):
            exec(code,scope)


    def test_controller_rerun_replaces_previous_output(self):
        pack_observation(self.legacy, self.output)
        original_input = self.output.read_bytes()
        primary = Path(__file__).resolve().parents[1] / 'eric_subtraction_1block.py'
        tree = ast.parse(primary.read_text())
        initialization = next(node for node in ast.walk(tree)
                              if isinstance(node, ast.With) and any(
                                  isinstance(item.context_expr, ast.Call)
                                  and len(item.context_expr.args) > 1
                                  and isinstance(item.context_expr.args[0], ast.Name)
                                  and item.context_expr.args[0].id == 'output_file'
                                  and isinstance(item.context_expr.args[1], ast.Constant)
                                  and item.context_expr.args[1].value == 'w'
                                  for item in node.items))
        # Exercise the real open mode, rounds group, and completion initialization.
        # Numerical payload serialization and GPU inference are outside this check.
        stage = ast.copy_location(ast.With(items=initialization.items,
                                          body=initialization.body[1:],
                                          type_comment=None), initialization)
        code = compile(ast.Module(body=[stage], type_ignores=[]), str(primary), 'exec')
        for case in ('complete', 'incomplete', 'symlink', 'broken_symlink'):
            with self.subTest(case=case):
                result = self.root / (case + '.h5')
                target = self.root / (case + '_target.h5')
                if case in ('symlink', 'broken_symlink'):
                    result.symlink_to(target)
                    storage = target
                else:
                    storage = result
                if case != 'broken_symlink':
                    with h5py.File(storage, 'w') as saved:
                        saved.create_dataset('stale_round', data=[99])
                        saved.attrs.update(complete=case != 'incomplete',
                                           single_file_complete=case != 'incomplete',
                                           last_accepted_round=9, old_configuration='old')
                for repetition in range(2):
                    exec(code, dict(output_file=result, h5py=h5py))
                    with h5py.File(result, 'r') as saved:
                        self.assertEqual(set(saved), {'rounds'})
                        self.assertFalse(saved.attrs['complete'])
                        self.assertFalse(saved.attrs['single_file_complete'])
                        self.assertEqual(saved.attrs['last_accepted_round'], 0)
                        self.assertNotIn('old_configuration', saved.attrs)
                    if repetition == 0:
                        with h5py.File(result, 'a') as saved:
                            saved.create_dataset('stale_round', data=[101])
                            saved.attrs.update(complete=True, single_file_complete=True,
                                               last_accepted_round=1)
                if case in ('symlink', 'broken_symlink'):
                    self.assertTrue(result.is_symlink())
        self.assertEqual(self.output.read_bytes(), original_input)

    def simulation_nodes(self):
        """Return the executable top-level simulation steps."""
        primary = Path(__file__).resolve().parents[1] / 'eric_simulation_DWD.py'
        return ast.parse(primary.read_text()).body

    def simulation_step(self, step, scope):
        """Run one real step without importing the native GPU libraries."""
        primary = Path(__file__).resolve().parents[1] / 'eric_simulation_DWD.py'
        source = primary.read_text()
        starts = {int(line.strip().split('.')[0].split()[-1]): index
                  for index, line in enumerate(source.splitlines(), 1)
                  if line.strip().startswith('# Step ')}
        end = starts.get(step + 1, len(source.splitlines()) + 1)
        nodes = [node for node in self.simulation_nodes()
                 if starts[step] < node.lineno < end]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(primary), 'exec'), scope)

    def simulation_finalize(self, scope):
        """Exercise the real full-duration TD save and publication."""
        primary = Path(__file__).resolve().parents[1] / 'eric_simulation_DWD.py'
        nodes = self.simulation_nodes()
        first = next(i for i, node in enumerate(nodes)
                     if isinstance(node, ast.With) and 'population_tdi.h5' in ast.dump(node.items[0].context_expr))
        exec(compile(ast.Module(body=nodes[first-1:], type_ignores=[]), str(primary), 'exec'), scope)

    def test_simulation_has_direct_full_duration_steps_without_main_guard_or_cli(self):
        primary = Path(__file__).resolve().parents[1] / 'eric_simulation_DWD.py'
        source = primary.read_text()
        tree = ast.parse(source)
        functions = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
        self.assertEqual(functions, [])
        gpu_stage = next(node for node in self.simulation_nodes()
                         if isinstance(node, ast.Assign) and any(
                             isinstance(target, ast.Name) and target.id == 'links'
                             for target in node.targets))
        self.assertEqual(gpu_stage.value.func.id, 'DWD_simulation_gpu')
        self.assertNotIn('serial_scheduler', source)
        self.assertNotIn('cfgscheduler=', source)
        self.assertFalse(any(isinstance(node, (ast.AsyncFunctionDef, ast.ClassDef,
                                              ast.Lambda, ast.Raise))
                             for node in ast.walk(tree)))
        for name in ('argparse', 'parse_args', 'noise_reference_file'):
            self.assertNotIn(name, source)
        labels = [line.strip().split('.')[0] for line in source.splitlines()
                  if line.strip().startswith('# Step ')]
        self.assertEqual(labels, [f'# Step {step}' for step in range(1, 6)])
        self.assertNotIn('np.fft.', source)
        self.assertNotIn('__name__', source)
        self.assertNotIn('__main__', source)
        self.assertNotIn('ProcessPoolExecutor', source)
        self.assertNotIn('simulate_one_day', source)
        self.assertNotIn('multiprocessing', source)
        self.assertTrue(any(isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == 'workspace'
            for target in node.targets) for node in tree.body))

    def test_generator_regenerates_and_replaces_existing_observations(self):
        import types
        primary = Path(__file__).resolve().parents[1] / 'eric_simulation_DWD.py'
        source = primary.read_text()
        tree = ast.parse(source)
        setup_start = next(node.lineno for node in self.simulation_nodes()
                           if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == 'workspace'
                                   for target in node.targets))
        setup_end = next(index for index, line in enumerate(source.splitlines(), 1)
                         if line.strip().startswith('# Step 3.'))
        setup = [node for node in self.simulation_nodes() if setup_start <= node.lineno < setup_end]
        for kind in ('complete', 'incomplete', 'plain_file', 'symlink', 'broken_symlink'):
            with self.subTest(kind=kind):
                output = self.root / (kind + '.h5')
                target = self.root / (kind + '_target.h5')
                target_bytes = None
                if kind == 'complete':
                    pack_observation(self.legacy, output)
                    with h5py.File(output, 'a') as saved:
                        saved.create_dataset('old_run', data=[1])
                        saved.attrs['old_run'] = True
                elif kind == 'incomplete':
                    with h5py.File(output, 'x') as saved:
                        saved.attrs['complete'] = False
                        saved.create_dataset('old_run', data=[1])
                elif kind == 'plain_file':
                    output.write_bytes(b'old non-HDF5 output')
                else:
                    if kind == 'symlink':
                        pack_observation(self.legacy, target)
                        target_bytes = target.read_bytes()
                    output.symlink_to(target)
                for generation in (1, 2):
                    scope = dict(
                        observation_file=output, Path=Path, tempfile=tempfile,
                        T_obs=120., fs=.1, gpu_batch_size=10000, minimum_envelope_samples=512, SchedulerConfigSerial=lambda **kwargs: kwargs,
                    )
                    exec(compile(ast.Module(body=setup, type_ignores=[]),
                                 str(primary), 'exec'), scope)
                    workspace = scope['workspace']
                    self.assertTrue(workspace.is_dir())
                    for name in CACHED_FILES:
                        shutil.copy2(self.legacy / name, workspace / name)
                    (workspace / 'observation.h5').unlink()
                    (workspace / 'population_tdi.h5').unlink()
                    total = self.total + generation * 10
                    noise = np.full_like(total, generation)
                    scope.update(
                        source_count=2, N_acc=3.e-15, delta_x=15.e-12,
                        tdi_results=dict(total=total, noise=noise, signal=total-noise),
                        signal_conversion_runtime=0., started=0., time=__import__('time'),
                        h5py=h5py, np=np, json=__import__('json'),
                        importlib=types.SimpleNamespace(metadata=types.SimpleNamespace(
                            version=lambda name: 'test-version')),
                        pack_observation=pack_observation, shutil=shutil,
                    )
                    self.simulation_finalize(scope)
                    self.assertFalse(workspace.exists())
                    self.assertFalse(output.is_symlink())
                    with h5py.File(output, 'r') as saved:
                        self.assertNotIn('old_run', saved)
                        self.assertNotIn('old_run', saved.attrs)
                        np.testing.assert_array_equal(saved['observation/total_xyz_td'], total)
                        np.testing.assert_array_equal(saved['observation/noise_xyz_td'], noise)
                    if target_bytes is not None:
                        self.assertEqual(target.read_bytes(), target_bytes)
                    elif kind == 'broken_symlink':
                        self.assertFalse(target.exists())


    def test_generator_publishes_complete_observation_and_companion_orbit(self):
        import types
        workspace = self.root / 'work'
        shutil.copytree(self.legacy, workspace)
        (workspace/'observation.h5').unlink()
        (workspace/'population_tdi.h5').unlink()
        scope = dict(workspace=workspace, observation_file=self.output,
                     source_count=2, N=12, N_day=4, T_obs=120., fs=.1,
                     detector_config=dict(T_obs=120., dt=10., orbits=str(workspace/'orbit.h5')),
                     tdi_results=dict(total=self.total, noise=np.ones_like(self.total), signal=self.total-1),
                     signal_conversion_runtime=0., started=0., time=__import__('time'),
                     N_acc=3.e-15, delta_x=15.e-12, h5py=h5py, np=np,
                     json=__import__('json'), importlib=types.SimpleNamespace(
                         metadata=types.SimpleNamespace(version=lambda name:'test-version')),
                     pack_observation=pack_observation, shutil=shutil)
        self.simulation_finalize(scope)
        self.assertEqual(sorted(path.name for path in self.root.iterdir()),
                         ['dwd_observation.h5','dwd_observation.orbits.h5','legacy'])
        with h5py.File(self.output,'r') as saved:
            np.testing.assert_array_equal(saved['observation/total_xyz_td'],self.total)
            self.assertEqual(saved['observation'].id,saved['population_tdi'].id)
            self.assertEqual(saved.attrs['data_domain'],'time')
            self.assertNotIn('frequency',saved['observation'])
            self.assertFalse(any(name.endswith('_xyz_fd') for name in saved['observation']))
            self.assertNotIn('noise_seed', saved['observation'].attrs)
            self.assertNotIn('noise_xyz_td', saved['population'])

    def test_simulation_failure_retains_intermediates_without_publishing(self):
        workspace = self.root / 'work'
        shutil.copytree(self.legacy, workspace)
        def failed_simulation(*args, **kwargs):
            raise RuntimeError('controlled simulation failure')
        primary = Path(__file__).resolve().parents[1] / 'eric_simulation_DWD.py'
        node = next(node for node in self.simulation_nodes()
                    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 'links')
        with self.assertRaisesRegex(RuntimeError,'controlled simulation failure'):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(primary), 'exec'), dict(
                catalogue=np.empty((1, 8)), detector_config={},
                DWD_simulation_gpu=failed_simulation,
            ))
        self.assertFalse(self.output.exists())
        self.assertTrue(workspace.exists())
        self.assertEqual((workspace/'population.h5').read_bytes(),
                         (self.legacy/'population.h5').read_bytes())

    def test_full_catalogue_truth_view_selects_only_local_audit_rows(self):
        with h5py.File(self.legacy/'population.h5','a') as saved:
            saved.attrs.clear()
            del saved['dwd_id'], saved['parameters']
            records = np.zeros(2,dtype=[('ID','i8')]+[(field,'f8') for field in PARAMETER_FIELDS.values()])
            records['ID'] = [101,202]
            records['Frequency'] = [.00329225,.02]
            saved.create_dataset('catalogues/all',data=records)
        with h5py.File(self.legacy/'population_links.h5','a') as saved:
            saved.attrs['processed_sources'] = 2
        with h5py.File(self.legacy/'observation.h5','a') as saved:
            saved.attrs['injected_dwd_count'] = 2
        pack_observation(self.legacy,self.output)
        work = self.root/'work'; work.mkdir()
        prepare_cached_inputs(self.output,work,frequency_range=[.003287,.003297],duration=360*86400.)
        with h5py.File(work/'population.h5') as saved:
            np.testing.assert_array_equal(saved['catalogues/all']['ID'],[101])
            self.assertEqual(saved.attrs['full_injected_source_count'],2)
            self.assertEqual(list(saved.keys()), ['catalogues'])
        with h5py.File(self.output) as saved:
            self.assertEqual(len(saved['population/catalogues/all']),2)
            np.testing.assert_array_equal(saved['observation/total_xyz_td'],self.total)

    def test_generator_reads_every_row_from_both_catalogues(self):
        import time
        catalogue = self.root/'catalogue.h5'
        field_names = dict(A='Amplitude',f='Frequency',df='FrequencyDerivative',phi0='InitialPhase',
                           iota='Inclination',longitude='EclipticLongitude',latitude='EclipticLatitude',psi_bme='Polarization')
        with h5py.File(catalogue,'x') as saved:
            for kind in ('dgb','igb'):
                dtype = [('Name','i8' if kind=='dgb' else 'S7')]+[(name,'f4' if name=='FrequencyDerivative' and kind=='igb' else 'f8') for name in field_names.values()]
                rows = np.zeros((2,1),dtype=dtype)
                rows['Name'][:,0] = [101,7] if kind=='dgb' else [b'first',b'second']
                rows['Frequency'][:,0] = [.0001,.04]
                rows['Amplitude'][:,0] = [1.e-22,2.e-22]
                rows['FrequencyDerivative'][:,0] = [1.e-20,-1.e-20]
                saved.create_dataset('sky/'+kind+'/cat',data=rows).attrs['description'] = kind + ' original catalogue'
        received = {}
        def stop_before_gpu(table, detector_config):
            received.update(table=table.copy(), config=detector_config.copy())
            raise RuntimeError('GPU belongs to the separate real-device smoke test')
        work = self.root/'all_rows'; work.mkdir()
        scope = dict(Path=Path,h5py=h5py,np=np,catalogue_file=catalogue,workspace=work,
            observation_file=self.output,parameter_fields=field_names,T_obs=120.,fs=.1,
            L=2.5e9,time=time,
            orbit_file=work/'orbit.h5',gpu_batch_size=4,minimum_envelope_samples=512,
            N_acc=3.e-15,delta_x=15.e-12,DWD_simulation_gpu=stop_before_gpu,
            tqdm=lambda it,**kw:it, detector_config=dict(T_obs=120., dt=10., orbits=str(work/'orbit.h5'), gpu_batch_size=4, minimum_envelope_samples=512, include_padding=True))
        with self.assertRaisesRegex(RuntimeError,'separate real-device smoke test'):
            self.simulation_step(4, scope)
        with h5py.File(work/'population.h5') as saved,h5py.File(catalogue) as source:
            self.assertEqual(dict(saved.attrs), {})
            self.assertEqual(list(saved.keys()), ['catalogues'])
            table = saved['catalogues/all']
            self.assertEqual(table.shape, (4,))
            self.assertEqual(dict(table.attrs), {})
            self.assertEqual(table.dtype.names, ('ID', *field_names.values()))
            np.testing.assert_array_equal(table['ID'],[1,2,3,4])
            self.assertEqual(len(np.unique(table['ID'])),4)
            for field in field_names.values():
                expected = np.concatenate((source['sky/dgb/cat'][field][:,0], source['sky/igb/cat'][field][:,0]))
                np.testing.assert_array_equal(table[field], expected)
            self.assertEqual(table.dtype['FrequencyDerivative'], np.dtype('f8'))
        np.testing.assert_array_equal(received['table']['ID'], [1, 2, 3, 4])
        self.assertEqual(received['config']['orbits'], str(work/'orbit.h5'))
        self.assertTrue(received['config']['include_padding'])
        self.assertEqual(received['config']['gpu_batch_size'], 4)

        # Read corresponding saved SNR rows through the production bank selector.
        snr_file = work/'snr.h5'
        with h5py.File(snr_file,'x') as saved,h5py.File(catalogue) as source:
            for kind in ('dgb','igb'):
                raw = source['sky/'+kind+'/cat'][:]
                rows = np.empty((len(raw),1),dtype=[(field,'f8') for field in field_names.values()]
                                + [('SNR_absolute','f8'),('SNR_relative','f8')])
                for field in field_names.values(): rows[field] = raw[field]
                rows['SNR_absolute'],rows['SNR_relative'] = 10.,8.
                saved.create_dataset('sky/'+kind+'/cat',data=rows)
            saved.create_dataset('settings/observation_duration_s',data=120.)
            saved.create_dataset('settings/detector_sampling_rate_hz',data=.1)
            saved.create_dataset('status/complete',data=True)
        primary = Path(__file__).resolve().parents[1]/'eric_subtraction_1block.py'
        body = ast.parse(primary.read_text()).body
        start = next(i for i,node in enumerate(body) if isinstance(node,ast.With)
                     and 'sky/dgb/cat' in ast.dump(node))
        end = next(i for i,node in enumerate(body) if isinstance(node,ast.Assign)
                   and any(isinstance(target,ast.Name) and target.id=='bank_doppler' for target in node.targets))
        bank = dict(snr_file=snr_file,h5py=h5py,np=np,T_obs=120.,fs=.1,SNR_a_threshold=7.,
                    freq_block_range=np.array([.00005,.0002]),parameter_fields=field_names)
        exec(compile(ast.Module(body=body[start:end+1],type_ignores=[]),str(primary),'exec'),bank)
        np.testing.assert_array_equal(bank['bank_ids'],[1,3])
        np.testing.assert_array_equal(bank['bank_parameters']['f'],[.0001,.0001])

        # The post-fit truth reader consumes the same single table and IDs.
        from types import SimpleNamespace
        godview = Path(__file__).resolve().parents[1]/'script/oneblock_godview.py'
        reader = next(node for node in ast.walk(ast.parse(godview.read_text()))
                      if isinstance(node,ast.With) and "attr='directory'" in ast.dump(node))
        truth = dict(self=SimpleNamespace(directory=work),h5py=h5py,PARAMETER_FIELDS=PARAMETER_FIELDS)
        exec(compile(ast.Module(body=[reader],type_ignores=[]),str(godview),'exec'),truth)
        np.testing.assert_array_equal(truth['self'].ids,[1,2,3,4])
        np.testing.assert_array_equal(truth['parameters']['f'],[.0001,.04,.0001,.04])
        np.testing.assert_array_equal(received['table']['ID'], truth['self'].ids)
        for name, field in field_names.items():
            np.testing.assert_array_equal(received['table'][field], truth['parameters'][name])


    def test_noise_is_one_continuous_native_draw_before_tdi(self):
        from types import SimpleNamespace
        calls = []
        class Instrument:
            def __init__(self, **settings):
                calls.append(('instrument', settings))
            def disable_all_noises(self, **settings):
                calls.append(('noises', settings))
            def disable_dopplers(self):
                calls.append(('dopplers',))
            def export_numpy_full(self, **settings):
                calls.append(('export', settings))
                return 'continuous measurement arrays'
        scope = dict(
            detector_config=dict(dt=10., orbits='orbit.h5'), N=19440,
            orbit_file=Path('orbit.h5'), L=2.5e9,
            lisaorbits=SimpleNamespace(KeplerianOrbits=lambda **settings:SimpleNamespace(
                write=lambda *args, **settings:None)),
            N_day=8640, fs=.1,
            N_acc=3.e-15, delta_x=15.e-12, serial_scheduler='serial chunks',
            lisainstrument=SimpleNamespace(Instrument=Instrument),
            pytdi=SimpleNamespace(Data=SimpleNamespace(from_instrument=lambda result:result)),
        )
        self.simulation_step(3, scope)
        self.assertEqual(len([item for item in calls if item[0]=='instrument']), 1)
        settings = calls[0][1]
        self.assertEqual(settings['size'], 19440 + 2*8640)
        self.assertEqual(settings['t0'], -86400.)
        self.assertNotIn('seed', settings)
        self.assertEqual(settings['testmass_asds'], 3.e-15)
        self.assertEqual(settings['oms_asds'], (15.e-12, 0., 0., 0., 0., 0.))
        self.assertIn(('noises', dict(excluding={'test-mass','oms'})), calls)
        self.assertIn(('export', {}), calls)
        self.assertEqual(scope['noise_data'], 'continuous measurement arrays')

    def full_duration_fixture(self, fail_tdi=False):
        from types import SimpleNamespace
        import time
        work = self.root / 'continuous_work'
        work.mkdir()
        for name in ('population.h5', 'orbit.h5', 'population_links.h5'):
            shutil.copy2(self.legacy/name, work/name)
        N_day, N = 4, 10
        fs = N_day / 86400.
        noise = np.arange(N + 2*N_day, dtype=float)*3 + 7
        calls = []
        class Instrument:
            central_freq = 2.
            def __init__(self, **settings):
                calls.append(settings)
                self.settings = settings
            def disable_all_noises(self):
                pass
            def disable_dopplers(self):
                pass
            def export_numpy_full(self):
                start = int(round(self.settings['t0']*fs))
                signal = np.arange(start, start+self.settings['size'], dtype=float)*5 + 11
                return SimpleNamespace(measurements={'isc12':signal}, args={})
        def operator(scale):
            def evaluate(measurements):
                if fail_tdi:
                    raise RuntimeError('controlled TDI failure')
                values = measurements['isc12']
                return scale*(values + np.roll(values, 1))
            return SimpleNamespace(build=lambda **args:evaluate)
        def simulate_tdi(measurements, args, *, keep):
            return np.stack([operator(scale).build(**args)(measurements)[keep]
                             for scale in (1, 2, 3)])
        scope = dict(workspace=work, observation_file=self.output, N=N, N_day=N_day,
                     fs=fs, T_obs=N/fs, source_count=2, N_acc=3.e-15, delta_x=15.e-12,
                     links_file=work/'population_links.h5', detector_config=dict(dt=1/fs),
                     noise_data=SimpleNamespace(measurements={'isc12':noise}), noise_instrument=object(),
                     lisainstrument=SimpleNamespace(Instrument=Instrument),
                     pytdi=SimpleNamespace(Data=SimpleNamespace(from_instrument=lambda data:data),
                         michelson=SimpleNamespace(X2=operator(1), Y2=operator(2), Z2=operator(3))),
                     tdi_gpu=simulate_tdi,
                     time=time, h5py=h5py, np=np, json=__import__('json'),
                     importlib=SimpleNamespace(metadata=SimpleNamespace(version=lambda name:'test-version')),
                     pack_observation=pack_observation, shutil=shutil)
        source = (Path(__file__).resolve().parents[1]/'eric_simulation_DWD.py').read_text()
        start = next(node.lineno for node in self.simulation_nodes() if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name) and node.targets[0].id=='signal_instrument')
        end = next(i for i,line in enumerate(source.splitlines(),1) if line.startswith('# Step 5.'))
        nodes = [node for node in self.simulation_nodes() if start <= node.lineno < end]
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'eric_simulation_DWD.py','exec'),scope)
        return work, scope, calls, noise

    def test_signal_and_noise_share_the_full_padded_grid_before_tdi(self):
        work, scope, calls, noise = self.full_duration_fixture()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['size'], 18)
        self.assertEqual(calls[0]['t0'], -86400.)
        self.assertEqual(calls[0]['gws'], str(work/'population_links.h5'))
        np.testing.assert_array_equal(scope['signal_measurements']['isc12'], np.arange(-4,14)*5+11)
        np.testing.assert_array_equal(scope['noise_measurements']['isc12'], noise)
        self.assertNotIn('tdi_results', scope)
        self.assertFalse(self.output.exists())

    def test_full_duration_tdi_keeps_padding_alignment_and_saves_only_retained_samples(self):
        work, scope, calls, noise = self.full_duration_fixture()
        self.simulation_step(5, scope)
        self.assertFalse(work.exists())
        positions = np.arange(10)
        signal_expected = ((positions*5+11) + ((positions-1)*5+11))/2
        noise_expected = (noise[positions+4] + noise[positions+3])/2
        with h5py.File(self.output) as saved:
            with h5py.File(self.output.with_suffix('.orbits.h5')) as orbit:
                self.assertEqual(dict(orbit.attrs), dict(saved['orbit'].attrs))
                np.testing.assert_array_equal(orbit['tcb/ltt'][:], saved['orbit/tcb/ltt'][:])
            group = saved['observation']
            self.assertEqual(group.id, saved['population_tdi'].id)
            self.assertEqual(group.attrs['duration'], scope['T_obs'])
            self.assertEqual(group.attrs['rate'], scope['fs'])
            config = __import__('json').loads(group.attrs['detector_config'])
            self.assertEqual(config['dt'], 1 / scope['fs'])
            self.assertEqual(config['orbits'], str(self.output.with_suffix('.orbits.h5')))
            self.assertNotIn('completed_days', group)
            for index, scale in enumerate((1,2,3)):
                np.testing.assert_array_equal(group['signal_xyz_td'][index], scale*signal_expected)
                np.testing.assert_array_equal(group['noise_xyz_td'][index], scale*noise_expected)
            np.testing.assert_array_equal(group['total_xyz_td'][:],
                                          group['signal_xyz_td'][:] + group['noise_xyz_td'][:])

    def test_failed_full_duration_tdi_retains_workspace_and_preserves_output(self):
        work, scope, calls, noise = self.full_duration_fixture(fail_tdi=True)
        self.output.write_bytes(b'existing published observation')
        with self.assertRaisesRegex(RuntimeError, 'controlled TDI failure'):
            self.simulation_step(5, scope)
        self.assertFalse((work/'population_tdi.h5').exists())
        self.assertEqual(self.output.read_bytes(), b'existing published observation')
        self.assertTrue(work.exists())

    def test_checkpoint_snapshot_survives_source_removal(self):
        from script.oneblock_observation import read_checkpoint, write_checkpoint
        source_path = self.root / 'checkpoint.h5'
        with h5py.File(source_path, 'w') as saved:
            saved.attrs['iteration'] = 17
            branch = saved.create_group('mcmc/branch')
            branch.attrs['labels'] = ['f', 'df']
            values = np.arange(24.).reshape(3, 8)
            branch.create_dataset('chain', data=values, chunks=(1, 8), maxshape=(None, 8),
                                  compression='gzip', shuffle=True).attrs['unit'] = 'Hz'
            saved.create_dataset('mask', data=[True, False])
            saved.create_dataset('name', data='native checkpoint', dtype=h5py.string_dtype())
        with h5py.File(source_path, 'r') as source:
            snapshot = read_checkpoint(source)
        source_path.unlink()
        with h5py.File(self.root / 'result.h5', 'w') as saved:
            write_checkpoint(saved.create_group('rounds/01/checkpoints/native'), snapshot)
        with h5py.File(self.root / 'result.h5') as saved:
            checkpoint = saved['rounds/01/checkpoints/native']
            self.assertEqual(checkpoint.attrs['iteration'], 17)
            chain = checkpoint['mcmc/branch/chain']
            np.testing.assert_array_equal(chain, values)
            self.assertEqual(chain.dtype, values.dtype)
            self.assertEqual(chain.maxshape, (None, 8))
            self.assertEqual(chain.compression, 'gzip')
            self.assertTrue(chain.shuffle)
            self.assertEqual(chain.attrs['unit'], 'Hz')
            self.assertEqual(checkpoint['name'].asstr()[()], 'native checkpoint')
            np.testing.assert_array_equal(checkpoint['mask'], [True, False])



if __name__ == '__main__':
    unittest.main()

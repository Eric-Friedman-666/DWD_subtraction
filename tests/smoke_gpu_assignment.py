"""Allocated OzSTAR-only 4x16 capacity smoke with full-year FD waveforms.

No production inputs are read. Each of 64 real isolated CUDA contexts computes a
128-template full-360-day/900-second-grid basis in the real prior-padded window,
then remains resident until all 64 contexts have reached the same barrier. This
is bounded waveform residency/capacity evidence, NOT 64 concurrent Eryn fits.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('DWD_WORKER_GPU_MEM_FRACTION', '.045')
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from datetime import datetime, timezone
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from script.dwd_worker_supervisor import run_block_jobs, worker_progress


def save(path, value):
    temporary = Path(str(path) + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def waveform_resident(index, options, *, work, label):
    import gc
    import numpy as np
    from script.eric_generate_waveform import _initialize_gpu_backend
    _initialize_gpu_backend()
    import jax
    jax.config.update('jax_enable_x64', True)
    from script.dwd_frequency_support import response_grid, chirp_frequency_padding, RESPONSE_STEP_SECONDS
    from script.spectrum import prepare_xyz_geometry
    from script.eric_generate_waveform import GPUResponse
    import lisagwresponse
    work = Path(work)
    assert len(jax.devices('gpu')) == 1
    assert os.environ['CUDA_VISIBLE_DEVICES'] == os.environ['DWD_ASSIGNED_GPU']
    assert RESPONSE_STEP_SECONDS == 900.
    duration, rate = 31104000., .1
    bounds = np.array([[-4.e-12, -7.60863249122299e-23],
                        [7.60863249122299e-23, 4.e-12]])
    frequency_center, fdot = .04296793987485368, 2.758292489484066e-12
    lower = .0001 + np.floor((frequency_center - .0001) / 1.e-5) * 1.e-5
    padding = chirp_frequency_padding(duration, 1.e-5, .05, bounds)
    bins = np.arange(max(1, int(np.ceil((lower-padding)*duration))),
        int(np.floor((lower+1.e-5+padding)*duration))+1)
    frequency = bins / duration
    times = response_grid(duration, rate)
    model = lisagwresponse.GalacticBinary(A=1., f=.003, ra=0., dec=0.,
        orbits=str(work/'synthetic_orbits.h5'), orbit_interp_order=5, shift_sun='never')
    geometry = prepare_xyz_geometry(times, model)
    covariance = np.broadcast_to(np.eye(3), (len(frequency), 3, 3)).copy()
    engine = GPUResponse(covariance, np.zeros((3, len(frequency)), complex),
        times, geometry, frequency, duration, rate)
    points = np.tile([frequency_center, fdot, 1.2, -.3], (128, 1))
    points[:, 0] += np.linspace(-1.e-7, 1.e-7, 128)
    worker_progress('waveform_basis', force=True, basis_batch_size=128)
    started = time.monotonic()
    basis = engine.build(points)
    assert np.isfinite(basis).all() and float(np.sum(abs(basis)**2)) > 0.
    # First real waveform synchronization releases only this GPU's startup gate.
    worker_progress('f_screening', force=True)
    # Retain the engine + real device buffers after construction, rather than
    # claiming capacity from 64 tiny toy allocations.
    retained = jax.device_put(basis[:4])
    retained.block_until_ready()
    stats = jax.devices('gpu')[0].memory_stats()
    audit = dict(index=index, pid=os.getpid(), gpu=os.environ['DWD_ASSIGNED_GPU'],
        gpu_slot=int(os.environ['DWD_GPU_SLOT_INDEX']),
        global_slot=int(os.environ['DWD_WORKER_SLOT_INDEX']),
        response_seconds=RESPONSE_STEP_SECONDS, duration_seconds=duration,
        basis_batch_size=128, frequency_bins=len(frequency), memory_stats=stats,
        first_waveform_seconds=time.monotonic()-started, waveform_finite=True)
    save(work/f'resident_{index:05d}.json', audit)
    deadline = time.monotonic()+options.get('residency_timeout', 1800.)
    while len(list(work.glob('resident_*.json'))) < 64:
        if time.monotonic() >= deadline:
            raise TimeoutError('64 real waveform contexts did not become concurrently resident')
        worker_progress('sampling', force=True, residency_wait=True)
        time.sleep(.2)
    if index == 0:
        snapshot = subprocess.check_output(['nvidia-smi',
            '--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True)
        contexts = subprocess.check_output(['nvidia-smi',
            '--query-compute-apps=gpu_uuid,pid,used_memory',
            '--format=csv,noheader,nounits'], text=True)
        save(work/'all_resident_gpu_snapshot.json', dict(gpus=snapshot, contexts=contexts,
            collected_utc=datetime.now(timezone.utc).isoformat()))
        (work/'release_all_resident').touch()
    while not (work/'release_all_resident').exists():
        worker_progress('sampling', force=True, residency_wait=True)
        time.sleep(.1)
    # Exercise another synchronized real basis while every context is resident.
    checked = engine.build(points[:4])
    assert np.isfinite(checked).all()
    until = time.monotonic()+.2
    while time.monotonic() < until:
        worker_progress('sampling', force=True)
        time.sleep(.05)
    del retained, checked
    gc.collect()
    bank = work/f'round_{label}_block_{index:05d}'
    bank.mkdir()
    save(bank/'result.pkl', audit)  # JSON fixture filename, not a pickle result.


def main():
    if os.uname().sysname == 'Darwin' or not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Use an allocated OzSTAR Slurm compute node; never run on login/Mac')
    devices = tuple(device.strip() for device in os.environ.get('DWD_GPU_DEVICES',
        os.environ.get('CUDA_VISIBLE_DEVICES', '')).split(',') if device.strip())
    assert len(devices) == 4 and len(set(devices)) == 4, devices
    destination = Path(sys.argv[1]).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    import lisaorbits
    lisaorbits.KeplerianOrbits(L=2.5e9).write(str(destination/'synthetic_orbits.h5'),
        t0=-20e6, dt=86400., size=825, mode='w')
    report = dict(status='running', scope=__doc__, devices=list(devices),
        started_utc=datetime.now(timezone.utc).isoformat(),
        slurm_job_id=os.environ['SLURM_JOB_ID'], workers_per_gpu=16,
        preallocation=os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'],
        worker_memory_fraction=os.environ['DWD_WORKER_GPU_MEM_FRACTION'],
        controller_memory_fraction=os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION', 'JAX default'),
        source_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in ('script/dwd_worker_supervisor.py', 'script/dwd_frequency_support.py',
                         'script/eric_generate_waveform.py')})
    save(destination/'smoke.json', report)
    try:
        summary = run_block_jobs(partial(waveform_resident, work=destination, label='01'),
            [(i, {}) for i in range(64)], max_workers=64,
            gpu_devices=devices, workers_per_gpu=16,
            worker_gpu_memory_fraction=float(os.environ['DWD_WORKER_GPU_MEM_FRACTION']),
            work=destination, label='01', progress_timeout=600., max_attempts=1,
            startup_ready_phase='f_screening', status_file=destination/'controller.status.json', run_status={})
        rows = [json.loads(path.read_text()) for path in destination.glob('resident_*.json')]
        assert len(rows) == 64 and len({row['pid'] for row in rows}) == 64
        assert summary['maximum_active_by_gpu'] == {device: 16 for device in devices}
        snapshot = json.loads((destination/'all_resident_gpu_snapshot.json').read_text())
        contexts = [line.split(',') for line in snapshot['contexts'].splitlines() if line.strip()]
        by_pid = {int(fields[1].strip()): fields[0].strip() for fields in contexts}
        assert all(row['pid'] in by_pid for row in rows), 'A claimed resident worker had no CUDA context'
        physical = {}
        for row in rows:
            uuid = by_pid[row['pid']]
            physical.setdefault(uuid, []).append(row['pid'])
        assert len(physical) == 4 and all(len(pids) == 16 for pids in physical.values()), physical
        for row in rows:
            try:
                os.kill(row['pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise AssertionError(f'Worker remains after round barrier: {row["pid"]}')
        gpu_rows = [[field.strip() for field in line.split(',')]
            for line in snapshot['gpus'].splitlines() if line.strip()]
        assert all(int(fields[4]) >= 1024 for fields in gpu_rows if fields[1] in physical), 'Less than 1 GiB free'
        report.update(status='passed', summary=summary, context_pids_by_physical_gpu=physical,
            simultaneous_real_waveform_contexts=64, per_worker=rows,
            simultaneous_memory_snapshot=snapshot, all_workers_reaped=True,
            maximum_sources_100_rjmcmc_capacity_proved=False)
    except BaseException as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc()[-12000:])
        raise
    finally:
        report['completed_utc'] = datetime.now(timezone.utc).isoformat()
        save(destination/'smoke.json', report)
        print('GPU_ASSIGNMENT_SMOKE_FINAL', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

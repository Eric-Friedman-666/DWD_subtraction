# Step 1. Import packages.
import os
for thread_variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[thread_variable] = '1'
import importlib.metadata
import json
from pathlib import Path
import shutil
import tempfile
import time
import h5py
import lisaorbits
import lisagwresponse
import lisainstrument
import numpy as np
import pytdi
from script.oneblock_observation import pack_observation
from eric_generate_waveform import DWD_simulation_gpu, tdi_gpu


# Step 2. Configure parameter settings.
catalogue_file = Path('./data/LDC2_sangria_training_v2.h5')
observation_file = Path('./data/dwd_observation_full_catalogue.h5')
T_obs = 360. * 86400.
fs = 0.1
L = 2.5e9
N_acc = 3.0e-15
delta_x = 15.0e-12
gpu_batch_size = 10000
minimum_envelope_samples = 512
parameter_fields = dict(
    A='Amplitude', f='Frequency', df='FrequencyDerivative', phi0='InitialPhase',
    iota='Inclination', longitude='EclipticLongitude', latitude='EclipticLatitude',
    psi_bme='Polarization',
)


workspace = Path(tempfile.mkdtemp(prefix='.dwd_simulation_', dir=observation_file.parent))
orbit_file = workspace / 'orbit.h5'
N = int(round(T_obs * fs))
N_day = int(round(86400 * fs))
detector_config = dict(
    T_obs=T_obs, gpu_batch_size=gpu_batch_size,
    minimum_envelope_samples=minimum_envelope_samples, include_padding=True,
    dt=1 / fs, orbits=str(orbit_file), orbit_dataset='tcb/ltt',
    aafilter=('kaiser', 240., .275 * fs, .725 * fs),
    clock_offsets=0., clock_freqoffsets=0., clock_freqlindrifts=0., clock_freqquaddrifts=0.,
)

# Step 3. Simulate instrumental noise with the LISA packages.
lisaorbits.KeplerianOrbits(L=L).write(
    str(orbit_file), t0=-20e6, dt=86400., size=825, mode='w',
)
print('Simulating one continuous ACC+OMS noise realization.', flush=True)
noise_instrument = lisainstrument.Instrument(
    **{key: value for key, value in detector_config.items()
       if key not in ('T_obs', 'gpu_batch_size', 'minimum_envelope_samples', 'include_padding')}, size=N + 2 * N_day,
    t0=-N_day / fs,
    testmass_asds=N_acc, oms_asds=(delta_x, 0., 0., 0., 0., 0.),
)
noise_instrument.disable_all_noises(excluding={'test-mass', 'oms'})
noise_instrument.disable_dopplers()
noise_data = pytdi.Data.from_instrument(
    noise_instrument.export_numpy_full(),
)

# Step 4. Simulate waveforms of all DWDs on the GPU.
with h5py.File(catalogue_file) as source:
    dgb = source['sky/dgb/cat'][:]
    igb = source['sky/igb/cat'][:]

source_count = len(dgb) + len(igb)
catalogue = np.empty(source_count, dtype=[('ID', 'i8')] + [
    (field, 'f8') for field in parameter_fields.values()
])
catalogue['ID'] = np.arange(1, source_count + 1, dtype=np.int64)
for field in parameter_fields.values():
    catalogue[field] = np.concatenate((dgb[field][:, 0], igb[field][:, 0]))
del dgb, igb

with h5py.File(workspace / 'population.h5', 'x') as saved:
    saved.create_dataset('catalogues/all', data=catalogue, chunks=True)

links_file = workspace / 'population_links.h5'
print(f'Simulating all {source_count:,} DWDs with DWD_simulation_gpu.', flush=True)
started = time.perf_counter()
links = DWD_simulation_gpu(catalogue, detector_config)
waveform_runtime = time.perf_counter() - started
print(f'GPU injection completed: {source_count:,} DWDs in {waveform_runtime:.1f}s.', flush=True)
del catalogue
response_model = lisagwresponse.GalacticBinary(
    A=1., f=.003, ra=0., dec=0., orbits=str(orbit_file),
    orbit_interp_order=5, shift_sun='never',
)
with h5py.File(links_file, 'x') as saved:
    response_model.init_response_file(saved, detector_config['dt'], len(links), -2 * N_day / fs, 'tcb')
    saved['tcb/y'][:] = links
del links

# Convert the summed links into full-duration detector measurements.
signal_instrument = lisainstrument.Instrument(
    **{key: value for key, value in detector_config.items()
       if key not in ('T_obs', 'gpu_batch_size', 'minimum_envelope_samples', 'include_padding')}, gws=str(links_file), size=N + 2 * N_day,
    t0=-N_day / fs,
)
signal_instrument.disable_all_noises()
signal_instrument.disable_dopplers()
signal_data = pytdi.Data.from_instrument(signal_instrument.export_numpy_full())
signal_measurements = signal_data.measurements
noise_measurements = noise_data.measurements
del noise_data, noise_instrument

# Step 5. Combine the full observation, apply TDI, and save.
total_measurements = {
    name: signal_measurements[name] + noise_measurements[name]
    for name in signal_measurements
}
tdi_results = {}
for name, measurements in (('total', total_measurements),
                           ('signal', signal_measurements), ('noise', noise_measurements)):
    tdi_results[name] = tdi_gpu(
        measurements, signal_data.args, keep=slice(N_day, N_day + N),
    ) / signal_instrument.central_freq
detector_config['orbits'] = str(observation_file.with_suffix('.orbits.h5'))
with h5py.File(workspace / 'population_tdi.h5', 'x') as saved:
    saved.attrs.update(duration=T_obs, rate=fs)
    saved.attrs['detector_config'] = json.dumps(detector_config)
    for name, value in tdi_results.items():
        saved.create_dataset(name + '_xyz_td', data=value, chunks=(3, min(N_day, N)))

pack_observation(workspace, observation_file, overwrite=True, time_domain_only=True)
shutil.copy2(workspace / 'orbit.h5', observation_file.with_suffix('.orbits.h5'))
shutil.rmtree(workspace)
print(f'Saved {source_count:,} DWDs: {observation_file.resolve()}', flush=True)

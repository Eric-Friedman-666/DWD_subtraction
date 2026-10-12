import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
import logging
import time
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import lu_factor, lu_solve
from jax.scipy.signal import fftconvolve
from tqdm.auto import tqdm

from script.eric_generate_waveform import PARAMETER_FIELDS, generate_waveform_td
from script.spectrum import DetectorFunction

jax.config.update('jax_enable_x64', True)

logging.getLogger('lisainstrument.instru.instru_fplan').addFilter(
    lambda record: not record.getMessage().startswith(
        'Using default set of locking beatnote frequencies'))


LDC_file = Path('./data/LDC2_sangria_training_v2.h5')
orbit_file = Path('./data/lisa_snr_orbits.h5')
output_file = Path('./data/ldc_dwd_data.h5')
catalogue_groups = ('dgb', 'igb')
T_obs = 360 * 86400
fs = 0.1
N_acc = 3.0e-15
delta_x = 15.0e-12
SNR_threshold = 7.0
covariance_regularization = 1.0e-12  # Add this times trace(S_instrument)/3 to all channels.
catalogue_chunk_size = 65536

started = time.perf_counter()
output_file.parent.mkdir(parents=True, exist_ok=True)
temporary_directory = TemporaryDirectory(prefix='ldc_snr_', dir=output_file.parent)
work_directory = Path(temporary_directory.name)
final_output_file = output_file
output_file = work_directory/final_output_file.name
background_file = work_directory/'background.h5'
history = []

# -----------------------------------------------------------------------
# 1. Read every DWD catalogue and prepare the native physical frequency grid.
# -----------------------------------------------------------------------
print('1/4  Read the complete LDC DWD catalogue', flush=True)
# Positive FFT bins, excluding DC and Nyquist.
first_bin = 1
last_bin = int(T_obs*fs)//2-1
frequency = np.arange(first_bin, last_bin+1)/T_obs
catalogue_sizes = {}
with h5py.File(LDC_file, 'r') as ldc:
    for group in catalogue_groups:
        catalogue = ldc[f'sky/{group}/cat']
        catalogue_sizes[group] = len(catalogue)
catalogue_offsets = {'dgb': 0, 'igb': catalogue_sizes['dgb']}
source_count = sum(catalogue_sizes.values())
print(f'     {source_count:,} binaries; one {T_obs/86400:g}-day observation; '
      f'{len(frequency):,} native bins; '
      f'GPU TD waveform batches; GPU FFT/SNR and foreground', flush=True)


@jax.jit
def SNR_gpu(waveform_fd, covariance_factor, T_obs, fs):
    vectors = waveform_fd.transpose(2, 1, 0)
    weighted = lu_solve(covariance_factor, vectors)
    power = jnp.sum((vectors.conj()*weighted).real, axis=(0, 1))
    return jnp.sqrt(4/(T_obs*fs**2)*power)


class GPUJobs:
    def __init__(self, T_obs, fs, orbit_file, first_bin):
        self.T_obs = T_obs
        self.fs = fs
        self.orbit_file = orbit_file
        self.first_bin = first_bin
        self.noise_input = None
        self.noise_covariance = None
        self.covariance_factor = None
        self.foreground_lines = None

    def snr(self, dwd_ids, noise_covariance, chunk=None):
        return self._results(dwd_ids, noise_covariance, None, chunk)

    def foreground(self, dwd_ids, noise_covariance, chunk=None):
        return self._results(dwd_ids, noise_covariance, SNR_threshold, chunk)

    @partial(jax.jit, static_argnums=(0, 3), donate_argnums=(4,))
    def _process(self, xyz, covariance_factor, threshold, foreground_lines):
        waveform_fd = jnp.fft.rfft(
            xyz, n=int(round(self.T_obs*self.fs)), axis=-1)
        waveform_fd = waveform_fd[
            :, :, self.first_bin:self.first_bin+len(covariance_factor[0])]
        snr = SNR_gpu(waveform_fd, covariance_factor, self.T_obs, self.fs)
        if threshold is not None:
            unresolved = (snr <= threshold).astype(waveform_fd.real.dtype)
            foreground_lines += 2/(self.T_obs*self.fs**2)*jnp.einsum(
                'bik,bjk,b->kij', waveform_fd, waveform_fd.conj(), unresolved)
        return snr, foreground_lines

    def _results(self, dwd_ids, noise_covariance, threshold, chunk=None):
        description = 'Generate full TD XYZ'
        if chunk is not None:
            description += f' [{chunk[0]}/{chunk[1]}]'
        for batch_ids, _, xyz in generate_waveform_td(
                self.fs, self.T_obs, dwd_ids, self.orbit_file,
                progress_desc=description):
            if noise_covariance is not self.noise_input:
                self.noise_covariance = jax.device_put(
                    noise_covariance, jax.devices('gpu')[0])
                self.covariance_factor = lu_factor(self.noise_covariance)
                self.noise_input = noise_covariance
            if self.foreground_lines is None:
                self.foreground_lines = jnp.zeros_like(self.noise_covariance)
            snr, self.foreground_lines = self._process(
                xyz, self.covariance_factor, threshold, self.foreground_lines)
            jax.block_until_ready((snr, self.foreground_lines))
            yield batch_ids, np.asarray(snr)

    def confusion_covariance(self):
        with jax.default_device(self.foreground_lines.device):
            offsets = jnp.arange(-128, 129)
            kernel = 0.5*(1+jnp.cos(jnp.pi*offsets/128))
            kernel /= kernel.sum()
            covariance = fftconvolve(
                self.foreground_lines, jnp.broadcast_to(kernel[:, None, None],
                    (len(kernel), 3, 3)), mode='same', axes=0)
        return 0.5*(covariance+covariance.conj().swapaxes(-1, -2))

# -----------------------------------------------------------------------
# 2. Instrument-only XYZ covariance, then absolute SNRs for every source.
# -----------------------------------------------------------------------
print('2/4  Instrumental covariance and round 1 absolute SNRs', flush=True)
t = T_obs/2
with tqdm(total=1, desc='Instrumental covariance', unit='calculation') as progress:
    detector = DetectorFunction(
        fs=fs, orbit_file=orbit_file, orbit_interp_order=5)
    instrument_covariance, _ = detector.noise_covariance(
        frequency, t,
        acceleration_asd=N_acc,
        oms_asd=delta_x,
        generation='2nd',
        combination_type='XYZ')
    progress.update(1)
scale = np.trace(instrument_covariance, axis1=-2, axis2=-1).real/3
instrument_covariance += covariance_regularization*scale[:, None, None]*np.eye(3)
active_chunks = []

# Save settings as datasets; no HDF5 attributes.
with h5py.File(output_file, 'x') as output:
    settings = output.create_group('settings')
    settings.create_dataset('source_catalogue', data=str(LDC_file.resolve()))
    settings.create_dataset('observation_duration_s', data=T_obs)
    settings.create_dataset('detector_sampling_rate_hz', data=fs)
    settings.create_dataset('frequency_spacing_hz', data=1/T_obs)
    settings.create_dataset('smoothing_half_width_hz', data=128/T_obs)
    settings.create_dataset('signal_time_window', data='rectangular full observation')
    settings.create_dataset('signal_window_mean_square', data=1.)
    settings.create_dataset('tdi_channels', data='X2,Y2,Z2')
    settings.create_dataset('snr_threshold', data=SNR_threshold)
    settings.create_dataset('transfer_factor_threshold', data=1)
    settings.create_dataset('neighboring_bins', data=128)
    settings.create_dataset('noise_acceleration_asd', data=N_acc)
    settings.create_dataset('noise_oms_asd', data=delta_x)
    settings.create_dataset('covariance_regularization', data=covariance_regularization)
    settings.create_dataset('waveform_device', data='gpu')
    settings.create_dataset('post_waveform_device', data='gpu')
    settings.create_dataset('snr_cpu_threads', data=0)

    status = output.create_group('status')
    status.create_dataset('complete', data=False)
    status.create_dataset('source_count', data=source_count)
    output.create_group('sky')

fields = ['Name', *PARAMETER_FIELDS]
number_chunks = sum((size+catalogue_chunk_size-1)//catalogue_chunk_size
                    for size in catalogue_sizes.values())
gpu = GPUJobs(T_obs, fs, orbit_file, first_bin)
chunk_index = 0
with tqdm(total=source_count, desc='Round 1: all absolute SNRs',
          unit='DWD', unit_scale=True) as progress:
    for group in catalogue_groups:
        dataset_path = f'sky/{group}/cat'
        group_size = catalogue_sizes[group]
        with h5py.File(LDC_file, 'r') as ldc:
            catalogue_dtype = ldc[dataset_path].dtype
        # Preserve source IDs; store every numerical field without decimal rounding.
        dtype = np.dtype([('Name', catalogue_dtype.fields['Name'][0])]
                         + [(name, '<f8') for name in PARAMETER_FIELDS]
                         + [('SNR_absolute', '<f8'), ('SNR_relative', '<f8')])
        chunk_shape = (min(4096, group_size), 1)
        with h5py.File(output_file, 'a') as output:
            catalogue_group = output['sky'].create_group(group)
            catalogue_group.create_dataset('cat', shape=(group_size, 1), dtype=dtype,
                                           chunks=chunk_shape, compression='lzf', shuffle=True)

        for first in range(0, group_size, catalogue_chunk_size):
            chunk_index += 1
            last = min(first+catalogue_chunk_size, group_size)
            # Read one chunk.
            with h5py.File(LDC_file, 'r') as ldc:
                records = ldc[dataset_path].fields(fields)[first:last, 0]

            # Calculate with the files closed.
            rows = np.empty(len(records), dtype=dtype)
            for name in fields:
                rows[name] = records[name]
            rows['SNR_relative'] = np.nan
            dwd_ids = catalogue_offsets[group]+np.arange(first, last)
            for batch_ids, snr in gpu.foreground(
                    dwd_ids, instrument_covariance, chunk=(chunk_index, number_chunks)):
                rows['SNR_absolute'][batch_ids-catalogue_offsets[group]-first] = snr
                progress.update(len(batch_ids))
            keep = rows['SNR_absolute'] > SNR_threshold
            active_chunks.append(dwd_ids[keep])

            # Save the calculated chunk.
            with h5py.File(output_file, 'a') as output:
                output[dataset_path][first:last, 0] = rows

active = np.concatenate(active_chunks) if active_chunks else np.empty(0, dtype=np.int64)
del active_chunks
resolved_count = len(active)
history.append(dict(round=1, evaluated=source_count, resolved=resolved_count,
                    unresolved=source_count-resolved_count, transfer_factor=None))
print(f'Round 1: {resolved_count:,} resolved; {source_count-resolved_count:,} initially unresolved', flush=True)

instrument_covariance = gpu.noise_covariance
confusion_covariance = gpu.confusion_covariance()
with h5py.File(background_file, 'x') as background:
    settings = background.create_group('settings')
    settings.create_dataset('observation_duration_s', data=T_obs)
    settings.create_dataset('detector_sampling_rate_hz', data=fs)
    settings.create_dataset('frequency_spacing_hz', data=1/T_obs)
    settings.create_dataset('smoothing_half_width_hz', data=128/T_obs)
    settings.create_dataset('signal_time_window', data='rectangular full observation')
    settings.create_dataset('signal_window_mean_square', data=1.)
    settings.create_dataset('covariance_axes', data='frequency,XYZ,XYZ')
    settings.create_dataset('neighboring_bins', data=128)

    status = background.create_group('status')
    status.create_dataset('complete', data=False)
    background.create_dataset('frequency_hz', data=frequency)
    background.create_dataset('instrument_covariance', data=np.asarray(instrument_covariance), compression='lzf')
    background.create_dataset('initial_confusion_covariance',
                              data=np.asarray(confusion_covariance),
                              compression='lzf')

# -----------------------------------------------------------------------
# 3. Re-evaluate only resolved sources and accumulate each transfer once.
# -----------------------------------------------------------------------
print('3/4  Iteratively update the unresolved foreground', flush=True)
round_number = 2
while True:
    effective_covariance = instrument_covariance+confusion_covariance
    evaluated = resolved_count
    transfer_factor = 0
    remaining = []
    with tqdm(total=evaluated, desc=f'Round {round_number}: resolved SNRs',
              unit='DWD', unit_scale=True) as progress:
        for batch_ids, snr in gpu.foreground(active, effective_covariance, chunk=(1, 1)):
            transfer = snr <= SNR_threshold
            transfer_factor += np.count_nonzero(transfer)
            remaining.append(batch_ids[~transfer])
            progress.update(len(batch_ids))
    active = np.concatenate(remaining) if remaining else np.empty(0, dtype=active.dtype)
    resolved_count = evaluated-transfer_factor
    history.append(dict(round=round_number, evaluated=evaluated, resolved=resolved_count,
                        unresolved=source_count-resolved_count, transfer_factor=transfer_factor))
    print(f'Round {round_number}: transfer_factor={transfer_factor:,}; '
          f'{resolved_count:,} resolved; {source_count-resolved_count:,} unresolved', flush=True)
    if transfer_factor == 0:
        break
    confusion_covariance = gpu.confusion_covariance()
    round_number += 1

# -----------------------------------------------------------------------
# 4. With the final foreground fixed, calculate every relative SNR again.
# -----------------------------------------------------------------------
print('4/4  Final relative SNRs for every LDC DWD', flush=True)
catalogue_columns = ('ID', *PARAMETER_FIELDS, 'SNR_absolute', 'SNR_relative')
with h5py.File(output_file, 'a') as output:
    output.create_dataset('catalogue', shape=(source_count, 11), dtype='f8',
                          chunks=(min(4096, source_count), 11), compression='lzf', shuffle=True)
    output['settings'].create_dataset('catalogue_columns',
        data=np.asarray(catalogue_columns, dtype=h5py.string_dtype('utf-8')))
# No transfers occurred in the last round, so its covariance is already final.
chunk_index = 0
with tqdm(total=source_count, desc='Final round: all relative SNRs',
          unit='DWD', unit_scale=True) as progress:
    for group in catalogue_groups:
        dataset_path = f'sky/{group}/cat'
        group_size = catalogue_sizes[group]
        for first in range(0, group_size, catalogue_chunk_size):
            chunk_index += 1
            last = min(first+catalogue_chunk_size, group_size)
            # Read the previously saved absolute SNRs.
            with h5py.File(output_file, 'r') as output:
                rows = output[dataset_path][first:last, 0]

            # Calculate with the files closed.
            dwd_ids = catalogue_offsets[group]+np.arange(first, last)
            for batch_ids, snr in gpu.snr(
                    dwd_ids, effective_covariance, chunk=(chunk_index, number_chunks)):
                rows['SNR_relative'][batch_ids-catalogue_offsets[group]-first] = snr
                progress.update(len(batch_ids))

            merged_rows = np.column_stack((dwd_ids + 1,
                *[rows[name] for name in catalogue_columns[1:]]))
            # Save the calculated chunk.
            with h5py.File(output_file, 'a') as output:
                output[dataset_path][first:last, 0] = rows
                output['catalogue'][catalogue_offsets[group] + first:catalogue_offsets[group] + last] = merged_rows

# Store one value per round in each dataset.
round_numbers = np.array([row['round'] for row in history])
evaluated_counts = np.array([row['evaluated'] for row in history])
resolved_counts = np.array([row['resolved'] for row in history])
unresolved_counts = np.array([row['unresolved'] for row in history])
# Round 1 has no transfers yet; NaN means not applicable.
transfer_counts = np.array([
    np.nan if row['transfer_factor'] is None else row['transfer_factor'] for row in history])
runtime_seconds = time.perf_counter()-started

with h5py.File(output_file, 'a') as output:
    rounds = output.create_group('rounds')
    rounds.create_dataset('round', data=round_numbers)
    rounds.create_dataset('evaluated', data=evaluated_counts)
    rounds.create_dataset('resolved', data=resolved_counts)
    rounds.create_dataset('unresolved', data=unresolved_counts)
    rounds.create_dataset('transfer_factor', data=transfer_counts)

    status = output['status']
    status.create_dataset('resolved_count', data=resolved_count)
    status.create_dataset('runtime_seconds', data=runtime_seconds)
    status['complete'][()] = True

with h5py.File(background_file, 'a') as background:
    background.create_dataset('final_confusion_covariance', data=np.asarray(confusion_covariance), compression='lzf')
    background.create_dataset('final_effective_covariance', data=np.asarray(effective_covariance), compression='lzf')

    rounds = background.create_group('rounds')
    rounds.create_dataset('round', data=round_numbers)
    rounds.create_dataset('evaluated', data=evaluated_counts)
    rounds.create_dataset('resolved', data=resolved_counts)
    rounds.create_dataset('unresolved', data=unresolved_counts)
    rounds.create_dataset('transfer_factor', data=transfer_counts)
    background['status/complete'][()] = True
# Publish only the completed catalogue, then remove the temporary files.
output_file = output_file.replace(final_output_file)
temporary_directory.cleanup()
print(f'Saved: {output_file.resolve()}', flush=True)
print(f'{resolved_count:,}/{source_count:,} above SNR_r > {SNR_threshold:g}; '
      f'{time.perf_counter()-started:.1f} s', flush=True)

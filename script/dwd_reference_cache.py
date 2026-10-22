"""Run-local, unweighted same-block reference waveforms for later-round matching."""
import hashlib
import inspect
import json
import gc
from contextlib import contextmanager
from pathlib import Path
from time import monotonic, sleep

import h5py
import numpy as np
from tqdm.auto import tqdm
from script.dwd_run_status import write_run_status
from script.dwd_chirp_policy import bank_padding_values


def _content_signature(value):
    """Hash values, not pickle memoization, NumPy scalar wrappers or strides."""
    if isinstance(value, np.ndarray):
        return dict(array_dtype=value.dtype.str, shape=list(value.shape),
            sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
    if isinstance(value, np.generic):
        return _content_signature(value.item())
    if isinstance(value, dict):
        return dict(mapping=[[key, _content_signature(value[key])] for key in sorted(value)])
    if isinstance(value, (tuple, list)):
        return dict(sequence=[_content_signature(item) for item in value])
    if isinstance(value, bytes):
        return dict(bytes_sha256=hashlib.sha256(value).hexdigest())
    if value is None or isinstance(value, (str, int, float, bool)):
        return value


def reference_cache_key(selected_blocks, template_tables, frequency_blocks,
        block_width, frequency, duration, rate, response_times, response_geometry,
        response_type, amplitude_coordinates, reconstruct_waveform, *, frequency_padding_hz=None,
        chirp_bounds_by_block=None):
    """Portable v4 identity includes per-bank support and policy implementation."""
    source_files = {Path(__file__), Path(inspect.getfile(response_type)),
                    Path(inspect.getfile(amplitude_coordinates)),
                    Path(inspect.getfile(reconstruct_waveform))}
    native_filter = Path(inspect.getfile(response_type)).with_name('native_lisa_filter.py')
    if native_filter.is_file():
        source_files.add(native_filter)
    support = Path(__file__).with_name('dwd_frequency_support.py')
    if support.is_file():
        source_files.add(support)
    policy = Path(__file__).with_name('dwd_chirp_policy.py')
    if policy.is_file():
        source_files.add(policy)
    inputs = (4, duration, rate, block_width, frequency_padding_hz, chirp_bounds_by_block, frequency_blocks, selected_blocks,
              template_tables, frequency, response_times, response_geometry,
              sorted((path.name, path.read_bytes()) for path in source_files))
    encoded = json.dumps(_content_signature(inputs), sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


@contextmanager
def _open_cache(path, mode, timeout=30.):
    deadline = monotonic() + timeout
    saved = None
    while saved is None and monotonic() < deadline:
        try:
            saved = h5py.File(path, mode)
        except BlockingIOError:
            sleep(.1)
    if saved is None:
        saved = h5py.File(path, mode)
    try:
        yield saved
    finally:
        saved.close()


def precompute_reference_waveforms(cache_file, selected_blocks, template_tables,
        frequency_blocks, block_width, frequency, duration, rate,
        response_times, response_geometry, response_type, amplitude_coordinates,
        reconstruct_waveform, *, frequency_padding_hz=None, chirp_bounds_by_block=None):
    """Persist fixed physical references; never cache a round's noise weighting."""
    cache_file = Path(cache_file)
    cache_key = reference_cache_key(selected_blocks, template_tables, frequency_blocks,
        block_width, frequency, duration, rate, response_times, response_geometry,
        response_type, amplitude_coordinates, reconstruct_waveform,
        frequency_padding_hz=frequency_padding_hz, chirp_bounds_by_block=chirp_bounds_by_block)
    padding_by_block = bank_padding_values(frequency_padding_hz,
        block_width, len(frequency_blocks))
    cache, loaded, generated = {}, 0, 0
    progress = dict(phase='reference_cache', complete=False, cache_key=cache_key,
        nonempty_block_total=sum(bool(len(table)) for table in template_tables),
        reference_sources=sum(len(table) for table in template_tables))
    status_file = cache_file.with_suffix('.status.json')
    write_run_status(status_file, **progress, loaded_blocks=0, generated_blocks=0)
    nonempty_banks = ((block_index, table) for block_index, table
        in zip(selected_blocks, template_tables) if len(table))
    for block_index, table in tqdm(nonempty_banks,
            total=progress['nonempty_block_total'], desc='Reference cache', unit='nonempty bank'):
        block_range = frequency_blocks[block_index]
        padding = padding_by_block[block_index]
        retained_bins = np.flatnonzero((frequency >= block_range[0] - padding)
            & (frequency <= block_range[1] + padding) & (frequency > 0))
        ids = table[:, 0].astype(np.int64)
        group_name = f'{cache_key}/block_{block_index:05d}'
        values = None
        with _open_cache(cache_file, 'a') as saved:
            if group_name in saved and saved[group_name].attrs.get('complete', False):
                group = saved[group_name]
                values = {name: group[name][:] for name in ('dwd_id', 'retained_bins', 'waveforms')}
        if values is not None:
            loaded += 1
        else:
            physical_frequency = frequency[retained_bins]
            response = response_type(np.broadcast_to(np.eye(3), (len(retained_bins), 3, 3)),
                np.zeros((3, len(retained_bins)), complex), response_times, response_geometry,
                physical_frequency, duration, rate)
            doppler = table[:, [2, 3, 6, 7]]
            cos_iota = np.cos(table[:, 5])
            coefficients = amplitude_coordinates(table[:, 1] * (1 + cos_iota**2),
                2 * table[:, 1] * cos_iota, table[:, 8], table[:, 4]) / 1.e-22
            waveforms = np.empty((len(ids), 3, len(retained_bins)), dtype=complex)
            for first in range(0, len(ids), 32):
                waveforms[first:first + 32] = reconstruct_waveform(coefficients[first:first + 32],
                    response.build(doppler[first:first + 32]))
            values = dict(dwd_id=ids, retained_bins=retained_bins, waveforms=waveforms)
            del response, doppler, coefficients, waveforms
            # GPUResponse's bound JIT methods form Python reference cycles.
            # Collect each bank before constructing another full-year response;
            # the refined grid must not accumulate hundreds of device geometries.
            gc.collect()
            with _open_cache(cache_file, 'a') as saved:
                if group_name in saved:
                    del saved[group_name]
                group = saved.create_group(group_name)
                for name, value in values.items():
                    group.create_dataset(name, data=value)
                group.attrs['complete'] = True
            generated += 1
        cache[int(block_index)] = values
        if (loaded + generated) % 32 == 0:
            write_run_status(status_file, **progress, loaded_blocks=loaded, generated_blocks=generated)
    progress.update(phase='reference_cache_complete', complete=True)
    write_run_status(status_file, **progress, loaded_blocks=loaded, generated_blocks=generated)
    print(f'Reference cache: {sum(len(value["dwd_id"]) for value in cache.values())} sources; '
          f'{generated} blocks generated, {loaded} loaded', flush=True)
    return cache

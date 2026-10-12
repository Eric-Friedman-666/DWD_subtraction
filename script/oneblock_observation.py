"""Store one-block simulation inputs in one portable, grouped HDF5 file."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np


CACHED_FILES = (
    'population.h5', 'orbit.h5', 'observation.h5',
    'population_links.h5', 'population_tdi.h5',
)
SCHEMA = 'dwd-oneblock-observation'
SCHEMA_VERSION = 1
PARAMETER_FIELDS = dict(
    A='Amplitude', f='Frequency', df='FrequencyDerivative', phi0='InitialPhase',
    iota='Inclination', longitude='EclipticLongitude', latitude='EclipticLatitude',
    psi_bme='Polarization',
)
PARAMETERS = tuple(PARAMETER_FIELDS)


def read_checkpoint(group):
    """Read a checkpoint hierarchy into memory while its file is open."""
    contents = {}
    for name, item in group.items():
        if isinstance(item, h5py.Group):
            contents[name] = read_checkpoint(item)
        else:
            options = dict(dtype=item.dtype)
            if item.chunks is not None:
                options.update(chunks=item.chunks, maxshape=item.maxshape,
                               compression=item.compression, compression_opts=item.compression_opts,
                               shuffle=item.shuffle, fletcher32=item.fletcher32,
                               scaleoffset=item.scaleoffset)
            contents[name] = (item[()], dict(item.attrs), options)
    return dict(attributes=dict(group.attrs), contents=contents)


def write_checkpoint(group, snapshot):
    """Save a previously read checkpoint without opening its source file."""
    group.attrs.update(snapshot['attributes'])
    for name, value in snapshot['contents'].items():
        if isinstance(value, dict):
            write_checkpoint(group.create_group(name), value)
        else:
            data, attributes, options = value
            dataset = group.create_dataset(name, data=data, **options)
            dataset.attrs.update(attributes)




def pack_observation(directory, output_file, *, overwrite=False, time_domain_only=False):
    """Publish a portable input, optionally sharing the saved TD observation."""
    directory = Path(directory).expanduser().resolve()
    output = Path(output_file).expanduser().absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.dwd_observation_', dir=output.parent) as temporary:
        pending = Path(temporary) / 'observation.h5'
        with h5py.File(pending, 'x') as target:
            target.attrs.update(schema=SCHEMA, schema_version=SCHEMA_VERSION, complete=False)
            for name in CACHED_FILES:
                if time_domain_only and name == 'observation.h5':
                    continue
                with h5py.File(directory / name, 'r') as source:
                    source.copy('/', target, name=Path(name).stem,
                                expand_soft=True, expand_external=True, expand_refs=True)
            if time_domain_only:
                target['observation'] = target['population_tdi']
                target.attrs['data_domain'] = 'time'
            target.attrs['complete'] = True
            population = target['population']
            if 'catalogues/all' in population and 'ID' in (population['catalogues/all'].dtype.names or ()):
                target.attrs.update(injection_scope='full_catalogue',
                                    injected_dwd_count=len(population['catalogues/all']))
            elif population.attrs.get('injection_scope') == 'full_catalogue':
                target.attrs.update(injection_scope='full_catalogue',
                                    injected_dwd_count=int(population.attrs['source_count']))
            target.flush()
        # Publish the closed bundle atomically on the same filesystem.
        if overwrite:
            os.replace(pending, output)
        else:
            os.link(pending, output)
    return output


def prepare_cached_inputs(observation_file, work, cached_files=CACHED_FILES, *, frequency_range=None, duration=None):
    """Create disposable root-layout views for existing file-based libraries.

    Every dataset stays in the bundle. Only small HDF5 link files are made in
    the recovery workspace; deleting them does not delete or copy input data.
    """
    observation_file = Path(observation_file).expanduser().resolve(strict=True)
    work = Path(work)
    with h5py.File(observation_file, 'r') as saved:
        for name in cached_files:
            source = saved[Path(name).stem]
            with h5py.File(work / name, 'x') as view:
                for attr in source.attrs:
                    view.attrs.create(attr, source.attrs[attr], dtype=source.attrs.get_id(attr).dtype)
                if name == 'population.h5' and (source.attrs.get('injection_scope') == 'full_catalogue'
                        or ('catalogues/all' in source and 'ID' in (source['catalogues/all'].dtype.names or ()))):
                    low,high = np.asarray(frequency_range,dtype=float)
                    # This limits a post-fit truth audit, never the saved injection.
                    support = np.array([low,high])+np.array([-2304.,2304.])/duration
                    table_layout = ('catalogues/all' in source
                                    and 'ID' in (source['catalogues/all'].dtype.names or ()))
                    count = len(source['catalogues/all'] if table_layout else source['dwd_id'])
                    positions = []
                    for first in range(0,count,262144):
                        if table_layout:
                            records = source['catalogues/all'][first:first+262144]
                            f,df = records['Frequency'],records['FrequencyDerivative']
                        else:
                            f = source['parameters/f'][first:first+262144]
                            df = source['parameters/df'][first:first+262144]
                        margin = np.maximum(f,f+df*duration)*1.e-4+8/duration
                        selected = ((np.minimum(f,f+df*duration)-margin <= support[1])
                                    & (np.maximum(f,f+df*duration)+margin >= support[0]))
                        positions.append(first+np.flatnonzero(selected))
                    positions = np.concatenate(positions)
                    if table_layout:
                        view.create_dataset('catalogues/all',data=source['catalogues/all'][positions])
                    else:
                        for key in ('dwd_id','catalogue_index'):
                            view.create_dataset(key,data=source[key][positions])
                        for key in PARAMETERS:
                            view.create_dataset('parameters/'+key,data=source['parameters/'+key][positions])
                        frequency = view['parameters/f'][:]
                        view.create_dataset('core',data=(frequency>=low)&(frequency<high))
                    view.attrs.update(truth_view=True,source_count=len(positions),
                                      full_injected_source_count=count)
                    continue
                for key in source:
                    view[key] = h5py.ExternalLink(str(observation_file), f'{source.name}/{key}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='Directory containing the five legacy HDF5 inputs.')
    parser.add_argument('output_file', type=Path, help='New combined HDF5 file; an existing path is rejected.')
    args = parser.parse_args(argv)
    print(pack_observation(args.directory, args.output_file))


if __name__ == '__main__':
    main()

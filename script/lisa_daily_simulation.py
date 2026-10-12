"""Day-by-day LISA signal simulation; one continuous coloured-noise draw."""
import os
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
import h5py
import numpy as np
import lisagwresponse
import lisainstrument
import pytdi
import pytdi.michelson
from threadpoolctl import threadpool_limits
from tqdm.auto import tqdm
from script.tool import _bme_to_icrs


logging.getLogger('lisainstrument.instru.instru_fplan').addFilter(
    lambda record: not (
        record.levelno == logging.WARNING and record.getMessage() ==
        'Using default set of locking beatnote frequencies '
        'might cause interferometric beatnote frequencies to fall '
        'outside the requirement range of 5..25 MHz'
    )
)


def _instrument_settings(first_sample, sample_count, orbit, rate):
    padding=int(round(86400.*rate))
    return dict(size=sample_count+2*padding,dt=1./rate,t0=(first_sample-padding)/rate,
        orbits=str(orbit),orbit_dataset='tcb/ltt',
        aafilter=('kaiser',240.,.275*rate,.725*rate),
        clock_offsets=0.,clock_freqoffsets=0.,clock_freqlindrifts=0.,clock_freqquaddrifts=0.)


def _trimmed_xyz(instrument, sample_count, rate):
    data=pytdi.Data.from_instrument(instrument.export_numpy_full())
    first=int(round(86400.*rate))
    keep=slice(first,first+sample_count)
    return np.stack([getattr(pytdi.michelson,name).build(**data.args)(data.measurements)[keep]
                     /instrument.central_freq for name in ('X2','Y2','Z2')])


def _source_signal(parameters, index, first_sample, sample_count, orbit, rate, gw_file):
    padding=int(round(86400.*rate))
    source=lisagwresponse.GalacticBinary(
        **{name:float(parameters[name][index]) for name in ('A','f','df','phi0','iota','psi','ra','dec')},
        t_init=0.,orbits=str(orbit),orbit_interp_order=5,shift_sun='never')
    source.write(str(gw_file),t0=(first_sample-2*padding)/rate,dt=1./rate,
        size=sample_count+4*padding,timeframe='tcb',mode='x',chunks=100000)
    instrument=lisainstrument.Instrument(**_instrument_settings(first_sample,sample_count,orbit,rate),
        gws=str(gw_file))
    instrument.disable_all_noises()
    instrument.disable_dopplers()
    return _trimmed_xyz(instrument,sample_count,rate)


def _daily_signals(job):
    day,first,size,parameters,orbit,rate,directory,cpus=job
    if cpus:
        os.sched_setaffinity(0,{cpus[day % len(cpus)]})
    path=Path(directory)/f'day_{day:03d}.h5'
    with threadpool_limits(limits=1), h5py.File(path,'x') as saved:
        signals=saved.create_dataset('source_xyz_td',shape=(len(parameters['A']),3,size),dtype='f8')
        with TemporaryDirectory(prefix=f'links_{day:03d}_',dir=directory) as temporary:
            for index in range(len(parameters['A'])):
                signals[index]=_source_signal(parameters,index,first,size,orbit,rate,
                    Path(temporary)/f'source_{index}.h5')
        saved.attrs['first_sample']=first
    return str(path)


def simulate_lisa_daily(parameters, orbit_file, duration, sampling_rate,
        N_acc, delta_x, workers=30, *, noise_data=None):
    """Return per-source XYZ time series, pure noise, and its recorded seed.

    Daily signal workers share t_init=0 and trim filter/TDI padding before
    chronological stitching. Noise is drawn continuously across all days;
    independent daily noise restarts would lose long-time correlations.
    The Linux fork context avoids re-executing Eric's sequential top-level code.
    noise_data=(XYZ time series, seed) reuses an existing realization unchanged;
    N_acc and delta_x apply only when drawing new noise.
    """
    sample_count=int(round(duration*sampling_rate))
    day_samples=int(round(86400.*sampling_rate))
    days=(sample_count+day_samples-1)//day_samples
    right_ascension,declination,polarization=_bme_to_icrs(
        parameters['longitude'],parameters['latitude'],parameters['psi_bme'])
    sources={name:np.asarray(parameters[name]) for name in ('A','f','df','phi0','iota')}
    sources.update(ra=right_ascension,dec=declination,psi=polarization)
    cpus=sorted(os.sched_getaffinity(0))[:min(workers,30)] if hasattr(os,'sched_getaffinity') else []
    worker_count=min(30,workers,days,len(cpus) if cpus else workers)
    result=np.empty((len(parameters['A']),3,sample_count))
    with threadpool_limits(limits=1), TemporaryDirectory(prefix='daily_dwd_') as temporary:
        jobs=[(day,day*day_samples,min(day_samples,sample_count-day*day_samples),sources,
               str(Path(orbit_file).resolve()),sampling_rate,temporary,cpus) for day in range(days)]
        with ProcessPoolExecutor(max_workers=worker_count,
                mp_context=multiprocessing.get_context('fork')) as pool:
            for path in tqdm(pool.map(_daily_signals,jobs),total=days,desc='Daily LISA signals',unit='day'):
                with h5py.File(path,'r') as chunk:
                    first=int(chunk.attrs['first_sample'])
                    result[:,:,first:first+chunk['source_xyz_td'].shape[-1]]=chunk['source_xyz_td'][:]
        if noise_data is None:
            print('Simulating fresh ACC+OMS noise', flush=True)
            instrument=lisainstrument.Instrument(
                **_instrument_settings(0,sample_count,orbit_file,sampling_rate),
                testmass_asds=N_acc,oms_asds=(delta_x,0.,0.,0.,0.,0.))
            instrument.disable_all_noises(excluding={'test-mass','oms'})
            instrument.disable_dopplers()
            noise=_trimmed_xyz(instrument,sample_count,sampling_rate)
            seed=int(instrument.seed)
        else:
            print('Reusing saved ACC+OMS noise', flush=True)
            noise,seed=noise_data
    return result,noise,seed

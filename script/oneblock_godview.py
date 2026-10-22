"""Simulation-only, post-RJMCMC ghost assessment; never a sampler input."""
import h5py
import numpy as np
from script.oneblock_observation import PARAMETER_FIELDS
from script.dwd_bank_association import assign_bank_ids,classify_injection_recovery
from script.eric_Fstats import physical_amplitude_coordinates, reconstruct_waveform


def too_many_ghosts(ghost_count,source_count,threshold):
    return source_count > 0 and ghost_count/source_count > threshold


def new_recovered_dwd_ids(matched_ids,previously_recovered_ids):
    """New unique post-fit IDs; ghosts and reappearances are not discoveries."""
    current = np.asarray(matched_ids,dtype=np.int64)
    previous = np.asarray(previously_recovered_ids,dtype=np.int64)
    return np.setdiff1d(current[current>=0],previous[previous>=0])


class PopulationGodView:
    def __init__(self,directory,builder,indices,duration,rate,overlap_threshold=.9,*,
                 snr_file=None,frequency_range=None):
        self.directory,self.builder,self.indices = directory,builder,indices
        self.duration,self.rate,self.overlap_threshold = duration,rate,overlap_threshold
        self.snr_file,self.frequency_range = snr_file,frequency_range
        self.waveforms = None

    def __call__(self,recovered,covariance):
        if self.waveforms is None:
            # First called only after a complete RJMCMC round has returned.
            if self.snr_file is not None:
                # Audit all nearby injected sources, without the candidate SNR cut.
                with h5py.File(self.snr_file) as saved:
                    dgb = saved['sky/dgb/cat'][:, 0]
                    igb = saved['sky/igb/cat'][:, 0]
                offset = 0
                rows, ids = [], []
                low,high = np.asarray(self.frequency_range) + np.array([-2304.,2304.])/self.duration
                for records in (dgb, igb):
                    frequency,chirp = records['Frequency'],records['FrequencyDerivative']
                    end_frequency = frequency + chirp*self.duration
                    margin = np.maximum(frequency,end_frequency)*1.e-4 + 8/self.duration
                    selected = np.flatnonzero((np.minimum(frequency,end_frequency)-margin <= high)
                                              & (np.maximum(frequency,end_frequency)+margin >= low))
                    ids.append(offset + selected + 1)
                    rows.append({key:records[field][selected] for key,field in PARAMETER_FIELDS.items()})
                    offset += len(records)
                self.ids = np.concatenate(ids)
                parameters = {key:np.concatenate([row[key] for row in rows]) for key in PARAMETER_FIELDS}
                del dgb,igb,records,rows
            else:
                with h5py.File(self.directory/'population.h5') as saved:
                    if 'catalogues/all' in saved:
                        records = saved['catalogues/all'][:]
                        self.ids = records['ID']
                        parameters = {key:records[field] for key,field in PARAMETER_FIELDS.items()}
                    else:
                        self.ids = saved['dwd_id'][:]
                        parameters = {key:value[:] for key,value in saved['parameters'].items()}
            doppler = np.column_stack([parameters[key] for key in ('f','df','longitude','latitude')])
            amplitude = np.asarray(parameters['A'],dtype=np.float64)
            cos_iota = np.cos(np.asarray(parameters['iota'],dtype=np.float64))
            coefficients = physical_amplitude_coordinates(
                amplitude*(1+cos_iota**2),2*amplitude*cos_iota,
                parameters['psi_bme'],parameters['phi0'])/1e-22
            waveforms = []
            for first in range(0,len(doppler),32):
                basis = self.builder.build(doppler[first:first+32])[:,:,:,self.indices]
                waveforms.extend(reconstruct_waveform(coefficients[first:first+32],basis))
            self.waveforms = np.asarray(waveforms)
        association = assign_bank_ids(recovered,self.waveforms,self.ids,np.linalg.inv(covariance),
            self.duration,self.rate,minimum_overlap=self.overlap_threshold)
        classification = classify_injection_recovery(association,self.ids)
        ghosts = int(classification['ghost'].sum())
        return dict(ghost_count=ghosts,source_count=len(recovered),
            ghost_fraction=ghosts/max(len(recovered),1),overlap_threshold=self.overlap_threshold,
            overlap_comparison='strictly greater than',matched_dwd_id=association['matched_dwd_id'],
            matched_overlap=association['matched_overlap'],best_overlap=association['best_overlap'],
            ghost=classification['ghost'],scope='Simulation truth used after fitting for the stopping decision only.')

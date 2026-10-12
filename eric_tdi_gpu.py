"""Compatibility adapter; new pipelines call eric_generate_waveform.tdi_gpu."""
import numpy as np
from eric_generate_waveform import tdi_gpu


def TDI_simulation_gpu(data, components, *, keep=slice(None), central_freq=1.):
    names = tuple(dict.fromkeys(name for values in components.values() for name in values))
    measurements = {name: np.stack([np.broadcast_to(values.get(name, 0.),
                    (len(next(iter(data.args['delays'].values()))),))
                    for values in components.values()]) for name in names}
    xyz = tdi_gpu(measurements, data.args, keep=keep) / central_freq
    return dict(zip(components, xyz))

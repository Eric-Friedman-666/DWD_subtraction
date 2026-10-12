"""Complex anti-alias transfer used by the native LISA-package simulation."""
import inspect
import numpy as np
import scipy.signal
import lisainstrument


def native_signal_filter(frequency, detector_sampling_rate):
    """Preserve the causal FIR phase, not merely sqrt(filter power)."""
    upsampling = inspect.signature(lisainstrument.Instrument).parameters['physics_upsampling'].default
    physics_rate = float(detector_sampling_rate)*upsampling
    _, coefficients, delay = lisainstrument.instru.init_aafilter(
        ('kaiser', 240., .275*detector_sampling_rate, .725*detector_sampling_rate), physics_rate)
    _, transfer = scipy.signal.freqz(coefficients, worN=2*np.pi*np.asarray(frequency)/physics_rate)
    return transfer, float(delay)

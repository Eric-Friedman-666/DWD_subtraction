"""Prior-bounded likelihood support; ownership banks remain unchanged."""
import numpy as np


RESPONSE_STEP_SECONDS = 900.


def chirp_frequency_padding(duration, block_width, maximum_frequency, derivative_bounds):
    """Cover either-sign prior drift, orbital Doppler support and eight bins.

    This uses the declared prior, never catalogue truth derivatives or residual
    SNRs. It is a likelihood/reference window, not an ownership boundary.
    """
    values = np.asarray(derivative_bounds, dtype=float)
    if values.shape not in ((2,), (2, 2)) or not np.isfinite(values).all():
        raise ValueError('Derivative bounds must be finite intervals')
    duration, block_width, maximum_frequency = map(float, (duration, block_width, maximum_frequency))
    if min(duration, block_width, maximum_frequency) <= 0:
        raise ValueError('Positive duration, block width and maximum frequency required')
    return max(block_width, float(np.max(np.abs(values))) * duration
               + 1.e-4 * maximum_frequency + 8. / duration)


def response_grid(duration, rate, step_seconds=RESPONSE_STEP_SECONDS):
    """Midpoints of equal integer-native-sample blocks, including all samples."""
    duration, rate, step_seconds = map(float, (duration, rate, step_seconds))
    count = int(round(duration * rate))
    block = int(round(step_seconds * rate))
    if (duration <= 0 or rate <= 0 or block < 1
            or not np.isclose(count / rate, duration, rtol=0, atol=1.e-8)
            or not np.isclose(block / rate, step_seconds, rtol=0, atol=1.e-8)
            or count % block):
        raise ValueError('Response step must partition the complete native sample grid')
    return np.arange(count // block) * step_seconds + .5 * (block - 1) / rate


def response_sample_block(times, duration, rate):
    """Infer the FFT integration factor from its actual grid, never 3600 s."""
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
        raise ValueError('A finite nonempty response midpoint grid is required')
    step = float(duration) / len(times)
    expected = response_grid(duration, rate, step)
    if not np.allclose(times, expected, rtol=0, atol=1.e-7):
        raise ValueError('Response times do not match the complete uniform midpoint grid')
    return int(round(step * float(rate)))

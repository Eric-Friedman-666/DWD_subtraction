"""Measured covariance and analytical F-statistics for an existing DWD template bank."""
import numpy as np


def residual_covariance(data, duration, rate, *, neighboring_bins=256, chunk_size=16384):
    """Estimate the measured XYZ covariance without rejecting any frequency blocks."""
    from script.eric_preprocess import effective_noise_cov, periodogram

    data = np.asarray(data)
    frequencies = np.arange(data.shape[-1]) / duration
    covariance = np.zeros((len(frequencies), 3, 3), complex)
    for first in range(1, len(frequencies), chunk_size):
        last = min(first + chunk_size, len(frequencies))
        lower, upper = max(1, first - neighboring_bins), min(len(frequencies), last + neighboring_bins)
        psd = periodogram(data[:, lower:upper], duration, rate)
        if not np.any(psd):
            local = np.broadcast_to(np.eye(3) * 1.e-100, psd.shape).copy()
        else:
            local = effective_noise_cov(psd, neighboring_bins=neighboring_bins, downweight_peaks=True)
            scale = np.trace(local, axis1=-2, axis2=-1).real / 3
            floor = max(float(np.median(scale)) * 1.e-10, 1.e-100)
            local += floor * np.eye(3)
        retained = local[first - lower:last - lower]
        covariance[first:last] = retained
    covariance[0] = covariance[1]
    return covariance


def amplitude_f_statistic(data, bases, covariance, duration, rate):
    """Vectorized standard real four-amplitude projection, including XYZ cross terms."""
    bases = np.asarray(bases)
    inverse = np.linalg.inv(covariance)
    factor = 4 / (duration * rate**2)
    x = factor * np.einsum('smcf,fcd,df->sm', bases.conj(), inverse, data, optimize=True).real
    M = factor * np.einsum('smcf,fcd,sndf->smn', bases.conj(), inverse, bases, optimize=True).real
    M = .5 * (M + M.swapaxes(-1, -2))
    # A rank-deficient basis still defines a projection (especially near Nyquist).
    coefficients = np.einsum('smn,sn->sm', np.linalg.pinv(M, hermitian=True, rcond=1.e-12), x)
    two_f = np.maximum(np.einsum('sm,sm->s', x, coefficients), 0.)
    return dict(two_f=two_f, coefficients=coefficients, amplitude_matrix=M)


def template_f_statistic_search(data, covariance, builder, core_indices, duration, rate,
                                template_parameters, *, f_threshold=16., batch_size=128):
    """Evaluate F at every supplied template; maximize its four amplitudes analytically."""
    templates = np.asanyarray(template_parameters)
    if templates.dtype.names:
        doppler = np.column_stack([templates[name] for name in
            ('Frequency', 'FrequencyDerivative', 'EclipticLongitude', 'EclipticLatitude')])
    elif templates.ndim == 2 and templates.shape[1] == 11:
        doppler = templates[:, [2, 3, 6, 7]]
    else:
        templates = templates.reshape(-1, 4)
        doppler = templates
    scores = np.empty(len(templates))
    coefficients = np.empty((len(templates), 4))
    for first in range(0, len(templates), batch_size):
        points = doppler[first:first + batch_size]
        padded = np.vstack((points, np.repeat(points[-1:], batch_size - len(points), axis=0)))
        bases = builder.build(padded)[:len(points), :, :, core_indices]
        fit = amplitude_f_statistic(data, bases, covariance, duration, rate)
        scores[first:first + len(points)] = fit['two_f']
        coefficients[first:first + len(points)] = fit['coefficients']
    retained = scores >= 2 * f_threshold
    return dict(template_dwd=templates[retained], doppler=doppler[retained], two_f=scores[retained],
        coefficients=coefficients[retained], template_parameters=templates,
        template_two_f=scores, retained_template_indices=np.flatnonzero(retained),
        template_evaluations=len(templates))

"""Analytical F-statistics for an existing DWD template bank."""
import numpy as np
from script.eric_preprocess import inner_product


def F_statistic(data, waveform_table_fd, *, covariance, T_obs, fs):
    bases = np.asarray(waveform_table_fd)
    x = inner_product(bases, data, covariance, T_obs, fs)
    M = inner_product(bases[:, :, None], bases[:, None, :], covariance, T_obs, fs)
    M = .5 * (M + M.swapaxes(-1, -2))
    coefficients = np.einsum('smn,sn->sm', np.linalg.pinv(M, hermitian=True, rcond=1.e-12), x)
    return .5 * np.maximum(np.einsum('sm,sm->s', x, coefficients), 0.)

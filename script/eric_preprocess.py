import numpy as np
from scipy.signal import fftconvolve

from .eric_generate_waveform import PARAMETER_FIELDS, _single_waveform_td, generate_waveform_td


def periodogram(data,duration,rate):
    return 2/(duration*rate**2)*np.einsum('ik,jk->kij',data,data.conj())


def relative_covariance_change(first,second):
    lower = np.linalg.cholesky(first)
    inv = np.linalg.inv(lower)
    whitened = inv@second@inv.conj().swapaxes(-1,-2)
    return float(np.sqrt(np.mean(np.sum(abs(whitened-np.eye(3))**2,axis=(-2,-1))/3)))


def effective_noise_cov(psd, neighboring_bins=128, *, downweight_peaks=False):
    psd = np.asarray(psd, dtype=np.complex128)
    offsets = np.arange(-neighboring_bins, neighboring_bins+1)
    kernel = 0.5*(1+np.cos(np.pi*offsets/neighboring_bins))
    kernel /= kernel.sum()
    if not downweight_peaks:
        covariance = fftconvolve(psd, kernel[:, None, None], mode='same', axes=0)
        return 0.5*(covariance+covariance.conj().swapaxes(-1, -2))
    cutoff = 15.0
    normalization = 0.8503365527522067
    def weighted_average(weights, correction):
        denominator = fftconvolve(weights, kernel, mode='same')
        numerator = fftconvolve(psd*weights[:, None, None], kernel[:, None, None],
                                mode='same', axes=0)
        covariance = numerator/denominator[:, None, None]/correction
        covariance = 0.5*(covariance+covariance.conj().swapaxes(-1, -2))
        scale = np.trace(covariance, axis1=-2, axis2=-1).real/3
        return covariance+1.0e-5*scale[:, None, None]*np.eye(3)
    covariance = weighted_average(np.ones(len(psd)), 1.0)
    for _ in range(12):
        inverse = np.linalg.inv(covariance)
        power = np.einsum('kij,kji->k', inverse, psd).real
        weights = np.maximum(1-power/cutoff, 0.0)**2
        updated = weighted_average(weights, normalization)
        change = np.linalg.norm(updated-covariance)/np.linalg.norm(covariance)
        covariance = updated
        if change < 1.0e-5:
            break
    return covariance


def generate_waveform_fd(waveform_td, T_obs, fs):
    return np.fft.rfft(waveform_td[1:], n=int(round(T_obs*fs)), axis=-1)


def inner_product(a_fd, b_fd, noise_covariance, T_obs, fs):
    weighted_b = np.linalg.solve(noise_covariance, np.swapaxes(b_fd, -1, -2)[..., None])[..., 0]
    return 4/(T_obs*fs**2)*np.einsum(
        '...fc,...fc->...', np.conjugate(np.swapaxes(a_fd, -1, -2)), weighted_b, optimize=True
    ).real


def calculate_snr(dwd_parameters, noise_covariance, T_obs, fs, orbit_file,
                  first_bin=1):
    waveform_td = _single_waveform_td(dwd_parameters, T_obs, fs, orbit_file)
    waveform_fd = generate_waveform_fd(waveform_td, T_obs, fs)
    waveform_fd = waveform_fd[:, first_bin:first_bin+len(noise_covariance)]
    snr = np.sqrt(inner_product(waveform_fd, waveform_fd, noise_covariance, T_obs, fs))
    return snr

"""GPU DWD link synthesis and native LISA time-delay interferometry."""
import numpy as np


def DWD_simulation_gpu(DWD_parameters, detector_config):
    """Return the summed six-link response of the supplied DWD parameter rows.

    DWD_parameters is a float array with shape (n_DWD, 8), with columns:
    A, f, df, phi0, iota, ecliptic longitude, ecliptic latitude, polarization.
    An (n_DWD, 11) SNR table is also accepted: ID, those eight parameters,
    SNR_absolute, SNR_relative. The ID and SNR columns are ignored.
    Alternatively, pass a one-dimensional structured table with fields:
    Amplitude, Frequency, FrequencyDerivative, InitialPhase, Inclination,
    EclipticLongitude, EclipticLatitude, Polarization.
    Additional fields such as ID and SNR are ignored. Every supplied row
    contributes once, including repeated parameter sets. No catalogue or ID
    lookup is performed; sources need not appear in an LDC file.

    detector_config requires T_obs [s], dt [s], and orbits (orbit-file path).
    The equivalent fs [Hz] and orbit_file keys remain supported. Arm lengths
    and spacecraft motion come from that orbit file. Optional settings are
    gpu_batch_size (default 10000) and minimum_envelope_samples (default 512).
    include_padding=True retains two days on each side for instrument/TDI use.

    By default, the result is a float64 NumPy array with shape (round(T_obs * fs), 6),
    sampled at t = arange(N) / fs in native LISA link order
    (12, 23, 31, 13, 32, 21). Values are dimensionless link fractional
    frequency shifts, before instrument and TDI processing. No noise is added
    and no output file is written. CUDA is required for a nonempty table.
    With include_padding=True, N_pad = round(2 * 86400 * fs) samples are retained
    at each end; the first sample is at t = -N_pad / fs.
    """
    T_obs = float(detector_config['T_obs'])
    fs = float(detector_config['fs']) if 'fs' in detector_config else 1 / float(detector_config['dt'])
    N = int(round(T_obs * fs))
    include_padding = bool(detector_config.get('include_padding', False))
    padding = int(round(2 * 86400 * fs)) if include_padding else 0
    batch_size = int(detector_config.get('gpu_batch_size', 10000))
    DWD_parameters = np.asarray(DWD_parameters)
    fields = ('Amplitude', 'Frequency', 'FrequencyDerivative', 'InitialPhase',
              'Inclination', 'EclipticLongitude', 'EclipticLatitude', 'Polarization')
    named = DWD_parameters.dtype.names is not None
    if len(DWD_parameters) == 0:
        return np.zeros((N + 2 * padding, 6), dtype=np.float64)

    from astropy import units as u
    from astropy.coordinates import BarycentricMeanEcliptic, CartesianRepresentation, SkyCoord
    from astropy.time import Time
    from script.dwd_population_gpu import PopulationLinksGPU
    # Rotate the supplied sky position and polarization basis into ICRS.
    rotation = np.empty((3, 3))
    frame = BarycentricMeanEcliptic(equinox=Time('J2000'))
    for column, axis in enumerate(np.eye(3)):
        rotation[:, column] = SkyCoord(
            CartesianRepresentation(*(axis * u.one)), frame=frame,
        ).icrs.cartesian.xyz.value
    engine = PopulationLinksGPU(
        (detector_config['orbit_file'] if 'orbit_file' in detector_config else detector_config['orbits']), T_obs, fs, batch_size=batch_size,
        minimum=int(detector_config.get('minimum_envelope_samples', 512)),
    )
    for first in range(0, len(DWD_parameters), 2 * batch_size):
        selected = DWD_parameters[first:first + 2 * batch_size]
        values = (np.column_stack([selected[field] for field in fields])
                  if named else np.asarray(selected, dtype=np.float64))
        if not named and values.shape[1] == 11:
            values = values[:, 1:9]
        longitude, latitude, polarization = values[:, 5], values[:, 6], values[:, 7]
        cos_lon, sin_lon = np.cos(longitude), np.sin(longitude)
        cos_lat, sin_lat = np.cos(latitude), np.sin(latitude)
        direction = np.column_stack((cos_lat * cos_lon, cos_lat * sin_lon, sin_lat)) @ rotation.T
        transported_u = np.column_stack((sin_lon, -cos_lon, np.zeros_like(longitude))) @ rotation.T
        ra = np.mod(np.arctan2(direction[:, 1], direction[:, 0]), 2 * np.pi)
        dec = np.arcsin(np.clip(direction[:, 2], -1., 1.))
        u_icrs = np.column_stack((np.sin(ra), -np.cos(ra), np.zeros_like(ra)))
        v_icrs = np.column_stack((-np.sin(dec) * np.cos(ra),
                                  -np.sin(dec) * np.sin(ra), np.cos(dec)))
        basis_rotation = np.arctan2(np.einsum('ij,ij->i', transported_u, v_icrs),
                                   np.einsum('ij,ij->i', transported_u, u_icrs))
        psi = np.mod(polarization + basis_rotation, np.pi)
        engine.add(np.column_stack((values[:, :5], ra, dec, psi)))
    response = engine.finish()
    return (response if include_padding else response[engine.padding:engine.padding + N]).copy()


# Lagrange interpolation adapted from PyTDI (BSD 3-Clause).
# Copyright (c) 2022, California Institute of Technology and
# Max Planck Institute for Gravitational Physics (Albert Einstein Institute).
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

_CUDA_SHIFT = r'''
extern "C" __global__ void shift(
    const double *data, const double *shifts, const double *ratios,
    double *result, const long n, const int batches, const int halfp) {
    long i=(long)blockDim.x*blockIdx.x+threadIdx.x;
    if (i>=n) return;
    double q=floor(shifts[i]);
    double f=__dsub_rn(shifts[i],q);
    double taps[32];
    double factor=__dmul_rn(f,__dsub_rn(1.,f));
    for (int j=1;j<halfp;++j) {
        factor=__dmul_rn(factor,ratios[j]);
        taps[halfp-1-j]=__ddiv_rn(factor,__dadd_rn((double)j,f));
        taps[halfp+j]=__ddiv_rn(factor,__dsub_rn((double)(j+1),f));
    }
    taps[halfp-1]=__dsub_rn(1.,f);
    taps[halfp]=f;
    for (int j=2;j<halfp;++j) {
        double a=__ddiv_rn(f,(double)j);
        double b=__dsub_rn(1.,__dmul_rn(a,a));
        for (int k=0;k<2*halfp;++k) taps[k]=__dmul_rn(taps[k],b);
    }
    if (halfp>1) {
        double b=__dmul_rn(__dadd_rn(1.,f),__dsub_rn(1.,__ddiv_rn(f,(double)halfp)));
        for (int k=0;k<2*halfp;++k) taps[k]=__dmul_rn(taps[k],b);
    }
    long center=i+(long)q;
    if (center<-(halfp+1)) center=-(halfp+1);
    if (center>n+halfp-1) center=n+halfp-1;
    for (int batch=0;batch<batches;++batch) {
        double value=0.;
        for (int k=0;k<2*halfp;++k) {
            long index=center-(halfp-1)+k;
            double sample=(index>=0 && index<n)?data[(long)batch*n+index]:0.;
            value=__dadd_rn(value,__dmul_rn(taps[k],sample));
        }
        result[(long)batch*n+i]=value;
    }
}
'''


_TDI_ORBITS = {}


def tdi_gpu(mosa_links, detector_config, combination='XYZ2', *, t0=0., keep=slice(None)):
    """Return native-LISA XYZ time streams from six MOSA link measurements.

    mosa_links is an array of shape (..., N, 6), in native link order
    (12, 23, 31, 13, 32, 21). These are science-link measurements, with
    any instrument filtering already applied. The output is float64 with
    shape (..., 3, N), in X/Y/Z order and in the same units as the input.
    No FFT, instrument simulation, or file output is performed here.

    A dictionary of native PyTDI measurements is also accepted. Include
    sci_ij, tmi_ij and ref_ij when using full instrument noise: keeping these
    auxiliary beatnotes preserves the standard noise combinations. Missing
    beatnotes are zero, as in native Data.from_orbits. Keys '12', ..., '21'
    are shorthand for the six sci_ij measurements. Leading axes batch
    independent streams; the final axis of each dictionary value is time.

    detector_config requires fs [Hz] or dt [s], and orbits (or orbit_file).
    Moving unequal-arm light-travel times and derivatives come from that
    file, using orbit_dataset='tcb/ltt' by default. t0 gives the first
    input sample time. Native Data.args (fs, delays, delay_derivatives) can
    be passed instead to reuse already evaluated moving-orbit geometry.
    combination='XYZ2' selects second-generation Michelson by default;
    'XYZ1' selects first generation. keep is a slice removing input padding.
    Native orders 5/31 and FP64 arithmetic are preserved; CUDA is required.
    """
    import ctypes
    import copy
    from collections.abc import Mapping
    from pathlib import Path
    import sys
    from types import FunctionType, MethodType, SimpleNamespace
    import pytdi
    import pytdi.michelson

    channel_names = tuple(axis + combination[-1] for axis in 'XYZ')
    if isinstance(mosa_links, Mapping):
        provided = {('sci_' + str(name) if str(name) in pytdi.Data.MOSAS else str(name)):
                    np.asarray(value, dtype=np.float64) for name, value in mosa_links.items()}
    else:
        values = np.asarray(mosa_links, dtype=np.float64)
        provided = {'sci_' + mosa: values[..., column]
                    for column, mosa in enumerate(pytdi.Data.MOSAS)}
    shapes = [value.shape for value in provided.values() if value.ndim]
    shape = np.broadcast_shapes(*shapes)
    size = shape[-1]
    fs = float(detector_config['fs']) if 'fs' in detector_config else 1 / float(detector_config['dt'])
    if 'delays' in detector_config:
        native_args = detector_config
    else:
        orbit = Path(detector_config.get('orbit_file', detector_config.get('orbits', ''))).expanduser().resolve()
        stat = orbit.stat()
        dataset = detector_config.get('orbit_dataset', 'tcb/ltt')
        key = (str(orbit), stat.st_size, stat.st_mtime_ns, size, fs, float(t0), dataset)
        if key not in _TDI_ORBITS:
            data = pytdi.Data.from_orbits(str(orbit), fs=fs, t0=float(t0), dataset=dataset,
                                         sci_12=np.zeros(size, dtype=np.float64))
            _TDI_ORBITS.clear()
            _TDI_ORBITS[key] = data.args
        native_args = _TDI_ORBITS[key]

    cuda_root = Path(sys.prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages/nvidia'
    handles = [ctypes.CDLL(str(cuda_root / package / 'lib' / library), mode=ctypes.RTLD_GLOBAL)
               for package, library in (('cuda_runtime', 'libcudart.so.12'), ('cuda_nvrtc', 'libnvrtc.so.12'))]
    import cupy as cp
    kernel = cp.RawKernel(_CUDA_SHIFT, 'shift', options=('--fmad=false',))
    ratios = {halfp: cp.asarray(np.array([(-1)*(1-j/halfp)/(1+j/halfp)
                                          for j in range(halfp)], dtype=np.float64))
              for halfp in (3, 16)}

    def timeshift(values, shifts, order=31):
        if np.isscalar(values):
            return values
        if np.isscalar(shifts) and shifts == 0:
            return values
        values = cp.ascontiguousarray(values, dtype=cp.float64)
        shifts = cp.asarray(shifts, dtype=cp.float64)
        if values.size == 1:
            return values.reshape(())
        if shifts.size != size:
            shifts = cp.broadcast_to(shifts, (size,)).copy()
        halfp = (order + 1) // 2
        result = cp.empty_like(values)
        batches = values.size // size
        kernel(((size + 127)//128,), (128,),
               (values, shifts, ratios[halfp], result, np.int64(size), np.int32(batches), np.int32(halfp)))
        return result

    gpu_dsp = SimpleNamespace(timeshift=timeshift, diff=pytdi.dsp.diff)
    methods = {}
    for name in ('build_shifts', 'build'):
        original = getattr(pytdi.TDICombination, name)
        method = FunctionType(original.__code__, {**original.__globals__, 'dsp': gpu_dsp},
                              original.__name__, original.__defaults__, original.__closure__)
        method.__kwdefaults__ = original.__kwdefaults__
        methods[name] = method
    args = {'delays': {name: cp.asarray(value, dtype=cp.float64) for name, value in native_args['delays'].items()},
            'delay_derivatives': {name: cp.asarray(value, dtype=cp.float64) for name, value in native_args['delay_derivatives'].items()},
            'fs': fs}
    names = tuple(dict.fromkeys(name for channel in channel_names
                               for name in getattr(pytdi.michelson, channel).components))
    measurements = {name: cp.asarray(np.ascontiguousarray(
        np.broadcast_to(provided.get(name, 0.), shape).reshape(-1, size))) for name in names}
    channels = []
    for name in channel_names:
        native_combination = copy.copy(getattr(pytdi.michelson, name))
        native_combination.build_shifts = MethodType(methods['build_shifts'], native_combination)
        operator = methods['build'](native_combination, **args)
        channels.append(cp.asnumpy(operator(measurements)[..., keep]))
    xyz = np.stack(channels, axis=-2)
    return xyz.reshape(*shape[:-1], 3, xyz.shape[-1])

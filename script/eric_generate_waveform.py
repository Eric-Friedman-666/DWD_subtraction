import ctypes
import math
import os
import sys
import time
import warnings
from functools import lru_cache, partial
from importlib import import_module
from pathlib import Path
from threading import RLock

import h5py
import jax
import jax.numpy as jnp
import jax.numpy as xp
import lisagwresponse
import lisainstrument
import numpy as np
from astropy import units as u
from astropy.coordinates import BarycentricMeanEcliptic, CartesianRepresentation, SkyCoord
from astropy.time import Time
from lisagwresponse.response import LINKS
from lisainstrument.gwsource.gw_source import GWSource
from lisainstrument.sigpro.types_numpy import FuncOfTime
from lisainstrument.streams.streams import StreamBundle
from lisainstrument.streams.expression import StreamExpression
from lisainstrument.streams.delay import StreamDelayDyn
from lisainstrument.streams.firfilter import StreamFIR
from lisainstrument.streams.time import StreamDowsample
from lisainstrument.streams.segments import SegmentArray
from pytdi.michelson import X2, Y2, Z2
from scipy.interpolate import BSpline, make_interp_spline
from scipy.fft import next_fast_len
from tqdm.auto import tqdm

from script.native_lisa_filter import native_signal_filter
from script.tool import _bme_to_icrs_rotation

PARAMETER_FIELDS = (
    'Amplitude', 'Frequency', 'FrequencyDerivative', 'InitialPhase',
    'Inclination', 'EclipticLongitude', 'EclipticLatitude', 'Polarization',
)

_waveform_lock = RLock()
_waveform_state = {}


def generate_waveform_td(fs, T_obs, dwd_ids, orbit_file, *,
                         progress_desc='Generate full TD XYZ'):
    fs, T_obs = float(fs), float(T_obs)
    ids = np.asarray(dwd_ids)

    # 1. Keep the GPU implementation here; initialize its definitions only once.
    with _waveform_lock:
        if 'backend' not in _waveform_state:

            # Load the installed CUDA runtime, without importing another project module.
            cuda_root = (Path(sys.prefix)/'lib'
                         /f'python{sys.version_info.major}.{sys.version_info.minor}'
                         /'site-packages'/'nvidia')
            cuda_handles = []
            for package, library in (
                ('cuda_runtime', 'libcudart.so.12'), ('nvjitlink', 'libnvJitLink.so.12'),
                ('cublas', 'libcublasLt.so.12'), ('cublas', 'libcublas.so.12'),
                ('cusparse', 'libcusparse.so.12'), ('cusolver', 'libcusolver.so.11'),
                ('cufft', 'libcufft.so.11'), ('cudnn', 'libcudnn.so.9'),
                ('cuda_cupti', 'libcupti.so.12'), ('nccl', 'libnccl.so.2'),
            ):
                path = cuda_root/package/'lib'/library
                if path.exists():
                    cuda_handles.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))
            os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
            os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false')
            jax.config.update('jax_enable_x64', True)
            jax.config.update('jax_default_matmul_precision', 'highest')


            @lru_cache(maxsize=1)
            def _bme_to_icrs_rotation() -> np.ndarray:
                frame = BarycentricMeanEcliptic(equinox=Time("J2000"))
                columns = []
                for axis in np.eye(3):
                    coordinate = SkyCoord(
                        CartesianRepresentation(*(axis * u.one)),
                        frame=frame,
                    ).icrs
                    columns.append(coordinate.cartesian.xyz.value)
                return np.column_stack(columns)


            def _bme_to_icrs(
                longitude: np.ndarray,
                latitude: np.ndarray,
                polarization: np.ndarray,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                rotation = _bme_to_icrs_rotation()
                cos_longitude = np.cos(longitude)
                sin_longitude = np.sin(longitude)
                cos_latitude = np.cos(latitude)
                sin_latitude = np.sin(latitude)
                direction_bme = np.column_stack(
                    (
                        cos_latitude * cos_longitude,
                        cos_latitude * sin_longitude,
                        sin_latitude,
                    )
                )
                u_bme = np.column_stack(
                    (sin_longitude, -cos_longitude, np.zeros_like(longitude))
                )
                direction_icrs = direction_bme @ rotation.T
                transported_u = u_bme @ rotation.T

                ra = np.mod(
                    np.arctan2(direction_icrs[:, 1], direction_icrs[:, 0]),
                    2.0 * np.pi,
                )
                dec = np.arcsin(np.clip(direction_icrs[:, 2], -1.0, 1.0))
                u_icrs = np.column_stack(
                    (np.sin(ra), -np.cos(ra), np.zeros_like(ra))
                )
                v_icrs = np.column_stack(
                    (
                        -np.sin(dec) * np.cos(ra),
                        -np.sin(dec) * np.sin(ra),
                        np.cos(dec),
                    )
                )
                basis_rotation = np.arctan2(
                    np.einsum("ij,ij->i", transported_u, v_icrs),
                    np.einsum("ij,ij->i", transported_u, u_icrs),
                )
                psi = np.mod(polarization + basis_rotation, np.pi)

                return ra, dec, psi


            class GPUFullLinks:
                def __init__(self, parameters, orbit_file, source_batch=32):
                    self.jax = jax
                    try:
                        self.device = jax.devices('gpu')[0]
                    except RuntimeError:
                        self.device = jax.devices('cpu')[0]
                        warnings.warn('GPU unavailable; using CPU for DWD waveform link generation.',
                                      RuntimeWarning, stacklevel=2)
                    self.model = lisagwresponse.GalacticBinary(
                        A=1., f=.003, ra=0., dec=0., orbits=str(orbit_file),
                        orbit_interp_order=5, shift_sun='never')
                    self.links = np.array([12, 23, 31, 13, 32, 21])
                    # Obtain the package's column convention, rather than guessing link order.
                    self.links = np.asarray(LINKS)
                    ra, dec, psi = _bme_to_icrs(parameters['longitude'], parameters['latitude'],
                                              parameters['psi_bme'])
                    values = np.column_stack([parameters[name] for name in ('A','f','df','phi0','iota')]
                                             + [ra, dec, psi])
                    self.source_count = len(values)
                    padded = np.zeros(((len(values)+source_batch-1)//source_batch*source_batch, 8))
                    padded[:len(values)] = values
                    self.parameters = jax.device_put(padded.reshape(-1, source_batch, 8), self.device)
                    self.integrate = jax.jit(self._integrate)

                def geometry(self, times):
                    times = np.asarray(times)
                    trec = np.broadcast_to(times[:, None], (len(times), 6)).copy()
                    temi = trec.copy()
                    xrec, xemi = np.empty((len(times),6,3)), np.empty((len(times),6,3))
                    for column, link in enumerate(self.links):
                        rec, emi = int(link)//10, int(link)%10
                        temi[:,column] -= self.model.ltt[int(link)](times)
                        for axis, coordinate in enumerate((self.model.x,self.model.y,self.model.z)):
                            xrec[:,column,axis] = coordinate[rec](times)
                            xemi[:,column,axis] = coordinate[emi](temi[:,column])
                    direction = xrec-xemi
                    direction /= np.linalg.norm(direction,axis=-1)[...,None]
                    return trec, temi, xrec, xemi, direction

                @staticmethod
                def _integrate(parameters, groups, geometry):
                    trec, temi, xrec, xemi, direction = geometry
                    def batch(carry, inputs):
                        p, group = inputs
                        amplitude, f, df, phi, inc, ra, dec, psi = [p[:,i,None,None] for i in range(8)]
                        ra0, dec0 = p[:,5], p[:,6]
                        k = xp.stack([-xp.cos(dec0)*xp.cos(ra0),-xp.cos(dec0)*xp.sin(ra0),-xp.sin(dec0)],-1)
                        u = xp.stack([xp.sin(ra0),-xp.cos(ra0),xp.zeros_like(ra0)],-1)
                        v = xp.stack([-xp.sin(dec0)*xp.cos(ra0),-xp.sin(dec0)*xp.sin(ra0),xp.cos(dec0)],-1)
                        nk, nu, nv = [xp.einsum('tlc,sc->stl',direction,vec) for vec in (k,u,v)]
                        tr = trec[None]-xp.einsum('tlc,sc->stl',xrec,k)/299792458.
                        te = temi[None]-xp.einsum('tlc,sc->stl',xemi,k)/299792458.
                        phase_r = xp.pi*df*tr**2+2*xp.pi*f*tr-phi
                        phase_e = xp.pi*df*te**2+2*xp.pi*f*te-phi
                        plus, cross = nu**2-nv**2, 2*nu*nv
                        aplus = -amplitude*(1+xp.cos(inc)**2)
                        across = -2*amplitude*xp.cos(inc)
                        wave = (aplus*(plus*xp.cos(2*psi)+cross*xp.sin(2*psi))
                                *(xp.cos(phase_e)-xp.cos(phase_r))
                                + across*(-plus*xp.sin(2*psi)+cross*xp.cos(2*psi))
                                *(xp.sin(phase_e)-xp.sin(phase_r)))/(2*(1-nk))
                        return carry+xp.einsum('sg,stl->gtl',group,wave), None
                    initial = xp.zeros((groups.shape[-1],trec.shape[0],6),dtype=xp.float64)
                    return jax.lax.scan(batch,initial,(parameters,groups))[0]

                def evaluate(self, times, core=None):
                    if core is None:
                        core = np.ones(self.source_count,dtype=bool)
                    groups = np.zeros((np.prod(self.parameters.shape[:2]),2))
                    groups[:self.source_count,0] = core
                    groups[:self.source_count,1] = ~np.asarray(core,dtype=bool)
                    groups = self.jax.device_put(groups.reshape(*self.parameters.shape[:2],2),self.device)
                    geometry = tuple(self.jax.device_put(value,self.device) for value in self.geometry(times))
                    return np.asarray(self.integrate(self.parameters,groups,geometry))


            @partial(jax.jit, static_argnames=('order', 'offset'))
            def _lagrange_shift(data, shifts, order=31, offset=0):
                half = (order+1)//2
                integer = jnp.floor(shifts).astype(jnp.int64)
                fraction = shifts-integer
                factor = fraction*(1-fraction)
                taps = [None]*(2*half)
                for j in range(1, half):
                    factor = factor*(-(1-j/half)/(1+j/half))
                    taps[half-1-j] = factor/(j+fraction)
                    taps[half+j] = factor/(j+1-fraction)
                taps[half-1] = 1-fraction
                taps[half] = fraction
                common = jnp.ones_like(fraction)
                for j in range(2, half):
                    common = common*(1-(fraction/j)**2)
                common = common*(1+fraction)*(1-fraction/half)

                indices = jnp.arange(shifts.size)+integer+offset-(half-1)
                result = jnp.zeros_like(shifts)
                for j in range(2*half):
                    index = indices+j
                    sample = jnp.where((index >= 0) & (index < data.size),
                                       data[jnp.clip(index, 0, data.size-1)], 0.)
                    result = result+taps[j]*common*sample
                return result


            @jax.jit
            def _interpolate_spline(coefficients, first, weights):
                result = jnp.zeros((first.size, coefficients.shape[1]), dtype=jnp.float64)
                for j in range(6):
                    result = result+weights[:, j, None]*coefficients[first+j]
                return result


            @jax.jit
            def _fir(data, coefficients):
                return jnp.convolve(data, coefficients[::-1], mode='valid')


            @jax.jit
            def _add_tdi_term(result, measurement, shift_samples, derivative, factor):
                return result+factor*(1+derivative)*_lagrange_shift(measurement, shift_samples)


            @jax.jit
            def _quintic_coefficients(values, symbol, impulse, correction, rows, columns):
                length = impulse.size
                base = jnp.fft.irfft(jnp.fft.rfft(values, n=length, axis=0)/symbol[:, None],
                                     n=length, axis=0)
                delta = correction@base[columns]
                result = base[:values.shape[0]]
                index = jnp.arange(values.shape[0])
                for j in range(rows.size):
                    result = result-impulse[(index-rows[j]) % length, None]*delta[j]
                return result


            class _UniformQuintic:
                def __init__(self, times, device):
                    size = len(times)
                    length = next_fast_len(size, real=True)
                    self.knots = np.r_[np.repeat(times[0], 6), times[3:-3],
                                       np.repeat(times[-1], 6)]
                    rows = np.r_[np.arange(7), np.arange(size-7, size)]
                    native = BSpline.design_matrix(times[rows], self.knots, 5)
                    offsets = np.arange(-2, 3)
                    columns = np.unique(np.r_[native.indices,
                                              ((rows[:, None]+offsets) % length).ravel()])
                    boundary = np.zeros((len(rows), len(columns)))
                    for i, row in enumerate(rows):
                        first, last = native.indptr[i:i+2]
                        boundary[i, np.searchsorted(columns, native.indices[first:last])] = native.data[first:last]
                        boundary[i, np.searchsorted(columns, (row+offsets) % length)] -= np.array([1, 26, 66, 26, 1])/120
                    angle = 2*np.pi*np.arange(length//2+1)/length
                    symbol = (66+52*np.cos(angle)+2*np.cos(2*angle))/120
                    impulse = np.fft.irfft(1/symbol, n=length)
                    selected = impulse[(columns[:, None]-rows) % length]
                    correction = np.linalg.solve(np.eye(len(rows))+boundary@selected, boundary)
                    self.arguments = tuple(jax.device_put(value, device) for value in
                                           (symbol, impulse, correction, rows, columns))

                def __call__(self, values):
                    return _quintic_coefficients(values, *self.arguments)


            class _GWInputs(GWSource):
                def link_gw(self, mosa):
                    return FuncOfTime(lambda times: np.zeros_like(times))


            class _WaveformEngine:
                def __init__(self, parameters, T_obs, fs, orbit_file):
                    started = time.perf_counter()
                    self.size = int(round(T_obs*fs))
                    self.padding = int(round(800*fs))
                    self.fs = fs
                    self.links = GPUFullLinks(parameters, orbit_file, source_batch=1)
                    self.device = self.links.device
                    size, padding = self.size, self.padding
                    self.times = -2*padding/fs+np.arange(size+4*padding)*(1/fs)
                    self.geometry = tuple(jax.device_put(value, self.device)
                                          for value in self.links.geometry(self.times))

                    # Use the original native model: 4x internal sampling, same causal
                    # anti-alias filter, laser locking, moving delays, padding and units.
                    instrument = lisainstrument.Instrument(
                        size=size+2*padding, dt=1/fs, t0=-padding/fs,
                        orbits=str(orbit_file), orbit_dataset='tcb/ltt',
                        aafilter=('kaiser', 240., .275*fs, .725*fs),
                        clock_offsets=0., clock_freqoffsets=0., clock_freqlindrifts=0.,
                        clock_freqquaddrifts=0.)
                    instrument.disable_all_noises()
                    instrument.disable_dopplers()
                    instrument.gwsrc = _GWInputs()
                    self.central_freq = instrument.central_freq
                    self.model = instrument.get_model()
                    self.bundle = StreamBundle()
                    self.outputs = {}
                    for field in ('sci', 'ref', 'tmi'):
                        for mosa, node in getattr(self.model, field+'_carrier_fluctuations').items():
                            name = field+'_'+mosa
                            self.outputs[name] = node.id
                            self.bundle.add((name,), node, (0, size+2*padding))
                    for mosa, node in self.model.mprs.items():
                        self.bundle.add(('d_'+mosa,), node, (0, size+2*padding))

                    self.nodes = self.bundle.get_streams()
                    self.ranges = self.bundle._req_range
                    self.leaves = {node.id: mosa for mosa, node in self.model.gws.items()}
                    self.dynamic = {}
                    self.order = []

                    def visit(node):
                        if node.id in self.dynamic:
                            return self.dynamic[node.id]
                        dependencies = [visit(dep.stream) for dep in node.dependencies]
                        self.dynamic[node.id] = node.id in self.leaves or any(dependencies)
                        self.order.append(node.id)
                        return self.dynamic[node.id]

                    for node in self.nodes.values():
                        visit(node)

                    # Source-independent stream values are prepared once, not per DWD.
                    memo = {}

                    def native(sid):
                        if sid not in memo:
                            node = self.nodes[sid]
                            first, last = self.ranges[sid]
                            segment, state = node.generate(
                                None, [native(dep.stream.id) for dep in node.dependencies],
                                first, last)
                            memo[sid] = segment
                        return memo[sid]

                    constants = {dep.stream.id for sid in self.order if self.dynamic[sid]
                                 for dep in self.nodes[sid].dependencies
                                 if not self.dynamic[dep.stream.id]}
                    constants.update(sid for sid in self.outputs.values() if not self.dynamic[sid])
                    self.constants = {}
                    for sid in constants:
                        segment = native(sid)
                        value = segment.data if isinstance(segment, SegmentArray) else segment.const
                        self.constants[sid] = jax.device_put(value, self.device)

                    delays = {}
                    for mosa in self.model.mprs:
                        sid = self.bundle.output_from_dsid(('d_'+mosa,))
                        value = np.empty(size+2*padding)
                        native(sid).write(value, istart=0, istop=len(value))
                        delays['d_'+mosa] = value

                    self.operations = {}
                    self.filters = {}
                    for sid in self.order:
                        if not self.dynamic[sid]:
                            continue
                        node = self.nodes[sid]
                        if sid in self.leaves:
                            continue
                        if isinstance(node, StreamExpression):
                            self.operations[sid] = jax.jit(node._func)
                        elif isinstance(node, StreamFIR):
                            self.filters[sid] = jax.device_put(
                                np.asarray(node._firdef.filter_coeffs), self.device)

                    # Match the native HDF5 reader's 100-sample interpolation margins.
                    tmin = -padding/fs
                    tmax = tmin+instrument.physics_dt*instrument.physics_size
                    self.k0 = max(0, int(np.floor((tmin-self.times[0])*fs))-100)
                    self.k1 = min(len(self.times), int(np.ceil((tmax-self.times[0])*fs))+101)
                    times = self.times[self.k0:self.k1]
                    steps = np.diff(times)
                    self.spline_solver = (_UniformQuintic(times, self.device)
                                          if len(times) >= 32 and np.all(steps == steps[0]) else None)
                    if self.spline_solver is not None:
                        self._prepare_spline(self.spline_solver.knots)
                    self._prepare_tdi(delays)
                    self.initialization_seconds = time.perf_counter()-started

                def _prepare_tdi(self, delays):
                    self.shifts = {(): jnp.zeros(self.size+2*self.padding)}
                    self.derivatives = {(): jnp.zeros(self.size+2*self.padding)}
                    for name, value in delays.items():
                        operator = ('D_'+name[2:],)
                        self.shifts[operator] = jax.device_put(-value, self.device)
                        self.derivatives[operator] = jax.device_put(
                            -np.gradient(value, 1/self.fs), self.device)
                    operators = {tuple(operator) for channel in (X2, Y2, Z2)
                                 for polynomial in channel.components.values()
                                 for _, operator in polynomial}
                    for operator in sorted(operators, key=len):
                        if operator in self.shifts:
                            continue
                        shift = jnp.zeros(self.size+2*self.padding)
                        derivative = jnp.zeros_like(shift)
                        for atom in operator:
                            derivative = derivative+_lagrange_shift(
                                self.derivatives[(atom,)], shift*self.fs, order=5)
                            shift = shift+_lagrange_shift(
                                self.shifts[(atom,)], shift*self.fs, order=5)
                        self.shifts[operator] = shift
                        self.derivatives[operator] = derivative
                    self.shifts = {operator: value*self.fs for operator, value in self.shifts.items()}
                    jax.block_until_ready((self.shifts, self.derivatives))

                def _prepare_spline(self, knots):
                    self.spline_start = min(self.ranges[sid][0] for sid in self.leaves)
                    stop = max(self.ranges[sid][1] for sid in self.leaves)
                    sid = next(iter(self.leaves))
                    time_node = self.nodes[sid].dependencies[0].stream
                    times = np.arange(self.spline_start, stop)*time_node._dt+time_node._t0
                    low, high = self.times[self.k0], self.times[self.k1-1]
                    matrix = BSpline.design_matrix(np.clip(times, low, high), knots, 5)
                    weights = matrix.data.reshape(-1, 6)
                    weights[(times < low) | (times > high)] = 0.
                    self.spline_first = jax.device_put(matrix.indices.reshape(-1, 6)[:, 0], self.device)
                    self.spline_weights = jax.device_put(weights, self.device)

                def evaluate(self, parameters):
                    started = time.perf_counter()
                    ra, dec, psi = _bme_to_icrs(parameters['longitude'], parameters['latitude'],
                                              parameters['psi_bme'])
                    values = np.column_stack([parameters[name] for name in ('A', 'f', 'df', 'phi0', 'iota')]
                                             + [ra, dec, psi])
                    source = jax.device_put(values.reshape(1, 1, 8), self.device)
                    raw = self.links.integrate(source, jnp.ones((1, 1, 1)), self.geometry)[0]
                    if self.spline_solver is not None:
                        coefficients = self.spline_solver(raw[self.k0:self.k1])
                    else:
                        spline = make_interp_spline(self.times[self.k0:self.k1],
                                                   np.asarray(raw[self.k0:self.k1]), k=5)
                        coefficients = jax.device_put(spline.c, self.device)
                        if not hasattr(self, 'spline_weights'):
                            self._prepare_spline(spline.t)
                    interpolated = _interpolate_spline(
                        coefficients, self.spline_first, self.spline_weights)

                    samples = dict(self.constants)
                    for sid, mosa in self.leaves.items():
                        first, last = self.ranges[sid]
                        column = list(self.links.links).index(int(mosa))
                        samples[sid] = interpolated[first-self.spline_start:last-self.spline_start, column]
                    self.link_seconds = time.perf_counter()-started

                    def sliced(sid, first, last, step=1):
                        value = samples[sid]
                        if value.ndim == 0:
                            return value
                        return value[first-self.ranges[sid][0]:last-self.ranges[sid][0]:step]

                    for sid in self.order:
                        if not self.dynamic[sid] or sid in self.leaves:
                            continue
                        node = self.nodes[sid]
                        first, last = self.ranges[sid]
                        dependencies = [dep.stream.id for dep in node.dependencies]
                        if isinstance(node, StreamExpression):
                            samples[sid] = self.operations[sid](
                                *[sliced(dep, first, last) for dep in dependencies])
                        elif isinstance(node, StreamDelayDyn):
                            samples[sid] = _lagrange_shift(
                                samples[dependencies[0]],
                                -sliced(dependencies[1], first, last)*node._sample_rate,
                                offset=first-self.ranges[dependencies[0]][0])
                        elif isinstance(node, StreamFIR):
                            samples[sid] = _fir(
                                sliced(dependencies[0], first-node._opfir.margin_left,
                                       last+node._opfir.margin_right), self.filters[sid])
                        elif isinstance(node, StreamDowsample):
                            samples[sid] = sliced(
                                dependencies[0], first*node._ratio+node._offs,
                                (last-1)*node._ratio+node._offs+1, node._ratio)
                    measurements = {}
                    for name, sid in self.outputs.items():
                        value = sliced(sid, 0, self.size+2*self.padding)
                        measurements[name] = (jnp.full(self.size+2*self.padding, value)
                                              if value.ndim == 0 else value)
                    jax.block_until_ready(measurements)
                    self.instrument_seconds = time.perf_counter()-started-self.link_seconds

                    result = []
                    for channel in (X2, Y2, Z2):
                        output = jnp.zeros(self.size+2*self.padding)
                        for name, polynomial in channel.components.items():
                            for factor, operator in polynomial:
                                operator = tuple(operator)
                                output = _add_tdi_term(
                                    output, measurements[name], self.shifts[operator],
                                    self.derivatives[operator], float(factor))
                        result.append(np.asarray(
                            output[self.padding:self.padding+self.size]/self.central_freq))
                    self.total_seconds = time.perf_counter()-started
                    return np.vstack((np.arange(self.size)/self.fs, result))


            class FusedFullRate:
                def __init__(self, engine):
                    self.engine = engine
                    self.arguments = (engine.geometry, engine.spline_solver.arguments,
                        engine.spline_first, engine.spline_weights, engine.constants,
                        engine.filters, engine.shifts, engine.derivatives)
                    self.links = jax.jit(self._links)
                    self.instrument = jax.jit(self._instrument)
                    self.tdi = jax.jit(self._tdi)
                    self.all_stages = jax.jit(self._all_stages)

                def source(self, parameters):
                    ra, dec, psi = _bme_to_icrs(parameters['longitude'], parameters['latitude'], parameters['psi_bme'])
                    values = np.column_stack([parameters[name] for name in ('A','f','df','phi0','iota')]+[ra,dec,psi])
                    return jax.device_put(values.reshape(1,1,8), self.engine.device)

                def _links(self, source, geometry, spline_arguments, first, weights):
                    engine = self.engine
                    raw = engine.links.integrate(source, jnp.ones((1,1,1)), geometry)[0]
                    coefficients = _quintic_coefficients(raw[engine.k0:engine.k1], *spline_arguments)
                    return _interpolate_spline(coefficients, first, weights)

                def _instrument(self, interpolated, constants, filters):
                    engine = self.engine
                    samples = dict(constants)
                    for sid, mosa in engine.leaves.items():
                        first,last = engine.ranges[sid]
                        column = list(engine.links.links).index(int(mosa))
                        samples[sid] = interpolated[first-engine.spline_start:last-engine.spline_start,column]

                    def sliced(sid, first, last, step=1):
                        value = samples[sid]
                        if value.ndim == 0:
                            return value
                        return value[first-engine.ranges[sid][0]:last-engine.ranges[sid][0]:step]

                    for sid in engine.order:
                        if not engine.dynamic[sid] or sid in engine.leaves:
                            continue
                        node = engine.nodes[sid]
                        first,last = engine.ranges[sid]
                        dependencies = [dep.stream.id for dep in node.dependencies]
                        if isinstance(node, StreamExpression):
                            samples[sid] = node._func(*[sliced(dep,first,last) for dep in dependencies])
                        elif isinstance(node, StreamDelayDyn):
                            samples[sid] = _lagrange_shift(samples[dependencies[0]],
                                -sliced(dependencies[1],first,last)*node._sample_rate,
                                offset=first-engine.ranges[dependencies[0]][0])
                        elif isinstance(node, StreamFIR):
                            samples[sid] = _fir(sliced(dependencies[0], first-node._opfir.margin_left,
                                last+node._opfir.margin_right), filters[sid])
                        elif isinstance(node, StreamDowsample):
                            samples[sid] = sliced(dependencies[0], first*node._ratio+node._offs,
                                (last-1)*node._ratio+node._offs+1, node._ratio)
                    measurements = {}
                    for name,sid in engine.outputs.items():
                        value = sliced(sid,0,engine.size+2*engine.padding)
                        measurements[name] = (jnp.full(engine.size+2*engine.padding,value)
                                              if value.ndim == 0 else value)
                    return measurements

                def _tdi(self, measurements, shifts, derivatives):
                    engine = self.engine
                    result = []
                    for channel in (X2,Y2,Z2):
                        output = jnp.zeros(engine.size+2*engine.padding)
                        for name,polynomial in channel.components.items():
                            for factor,operator in polynomial:
                                operator = tuple(operator)
                                output = _add_tdi_term(output, measurements[name], shifts[operator],
                                                       derivatives[operator], float(factor))
                        result.append(output[engine.padding:engine.padding+engine.size]/engine.central_freq)
                    return jnp.stack(result)

                def _all_stages(self, source, geometry, spline_arguments, first, weights,
                                constants, filters, shifts, derivatives):
                    links = self._links(source,geometry,spline_arguments,first,weights)
                    measurements = self._instrument(links,constants,filters)
                    return self._tdi(measurements,shifts,derivatives)

                def device_waveform(self, source, mode='fused'):
                    if mode == 'fused':
                        return self.all_stages(source,*self.arguments)
                    geometry,spline_arguments,first,weights,constants,filters,shifts,derivatives = self.arguments
                    links = self.links(source,geometry,spline_arguments,first,weights)
                    measurements = self.instrument(links,constants,filters)
                    return self.tdi(measurements,shifts,derivatives)

                def host_waveform(self, source, mode='fused'):
                    xyz = np.asarray(self.device_waveform(source,mode))
                    return np.vstack((np.arange(self.engine.size)/self.engine.fs,xyz))


            class WindowEngine(_WaveformEngine):
                def __init__(self, parameters, T_obs, fs, orbit_file, epoch):
                    started = time.perf_counter()
                    self.size = int(round(T_obs*fs))
                    self.padding = int(round(800*fs))
                    self.fs = fs
                    self.links = GPUFullLinks(parameters, orbit_file, source_batch=1)
                    self.device = self.links.device
                    size, padding = self.size, self.padding
                    self.times = epoch-2*padding/fs+np.arange(size+4*padding)*(1/fs)
                    self.geometry = tuple(jax.device_put(value, self.device)
                                          for value in self.links.geometry(self.times))

                    # Use the original native model: 4x internal sampling, same causal
                    # anti-alias filter, laser locking, moving delays, padding and units.
                    instrument = lisainstrument.Instrument(
                        size=size+2*padding, dt=1/fs, t0=epoch-padding/fs,
                        orbits=str(orbit_file), orbit_dataset='tcb/ltt',
                        aafilter=('kaiser', 240., .275*fs, .725*fs),
                        clock_offsets=0., clock_freqoffsets=0., clock_freqlindrifts=0.,
                        clock_freqquaddrifts=0.)
                    instrument.disable_all_noises()
                    instrument.disable_dopplers()
                    instrument.gwsrc = _GWInputs()
                    self.central_freq = instrument.central_freq
                    self.model = instrument.get_model()
                    self.bundle = StreamBundle()
                    self.outputs = {}
                    for field in ('sci', 'ref', 'tmi'):
                        for mosa, node in getattr(self.model, field+'_carrier_fluctuations').items():
                            name = field+'_'+mosa
                            self.outputs[name] = node.id
                            self.bundle.add((name,), node, (0, size+2*padding))
                    for mosa, node in self.model.mprs.items():
                        self.bundle.add(('d_'+mosa,), node, (0, size+2*padding))

                    self.nodes = self.bundle.get_streams()
                    self.ranges = self.bundle._req_range
                    self.leaves = {node.id: mosa for mosa, node in self.model.gws.items()}
                    self.dynamic = {}
                    self.order = []

                    def visit(node):
                        if node.id in self.dynamic:
                            return self.dynamic[node.id]
                        dependencies = [visit(dep.stream) for dep in node.dependencies]
                        self.dynamic[node.id] = node.id in self.leaves or any(dependencies)
                        self.order.append(node.id)
                        return self.dynamic[node.id]

                    for node in self.nodes.values():
                        visit(node)

                    # Source-independent stream values are prepared once, not per DWD.
                    memo = {}

                    def native(sid):
                        if sid not in memo:
                            node = self.nodes[sid]
                            first, last = self.ranges[sid]
                            segment, state = node.generate(
                                None, [native(dep.stream.id) for dep in node.dependencies],
                                first, last)
                            memo[sid] = segment
                        return memo[sid]

                    constants = {dep.stream.id for sid in self.order if self.dynamic[sid]
                                 for dep in self.nodes[sid].dependencies
                                 if not self.dynamic[dep.stream.id]}
                    constants.update(sid for sid in self.outputs.values() if not self.dynamic[sid])
                    self.constants = {}
                    for sid in constants:
                        segment = native(sid)
                        value = segment.data if isinstance(segment, SegmentArray) else segment.const
                        self.constants[sid] = jax.device_put(value, self.device)

                    delays = {}
                    for mosa in self.model.mprs:
                        sid = self.bundle.output_from_dsid(('d_'+mosa,))
                        value = np.empty(size+2*padding)
                        native(sid).write(value, istart=0, istop=len(value))
                        delays['d_'+mosa] = value

                    self.operations = {}
                    self.filters = {}
                    for sid in self.order:
                        if not self.dynamic[sid]:
                            continue
                        node = self.nodes[sid]
                        if sid in self.leaves:
                            continue
                        if isinstance(node, StreamExpression):
                            self.operations[sid] = jax.jit(node._func)
                        elif isinstance(node, StreamFIR):
                            self.filters[sid] = jax.device_put(
                                np.asarray(node._firdef.filter_coeffs), self.device)

                    # Match the native HDF5 reader's 100-sample interpolation margins.
                    tmin = epoch-padding/fs
                    tmax = tmin+instrument.physics_dt*instrument.physics_size
                    self.k0 = max(0, int(np.floor((tmin-self.times[0])*fs))-100)
                    self.k1 = min(len(self.times), int(np.ceil((tmax-self.times[0])*fs))+101)
                    times = self.times[self.k0:self.k1]
                    steps = np.diff(times)
                    self.spline_solver = (_UniformQuintic(times, self.device)
                                          if len(times) >= 32 and np.all(steps == steps[0]) else None)
                    if self.spline_solver is not None:
                        self._prepare_spline(self.spline_solver.knots)
                    self._prepare_tdi(delays)
                    self.initialization_seconds = time.perf_counter()-started


            class NativePointPipeline(FusedFullRate):
                def _tdi(self, measurements, shifts, derivatives):
                    e=self.engine
                    result=[]
                    for channel in (X2,Y2,Z2):
                        output=jnp.zeros(e.size)
                        for name,polynomial in channel.components.items():
                            for factor,operator in polynomial:
                                operator=tuple(operator)
                                shifted=_lagrange_shift(measurements[name],
                                    shifts[operator][e.padding:e.padding+e.size],offset=e.padding)
                                output=output+factor*(1+derivatives[operator][e.padding:e.padding+e.size])*shifted
                        result.append(output/e.central_freq)
                    return jnp.stack(result)


            def match_nodes(base, other):
                mapping={}
                def visit(a,b):
                    if a in mapping:
                        return
                    mapping[a]=b
                    left,right=base.nodes[a].dependencies,other.nodes[b].dependencies
                    for x,y in zip(left,right): visit(x.stream.id,y.stream.id)
                for name in base.outputs:
                    visit(base.outputs[name],other.outputs[name])
                return mapping


            class NativeSupportTD:
                def __init__(self,parameters,T_obs,fs,orbit_file,intervals=128,share_filters=False):
                    started=time.perf_counter()
                    self.size=int(round(T_obs*fs)); self.T_obs=T_obs; self.fs=fs
                    self.support_times=np.linspace(0,T_obs,intervals+1)
                    packed=[];base=None
                    for index,epoch in enumerate(self.support_times):
                        engine=WindowEngine(parameters,1/fs,fs,orbit_file,float(epoch))
                        if base is None:
                            base=engine
                            self.pipeline=NativePointPipeline(base)
                        remap=match_nodes(base,engine)
                        arguments=(engine.geometry,engine.spline_solver.arguments,
                            engine.spline_first,engine.spline_weights,
                            {key:engine.constants[remap[key]] for key in base.constants},
                            {key:engine.filters[remap[key]] for key in base.filters},
                            engine.shifts,engine.derivatives)
                        packed.append(arguments)
                        if index%32==0:
                            print(f'Native time-domain geometry {index+1}/{len(self.support_times)}',flush=True)
                    self.device=base.device
                    self.arguments=jax.tree.map(lambda *values:jnp.stack(values),*packed)
                    if share_filters:
                        # Native FIR coefficients do not depend on the epoch. Share them so
                        # the GPU sees batched convolution, not one different filter per group.
                        args=list(self.arguments);args[5]=packed[0][5];self.arguments=tuple(args)
                        axes=(None,0,0,0,0,0,None,0,0)
                    else:
                        axes=(None,0,0,0,0,0,0,0,0)
                    self.sample=jax.jit(jax.vmap(self.pipeline._all_stages,in_axes=axes))
                    self.solver=_UniformQuintic(self.support_times,self.device)
                    times=np.arange(self.size)/fs
                    basis=BSpline.design_matrix(times,self.solver.knots,5)
                    self.first=jax.device_put(basis.indices.reshape(-1,6)[:,0],self.device)
                    self.weights=jax.device_put(basis.data.reshape(-1,6),self.device)
                    self.times=jax.device_put(times,self.device)
                    self.nodes=jax.device_put(self.support_times,self.device)
                    self.reconstruct=jax.jit(self._reconstruct)
                    self.initialization_seconds=time.perf_counter()-started

                def _reconstruct(self,real,quadrature,carrier,solver_arguments,first,weights,times,nodes):
                    complex_samples=(real[:,:,0]+1j*quadrature[:,:,0])*jnp.exp(-2j*jnp.pi*carrier*nodes[:,None])
                    values=jnp.concatenate((complex_samples.real,complex_samples.imag),axis=1)
                    coefficients=_quintic_coefficients(values,*solver_arguments)
                    smooth=_interpolate_spline(coefficients,first,weights)
                    complex_values=smooth[:,:3]+1j*smooth[:,3:]
                    return (complex_values*jnp.exp(2j*jnp.pi*carrier*times[:,None])).real.T

                def support(self,source):
                    real=self.sample(source,*self.arguments)
                    quadrature=self.sample(source.at[0,0,3].add(jnp.pi/2),*self.arguments)
                    return real,quadrature

                def _device_with_arguments(self,source,arguments,solver_arguments,first,weights,times,nodes):
                    real=self.sample(source,*arguments)
                    quadrature=self.sample(source.at[0,0,3].add(jnp.pi/2),*arguments)
                    carrier=jnp.rint((source[0,0,1]+.5*source[0,0,2]*self.T_obs)*self.T_obs)/self.T_obs
                    return self._reconstruct(real,quadrature,carrier,solver_arguments,first,weights,times,nodes)

                def batch_waveforms(self,sources):
                    if not hasattr(self,'_batch_function'):
                        self._batch_function=jax.jit(jax.vmap(self._device_with_arguments,
                            in_axes=(0,None,None,None,None,None,None)))
                    return self._batch_function(sources,self.arguments,self.solver.arguments,
                                               self.first,self.weights,self.times,self.nodes)

                def device_waveform(self,source):
                    real,quadrature=self.support(source)
                    carrier=jnp.rint((source[0,0,1]+.5*source[0,0,2]*self.T_obs)*self.T_obs)/self.T_obs
                    return self.reconstruct(real,quadrature,carrier,self.solver.arguments,
                                            self.first,self.weights,self.times,self.nodes)

                def host_waveform(self,source):
                    xyz=np.asarray(self.device_waveform(source))
                    return np.vstack((np.arange(self.size)/self.fs,xyz))


            class NativeLinearSupportTD(NativeSupportTD):
                def __init__(self,parameters,T_obs,fs,orbit_file,intervals=128,share_filters=True):
                    super().__init__(parameters,T_obs,fs,orbit_file,intervals,True)
                    started=time.perf_counter()
                    e=self.pipeline.engine
                    raw=jnp.zeros((len(e.times),6),dtype=jnp.float64)
                    def from_raw(raw,spline_arguments,first,weights,constants,filters,shifts,derivatives):
                        coefficients=_quintic_coefficients(raw[e.k0:e.k1],*spline_arguments)
                        interpolated=_interpolate_spline(coefficients,first,weights)
                        measurements=self.pipeline._instrument(interpolated,constants,filters)
                        return self.pipeline._tdi(measurements,shifts,derivatives)[:,0]
                    axes=(None,0,0,0,0,None,0,0)
                    build=jax.jit(jax.vmap(jax.jacrev(from_raw,argnums=0),in_axes=axes))
                    print('Precomputing native linear time-domain response',flush=True)
                    response=build(raw,*self.arguments[1:])
                    jax.block_until_ready(response)
                    print('Native linear response',response.shape,'seconds',time.perf_counter()-started,flush=True)
                    self.response_seconds=time.perf_counter()-started
                    self.response_bytes=int(response.size*response.dtype.itemsize)
                    self.native_arguments=self.arguments
                    self.arguments=(self.arguments[0],response)
                    self.linear_sample=jax.jit(self._linear_sample)
                    self.initialization_seconds+=self.response_seconds

                def _linear_sample(self,source,geometry,response):
                    integrate=self.pipeline.engine.links.integrate
                    raw=jax.vmap(lambda g:integrate(source,jnp.ones((1,1,1)),g)[0])(geometry)
                    return jnp.einsum('mctl,mtl->mc',response,raw)[...,None]

                def support(self,source):
                    real=self.linear_sample(source,*self.arguments)
                    quadrature=self.linear_sample(source.at[0,0,3].add(jnp.pi/2),*self.arguments)
                    return real,quadrature

                def _device_with_arguments(self,source,arguments,solver_arguments,first,weights,times,nodes):
                    real=self._linear_sample(source,*arguments)
                    quadrature=self._linear_sample(source.at[0,0,3].add(jnp.pi/2),*arguments)
                    carrier=jnp.rint((source[0,0,1]+.5*source[0,0,2]*self.T_obs)*self.T_obs)/self.T_obs
                    return self._reconstruct(real,quadrature,carrier,solver_arguments,first,weights,times,nodes)


            class NativeBarycentricTD(NativeLinearSupportTD):
                def __init__(self,parameters,T_obs,fs,orbit_file,intervals=128,share_filters=True):
                    super().__init__(parameters,T_obs,fs,orbit_file,intervals,True)
                    started=time.perf_counter()
                    model=self.pipeline.engine.links.model
                    def barycentre(times):
                        return np.column_stack([sum(coordinate[sc](times) for sc in (1,2,3))/3
                                                for coordinate in (model.x,model.y,model.z)])
                    full=jax.device_put(barycentre(np.arange(self.size)/fs),self.device)
                    nodes=jax.device_put(barycentre(self.support_times),self.device)
                    self.arguments=(*self.arguments,full,nodes)
                    self.reconstruct_phase=jax.jit(self._reconstruct_phase)
                    self.initialization_seconds+=time.perf_counter()-started

                @staticmethod
                def _phase(source,times,positions):
                    p=source[0,0];ra,dec=p[5],p[6]
                    k=jnp.array([-jnp.cos(dec)*jnp.cos(ra),-jnp.cos(dec)*jnp.sin(ra),-jnp.sin(dec)])
                    barycentric=times-positions@k/299792458.
                    return jnp.pi*p[2]*barycentric**2+2*jnp.pi*p[1]*barycentric-p[3]

                def _reconstruct_phase(self,real,quadrature,source,solver_arguments,first,weights,times,nodes,full_positions,node_positions):
                    phase=self._phase(source,nodes,node_positions)
                    samples=(real[:,:,0]+1j*quadrature[:,:,0])*jnp.exp(-1j*phase[:,None])
                    values=jnp.concatenate((samples.real,samples.imag),axis=1)
                    coefficients=_quintic_coefficients(values,*solver_arguments)
                    smooth=_interpolate_spline(coefficients,first,weights)
                    phase=self._phase(source,times,full_positions)
                    return ((smooth[:,:3]+1j*smooth[:,3:])*jnp.exp(1j*phase[:,None])).real.T

                def support(self,source):
                    real=self.linear_sample(source,*self.arguments[:2])
                    quadrature=self.linear_sample(source.at[0,0,3].add(jnp.pi/2),*self.arguments[:2])
                    return real,quadrature

                def _device_with_arguments(self,source,arguments,solver_arguments,first,weights,times,nodes):
                    real=self._linear_sample(source,*arguments[:2])
                    quadrature=self._linear_sample(source.at[0,0,3].add(jnp.pi/2),*arguments[:2])
                    return self._reconstruct_phase(real,quadrature,source,solver_arguments,first,weights,times,nodes,*arguments[2:])

                def device_waveform(self,source):
                    real,quadrature=self.support(source)
                    return self.reconstruct_phase(real,quadrature,source,self.solver.arguments,
                        self.first,self.weights,self.times,self.nodes,*self.arguments[2:])


            def stable_complex_links(source,geometry,ltt):
                p=source[0,0]
                amplitude,f,df,phi,inc,ra,dec,psi=p
                trec,temi,xrec,xemi,direction=geometry
                k=jnp.array([-jnp.cos(dec)*jnp.cos(ra),-jnp.cos(dec)*jnp.sin(ra),-jnp.sin(dec)])
                u=jnp.array([jnp.sin(ra),-jnp.cos(ra),0.])
                v=jnp.array([-jnp.sin(dec)*jnp.cos(ra),-jnp.sin(dec)*jnp.sin(ra),jnp.cos(dec)])
                nk=jnp.einsum('tlc,c->tl',direction,k)
                nu=jnp.einsum('tlc,c->tl',direction,u)
                nv=jnp.einsum('tlc,c->tl',direction,v)
                tr=trec-jnp.einsum('tlc,c->tl',xrec,k)/299792458.
                dt=-ltt-jnp.einsum('tlc,c->tl',xemi-xrec,k)/299792458.
                delta=2*jnp.pi*f*dt+jnp.pi*df*dt*(2*tr+dt)
                phase=jnp.pi*df*tr**2+2*jnp.pi*f*tr-phi
                plus,cross=nu**2-nv**2,2*nu*nv
                aplus=-amplitude*(1+jnp.cos(inc)**2)
                across=-2*amplitude*jnp.cos(inc)
                coefficient=aplus*(plus*jnp.cos(2*psi)+cross*jnp.sin(2*psi))-1j*across*(-plus*jnp.sin(2*psi)+cross*jnp.cos(2*psi))
                return coefficient*jnp.exp(1j*(phase+.5*delta))*1j*jnp.sin(.5*delta)/(1-nk)


            def phase_rounding_residual(source,geometry,ltt):
                p=source[0,0];amplitude,f,df,phi,inc,ra,dec,psi=p
                trec,temi,xrec,xemi,direction=geometry
                k=jnp.array([-jnp.cos(dec)*jnp.cos(ra),-jnp.cos(dec)*jnp.sin(ra),-jnp.sin(dec)])
                u=jnp.array([jnp.sin(ra),-jnp.cos(ra),0.])
                v=jnp.array([-jnp.sin(dec)*jnp.cos(ra),-jnp.sin(dec)*jnp.sin(ra),jnp.cos(dec)])
                nk=jnp.einsum('tlc,c->tl',direction,k)
                nu=jnp.einsum('tlc,c->tl',direction,u)
                nv=jnp.einsum('tlc,c->tl',direction,v)
                tr=trec-jnp.einsum('tlc,c->tl',xrec,k)/299792458.
                te=temi-jnp.einsum('tlc,c->tl',xemi,k)/299792458.
                phase_r=jnp.pi*df*tr**2+2*jnp.pi*f*tr-phi
                phase_e=jnp.pi*df*te**2+2*jnp.pi*f*te-phi
                phase_e,phase_r=jax.lax.optimization_barrier((phase_e,phase_r))
                dt=-ltt-jnp.einsum('tlc,c->tl',xemi-xrec,k)/299792458.
                delta=2*jnp.pi*f*dt+jnp.pi*df*dt*(2*tr+dt)
                epsilon=(phase_e-phase_r)-delta
                plus,cross=nu**2-nv**2,2*nu*nv
                aplus=-amplitude*(1+jnp.cos(inc)**2)
                across=-2*amplitude*jnp.cos(inc)
                derivative=(-aplus*(plus*jnp.cos(2*psi)+cross*jnp.sin(2*psi))*jnp.sin(phase_e)
                    +across*(-plus*jnp.sin(2*psi)+cross*jnp.cos(2*psi))*jnp.cos(phase_e))/(2*(1-nk))
                return epsilon*derivative


            class NativeResidualTD(NativeBarycentricTD):
                def __init__(self,parameters,T_obs,fs,orbit_file,full_engine,intervals=128):
                    super().__init__(parameters,T_obs,fs,orbit_file,intervals,True)
                    started=time.perf_counter()
                    self.full_engine=full_engine
                    support_times=np.asarray(self.arguments[0][0])[:,:,0]
                    def delays(times):
                        return np.stack([full_engine.links.model.ltt[int(link)](times)
                                         for link in full_engine.links.links],axis=-1)
                    self.support_ltt=jax.device_put(delays(support_times),self.device)
                    self.full_geometry=full_engine.geometry
                    self.full_ltt=jax.device_put(delays(full_engine.times),self.device)
                    response=np.asarray(self.arguments[1])
                    base=response[len(response)//2]
                    u,s,vh=np.linalg.svd((response-base).reshape(len(response),-1),full_matrices=False)
                    filters=np.concatenate((base[None],vh[:2].reshape(2,*base.shape)),axis=0)
                    values=jax.device_put(np.column_stack((np.ones(len(response)),u[:,:2]*s[:2])),self.device)
                    coefficients=_quintic_coefficients(values,*self.solver.arguments)
                    self.time_weights=_interpolate_spline(coefficients,self.first,self.weights).T
                    self.kernel_size=response.shape[2]
                    self.fft_size=next_fast_len(len(full_engine.times)+self.kernel_size-1,real=True)
                    impulse=jax.device_put(filters[:,:,::-1,:].transpose(0,1,3,2),self.device)
                    self.transfer=jnp.fft.rfft(impulse,n=self.fft_size,axis=-1)
                    jax.block_until_ready((self.time_weights,self.transfer))
                    self.response_singular_values=s[:8].tolist()
                    self.stable_sample=jax.jit(self._stable_sample)
                    self.difference=jax.jit(self._difference)
                    self.brute_difference=jax.jit(self._brute_difference)
                    self.filter_residual=jax.jit(self._filter_residual)
                    self.corrected=jax.jit(self._corrected)
                    self.batched=jax.jit(jax.vmap(self._corrected,in_axes=(0,)+(None,)*11))
                    self.initialization_seconds+=time.perf_counter()-started

                def _stable_sample(self,source,geometry,response,ltt):
                    raw=jax.vmap(stable_complex_links,in_axes=(None,0,0))(source,geometry,ltt)
                    return jnp.einsum('mctl,mtl->mc',response,raw)

                def stable_waveform(self,source):
                    values=self.stable_sample(source,self.arguments[0],self.arguments[1],self.support_ltt)
                    return self.reconstruct_phase(values.real[...,None],values.imag[...,None],source,
                        self.solver.arguments,self.first,self.weights,self.times,self.nodes,*self.arguments[2:])

                def _difference(self,source,geometry,ltt):
                    return phase_rounding_residual(source,geometry,ltt)

                def _brute_difference(self,source,geometry,ltt):
                    original=self.full_engine.links.integrate(source,jnp.ones((1,1,1)),geometry)[0]
                    stable=stable_complex_links(source,geometry,ltt).real
                    return original-stable

                def _filter_residual(self,residual,transfer,time_weights):
                    spectrum=jnp.fft.rfft(residual.T,n=self.fft_size,axis=-1)
                    filtered=jnp.fft.irfft(jnp.einsum('rclk,lk->rck',transfer,spectrum),n=self.fft_size,axis=-1)
                    start=self.kernel_size-1
                    return jnp.sum(filtered[:,:,start:start+self.size]*time_weights[:,None,:],axis=0)

                def _corrected(self,source,arguments,support_ltt,geometry,ltt,transfer,time_weights,solver_arguments,first,weights,times,nodes):
                    values=self._stable_sample(source,arguments[0],arguments[1],support_ltt)
                    smooth=self._reconstruct_phase(values.real[...,None],values.imag[...,None],source,
                        solver_arguments,first,weights,times,nodes,*arguments[2:])
                    return smooth+self._filter_residual(self._difference(source,geometry,ltt),transfer,time_weights)

                def call_arguments(self,rank):
                    return (self.arguments,self.support_ltt,self.full_geometry,self.full_ltt,
                            self.transfer[:rank+1],self.time_weights[:rank+1],self.solver.arguments,
                            self.first,self.weights,self.times,self.nodes)

                def device_waveform(self,source,rank=0):
                    return self.corrected(source,*self.call_arguments(rank))

                def batch_waveforms(self,sources,rank=0):
                    return self.batched(sources,*self.call_arguments(rank))


            class FastResidualTD(NativeResidualTD):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    base = np.asarray(self.arguments[1])[len(self.support_times)//2]
                    self.variants = {}
                    cached = self.call_arguments(0)
                    self.variants['fft64'] = dict(
                        call=self.corrected, batch=jax.jit(jax.vmap(self._corrected, in_axes=(0,)+(None,)*11)),
                        arguments=cached, retained_filter_energy=1., kernel_samples=321)
                    for label, keep in (('fft32', None), ('conv32_9999', .9999), ('conv32_99', .99)):
                        offset = 0
                        if keep is None:
                            filter_data = cached[4].astype(jnp.complex64)
                            retained = 1.
                            length = 321
                        else:
                            energy = np.sum(base**2, axis=(0, 2))
                            cumulative = np.cumsum(energy)/energy.sum()
                            offset = int(np.searchsorted(cumulative, (1-keep)/2))
                            end = min(len(energy), int(np.searchsorted(cumulative, 1-(1-keep)/2))+1)
                            retained = float(energy[offset:end].sum()/energy.sum())
                            filter_data = jax.device_put(base[:, offset:end, :].transpose(0, 2, 1).astype(np.float32), self.device)
                            length = end-offset
                        def make_waveform(use_fft, first_sample):
                            def waveform(source, arguments, support_ltt, geometry, ltt, filt, time_weights,
                                         solver_arguments, first, weights, times, nodes):
                                values = self._stable_sample(source, arguments[0], arguments[1], support_ltt)
                                smooth = self._reconstruct_phase(values.real[..., None], values.imag[..., None], source,
                                    solver_arguments, first, weights, times, nodes, *arguments[2:])
                                # Only the tiny correction uses float32. Normalize first to avoid
                                # subnormal underflow at astrophysical strain amplitudes.
                                unit = source.at[0, 0, 0].set(1.)
                                residual = self._difference(unit, geometry, ltt).astype(jnp.float32)
                                if use_fft:
                                    correction = self._filter_residual(residual, filt, time_weights)
                                else:
                                    length = filt.shape[-1]
                                    cropped = residual[first_sample:first_sample+self.size+length-1].T[None]
                                    correction = jax.lax.conv_general_dilated(
                                        cropped, filt, (1,), 'VALID',
                                        dimension_numbers=('NCH', 'OIH', 'NCH'),
                                        precision=jax.lax.Precision.DEFAULT)[0]
                                return smooth+source[0, 0, 0]*correction.astype(jnp.float64)
                            return waveform
                        fn = make_waveform(keep is None, offset)
                        arguments = (*cached[:4], filter_data, cached[5].astype(jnp.float32), *cached[6:])
                        self.variants[label] = dict(call=jax.jit(fn),
                            batch=jax.jit(jax.vmap(fn, in_axes=(0,)+(None,)*11)),
                            arguments=arguments, retained_filter_energy=retained, kernel_samples=length)
                    jax.block_until_ready([v['arguments'] for v in self.variants.values()])


            def projected_phase_residual(source,geometry,ltt,solver_arguments,first,weights,times):
                amplitude,f,df,phi,inc,ra,dec,psi=source[0,0]
                _,_,xrec,xemi,direction=geometry
                k=jnp.array([-jnp.cos(dec)*jnp.cos(ra),-jnp.cos(dec)*jnp.sin(ra),-jnp.sin(dec)])
                u=jnp.array([jnp.sin(ra),-jnp.cos(ra),0.])
                v=jnp.array([-jnp.sin(dec)*jnp.cos(ra),-jnp.sin(dec)*jnp.sin(ra),jnp.cos(dec)])
                nk=jnp.einsum('tlc,c->tl',direction,k)
                nu=jnp.einsum('tlc,c->tl',direction,u)
                nv=jnp.einsum('tlc,c->tl',direction,v)
                pr=jnp.einsum('tlc,c->tl',xrec,k)/299792458.
                pe=jnp.einsum('tlc,c->tl',xemi,k)/299792458.
                pd=jnp.einsum('tlc,c->tl',xemi-xrec,k)/299792458.
                plus,cross=nu**2-nv**2,2*nu*nv
                aplus=-amplitude*(1+jnp.cos(inc)**2)
                across=-2*amplitude*jnp.cos(inc)
                bs=-aplus*(plus*jnp.cos(2*psi)+cross*jnp.sin(2*psi))/(2*(1-nk))
                bc=across*(-plus*jnp.sin(2*psi)+cross*jnp.cos(2*psi))/(2*(1-nk))
                coefficients=_quintic_coefficients(jnp.concatenate((pr,pe,pd,ltt,bs,bc),axis=1),*solver_arguments)
                values=_interpolate_spline(coefficients,first,weights).reshape(-1,6,6)
                pr,pe,pd,ltt,bs,bc=[values[:,index,:] for index in range(6)]
                emission=jax.lax.optimization_barrier(times[:,None]-ltt)
                tr=times[:,None]-pr
                te=emission-pe
                phase_r=jnp.pi*df*tr**2+2*jnp.pi*f*tr-phi
                phase_e=jnp.pi*df*te**2+2*jnp.pi*f*te-phi
                phase_e,phase_r=jax.lax.optimization_barrier((phase_e,phase_r))
                dt=-ltt-pd
                epsilon=(phase_e-phase_r)-(2*jnp.pi*f*dt+jnp.pi*df*dt*(2*tr+dt))
                return epsilon*(bs*jnp.sin(phase_e)+bc*jnp.cos(phase_e))


            class ProjectedResidualTD(FastResidualTD):
                def __init__(self,*args,**kwargs):
                    super().__init__(*args,**kwargs)
                    nodes=np.linspace(0,self.T_obs,1025)
                    links=self.full_engine.links
                    geometry=tuple(jax.device_put(value,self.device) for value in links.geometry(nodes))
                    ltt=jax.device_put(np.stack([links.model.ltt[int(link)](nodes) for link in links.links],-1),self.device)
                    solver=_UniformQuintic(nodes,self.device)
                    times=self.full_engine.times
                    basis=BSpline.design_matrix(times,solver.knots,5,extrapolate=True)
                    projected=(geometry,ltt,solver.arguments,
                        jax.device_put(basis.indices.reshape(-1,6)[:,0],self.device),
                        jax.device_put(basis.data.reshape(-1,6),self.device),
                        jax.device_put(times,self.device))
                    self.variants={f'projected_{label}':dict(value,arguments=(
                        *value['arguments'][:2],projected,jnp.asarray(0.),*value['arguments'][4:]))
                        for label,value in self.variants.items()}
                    jax.block_until_ready(projected)

                def _difference(self,source,geometry,ltt):
                    if len(geometry)==6:
                        return projected_phase_residual(source,*geometry)
                    return super()._difference(source,geometry,ltt)


            def polynomial_arguments(knots, times, device):
                n = len(knots)-6
                breaks = np.unique(knots)
                left, width = breaks[:-1], np.diff(breaks)
                indices = BSpline.design_matrix(left+.5*width, knots, 5).indices.reshape(-1, 6)
                basis = BSpline(knots, np.eye(n), 5)
                transform = np.stack([
                    basis.derivative(order)(left)[np.arange(len(left))[:, None], indices]
                    *width[:, None]**order/math.factorial(order) for order in range(6)], axis=1)
                segment = np.clip(np.searchsorted(breaks, times, side='right')-1, 0, len(left)-1).astype(np.int32)
                coordinate = (times-left[segment])/width[segment]
                return tuple(jax.device_put(x, device) for x in (indices.astype(np.int32), transform, segment, coordinate))


            def projected_coefficients(source, geometry, ltt, solver_arguments):
                amplitude,f,df,phi,inc,ra,dec,psi=source[0,0]
                _,_,xrec,xemi,direction=geometry
                k=jnp.array([-jnp.cos(dec)*jnp.cos(ra),-jnp.cos(dec)*jnp.sin(ra),-jnp.sin(dec)])
                u=jnp.array([jnp.sin(ra),-jnp.cos(ra),0.])
                v=jnp.array([-jnp.sin(dec)*jnp.cos(ra),-jnp.sin(dec)*jnp.sin(ra),jnp.cos(dec)])
                nk=jnp.einsum('tlc,c->tl',direction,k)
                nu=jnp.einsum('tlc,c->tl',direction,u)
                nv=jnp.einsum('tlc,c->tl',direction,v)
                pr=jnp.einsum('tlc,c->tl',xrec,k)/299792458.
                pe=jnp.einsum('tlc,c->tl',xemi,k)/299792458.
                pd=jnp.einsum('tlc,c->tl',xemi-xrec,k)/299792458.
                plus,cross=nu**2-nv**2,2*nu*nv
                aplus=-amplitude*(1+jnp.cos(inc)**2)
                across=-2*amplitude*jnp.cos(inc)
                bs=-aplus*(plus*jnp.cos(2*psi)+cross*jnp.sin(2*psi))/(2*(1-nk))
                bc=across*(-plus*jnp.sin(2*psi)+cross*jnp.cos(2*psi))/(2*(1-nk))
                return _quintic_coefficients(jnp.concatenate((pr,pe,pd,ltt,bs,bc),axis=1),*solver_arguments)


            def prepare_batch(engine, parameters, T_obs, fs, orbit_file):
                base = ProjectedResidualTD(parameters, T_obs, fs, orbit_file, engine, 128)
                args = base.variants['projected_conv32_99']['arguments']
                response = base.arguments[1][len(base.support_times)//2]
                energy = jnp.sum(response**2, axis=(0, 2))
                offset = int(jnp.searchsorted(jnp.cumsum(energy)/jnp.sum(energy), .005))
                nodes = np.linspace(0, base.T_obs, 1025)
                knots = np.r_[np.repeat(nodes[0], 6), nodes[3:-3], np.repeat(nodes[-1], 6)]
                residual_poly = polynomial_arguments(knots, np.asarray(base.full_engine.times), base.device)
                main_poly = polynomial_arguments(base.solver.knots, np.asarray(base.times), base.device)
                mode, fast, horner_residual, tiled = 1, True, True, True
                target = 'eric_full_horner_xyz'
                library = ctypes.CDLL(str(Path(__file__).with_name('td_full_horner.so')))
                jax.ffi.register_ffi_target(target, jax.ffi.pycapsule(library.EricsFullHornerXYZ), platform='CUDA')

                def waveform_batch(sources,arguments,support_ltt,projected,unused,filt,time_weights,
                                   solver_arguments,first,weights,times,nodes,residual_poly,main_poly):
                    def main_coefficients(source):
                        values=base._stable_sample(source,arguments[0],arguments[1],support_ltt)
                        phase=base._phase(source,nodes,arguments[3])
                        samples=values*jnp.exp(-1j*phase[:,None])
                        return _quintic_coefficients(
                            jnp.concatenate((samples.real,samples.imag),axis=1),*solver_arguments)
                    main=jax.vmap(main_coefficients)(sources)
                    geometry,ltt,solver,projected_first,projected_weights,full_times=projected
                    unit=sources.at[:,0,0,0].set(1.)
                    residual=jax.vmap(projected_coefficients,in_axes=(0,None,None,None))(
                        unit,geometry,ltt,solver)
                    ri,rt,rsegment,rcoord=residual_poly
                    mi,mt,msegment,mcoord=main_poly
                    if horner_residual:
                        residual=jnp.einsum('mij,bmjc->bmic',rt,residual[:,ri])
                    else:
                        # Hybrid: preserve the original residual coefficients and arithmetic.
                        rsegment=projected_first.astype(jnp.int32)
                        rcoord=projected_weights
                    main=jnp.einsum('mij,bmjc->bmic',mt,main[:,mi])
                    p=sources[:,0,0,:]
                    sky=jnp.stack((-jnp.cos(p[:,6])*jnp.cos(p[:,5]),
                        -jnp.cos(p[:,6])*jnp.sin(p[:,5]),-jnp.sin(p[:,6])),axis=-1)
                    shapes=(jax.ShapeDtypeStruct((len(sources),6,full_times.size),jnp.float32),
                            jax.ShapeDtypeStruct((len(sources),3,times.size),jnp.float64))
                    scratch,xyz=jax.ffi.ffi_call(target,shapes)(
                        p,residual,rsegment,rcoord,full_times,
                        main,msegment,mcoord,times,arguments[2],sky,filt,
                        mode=np.int64(mode),fast_trig=fast,offset=np.int64(offset),tiled=tiled)
                    return xyz

                return {'batch': jax.jit(waveform_batch),
                        'arguments': (*args, residual_poly, main_poly),
                        'times': base.times, 'device': base.device, 'library': library}

            _waveform_state['backend'] = {
                'jax': jax, 'coordinates': _bme_to_icrs,
                'engine': _WaveformEngine, 'prepare': prepare_batch,
                'cuda_handles': cuda_handles,
            }
    if not ids.size:
        return iter(())

    # 2. Read the requested DWD parameters and the fixed setup source.
    catalogue_file = Path(__file__).resolve().parents[1]/'data/LDC2_sangria_training_v2.h5'
    rows = np.empty(ids.size, dtype=[(field, '<f8') for field in PARAMETER_FIELDS])
    with h5py.File(catalogue_file, 'r') as catalogue:
        counts = [len(catalogue[f'sky/{group}/cat']) for group in ('dgb', 'igb')]
        ids = ids.astype(np.int64, copy=True)
        offset = 0
        for group, count in zip(('dgb', 'igb'), counts):
            positions = np.flatnonzero((ids >= offset) & (ids < offset+count))
            selected, inverse = np.unique(ids[positions]-offset, return_inverse=True)
            if selected.size:
                rows[positions] = catalogue[f'sky/{group}/cat'].fields(
                    PARAMETER_FIELDS)[selected, 0][inverse]
            offset += count
        prototype = catalogue['sky/dgb/cat'].fields(PARAMETER_FIELDS)[4, 0]

    # 3. Reuse the response setup for this sampling, duration and orbit file.
    backend = _waveform_state['backend']
    ra, dec, psi = backend['coordinates'](
        rows['EclipticLongitude'], rows['EclipticLatitude'], rows['Polarization'])
    packed_parameters = np.column_stack(
        [rows[field] for field in PARAMETER_FIELDS[:5]]+[ra, dec, psi])
    orbit_file = Path(orbit_file).expanduser().resolve()
    stat = orbit_file.stat()
    key = (fs, T_obs, str(orbit_file), stat.st_size, stat.st_mtime_ns)
    parameters = {name: np.atleast_1d(prototype[field])
                  for name, field in zip(
                      ('A', 'f', 'df', 'phi0', 'iota', 'longitude', 'latitude', 'psi_bme'),
                      PARAMETER_FIELDS)}
    with _waveform_lock:
        if _waveform_state.get('key') != key or 'batch' not in _waveform_state:
            with tqdm(total=1, desc='Waveform response setup + compilation', unit='setup') as progress:
                if _waveform_state.get('key') != key:
                    _waveform_state.pop('key', None)
                    _waveform_state.pop('batch', None)
                    _waveform_state.pop('engine', None)
                    engine = backend['engine'](parameters, T_obs, fs, orbit_file)
                    _waveform_state.update(key=key, engine=engine)
                engine = _waveform_state['engine']
                if engine.device.platform == 'gpu' and int(round(T_obs*fs)) % 128 == 0:
                    prepared = backend['prepare'](engine, parameters, T_obs, fs, orbit_file)
                    # Compile the same 64-source shape used by subsequent batches.
                    ra, dec, psi = backend['coordinates'](
                        parameters['longitude'], parameters['latitude'], parameters['psi_bme'])
                    prototype_packed = np.column_stack(
                        [parameters[name] for name in ('A', 'f', 'df', 'phi0', 'iota')]+[ra, dec, psi])
                    source = jax.device_put(
                        np.repeat(prototype_packed, 64, axis=0).reshape(64, 1, 1, 8),
                        prepared['device'])
                    warmup = prepared['batch'](source, *prepared['arguments'])
                    jax.block_until_ready(warmup)
                    del warmup, source
                else:
                    prepared = {
                        'engine': engine, 'device': engine.device,
                        'times': jax.device_put(np.arange(int(round(T_obs*fs)))/fs, engine.device),
                    }
                _waveform_state['batch'] = prepared
                progress.update(1)
        prepared = _waveform_state['batch']

    # 4. Generate full time-domain XYZ batches; advance only completed DWDs.
    def batches():
        with tqdm(total=len(ids), desc=progress_desc, unit='DWD') as progress:
            for first in range(0, len(ids), 64):
                last = min(first+64, len(ids))
                if 'engine' in prepared:
                    samples = []
                    with _waveform_lock, jax.default_device(prepared['device']):
                        for row in rows[first:last]:
                            parameters = {name: np.atleast_1d(row[field])
                                          for name, field in zip(
                                              ('A', 'f', 'df', 'phi0', 'iota', 'longitude', 'latitude', 'psi_bme'),
                                              PARAMETER_FIELDS)}
                            samples.append(prepared['engine'].evaluate(parameters)[1:])
                    xyz = jax.device_put(np.stack(samples), prepared['device'])
                else:
                    packed = np.empty((64, 8), dtype=np.float64)
                    packed[:last-first] = packed_parameters[first:last]
                    packed[last-first:] = packed_parameters[last-1]
                    source = jax.device_put(packed.reshape(64, 1, 1, 8), prepared['device'])
                    xyz = prepared['batch'](source, *prepared['arguments'])[:last-first]
                jax.block_until_ready(xyz)
                progress.update(last-first)
                yield ids[first:last], prepared['times'], xyz
                del xyz

    return batches()


def _single_waveform_td(dwd_parameters, T_obs, fs, orbit_file):
    # An empty request initializes definitions, but no GPU response or waveforms.
    generate_waveform_td(fs, T_obs, [], orbit_file)
    parameters = {name: np.atleast_1d(dwd_parameters[field])
                  for name, field in zip(
                      ('A', 'f', 'df', 'phi0', 'iota', 'longitude', 'latitude', 'psi_bme'),
                      PARAMETER_FIELDS)}
    orbit_file = Path(orbit_file).expanduser().resolve()
    stat = orbit_file.stat()
    key = (float(fs), float(T_obs), str(orbit_file), stat.st_size, stat.st_mtime_ns)
    with _waveform_lock:
        if _waveform_state.get('key') != key:
            _waveform_state.pop('key', None)
            _waveform_state.pop('batch', None)
            _waveform_state.pop('engine', None)
            engine = _waveform_state['backend']['engine'](parameters, T_obs, fs, orbit_file)
            _waveform_state.update(key=key, engine=engine)
        engine = _waveform_state['engine']
        with _waveform_state['backend']['jax'].default_device(engine.device):
            return engine.evaluate(parameters)


_gpu_backend_ready = False
_cuda_handles = []


def _initialize_gpu_backend():
    global _gpu_backend_ready, _cuda_handles, jax, jnp, jsl
    if _gpu_backend_ready:
        return
    # Load pip-installed CUDA libraries in this process, without restarting Eric
    # after simulation. System-installed CUDA remains usable when these are absent.
    _cuda_root = (Path(sys.prefix) / 'lib'
                  / f'python{sys.version_info.major}.{sys.version_info.minor}'
                  / 'site-packages' / 'nvidia')
    _cuda_handles = []
    for _package, _library in (
        ('cuda_runtime', 'libcudart.so.12'), ('nvjitlink', 'libnvJitLink.so.12'),
        ('cublas', 'libcublasLt.so.12'), ('cublas', 'libcublas.so.12'),
        ('cusparse', 'libcusparse.so.12'), ('cusolver', 'libcusolver.so.11'),
        ('cufft', 'libcufft.so.11'), ('cudnn', 'libcudnn.so.9'),
        ('cuda_cupti', 'libcupti.so.12'), ('nccl', 'libnccl.so.2'),
    ):
        _path = _cuda_root / _package / 'lib' / _library
        if _path.exists():
            _cuda_handles.append(ctypes.CDLL(str(_path), mode=ctypes.RTLD_GLOBAL))
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault('XLA_FLAGS', '--xla_cpu_multi_thread_eigen=false')

    jax = import_module('jax')
    jnp = import_module('jax.numpy')
    jsl = import_module('jax.scipy.linalg')

    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    _gpu_backend_ready = True


class GPUResponse:
    """Same moving unequal-arm delayed links, filter and native-bin FFT."""
    def __init__(self, covariance, data, response_times, response_geometry,
                 physical_frequency, duration, rate):
        _initialize_gpu_backend()
        self.device = jax.devices('gpu')[0]
        terms = response_geometry[1]
        factors = np.array([[term[0]*term[1] for term in channel] for channel in terms])
        geometry = [np.array([[term[2][i] for term in channel] for channel in terms])
                    for i in range(5)]
        transfer, _ = native_signal_filter(physical_frequency, rate)
        self.arguments = tuple(jax.device_put(value, self.device) for value in (
            factors, *geometry, response_times, physical_frequency,
            transfer, _bme_to_icrs_rotation()))
        self.duration, self.rate = duration, rate
        self.frequency_count = len(physical_frequency)
        self.sample_block = int(round(3600.*rate))
        self.block_count = len(response_times)
        self.inverse_lower = jax.device_put(
            np.linalg.inv(np.linalg.cholesky(covariance)), self.device)
        self.factor = np.sqrt(4./(duration*rate**2))
        self.single = jax.jit(self._basis)
        self.white_single = jax.jit(
            lambda point, *arguments: self.whiten(self._basis(point, *arguments)))
        self.white_data = jax.block_until_ready(self.whiten(jax.device_put(data, self.device)))
        self.data_norm = float(self.white_data @ self.white_data)

    def _basis(self, point, factors, reception, emission, receiver, emitter, direction,
               times, frequency, transfer, rotation):
        f, df, longitude, latitude = point
        cl, sl = jnp.cos(longitude), jnp.sin(longitude)
        cb, sb = jnp.cos(latitude), jnp.sin(latitude)
        sky = rotation @ jnp.array([cb*cl, cb*sl, sb])
        transported_u = rotation @ jnp.array([sl, -cl, 0.])
        ra = jnp.mod(jnp.arctan2(sky[1], sky[0]), 2*jnp.pi)
        dec = jnp.arcsin(jnp.clip(sky[2], -1., 1.))
        u = jnp.array([jnp.sin(ra), -jnp.cos(ra), 0.])
        v = jnp.array([-jnp.sin(dec)*jnp.cos(ra), -jnp.sin(dec)*jnp.sin(ra), jnp.cos(dec)])
        propagation = jnp.array([-jnp.cos(dec)*jnp.cos(ra), -jnp.cos(dec)*jnp.sin(ra), -jnp.sin(dec)])
        psi = jnp.mod(jnp.arctan2(transported_u @ v, transported_u @ u)
                      + jnp.array([0., jnp.pi/4]), jnp.pi)
        cos_iota = jnp.cos(jnp.pi/2)
        plus = -1.e-22*(1+cos_iota**2)
        cross = 2.e-22j*cos_iota
        hp = plus*jnp.cos(2*psi)-cross*jnp.sin(2*psi)
        hc = plus*jnp.sin(2*psi)+cross*jnp.cos(2*psi)
        nk, nu, nv = (jnp.einsum('...c,c->...', direction, vector)
                      for vector in (propagation, u, v))
        tr = reception-jnp.einsum('...c,c->...', receiver, propagation)/299792458.
        te = emission-jnp.einsum('...c,c->...', emitter, propagation)/299792458.
        pe = jnp.pi*df*te**2+2*jnp.pi*f*(te-times)
        pr = jnp.pi*df*tr**2+2*jnp.pi*f*(tr-times)
        link = factors*(jnp.exp(1j*pe)-jnp.exp(1j*pr))/(2*(1-nk))
        polarizations = (hp[:, None, None, None]*(nu**2-nv**2)
                         + hc[:, None, None, None]*(2*nu*nv))
        envelope = jnp.sum(polarizations*link[None], axis=2)
        carrier_bin = jnp.rint(f*self.duration).astype(jnp.int64)
        fractional = f*self.duration-carrier_bin
        envelope_fft = jnp.fft.fft(
            envelope*jnp.exp(2j*jnp.pi*fractional*jnp.arange(self.block_count)
                             / self.block_count), axis=-1)
        offsets = frequency-f
        bins = jnp.rint(frequency*self.duration).astype(jnp.int64)
        factor = (self.sample_block*jnp.sinc(self.sample_block*offsets/self.rate)
                  / jnp.sinc(offsets/self.rate))
        cosine = (.5*envelope_fft[:, :, (bins-carrier_bin) % self.block_count]
                  * jnp.exp(-2j*jnp.pi*times[0]*offsets)*factor*transfer)
        return jnp.stack((cosine[0], cosine[1], 1j*cosine[0], 1j*cosine[1]), axis=-3)

    def whiten(self, values):
        white = jnp.einsum('kij,...jk->...ki', self.inverse_lower, values)
        flat = white.reshape(*values.shape[:-2], -1)
        return self.factor*jnp.concatenate((flat.real, flat.imag), axis=-1)

    def build(self, points):
        points = np.asarray(points).reshape(-1, 4)
        if not len(points):
            return np.empty((0, 4, 3, self.frequency_count), complex)
        return np.stack([np.asarray(self.single(jax.device_put(point, self.device),
                                                *self.arguments)) for point in points])

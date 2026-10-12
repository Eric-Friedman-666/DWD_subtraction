"""GPU summation of GalacticBinary links; native LISA geometry and file format.

The analytic GalacticBinary source/link arithmetic is evaluated in FP64 on
CUDA. A smooth extension outside the saved interval makes compact envelope
FFTs accurate without tapering the requested observation. Instrument and TDI
operations are performed afterwards by lisainstrument and pytdi.
"""
import ctypes
import os
from pathlib import Path
import sys

import numpy as np

_cuda_handles = []


def gpu_runtime():
    """Require CUDA; never silently substitute a CPU for the population run."""
    root = Path(sys.prefix)/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages/nvidia'
    for package, library in (
        ('cuda_runtime','libcudart.so.12'),('nvjitlink','libnvJitLink.so.12'),
        ('cublas','libcublasLt.so.12'),('cublas','libcublas.so.12'),
        ('cusparse','libcusparse.so.12'),('cusolver','libcusolver.so.11'),
        ('cufft','libcufft.so.11'),('cudnn','libcudnn.so.9'),
        ('cuda_cupti','libcupti.so.12'),('nccl','libnccl.so.2')):
        path = root/package/'lib'/library
        if path.exists():
            _cuda_handles.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    import jax
    import jax.numpy as jnp
    jax.config.update('jax_enable_x64',True)
    device = jax.devices('gpu')[0]
    return jax,jnp,device


def envelope_sizes(frequency,derivative,duration,minimum=512):
    """Cover both signs of chirp, annual Doppler modulation, and taper support."""
    frequency,derivative = np.asarray(frequency),np.asarray(derivative)
    highest = np.maximum(np.abs(frequency),np.abs(frequency+derivative*duration))
    width = 2*np.abs(derivative)*duration**2+4.e-4*highest*duration+256
    return (2**np.ceil(np.log2(np.maximum(minimum,width)))).astype(np.int64)


class PopulationLinksGPU:
    """Keep only one source batch and six aggregate Fourier streams on CUDA."""
    def __init__(self,orbit_file,duration,rate,*,batch_size=2048,minimum=512):
        import lisagwresponse
        from lisagwresponse.response import LINKS
        self.jax,self.xp,self.device = gpu_runtime()
        self.model = lisagwresponse.GalacticBinary(A=1.,f=.003,ra=0.,dec=0.,
            orbits=str(orbit_file),orbit_interp_order=5,shift_sun='never')
        self.links = np.asarray(LINKS)
        self.rate,self.duration = float(rate),float(duration)
        self.padding = int(round(2*86400*rate))
        self.count = int(round(duration*rate))+2*self.padding
        self.extended_count = 2*self.count
        self.extended_duration = self.extended_count/rate
        self.t0 = -self.padding/rate-self.extended_duration/4
        self.keep = slice(self.count//2,self.count//2+self.count)
        self.batch_size,self.minimum = int(batch_size),int(minimum)
        self.prepared = {}
        self.geometries = {}
        self.accumulator = self.jax.device_put(
            np.zeros((6,self.extended_count),dtype=np.complex128),self.device)
        self.processed = 0

    def geometry(self,size):
        times = self.t0+np.arange(size)*self.extended_duration/size
        trec = np.broadcast_to(times[:,None],(size,6)).copy()
        temi = trec.copy()
        receiver,emitter = np.empty((size,6,3)),np.empty((size,6,3))
        for column,link in enumerate(self.links):
            rec,emi = int(link)//10,int(link)%10
            temi[:,column] -= self.model.ltt[int(link)](times)
            for axis,coordinate in enumerate((self.model.x,self.model.y,self.model.z)):
                receiver[:,column,axis] = coordinate[rec](times)
                emitter[:,column,axis] = coordinate[emi](temi[:,column])
        direction = receiver-emitter
        direction /= np.linalg.norm(direction,axis=-1)[...,None]
        # Degree-nine smoothstep: four derivatives vanish at either join.
        x = np.clip(np.minimum(np.arange(size)/size,1-np.arange(size)/size)*4,0,1)
        taper = x**5*(126+x*(-420+x*(540+x*(-315+70*x))))
        return tuple(self.jax.device_put(v,self.device) for v in
                     (trec,temi,receiver,emitter,direction,times,taper))

    def _envelope(self,p,geometry):
        xp = self.xp
        trec,temi,receiver,emitter,direction,times,taper = geometry
        amplitude,f,df,phi,iota,ra,dec,psi = p
        k = xp.array([-xp.cos(dec)*xp.cos(ra),-xp.cos(dec)*xp.sin(ra),-xp.sin(dec)])
        u = xp.array([xp.sin(ra),-xp.cos(ra),0.])
        v = xp.array([-xp.sin(dec)*xp.cos(ra),-xp.sin(dec)*xp.sin(ra),xp.cos(dec)])
        nk,nu,nv = (xp.einsum('tlc,c->tl',direction,a) for a in (k,u,v))
        tr = trec-xp.einsum('tlc,c->tl',receiver,k)/299792458.
        te = temi-xp.einsum('tlc,c->tl',emitter,k)/299792458.
        carrier = xp.rint((f+.5*df*self.duration)*self.extended_duration).astype(xp.int64)
        phase = 2*xp.pi*(f-carrier/self.extended_duration)*(times-self.t0)+2*xp.pi*f*self.t0-phi
        pr = xp.pi*df*tr**2+2*xp.pi*f*(tr-times[:,None])+phase[:,None]
        pe = xp.pi*df*te**2+2*xp.pi*f*(te-times[:,None])+phase[:,None]
        # The analytic versions of the native source-frame plus/cross strains.
        plus,cross = -amplitude*(1+xp.cos(iota)**2),2j*amplitude*xp.cos(iota)
        hp = plus*xp.cos(2*psi)-cross*xp.sin(2*psi)
        hc = plus*xp.sin(2*psi)+cross*xp.cos(2*psi)
        strain = hp*(nu**2-nv**2)+hc*(2*nu*nv)
        values = strain*(xp.exp(1j*pe)-xp.exp(1j*pr))/(2*(1-nk))
        return values.T*taper,carrier

    def prepare(self,size,source_count=None):
        # Hold the envelope workload fixed as chirps require wider waveforms.
        batch = max(1,min(self.batch_size,self.batch_size*512//size,
                          source_count if source_count is not None else self.batch_size))
        key = size,batch
        if key not in self.prepared:
            jax,xp = self.jax,self.xp
            if size not in self.geometries:
                self.geometries[size] = self.geometry(size)
            geometry = self.geometries[size]
            calculate = jax.vmap(self._envelope,in_axes=(0,None))
            def add(accumulator,parameters,*geometry):
                offsets = (xp.fft.fftfreq(size)*size).astype(xp.int64)
                def add_link(accumulator,column):
                    link_geometry = tuple(jax.lax.dynamic_slice_in_dim(value,column,1,axis=1)
                                          for value in geometry[:5])+geometry[5:]
                    envelopes,carriers = calculate(parameters,link_geometry)
                    coefficients = .5*self.extended_count/size*xp.fft.fft(envelopes[:,0,:],axis=-1)
                    bins = (carriers[:,None]+offsets[None,:])%self.extended_count
                    channel = accumulator[column]
                    channel = channel.at[bins.reshape(-1)].add(coefficients.reshape(-1))
                    channel = channel.at[(-bins).reshape(-1)%self.extended_count].add(coefficients.conj().reshape(-1))
                    return accumulator.at[column].set(channel),None
                accumulator,_ = jax.lax.scan(add_link,accumulator,xp.arange(len(self.links)))
                return accumulator
            self.prepared[key] = geometry,jax.jit(add,donate_argnums=(0,)),batch
        return self.prepared[key]

    def add(self,parameters):
        """parameters columns: A,f,df,phi0,iota,ra,dec,psi; no selection here."""
        sizes = envelope_sizes(parameters[:,1],parameters[:,2],self.extended_duration,self.minimum)
        counts = {}
        for size in np.unique(sizes):
            positions = np.flatnonzero(sizes==size)
            geometry,calculate,batch = self.prepare(int(size),len(positions))
            counts[str(int(size))] = len(positions)
            for first in range(0,len(positions),batch):
                selected = positions[first:first+batch]
                packed = np.zeros((batch,8))
                packed[:,1] = .003  # finite geometry for zero-amplitude padding
                packed[:len(selected)] = parameters[selected]
                self.accumulator = calculate(self.accumulator,
                    self.jax.device_put(packed,self.device),*geometry)
            self.jax.block_until_ready(self.accumulator)
        self.processed += len(parameters)
        return counts

    def finish(self):
        """Return the untapered requested interval in the native link order."""
        values = self.xp.fft.ifft(self.accumulator,axis=-1).real[:,self.keep]
        return np.asarray(self.jax.block_until_ready(values)).T

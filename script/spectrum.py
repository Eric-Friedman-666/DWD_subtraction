import inspect

import healpy as hp
import numpy as np
import scipy

import lisaconstants
import lisagwresponse
import lisainstrument
import lisaorbits
import pytdi
import pytdi.michelson
import pytdi.naming
import pytdi.ortho
from lisagwresponse.utils import emitter, receiver


class DetectorFunction:
    """Segment-dependent TDI transfer functions and GW responses.

    Every cross component follows the right-conjugate convention
    ``C_ab(f) = <a(f) b(f)*>``.
    """

    class FrequencyBasis(lisagwresponse.ResponseFromStrain):
        def __init__(self, frequency, **kwargs):
            self.frequency = np.asarray(frequency)
            self.polarization = "plus"
            self.quadrature = "cos"
            super().__init__(**kwargs)

        def strain(self, values):
            phase = 2.0 * np.pi * self.frequency * np.asarray(values)
            if self.quadrature == "cos":
                return np.cos(phase)
            return np.sin(phase)

        def compute_hplus(self, values):
            if self.polarization == "plus":
                return self.strain(values)
            return np.zeros_like(values)

        def compute_hcross(self, values):
            if self.polarization == "cross":
                return self.strain(values)
            return np.zeros_like(values)

    def __init__(
        self,
        fs,
        orbit_file,
        links=None,
        nside=5,
        orbit_interp_order=2,
    ):
        self.fs = fs
        self.orbit_file = orbit_file
        self.nside = nside
        self.orbit_interp_order = int(orbit_interp_order)
        if links is None:
            links = lisaorbits.LINKS
        self.links = tuple(int(link) for link in links)
        self.orbit_model = self.FrequencyBasis(
            np.array([0.0]),
            ra=0.0,
            dec=0.0,
            orbits=orbit_file,
            orbit_interp_order=self.orbit_interp_order,
            shift_sun="never",
        )

    def light_travel_times(self, middle_time):
        time = np.array([middle_time])
        return {
            link: float(self.orbit_model.ltt[link](time)[0])
            for link in self.links
        }

    @staticmethod
    def _covariance(transfer):
        return np.einsum(
            "fcl,fdl->fcd",
            transfer,
            transfer.conj(),
            optimize=True,
        )

    @staticmethod
    def _covariance_components(covariance, combination_type):
        channels = tuple(combination_type)
        pairs = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
        return {
            channels[left] + channels[right]: (
                np.real(covariance[:, left, right])
                if left == right
                else covariance[:, left, right]
            )
            for left, right in pairs
        }

    def _tdi_source_transfer(
        self,
        f,
        middle_time,
        generation,
        combination_type,
        measurement_family,
        measurement_amplitude,
    ):
        f = np.asarray(f)
        light_travel_times = self.light_travel_times(middle_time)
        generation_number = {"1st": 1, "2nd": 2}[generation]
        channels = tuple(combination_type)
        package = pytdi.michelson if combination_type == "XYZ" else pytdi.ortho
        combinations = {
            channel: getattr(package, f"{channel}{generation_number}")
            for channel in channels
        }
        delays = {
            f"d_{link}": light_travel_times[link]
            for link in self.links
        }
        built = {
            channel: combination.build(delays=delays, fs=self.fs)
            for channel, combination in combinations.items()
        }
        measurement_names = set().union(
            *(combination.measurements for combination in combinations.values())
        )
        longest_path = (
            (12 if generation == "2nd" else 6)
            * max(light_travel_times.values())
            * self.fs
        )
        minimum_size = max(512, int(2 * (longest_path + 64)))
        impulse_size = 2 ** int(np.ceil(np.log2(minimum_size)))
        impulse_index = impulse_size // 2
        zero = np.zeros(impulse_size)
        transfer = np.empty((len(f), 3, len(self.links)), complex)

        for source_index, link in enumerate(self.links):
            measurements = {name: zero.copy() for name in measurement_names}
            measurements[f"{measurement_family}_{link}"][impulse_index] = (
                measurement_amplitude
            )
            for channel_index, channel in enumerate(channels):
                impulse = built[channel](measurements)
                _, response = scipy.signal.freqz(
                    impulse,
                    worN=2.0 * np.pi * f / self.fs,
                )
                transfer[:, channel_index, source_index] = response
        return transfer

    def _response_covariance(
        self,
        f,
        middle_time,
        science_transfer,
    ):
        f = np.asarray(f)
        basis = self.FrequencyBasis(
            f,
            ra=0.0,
            dec=0.0,
            x=self.orbit_model.x,
            y=self.orbit_model.y,
            z=self.orbit_model.z,
            ltt=self.orbit_model.ltt,
            orbits=self.orbit_file,
            orbit_interp_order=self.orbit_interp_order,
            shift_sun="never",
        )
        response = np.zeros((len(f), 3, 3), complex)
        response_times = np.full(len(f), middle_time)
        pixel_count = hp.nside2npix(self.nside)
        theta, right_ascension = hp.pix2ang(
            self.nside,
            np.arange(pixel_count),
            nest=False,
        )
        declination = np.pi / 2.0 - theta

        for ra, dec in zip(right_ascension, declination):
            basis.set_position(ra, dec)
            for polarization in ("plus", "cross"):
                basis.polarization = polarization
                basis.quadrature = "cos"
                cosine = basis.compute_gw_response(response_times, self.links)
                basis.quadrature = "sin"
                sine = basis.compute_gw_response(response_times, self.links)
                link_response = cosine + 1j * sine
                tdi_response = np.einsum(
                    "fcl,fl->fc",
                    science_transfer,
                    link_response,
                    optimize=True,
                )
                response += np.einsum(
                    "fi,fj->fij",
                    tdi_response,
                    tdi_response.conj(),
                    optimize=True,
                )
        return response / pixel_count

    def acc_noise_transfer(
        self,
        f,
        middle_time,
        generation="2nd",
        combination_type="XYZ",
    ):
        transfer = self._tdi_source_transfer(
            f=f,
            middle_time=middle_time,
            generation=generation,
            combination_type=combination_type,
            measurement_family="tmi",
            measurement_amplitude=-2.0 / lisaconstants.SPEED_OF_LIGHT,
        )
        return self._covariance_components(
            self._covariance(transfer),
            combination_type,
        )

    def oms_noise_transfer(
        self,
        f,
        middle_time,
        generation="2nd",
        combination_type="XYZ",
    ):
        transfer = self._tdi_source_transfer(
            f=f,
            middle_time=middle_time,
            generation=generation,
            combination_type=combination_type,
            measurement_family="sci",
            measurement_amplitude=1.0,
        )
        return self._covariance_components(
            self._covariance(transfer),
            combination_type,
        )

    def response_function(
        self,
        f,
        middle_time,
        generation="2nd",
        combination_type="XYZ",
    ):
        science_transfer = self._tdi_source_transfer(
            f=f,
            middle_time=middle_time,
            generation=generation,
            combination_type=combination_type,
            measurement_family="sci",
            measurement_amplitude=1.0,
        )
        response = self._response_covariance(
            f=f,
            middle_time=middle_time,
            science_transfer=science_transfer,
        )
        return self._covariance_components(response, combination_type)

    def detector_tables(
        self,
        f,
        middle_time,
        generation="2nd",
        combination_type="XYZ",
    ):
        acc_transfer = self._tdi_source_transfer(
            f=f,
            middle_time=middle_time,
            generation=generation,
            combination_type=combination_type,
            measurement_family="tmi",
            measurement_amplitude=-2.0 / lisaconstants.SPEED_OF_LIGHT,
        )
        science_transfer = self._tdi_source_transfer(
            f=f,
            middle_time=middle_time,
            generation=generation,
            combination_type=combination_type,
            measurement_family="sci",
            measurement_amplitude=1.0,
        )
        return {
            "light_travel_times": self.light_travel_times(middle_time),
            "acc_transfer": self._covariance_components(
                self._covariance(acc_transfer),
                combination_type,
            ),
            "oms_transfer": self._covariance_components(
                self._covariance(science_transfer),
                combination_type,
            ),
            "response": self._covariance_components(
                self._response_covariance(f, middle_time, science_transfer),
                combination_type,
            ),
        }

    def source_psds(self, f):
        """Return unit-ACC, unit-OMS, and GW anti-alias filter spectra."""
        f = np.asarray(f)
        defaults = inspect.signature(lisainstrument.Instrument).parameters
        upsampling = int(defaults["physics_upsampling"].default)
        physics_fs = self.fs * upsampling
        aafilter = ("kaiser", 240.0, 0.275 * self.fs, 0.725 * self.fs)
        _, coefficients, _ = lisainstrument.instru.init_aafilter(
            aafilter,
            physics_fs,
        )
        acceleration = lisainstrument.noisy.NoiseDefTestMass(
            accel_asd_onesided=1.0e-15,
            shape=defaults["testmass_shape"].default,
            f_min_hz=defaults["noises_f_min_hz"].default,
            f_knee_hz=defaults["testmass_fknees"].default,
            f_break_hz=defaults["testmass_fbreak"].default,
            f_relax_hz=defaults["testmass_frelax"].default,
            f_sample=physics_fs,
            name="unit_testmass",
        )
        oms = lisainstrument.noisy.NoiseDefOMS(
            displacement_asd_onesided=1.0e-12,
            f_min_hz=defaults["noises_f_min_hz"].default,
            f_knee_hz=defaults["oms_fknees"].default,
            f_sample=physics_fs,
            name="unit_oms",
        )
        acceleration_psd = np.zeros(len(f))
        oms_psd = np.zeros(len(f))
        coefficients = np.asarray(coefficients)

        for alias in range(upsampling):
            aliased = f + alias * self.fs
            aliased = np.where(
                aliased >= physics_fs / 2.0,
                aliased - physics_fs,
                aliased,
            )
            _, filter_amplitude = scipy.signal.freqz(
                coefficients,
                worN=2.0 * np.pi * aliased / physics_fs,
            )
            filter_power = np.abs(filter_amplitude) ** 2
            acceleration_psd += (
                filter_power * acceleration.discrete_psd_twosided(aliased)
            )
            oms_psd += filter_power * oms.discrete_psd_twosided(aliased)

        one_sided = np.full(len(f), 2.0)
        one_sided[np.isclose(f, self.fs / 2.0)] = 1.0
        _, signal_filter = scipy.signal.freqz(
            coefficients,
            worN=2.0 * np.pi * f / physics_fs,
        )
        return (
            one_sided * acceleration_psd,
            one_sided * oms_psd,
            np.abs(signal_filter) ** 2,
        )

    def noise_covariance(
        self,
        f,
        middle_time,
        acceleration_asd=3.0e-15,
        oms_asd=15.0e-12,
        generation="2nd",
        combination_type="XYZ",
    ):
        """Return the full ACC+OMS covariance and GW filter power.

        The physical ASD arguments use SI units.  Transfer functions, source
        spectra, anti-alias filtering, and cross-channel phases are evaluated
        through the same public ``DetectorFunction`` path used by the spectral
        analysis notebook.
        """
        f = np.asarray(f, dtype=np.float64)
        acceleration_transfer = self._tdi_source_transfer(
            f=f,
            middle_time=middle_time,
            generation=generation,
            combination_type=combination_type,
            measurement_family="tmi",
            measurement_amplitude=-2.0 / lisaconstants.SPEED_OF_LIGHT,
        )
        oms_transfer = self._tdi_source_transfer(
            f=f,
            middle_time=middle_time,
            generation=generation,
            combination_type=combination_type,
            measurement_family="sci",
            measurement_amplitude=1.0,
        )
        acceleration_psd, oms_psd, signal_filter_power = self.source_psds(f)
        acceleration_psd *= (acceleration_asd / 1.0e-15) ** 2
        oms_psd *= (oms_asd / 1.0e-12) ** 2
        covariance = self._covariance(acceleration_transfer) * (
            acceleration_psd[:, np.newaxis, np.newaxis]
        )
        covariance += self._covariance(oms_transfer) * (
            oms_psd[:, np.newaxis, np.newaxis]
        )
        covariance = 0.5 * (
            covariance + np.conjugate(np.swapaxes(covariance, 1, 2))
        )
        return covariance, signal_filter_power

    def inverse_noise_covariance(
        self,
        f,
        middle_time,
        acceleration_asd=3.0e-15,
        oms_asd=15.0e-12,
        generation="2nd",
        combination_type="XYZ",
        rcond=1.0e-12,
    ):
        """Return the Hermitian pseudoinverse of the full noise covariance."""
        covariance, signal_filter_power = self.noise_covariance(
            f=f,
            middle_time=middle_time,
            acceleration_asd=acceleration_asd,
            oms_asd=oms_asd,
            generation=generation,
            combination_type=combination_type,
        )
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        largest = np.maximum(
            eigenvalues[:, -1],
            np.finfo(np.float64).tiny,
        )
        retained = eigenvalues > rcond * largest[:, np.newaxis]
        inverse_eigenvalues = np.zeros_like(eigenvalues)
        np.divide(
            1.0,
            eigenvalues,
            out=inverse_eigenvalues,
            where=retained,
        )
        inverse_covariance = np.einsum(
            "eik,ek,ejk->eij",
            eigenvectors,
            inverse_eigenvalues,
            np.conjugate(eigenvectors),
            optimize=True,
        )
        return inverse_covariance, np.sqrt(signal_filter_power)

    def snr(
        self,
        frequency_response,
        f,
        middle_time,
        observation_duration,
        acceleration_asd=3.0e-15,
        oms_asd=15.0e-12,
        generation="2nd",
        combination_type="XYZ",
        rcond=1.0e-12,
    ):
        r"""Evaluate the full-covariance trace SNR for each source.

        ``frequency_response`` has shape ``(channel, source, Fourier bin)``
        and contains dimensionless, unnormalized DFT coefficients.  This
        follows the Godlog convention

        .. math::

           \widetilde{\mathbf s}_k
           = \sum_{n=0}^{N-1}\mathbf s_n e^{-2\pi i kn/N},
           \qquad
           \rho^2 = \frac{4}{T f_s^2}\sum_k
           \operatorname{Tr}[\mathbf C^+(f_0)\mathbf Q_k],
           \qquad
           \mathbf Q_k=\widetilde{\mathbf s}_k
           \widetilde{\mathbf s}_k^\dagger.

        Here ``f_s=N/T`` is inferred from the Fourier-bin count.  For a narrow
        DWD comb, the covariance is evaluated at each source carrier ``f`` and
        held across its annual sidebands.
        """
        frequency_response = np.asarray(frequency_response)
        if frequency_response.ndim != 3:
            raise ValueError(
                "frequency_response must have shape "
                "(channel, source, Fourier bin)"
            )
        inverse_covariance, signal_filter_amplitude = (
            self.inverse_noise_covariance(
                f=f,
                middle_time=middle_time,
                acceleration_asd=acceleration_asd,
                oms_asd=oms_asd,
                generation=generation,
                combination_type=combination_type,
                rcond=rcond,
            )
        )
        dft_sampling_rate = (
            frequency_response.shape[-1] / observation_duration
        )
        # Dividing the raw DFT by f_s before the contraction is algebraically
        # equivalent to the Godlog prefactor 4 / (T f_s^2).  This ordering also
        # preserves the established SNR floating-point operation order.
        filtered_response = (
            frequency_response
            / dft_sampling_rate
            * signal_filter_amplitude[np.newaxis, :, np.newaxis]
        )
        signal_matrix = np.einsum(
            "cef,def->ecd",
            filtered_response,
            np.conjugate(filtered_response),
            optimize=True,
        )
        snr_squared = (4.0 / observation_duration) * np.einsum(
            "ecd,edc->e",
            inverse_covariance,
            signal_matrix,
            optimize=True,
        ).real
        return np.sqrt(np.maximum(snr_squared, 0.0))

    def theoretical_psd_table(
        self,
        f,
        middle_time,
        para_list=[],
        model_type="noise+ast",
        generation="2nd",
        combination_type="XYZ",
    ):
        if model_type == "noise":
            N_acc, delta_x = para_list
            log_Omega_ast, alpha = 0.0, 0.0
        elif model_type == "ast":
            log_Omega_ast, alpha = para_list
            N_acc, delta_x = 0.0, 0.0
        else:
            N_acc, delta_x, log_Omega_ast, alpha = para_list

        f = np.asarray(f)
        tables = self.detector_tables(
            f=f,
            middle_time=middle_time,
            generation=generation,
            combination_type=combination_type,
        )
        acc_source_psd, oms_source_psd, signal_filter_power = self.source_psds(f)
        h0 = 67.66 * 1000.0 / (1.0e6 * lisaconstants.PARSEC_METER)
        f_ref = 3.162277660e-3
        omega = 10.0**log_Omega_ast * (f / f_ref) ** alpha
        strain_psd = 3.0 * h0**2 * omega / (4.0 * np.pi**2 * f**3)

        if model_type == "noise":
            strain_psd = np.zeros_like(f)
        return build_psd_table(
            response_table=tables["response"],
            acc_transfer_table=tables["acc_transfer"],
            oms_transfer_table=tables["oms_transfer"],
            signal_source_psd=strain_psd * signal_filter_power,
            acc_source_psd=N_acc**2 * acc_source_psd,
            oms_source_psd=delta_x**2 * oms_source_psd,
        )

    def theoretical_psd(
        self,
        f,
        middle_time,
        para_list=[],
        model_type="noise+ast",
        generation="2nd",
        combination_type="XYZ",
    ):
        return self.theoretical_psd_table(
            f=f,
            middle_time=middle_time,
            para_list=para_list,
            model_type=model_type,
            generation=generation,
            combination_type=combination_type,
        )["total"]


def build_psd_table(
    response_table,
    acc_transfer_table,
    oms_transfer_table,
    signal_source_psd,
    acc_source_psd,
    oms_source_psd,
):
    signal_psd = {
        component: response_table[component] * signal_source_psd
        for component in response_table
    }
    noise_psd = {
        component: (
            acc_transfer_table[component] * acc_source_psd
            + oms_transfer_table[component] * oms_source_psd
        )
        for component in response_table
    }
    total_psd = {
        component: signal_psd[component] + noise_psd[component]
        for component in response_table
    }
    return {
        "signal": signal_psd,
        "noise": noise_psd,
        "total": total_psd,
    }

def dwd_waveform(
    f,
    dwd_parameters,
    duration=86400.0,
    *,
    time_domain=False,
    icrs_polarization=False,
    include_initial_phase=True,
):
    """Return intrinsic h+ and hx for an eight-parameter DWD.

    By default, ``f`` is a frequency grid and the returned lists contain the
    positive-frequency continuous Fourier coefficients.  With
    ``time_domain=True``, ``f`` is instead a time grid and the returned arrays
    are the complex positive-frequency polarizations.  The moving-orbit link
    response requests the reference polarization coefficients without the
    initial phase, then applies that phase at the retarded link times.
    """
    from script.tool import _bme_to_icrs

    (
        amplitude,
        source_frequency,
        frequency_derivative,
        phase0,
        inclination,
        polarization,
        longitude,
        latitude,
    ) = dwd_parameters

    if icrs_polarization:
        psi = np.asarray(polarization, dtype=np.float64)
    else:
        _, _, psi = _bme_to_icrs(
            np.atleast_1d(np.asarray(longitude, dtype=np.float64)),
            np.atleast_1d(np.asarray(latitude, dtype=np.float64)),
            np.atleast_1d(np.asarray(polarization, dtype=np.float64)),
        )
        if np.ndim(polarization) == 0:
            psi = psi[0]

    cos_iota = np.cos(inclination)
    plus_source = -np.asarray(amplitude) * (1.0 + cos_iota**2)
    cross_source = 2.0j * np.asarray(amplitude) * cos_iota
    cos_2psi = np.cos(2.0 * psi)
    sin_2psi = np.sin(2.0 * psi)
    plus_amplitude = plus_source * cos_2psi - cross_source * sin_2psi
    cross_amplitude = plus_source * sin_2psi + cross_source * cos_2psi

    if time_domain:
        times = np.asarray(f, dtype=np.float64)

        def match_time_grid(value):
            value = np.asarray(value)
            return value.reshape(value.shape + (1,) * (times.ndim - value.ndim))

        initial_phase = phase0 if include_initial_phase else 0.0
        phase = (
            np.pi * match_time_grid(frequency_derivative) * times**2
            + 2.0 * np.pi * match_time_grid(source_frequency) * times
            - match_time_grid(initial_phase)
        )
        phase_factor = np.exp(1.0j * phase)
        return (
            match_time_grid(plus_amplitude) * phase_factor,
            match_time_grid(cross_amplitude) * phase_factor,
        )

    frequencies = np.atleast_1d(np.asarray(f, dtype=np.float64))
    frequency_offset = source_frequency - frequencies
    if abs(frequency_derivative) * duration**2 < 1.0e-8:
        chirp_integral = duration * np.sinc(frequency_offset * duration)
        chirp_integral = chirp_integral * np.exp(
            1.0j * (np.pi * frequency_offset * duration - phase0)
        )
    else:
        chirp_scale = np.sqrt(2.0 * abs(frequency_derivative))
        shifted_start = frequency_offset / frequency_derivative
        fresnel_start = chirp_scale * shifted_start
        fresnel_stop = chirp_scale * (duration + shifted_start)
        sine_start, cosine_start = scipy.special.fresnel(fresnel_start)
        sine_stop, cosine_stop = scipy.special.fresnel(fresnel_stop)
        fresnel_difference = (
            cosine_stop
            - cosine_start
            + 1.0j
            * np.sign(frequency_derivative)
            * (sine_stop - sine_start)
        )
        chirp_integral = (
            np.exp(
                -1.0j * phase0
                - 1.0j
                * np.pi
                * frequency_offset**2
                / frequency_derivative
            )
            * fresnel_difference
            / chirp_scale
        )

    h_plus = 0.5 * plus_amplitude * chirp_integral
    h_cross = 0.5 * cross_amplitude * chirp_integral
    return h_plus.tolist(), h_cross.tolist()


def dwd_xyz_response(
    dwd_parameters,
    sample_times,
    orbit_model,
    *,
    carrier_frequency=None,
    prepared_geometry=None,
):
    r"""Return the moving-orbit second-generation DWD response in XYZ.

    ``dwd_parameters`` contains the eight fields ``A``, ``f``, ``df``,
    ``phi0``, ``iota``, ``longitude``, ``latitude``, and ``psi_bme``.  Each
    value may describe one source or a source batch.  The channel order is
    ``(X2, Y2, Z2)`` and the returned array has shape
    ``(3, number_sources, number_times)``:

    .. math::

       \mathbf s_{\kappa n}
       = (X_2^{\rm DWD},Y_2^{\rm DWD},Z_2^{\rm DWD})^T(t_{\kappa n}).

    The real part is the physical fractional-frequency TDI signal.  One
    carrier frequency per source may be supplied to return a heterodyned
    envelope.  Reusing ``prepared_geometry`` avoids repeating the orbit and
    nested-delay interpolation for source batches on the same time grid.
    """
    from script.tool import _bme_to_icrs

    parameter_names = (
        "A",
        "f",
        "df",
        "phi0",
        "iota",
        "longitude",
        "latitude",
        "psi_bme",
    )
    parameters = {
        name: np.atleast_1d(np.asarray(dwd_parameters[name], dtype=np.float64))
        for name in parameter_names
    }
    right_ascension, declination, polarization = _bme_to_icrs(
        parameters["longitude"],
        parameters["latitude"],
        parameters["psi_bme"],
    )
    parameters["ra"] = np.ascontiguousarray(right_ascension)
    parameters["dec"] = np.ascontiguousarray(declination)
    parameters["psi"] = np.ascontiguousarray(polarization)

    sample_times = np.asarray(sample_times, dtype=np.float64)
    source_count = len(parameters["A"])
    if carrier_frequency is None:
        carrier_frequency = np.zeros(source_count, dtype=np.float64)
    else:
        carrier_frequency = np.atleast_1d(
            np.asarray(carrier_frequency, dtype=np.float64)
        )

    # The link response evaluates phase at distinct retarded emission and
    # reception times.  Obtain fixed polarization coefficients here and apply
    # the source phase inside _one_way_link_response.
    h_plus_coefficient, h_cross_coefficient = dwd_waveform(
        np.zeros((source_count, 1), dtype=np.float64),
        (
            parameters["A"],
            parameters["f"],
            parameters["df"],
            parameters["phi0"],
            parameters["iota"],
            parameters["psi"],
            parameters["ra"],
            parameters["dec"],
        ),
        time_domain=True,
        icrs_polarization=True,
        include_initial_phase=False,
    )

    if prepared_geometry is None:
        prepared_geometry = prepare_xyz_geometry(sample_times, orbit_model)
    _, channel_terms = prepared_geometry

    response = np.zeros(
        (3, source_count, len(sample_times)),
        dtype=np.complex128,
    )
    for channel_index, terms in enumerate(channel_terms):
        for factor, frequency_factor, geometry in terms:
            response[channel_index] += (
                factor
                * frequency_factor[np.newaxis, :]
                * _one_way_link_response(
                    parameters,
                    sample_times,
                    geometry,
                    carrier_frequency,
                    h_plus_coefficient,
                    h_cross_coefficient,
                )
            )
    return response


def dwd_xyz_response_fd(
    dwd_parameters,
    sample_times,
    orbit_model,
    *,
    carrier_frequency=None,
    prepared_geometry=None,
):
    r"""Return dimensionless positive-frequency XYZ raw-DFT coefficients.

    The time-domain response and its frequency-domain coefficients use the
    dimensionless Godlog convention

    .. math::

       \widetilde{\mathbf s}_k
       = \sum_{n=0}^{N-1}\mathbf s_n e^{-2\pi i kn/N}.

    ``sample_times`` contains both segment endpoints and therefore has
    ``N + 1`` entries; the final endpoint is omitted from the DFT.  The factor
    one half extracts the positive-frequency part of the real physical signal
    from the complex analytic response.  No ``dt`` multiplier is applied, so
    both ``s`` and ``s_tilde`` are dimensionless.
    """
    sample_times = np.asarray(sample_times, dtype=np.float64)
    segment_duration = sample_times[-1] - sample_times[0]
    midpoint = 0.5 * (sample_times[0] + sample_times[-1])

    if carrier_frequency is None:
        midpoint_frequency = (
            np.atleast_1d(np.asarray(dwd_parameters["f"], dtype=np.float64))
            + np.atleast_1d(
                np.asarray(dwd_parameters["df"], dtype=np.float64)
            )
            * midpoint
        )
        carrier_frequency = (
            np.rint(midpoint_frequency * segment_duration)
            / segment_duration
        )

    xyz_response = dwd_xyz_response(
        dwd_parameters,
        sample_times,
        orbit_model,
        carrier_frequency=carrier_frequency,
        prepared_geometry=prepared_geometry,
    )
    positive_frequency_response = 0.5 * xyz_response[:, :, :-1]
    return np.fft.fftshift(
        np.fft.fft(positive_frequency_response, axis=-1),
        axes=-1,
    )


def prepare_xyz_geometry(sample_times, orbit_model):
    """Precompute every delayed science-link term in PyTDI X2/Y2/Z2."""
    sample_times = np.asarray(sample_times, dtype=np.float64)
    channel_terms = []
    for combination in (
        pytdi.michelson.X2,
        pytdi.michelson.Y2,
        pytdi.michelson.Z2,
    ):
        terms = []
        for measurement, polynomial in combination.components.items():
            if not measurement.startswith("sci_"):
                continue
            link = int(measurement.split("_")[-1])
            for factor, operators in polynomial:
                query_times = _delayed_times(
                    sample_times,
                    operators,
                    orbit_model,
                )
                query_before = _delayed_times(
                    sample_times - 0.5,
                    operators,
                    orbit_model,
                )
                query_after = _delayed_times(
                    sample_times + 0.5,
                    operators,
                    orbit_model,
                )
                frequency_factor = query_after - query_before
                terms.append(
                    (
                        float(factor),
                        frequency_factor,
                        _link_geometry(query_times, link, orbit_model),
                    )
                )
        channel_terms.append(terms)
    return sample_times, channel_terms


def _delayed_times(sample_times, operators, orbit_model):
    """Apply one nested PyTDI delay path to an output-time grid."""
    query_times = np.asarray(sample_times, dtype=np.float64).copy()
    for operator in operators:
        _, receiving_spacecraft, emitting_spacecraft = pytdi.naming.split_indices(
            operator
        )
        link = int(f"{receiving_spacecraft}{emitting_spacecraft}")
        query_times -= orbit_model.ltt[link](query_times)
    return query_times


def _link_geometry(query_times, link, orbit_model):
    """Return emission times, positions, and direction for one LISA link."""
    query_times = np.asarray(query_times, dtype=np.float64)
    link_array = np.array([link], dtype=int)
    receiving_spacecraft = int(receiver(link_array)[0])
    emitting_spacecraft = int(emitter(link_array)[0])

    reception_times = query_times
    emission_times = query_times - orbit_model.ltt[link](query_times)
    receiver_position = np.column_stack(
        (
            orbit_model.x[receiving_spacecraft](reception_times),
            orbit_model.y[receiving_spacecraft](reception_times),
            orbit_model.z[receiving_spacecraft](reception_times),
        )
    )
    emitter_position = np.column_stack(
        (
            orbit_model.x[emitting_spacecraft](emission_times),
            orbit_model.y[emitting_spacecraft](emission_times),
            orbit_model.z[emitting_spacecraft](emission_times),
        )
    )
    link_vector = receiver_position - emitter_position
    link_vector /= np.linalg.norm(link_vector, axis=1)[:, np.newaxis]
    return (
        reception_times,
        emission_times,
        receiver_position,
        emitter_position,
        link_vector,
    )


def _one_way_link_response(
    parameters,
    reference_times,
    geometry,
    carrier_frequency,
    h_plus_coefficient,
    h_cross_coefficient,
):
    """Evaluate one heterodyned moving-orbit one-way GW link response."""
    (
        reception_times,
        emission_times,
        receiver_position,
        emitter_position,
        link_vector,
    ) = geometry

    frequency = parameters["f"][:, np.newaxis]
    frequency_derivative = parameters["df"][:, np.newaxis]
    phase0 = parameters["phi0"][:, np.newaxis]
    right_ascension = parameters["ra"]
    declination = parameters["dec"]

    cos_ra = np.cos(right_ascension)
    sin_ra = np.sin(right_ascension)
    cos_dec = np.cos(declination)
    sin_dec = np.sin(declination)
    propagation = np.column_stack(
        (-cos_dec * cos_ra, -cos_dec * sin_ra, -sin_dec)
    )
    basis_u = np.column_stack(
        (sin_ra, -cos_ra, np.zeros_like(right_ascension))
    )
    basis_v = np.column_stack(
        (-sin_dec * cos_ra, -sin_dec * sin_ra, cos_dec)
    )

    dot_n_k = np.einsum(
        "tc,ec->et",
        link_vector,
        propagation,
        optimize=True,
    )
    dot_n_u = np.einsum(
        "tc,ec->et",
        link_vector,
        basis_u,
        optimize=True,
    )
    dot_n_v = np.einsum(
        "tc,ec->et",
        link_vector,
        basis_v,
        optimize=True,
    )
    xi_plus = dot_n_u**2 - dot_n_v**2
    xi_cross = 2.0 * dot_n_u * dot_n_v

    reception_sun = reception_times[np.newaxis, :] - np.einsum(
        "tc,ec->et",
        receiver_position,
        propagation,
        optimize=True,
    ) / lisaconstants.c
    emission_sun = emission_times[np.newaxis, :] - np.einsum(
        "tc,ec->et",
        emitter_position,
        propagation,
        optimize=True,
    ) / lisaconstants.c

    reference_times = reference_times[np.newaxis, :]
    carrier_frequency = carrier_frequency[:, np.newaxis]
    phase_offset = (
        2.0
        * np.pi
        * (frequency - carrier_frequency)
        * reference_times
        - phase0
    )
    emission_phase = (
        np.pi * frequency_derivative * emission_sun**2
        + 2.0 * np.pi * frequency * (emission_sun - reference_times)
        + phase_offset
    )
    reception_phase = (
        np.pi * frequency_derivative * reception_sun**2
        + 2.0 * np.pi * frequency * (reception_sun - reference_times)
        + phase_offset
    )
    phase_difference = np.exp(1.0j * emission_phase) - np.exp(
        1.0j * reception_phase
    )
    return (
        (h_plus_coefficient * xi_plus + h_cross_coefficient * xi_cross)
        * phase_difference
        / (2.0 * (1.0 - dot_n_k))
    )

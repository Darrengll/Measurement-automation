from copy import deepcopy
from scipy import fftpack

from lib2.Measurement import *
from lib2.IQPulseSequence import IQPulseBuilder


class DigitizerWithPowerSweepMeasurementBase(Measurement):
    """
    Class for measurements with a Spectrum digitizer and power sweep

    This one must do:
        create Measurement object, set up all devices and take them from the class;
        set up all the parameters
        make measurements:
         -- sweep power/if_freq of one/another/both of generators
            and/or central if_freq of EXA and measure single trace / list sweep for certain frequencies
         --
    """

    def __init__(self, name, sample_name, measurement_result_class, **devs_aliases):
        """
        Parameters
        ----------
        name : str
            name of current measurement
        sample_name : str
            name of measured sample
        measurement_result_class : MeasurementResult
            measurement result for appropriate data handling and visualization for thi measurement
        devs_aliases : dict[str, Any]
            same as for Measurement class

        Notes
        ---------
        vna and current source is optional

        list_devs_names: {exa_name: default_name, src_plus_name: default_name,
                         src_minus_name: default_name, vna_name: default_name, current_name: default_name}
        """

        self._dig = devs_aliases.pop("dig", None)[0]
        super().__init__(name, sample_name, devs_aliases)
        self._devs_aliases = list(devs_aliases.keys())
        self._measurement_result = measurement_result_class(name, sample_name)

        # measurement class specific parameters section
        self._cal = None
        self._adc_parameters = None
        self._lo_parameters = None
        self._waveform_functions = {"CONTINUOUS TWO WAVES": self.get_two_continuous_waves,
                                    "CONTINUOUS WAVE": self.get_continuous_wave,
                                    "CONTINUOUS TWO WAVES FG": self.get_two_continuous_waves_fg}
        self._chosen_waveform_function = self._waveform_functions["CONTINUOUS TWO WAVES"]
        self._delta = 0
        self._modulation_array = None
        self._sweep_powers = None
        self.pulse_builder = None

        self._start_idx = None
        self._end_idx = None
        self._frequencies = None

    def set_fixed_parameters(self, waveform_type, awg_parameters=[], adc_parameters=[], freq_limits=(), lo_parameters=[]):
        """
        Parameters
        ----------
        waveform_type : str
            Choose the desired mode of operation
             One of the following is possible:
            "CONTINUOUS TWO WAVES"
            "CONTINUOUS WAVE"
            "CONTINUOUS TWO WAVES FG"

        awg_parameters : list[dict[str,Any]]
            maybe it is iqawg parameters?

        adc_parameters : list[dict[str, Any]]
            "channels" :    [1], # a list of channels to measure
            "ch_amplitude":    200, # amplitude for every channel
            "dur_seg":    100e-6, # duration of a segment in us
            "n_avg":    80000, # number of averages
            "n_seg":    2, # number of segments
            "oversampling_factor":    2, # sample_rate = max_sample_rate / oversampling_factor
            "pretrigger": 32,

        freq_limits : tuple[float]
            fourier limits for visualization
        lo_parameters : list[dict[str, Any]]

        Returns
        -------
        None

        Examples
        ________
        .ipynb

        name = "CWM_P";
        sample_name = "QOP_2_probe";
        wmBase = FourWaveMixingBase(name, sample_name, dig=[dig], lo=[exg], iqawg=[iqawg]);
        dig.stop_card()
        #awg.trigger_output_config("OFF", channel=channelI)
        #awg.trigger_output_config("ON", channel=channelQ)
        adc_pars = {"channels" :    [1], # a list of channels to measure
                    "ch_amplitude":    200, # amplitude for every channel
                    "dur_seg":    50e-6, # duration of a segment in us
                    "n_avg":    20000, # number of averages
                    "n_seg":    8, # number of segments
                    "oversampling_factor":    4, # sample_rate = max_sample_rate / oversampling_factor
                    "pretrigger": 32,
                   }
        lo_pars = { "power": lo_power,
                    "if_freq": lo_freq,
                  }

        wmBase.set_fixed_parameters(delta = 20e3, awg_parameters=[{"calibration": ro_cal}],
                                    adc_parameters=[adc_pars], freq_limits=(19.5, 20.5), lo_parameters=[lo_pars])
        wmBase.set_swept_parameters(powers_limits=(-40, 0), n_powers=201)

        #awg.trigger_output_config("ON", channel=channelQ)
        """
        self._chosen_waveform_function = self._waveform_functions[waveform_type]

        if len(awg_parameters) > 0:
            self._cal = awg_parameters[0]["calibration"]
            self._amplitudes = deepcopy(self._cal._if_amplitudes)
            self.pulse_builder = WMPulseBuilder(self._cal)

        if len(adc_parameters) > 0:
            self._adc_parameters = adc_parameters[0]
            self._dig.set_oversampling_factor(self._adc_parameters["oversampling_factor"])
            self._segment_size_optimal = int(self._adc_parameters["dur_seg"] * self._dig.get_sample_rate())
            self._segment_size = self._segment_size_optimal + 32 - self._segment_size_optimal % 32
            self._bufsize = self._adc_parameters["n_seg"] * self._segment_size * 4 * len(self._adc_parameters["channels"])
            self._dig.setup_averaging_mode(self._adc_parameters["channels"], self._adc_parameters["ch_amplitude"],
                                           self._adc_parameters["n_seg"], self._segment_size,
                                           self._adc_parameters["pretrigger"],
                                           self._adc_parameters["n_avg"])

        self._freq_limits = freq_limits

        # optimal size calculation
        self.nfft = fftpack.helper.next_fast_len(self._adc_parameters["n_seg"] * self._segment_size_optimal)

        # obtaining frequencies (frequencies is duplicating)
        xf = np.fft.fftfreq(self.nfft, 1 / self._dig.get_sample_rate()) / 1e6
        self._start_idx = np.searchsorted(xf[:self.nfft // 2 - 1], self._freq_limits[0])
        self._end_idx = np.searchsorted(xf[:self.nfft // 2 - 1], self._freq_limits[1])
        self._frequencies = xf[self._start_idx:self._end_idx + 1]

        self._measurement_result.get_context().update({"calibration_results": self._cal.get_optimization_results(), \
                                                       "radiation_parameters": self._cal.get_radiation_parameters()})

        super().set_fixed_parameters(iqawg=awg_parameters, lo=lo_parameters)
        if waveform_type == "CONTINUOUS TWO WAVES FG":
            self._iqawg[0].output_continuous_two_freq_IQ_waves(self._delta)

    def set_swept_parameters(self, powers_limits, n_powers):
        self._sweep_powers = np.linspace(*powers_limits, n_powers)
        swept_parameters = {"powers at $\\omega_{p}$": (self._set_power, self._sweep_powers)}
        super().set_swept_parameters(**swept_parameters)
        par_name = list(swept_parameters.keys())[0]
        self._measurement_result.set_parameter_name(par_name)
        # self._sources_on()

    def close_devs(self, *devs_to_close):
        if "spcm" in devs_to_close:
            self._dig.close()
        Measurement.close_devs(devs_to_close)

    def _sources_on(self):
        iq_sequence = self.pulse_builder.add_zero_pulse(10000).build()
        self._iqawg[0].output_pulse_sequence(iq_sequence)
        self._lo[0].set_output_state("ON")

    def _sources_off(self):
        iq_sequence = self.pulse_builder.add_zero_pulse(10000).build()
        self._iqawg[0].output_pulse_sequence(iq_sequence)
        self._lo[0].set_output_state("OFF")

    def srcs_power_calibration(self):
        """
        To define powers to set in setter (not implemented yet)
        """
        pass

    def _set_power(self, power):
        k = np.power(10, power / 20)
        self._chosen_waveform_function(k)
        # iq_sequence = self._chosen_waveform_function(k)
        # self._iqawg[0].output_pulse_sequence(iq_sequence)

    def get_two_continuous_waves(self, k_ampl):
        duration = 2e9 * self._adc_parameters["dur_seg"]
        return self.pulse_builder.add_simultaneous_pulses(duration, self._delta, amplitude=k_ampl).build()

    def get_two_continuous_waves_fg(self, k_ampl):
        self._iqawg[0].change_amplitudes_of_cont_IQ_waves(k_ampl)
        self._iqawg[0].update_modulation_coefficient_of_IQ_waves(2.)

    def get_continuous_wave(self, k_ampl):
        duration = 1e9 * self._adc_parameters["dur_seg"]
        return self.pulse_builder.add_sine_pulse(duration, amplitude_mult=k_ampl).build()

    def _prepare_measurement_result_data(self, parameter_names, parameters_values):
        measurement_data = super()._prepare_measurement_result_data(parameter_names, parameters_values)
        measurement_data["if_freq"] = self._frequencies
        return measurement_data

    def _recording_iteration(self):
        data = self._dig.measure(self._bufsize)  # data in mV
        # deleting extra samples from segments
        a = np.arange(self._segment_size_optimal, len(data), self._segment_size)
        b = np.concatenate([a + i for i in range(0, self._segment_size - self._segment_size_optimal)])
        data_cut = np.delete(data, b)
        yf = np.abs(np.fft.fft(data_cut, self.nfft))[self._start_idx:self._end_idx + 1] * 2 / self.nfft
        self._measurement_result._iter += 1
        return yf


class WMPulseBuilder(IQPulseBuilder):
    """IQ Pulse builder for wave mixing and for other measurements for a single qubit in line """

    def add_simultaneous_pulses(self, duration, delta_freq, phase=0, amplitude=1,
                                window="rectangular", hd_amplitude=0):
        """
        Adds two simultaneous pulses with amplitudes defined by the iqmx_calibration at frequencies
        (f_lo-f_if) ± delta_freq (or simpler w0 ± dw) and some phase to the sequence. All sine pulses will be parts
        of the same continuous wave at if_freq of f_if

        Parameters:
        -----------
        duration: float, ns
            Duration of the pulse in nanoseconds. For pulses other than rectangular
            will be interpreted as t_g (see F. Motzoi et al. PRL (2009))
        delta_freq: int, Hz
            The shift of two sidebands from the central if_freq. Ought to be > 0 Hz
        phase: float, rad
            Adds a relative phase to the outputted trace.
        amplitude: float
            Calibration if_amplitudes will be scaled by the
            amplitude_value.
        window: string
            List containing the name and the description of the modulating
            window of the pulse.
            Implemented modulations:
            "rectangular"
                Rectangular window.
            "gaussian"
                Gaussian window, see F. Motzoi et al. PRL (2009).
            "hahn"
                Hahn sin^2 window
        hd_amplitude: float
            correction for the Half Derivative method, theoretically should be 1
        """
        freq_m = self._iqmx_calibration._if_frequency - delta_freq
        freq_p = self._iqmx_calibration._if_frequency + delta_freq

        if_offsets = self._iqmx_calibration._if_offsets
        if_amplitudes = self._iqmx_calibration._if_amplitudes
        sequence_m = IQPulseBuilder(self._iqmx_calibration).add_sine_pulse(duration, phase, amplitude, window,
                                                                           hd_amplitude,
                                                                           freq_m, if_offsets/2, if_amplitudes/2).build()
        sequence_p = IQPulseBuilder(self._iqmx_calibration).add_sine_pulse(duration, phase, amplitude, window,
                                                                           hd_amplitude,
                                                                           freq_p, if_offsets/2, if_amplitudes/2).build()
        final_seq = sequence_m.direct_add(sequence_p)

        self._pulse_seq_I += final_seq._i
        self._pulse_seq_Q += final_seq._q
        return self

import drivers.instr as instr
from drivers.IQAWG import IQAWG
from lib.iq_mixer_calibration import *
from drivers.Spectrum_m4x import *
from scipy.fftpack import fft, fftfreq
from time import sleep
from lib.data_management import *

class DacAdcVNA():

    def __init__(self, iqawg : IQAWG, iqadc, lo, sa):
        '''
        Driver for a virtual VNA based on the heterodyne scheme with up-conversion
        and down-conversion in IQ mixers in the SSBSC (single-sideband suppressed carrier)
        regime. Two-channel DAC, ADC, local oscillator (microwave source) and spectrum
        analyzer for SSBSC mixer calibration should be supplied

        Parameters:
        -----------------
        iqawg: drivers.IQAWG instance
            pointer to an iqawg object
        iqadc: TODO: no interface class yet, maybe we should make it
            pointer to an analog-digital converter object
        lo: microwave source instance
            pointer to a local oscillator object
        sa: spectrum analyser instance
            pointer to a spectrum analyser object
        '''

        self._iqawg = iqawg
        self._iqadc = iqadc
        self._sa = sa
        self._lo = lo
        self._if_frequency = 100e6  # the value IQ mixer was calibrated for
        self._cal = None
        self._nop = None
        self._power = None

        self._samples_I = []
        self._samples_Q = []
        self._averages = 200
        self._amplitude_window = 500
        self._recalibrate_mixer = False
        self._adc_trigger_delay = 0
        self._dac_overridden = False
        self._lo.set_power(14)

    # Setter methods
    def set_lo(self, lo):
        self._lo = lo

    def set_iqawg(self, iqawg):
        self._iqawg = iqawg

    def set_iqadc(self, iqadc):
        self._iqadc = iqadc

    def set_sa(self, sa):
        self._sa = sa

    def set_lo(self, lo):
        self._lo = lo

    def set_if_frequency(self, if_frequency):
        self._if_frequency = if_frequency

    def set_cal(self, cal):
        self._cal = cal

    def set_nop(self, nop):
        self._nop = nop

    def set_power(self, power):
        self._power = power

    def set_freq_limits(self, *freq_limits):
        if ptp(freq_limits) > 200e6:
            raise ValueError("Can only measure within single SSB calibration area, reduce span")
        self._freq_limits = freq_limits

    def set_trigger_type(self, trigger_type):
        pass

    def set_averages(self, averages):
        self._averages = averages
        self._iqadc_parameters_invalidated = True

    def set_bandwidth(self, if_bw):
        requested_sample_length = 1 / if_bw * self._iqadc.get_sample_rate()
        real_sample_length = 2 ** int(log2(requested_sample_length))
        real_bw = (1.25e9 / real_sample_length)
        self._bandwidth = real_bw
        self._sample_length = real_sample_length
        self._iqadc_parameters_invalidated = True

    def sweep_hold(self):
        pass

    def avg_clear(self):
        pass

    def autoscale_all(self):
        pass

    def set_adc_trigger_delay(self, delay):
        self._adc_trigger_delay = delay
        self._iqadc_parameters_invalidated = True

    def generate_iqadc_parameters(self):
        self._iqadc_parameters_invalidated = False
        return {"channels": [0, 1],  # a list of channels to measure
                "ch_amplitude": self._amplitude_window,
                # mV, amplitude for every channel (allowed values are 200, 500, 1000, 2500 mV)
                "dur_seg": self._sample_length / self._iqadc.get_sample_rate() * 1e9,  # duration of a segment in ns
                "n_avg": self._averages,  # number of averages
                "n_seg": 1,  # number of segments
                "oversampling_factor": 1,  # sample_rate = max_sample_rate / oversampling_factor
                "pretrigger": 32,  # samples
                "mode": SPCM_MODE.AVERAGING,
                "trig_source": SPCM_TRIGGER.EXT0,
                "digitizer_delay": self._adc_trigger_delay}


    def set_parameters(self, parameters_dict):
        """
        Method allowing to set all or some of the VNA parameters at once
        (bandwidth, nop, power, averages, freq_limits and sweep type)
        """
        if "bandwidth" in parameters_dict:
            self.set_bandwidth(parameters_dict["bandwidth"])
        if "averages" in parameters_dict:
            self.set_averages(parameters_dict["averages"])
        if "power" in parameters_dict:
            self.set_power(parameters_dict["power"])
        if "nop" in parameters_dict:
            self.set_nop(parameters_dict["nop"])
        if "freq_limits" in parameters_dict:
            self.set_freq_limits(*parameters_dict["freq_limits"])
        if "adc_trigger_delay" in parameters_dict:
            self.set_adc_trigger_delay(parameters_dict["adc_trigger_delay"])
        if "dac_overridden" in parameters_dict:
            self._dac_overridden = parameters_dict["dac_overridden"]
        else:
            self._dac_overridden = False

        self._calibrate_ssb(mean(self._freq_limits))
        if not self._dac_overridden:
            self._output_IF()


    # Getter methods

    def get_averages(self):
        return self._averages

    def get_frequencies(self):
        return linspace(*self._freq_limits, self._nop)

    def get_sdata(self):
        # s_data = []
        # N = self._samples_I[0].shape[0] * 2
        # freqs = fftfreq(N, 1 / self._iqadc.get_sample_rate())
        # mask = np.logical_and(freqs > 95e6, freqs < 105e6)
        # for sample in self._samples_I:
        #     sample_fft = fft(sample, n=N)[mask] * 2 / N
        #     max_idx = argmax(abs(sample_fft))
        #     s_data.append(real(sample_fft[max_idx]) + 1j * imag(sample_fft[max_idx]))

        Ts = arange(0, len(self._samples_I[0])) / self._iqadc.get_sample_rate()
        complex_IQ = (array(self._samples_I) + 1j * array(self._samples_Q))
        demodulated_IQ = complex_IQ * exp(-1j * self._if_frequency * Ts * 2 * pi)
        return mean(conj(demodulated_IQ), axis=1)

    def _output_IF(self):
        self._iqawg.set_parameters({"calibration": self._cal})
        pb = self._iqawg.get_pulse_builder()
        seq = pb.add_sine_pulse(1000).build()
        self._iqawg.output_pulse_sequence(seq)

    # OPERATING methods
    def _calibrate_ssb(self, frequency):

        frequency = round(frequency / 1e9, 2) * 1e9


        if self._cal is not None:
            if self._cal._ssb_power == self._power \
                    and self._cal._lo_frequency - self._if_frequency == frequency:
                return

        ig = {"dc_offsets": (-0.017, -0.04), \
              "if_amplitudes": (.1, .1), "if_phase": -pi * 0.54}  # initial guess

        db = load_IQMX_calibration_database("DAV", 0)
        if db is not None:
            cal = \
                db.get(frozenset(dict(lo_power=14,
                                      ssb_power=self._power,
                                      lo_frequency=self._if_frequency+frequency,
                                      if_frequency=self._if_frequency,
                                      waveform_resolution=1,
                                      sideband_to_maintain = 'left').items()))
            if cal is not None and not self._recalibrate_mixer:
                self._cal = cal
                return

        cal = IQCalibrator(self._iqawg, self._sa, self._lo, "DAV", 0, sidebands_to_suppress=6)
        self._cal = cal.calibrate(lo_frequency=frequency + self._if_frequency, if_frequency=self._if_frequency,
                                  lo_power=14,
                                  ssb_power=self._power, waveform_resolution=1, iterations=3, minimize_iterlimit=20,
                                  sa_res_bandwidth=500,
                                  initial_guess=ig)
        save_IQMX_calibration(self._cal)


    def sweep_single(self):

        frequencies = self.get_frequencies()

        if self._iqadc_parameters_invalidated:
            self._iqadc.set_parameters(self.generate_iqadc_parameters())

        self._samples_I = []
        self._samples_Q = []
        for freq in frequencies:
            self._lo.set_frequency(freq + self._if_frequency)
            self._lo.set_frequency(freq + self._if_frequency)

            # sleep(0.00001)
            data = self._iqadc.measure()
            data = data / 128 / self._averages * self._amplitude_window

            self._samples_I += [data[0::2]]
            self._samples_Q += [data[1::2]]

    def wait_for_stb(self):
        pass

    def prepare_for_stb(self):
        pass

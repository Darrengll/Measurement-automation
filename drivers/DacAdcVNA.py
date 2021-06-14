from tqdm.notebook import tqdm

from drivers.IQVectorGenerator import IQVectorGenerator
from lib.iq_mixer_calibration import *
from drivers.Spectrum_m4x import *
from lib.data_management import *


class DacAdcVNA:

    def __init__(self, iqadc, iqvg: IQVectorGenerator, sa):
        '''
        Driver for a virtual VNA based on the heterodyne scheme with up-conversion
        and down-conversion in IQ mixers in the SSBSC (single-sideband suppressed carrier)
        regime. Two-channel DAC, ADC, local oscillator (microwave source) and spectrum
        analyzer for SSBSC mixer calibration should be supplied

        Parameters:
        -----------------
        iqadc: TODO: no interface class yet, maybe we should make it
            pointer to an analog-digital converter object
        iqvg: drivers.IQVectorGenerator instance
            pointer to a vector generator object
        sa: spectrum analyser instance
            pointer to a spectrum analyser object
        '''

        self._iqvg = iqvg
        self._iqadc = iqadc
        self._sa = sa
        self._cal = None
        self._nop = None
        self._power = None
        self._freq_limits = None
        self._bandwidth = None
        self._adc_timeout = 10000  # ms
        self._recalibrate_mixer = False
        self._status_widget = widgets.HTML("Idle")

        self._reference_level = -10
        self._samples_I = []
        self._samples_Q = []
        self._averages = 200
        self._amplitude_window = 1000
        self._adc_trigger_delay = 0
        self._sample_length = None
        self._iqadc_parameters_invalidated = False

    def get_status_widget(self):
        return self._status_widget

    def set_if_frequency(self, if_frequency):
        self._iqvg.set_frequency(if_frequency)

    def set_cal(self, cal):
        self._cal = cal

    def set_nop(self, nop):
        self._nop = nop

    def set_power(self, power):
        self._power = power
        self._iqvg.set_power(power)

    def set_freq_limits(self, *freq_limits):
        self._iqvg.set_frequency(np.mean(freq_limits))
        self._freq_limits = freq_limits

    def set_output_state(self, state):
        pass  # this device does not support output states

    def set_trigger_type(self, trigger_type):
        pass

    def set_averages(self, averages):
        self._averages = averages
        self._iqadc_parameters_invalidated = True

    def set_bandwidth(self, if_bw):
        requested_sample_length = 1 / if_bw * self._iqadc.get_sample_rate()
        # real_sample_length = 2 ** int(log2(requested_sample_length))
        real_sample_length = floor(requested_sample_length / 32) * 32
        real_bw = (1.25e9 / real_sample_length)
        self._bandwidth = real_bw
        self._sample_length = real_sample_length
        self._iqadc_parameters_invalidated = True
        # spectrum has rearm time of 80 ns (100 samples at 1.25 Gs/s) between
        # triggers in averaging mode. this is important for continuous
        # wave regime
        self._iqvg.set_marker_period((self._sample_length+100) / self._iqadc.get_sample_rate() * 1e9)

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
                                           # TODO: when 2, phase jumps of around 0.2pi
                "pretrigger": 32,  # samples
                "mode": SPCM_MODE.AVERAGING,
                "trig_source": SPCM_TRIGGER.EXT0,
                "digitizer_delay": self._adc_trigger_delay}

    def get_parameters(self):
        return {"freq_limits": self._freq_limits,
                "bandwidth:": self._bandwidth,
                "sample_length": self._sample_length,
                "power": self._power,
                "averages": self._averages,
                "nop": self._nop,
                "adc_trigger_delay": self._adc_trigger_delay}

    def set_parameters(self, parameters_dict):
        """
        Method allowing to set all or some of the VNA parameters at once
        (bandwidth, nop, power, averages, freq_limits and sweep type)
        """
        if "freq_limits" in parameters_dict:
            self.set_freq_limits(*parameters_dict["freq_limits"])
        if "bandwidth" in parameters_dict:
            self.set_bandwidth(parameters_dict["bandwidth"])
        if "averages" in parameters_dict:
            self.set_averages(parameters_dict["averages"])
        if "power" in parameters_dict:
            self.set_power(parameters_dict["power"])
        if "nop" in parameters_dict:
            self.set_nop(parameters_dict["nop"])
        if "adc_trigger_delay" in parameters_dict:
            self.set_adc_trigger_delay(parameters_dict["adc_trigger_delay"])
        else:
            self.set_adc_trigger_delay(0)

    def select_S_param(self, s_parameter):
        if s_parameter is not "S21":
            raise ValueError("This VNA only measures S21, %s is not supported" % s_parameter)
        pass

    # Getter methods

    def get_averages(self):
        return self._averages

    def get_iqawg(self):
        return self._iqvg.get_iqawg()

    def get_frequencies(self):
        return linspace(*self._freq_limits, self._nop)

    def get_power(self):
        return self._power

    def demodulated_traces(self):
        Ts = arange(0, len(self._samples_I[0])) / self._iqadc.get_sample_rate()
        complex_IQ = (array(self._samples_I) + 1j * array(self._samples_Q))
        demodulated_IQ = complex_IQ * exp(-1j * self._iqvg.get_if_frequency() * Ts * 2 * pi)
        return real(demodulated_IQ), imag(demodulated_IQ)

    def get_sdata(self):
        Ts = arange(0, len(self._samples_I[0])) / self._iqadc.get_sample_rate()
        complex_IQ = (array(self._samples_I) + 1j * array(self._samples_Q))
        demodulated_IQ = complex_IQ * exp(-1j * self._iqvg.get_if_frequency() * Ts * 2 * pi)
        # Downconversion attenuation is cancelled by 20 dB IF amplifiers
        return mean(conj(demodulated_IQ), axis=1)/sqrt(10**(
                (self._power-self._reference_level)/10))

    def get_calibration(self, frequency, power):
        return self._iqvg.get_calibration(frequency, power)

    def sweep_single(self, progress_bar = False):
        frequencies = self.get_frequencies()[::-1]
        self._iqvg.set_frequency(frequencies[0])  # ensure

        if self._iqadc_parameters_invalidated:
            self._iqadc.set_parameters(self.generate_iqadc_parameters())

        self._samples_I = []
        self._samples_Q = []
        if progress_bar:
            frequencies = tqdm(frequencies)
        for idx, freq in enumerate(frequencies):
            self._status_widget.value = f"Current frequency {(freq/1e9):.4f} " \
                                        f"GHz. Sweep done: " \
                                        f"{(idx/self._nop*100):.1f} %"
            self._iqvg.set_frequency(freq)
            # time.sleep(.05)

            data = self._iqadc.measure(timeout=self._adc_timeout)

            self._samples_I += [data[0::2]]
            self._samples_Q += [data[1::2]]
        self._samples_I = self._samples_I[::-1]
        self._samples_Q = self._samples_Q[::-1]
        self._status_widget.value = "Idle"


    def sweep_continuous(self):
        pass

    def wait_for_stb(self):
        pass

    def prepare_for_stb(self):
        pass

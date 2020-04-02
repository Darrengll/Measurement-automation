import drivers.instr as instr
from drivers.IQAWG import IQAWG
from drivers.IQVectorGenerator import IQVectorGenerator
from lib.iq_mixer_calibration import *
from drivers.Spectrum_m4x import *
from scipy.fftpack import fft, fftfreq
from time import sleep
from lib.data_management import *

class DacAdcVNA():

    def __init__(self, iqadc, iqvg : IQVectorGenerator, sa):
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

        self._samples_I = []
        self._samples_Q = []
        self._averages = 200
        self._amplitude_window = 500
        self._recalibrate_mixer = False
        self._adc_trigger_delay = 0


    def set_if_frequency(self, if_frequency):
        self._iqvg.set_frequency(if_frequency)

    def set_cal(self, cal):
        self._cal = cal

    def set_nop(self, nop):
        self._nop = nop

    def set_power(self, power):
        self._iqvg.set_power(power)

    def set_freq_limits(self, *freq_limits):
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

    def select_S_param(self, s_parameter):
        if s_parameter is not "S21":
            raise ValueError("This VNA only measures S21, %s is not supported"%s_parameter)
        pass

    # Getter methods

    def get_averages(self):
        return self._averages

    def get_frequencies(self):
        return linspace(*self._freq_limits, self._nop)

    def get_sdata(self):
        Ts = arange(0, len(self._samples_I[0])) / self._iqadc.get_sample_rate()
        complex_IQ = (array(self._samples_I) + 1j * array(self._samples_Q))
        demodulated_IQ = complex_IQ * exp(-1j * self._iqvg.get_if_frequency() * Ts * 2 * pi)
        return mean(conj(demodulated_IQ), axis=1)

    def sweep_single(self):

        frequencies = self.get_frequencies()
        self._iqvg.set_frequency(frequencies[0])  # ensure

        if self._iqadc_parameters_invalidated:
            self._iqadc.set_parameters(self.generate_iqadc_parameters())

        self._samples_I = []
        self._samples_Q = []
        for freq in frequencies:
            self._iqvg.set_frequency(freq)

            data = self._iqadc.measure()
            data = data / 128 / self._averages * self._amplitude_window

            self._samples_I += [data[0::2]]
            self._samples_Q += [data[1::2]]

    def wait_for_stb(self):
        pass

    def prepare_for_stb(self):
        pass

import numpy as np
from typing import Tuple

from lib2.Measurement import Measurement, MeasurementResult
from drivers.IQAWG import IQAWG
from drivers.Spectrum_m4x import SPCM, SPCM_MODE,SPCM_TRIGGER

class DigVNA(Measurement):
    _freqs_range: tuple

    def __init__(self, name, sample_name, plot_update_interval=5,
                 lo=[], iqawg=[], dig=[]):
        """
        name : str
            name of current measurement
        sample_name : str
            name of measured sample
        comment: str
            comment for the measurement
        q_lo, q_iqawg, dig: arrays with references
            references to LO source, AWG and the digitizer
        """
        self._lo = None
        self._dig: list[SPCM] = None
        self._iqawg: list[IQAWG] = None
        devs_aliases_map = {"lo": lo,
                            "dig": dig,
                            "iqawg": iqawg}
        super().__init__(name, sample_name, devs_aliases_map, plot_update_interval)

        self._measurement_result = DigVNAResult(name, sample_name)

        ''' internal parameters '''
        # range of frequency scan
        self._freqs_range: Tuple[float, float] = None
        # number of points in frequency scan
        self._freqs_nop: int = None
        # frequencies that are scanned
        self._freqs: np.ndarray[float] = None
        # multplier that will be used to change _if_amplitudes in calibration
        self._iqawg_amplitudes: Tuple[int, int] = None
        # bandiwdth of the VNA being faked in Hz
        self._bandwidth: float = None

    def set_fixed_parameters(self, lo_params=None, iqawg_params=None, dig_params=None,
                             bandwidth=1e3, iqawg_amplitudes=(1, 1)):
        fixed_pars = {"lo": lo_params,
                      "iqawg": iqawg_params,
                      "dig": dig_params}
        dig_params["mode"] = SPCM_MODE.AVERAGING   # averaging mode is chosen
        dig_params["trig_source"] = SPCM_TRIGGER.AUTOTRIG  # triggering automatically
        super().set_fixed_parameters(**fixed_pars)

        self._iqawg_amplitudes = iqawg_amplitudes
        self._bandwidth = bandwidth
        # calculating index of the sideband
        dig = self._dig[0]
        iqawg = self._iqawg[0]
        fft_freqs = np.fft.fftshift(np.fft.fftfreq(dig._segment_size, 1/dig.get_sample_rate()))
        self._sideband_freq_idx = np.searchsorted(iqawg._calibration._if_frequency)

    def set_swept_parameters(self, frequencies):
        self._freqs_nop = frequencies.shape[0]
        self._freqs = frequencies
        self._freqs_range = (np.amin(frequencies), np.amax(frequencies))

        swept_pars = {"Frequency [Hz]": (self.set_frequency, self._freqs)}
        super().set_swept_parameters(**swept_pars)

    def set_frequency(self, frequency):
        iqawg = self._iqawg[0]
        center_freq = iqawg._calibration._sideband_to_maintain_freq
        if_freq = iqawg._calibration._if_frequency
        calibration_safe_zone = (center_freq - 2*if_freq, center_freq + 2*if_freq)

        # if 'frequency' is in "safe calibration zone"
        if (frequency < calibration_safe_zone[1]) and (frequency > calibration_safe_zone[0]):
            sideband = iqawg._calibration._sideband_to_maintain
            lo_freq = None
            # determine what 'lo_freq' will correspond to the desired 'frequency'
            if sideband == "right":
                lo_freq = frequency - if_freq
            else:  # calibration with left sideband
                lo_freq = frequency + if_freq
            # output desired frequency
            self._lo[0].set_frequency(lo_freq)
        else:  # 'frequency' is out of "safe calibration zone"
            raise ValueError("frequency is out of range")

    def _recording_iteration(self):
        dig = self._dig[0]
        iqawg = self._iqawg[0]

        data = dig.measure(dig._bufsize).astype(float)
        data_cut = SPCM.extract_useful_data(data, 2, dig._segment_size, dig.get_how_many_samples_to_drop_in_front(),
                                            dig.get_how_many_samples_to_drop_in_end())
        data_cut = (2 * (data_cut / dig.n_avg + 128) / 255 - 1) * dig.ch_amplitude
        dataI = data_cut[::2]
        dataQ = data_cut[1::2]

        fft_data = np.fft.fftshift(np.fft.fft(dataI + 1j * dataQ, self._nfft)) / self._nfft
        yf = fft_data[self._start_idx:self._end_idx + 1]
        self._measurement_result._iter += 1
        return yf




class DigVNAResult(MeasurementResult):
    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)

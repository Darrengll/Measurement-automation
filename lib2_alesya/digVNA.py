import numpy as np
from typing import Tuple
from importlib import reload

import lib2.Measurement
reload(lib2.Measurement)
from lib2.Measurement import Measurement

import lib2.MeasurementResult
reload(lib2.MeasurementResult)
from lib2.MeasurementResult import MeasurementResult

from drivers.IQAWG import IQAWG
from drivers.Spectrum_m4x import SPCM, SPCM_MODE,SPCM_TRIGGER
from drivers.E8257D import EXG

import matplotlib.pyplot as plt

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
        self._lo: list[EXG] = None
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
        self._iqawg_amplitudes_mul: np.ndarray[float,float] = None
        # original _if_amplitudes from calibration
        self._iqawg_amplitudes_calib: np.ndarray[float, float] = None
        # bandiwdth of the VNA being faked in Hz
        self._bandwidth: float = None
        # index in 'fftshift'ed array of the IF frequency
        # this is set once and for all during call of 'set_fixed_parameters'
        self._sideband_freq_idx: float = None

        # for debug purposes
        self.dataI = []
        self.dataQ = []

    def set_fixed_parameters(self, lo_params=None, iqawg_params=None, dig_params=None,
                             bandwidth=1e3, iqawg_amplitudes=(1, 1)):
        """

        Parameters
        ----------
        lo_params : list[dict[str,Any]]
        iqawg_params : list[dict[str,Any]]
        dig_params : list[dict[str,Any]]
        bandwidth : float
        iqawg_amplitudes : tuple[float]

        Returns
        -------
        None
        """
        fixed_pars = {"lo": lo_params,
                      "iqawg": iqawg_params,
                      "dig": dig_params}
        dig_params[0]["mode"] = SPCM_MODE.AVERAGING   # averaging mode is chosen
        dig_params[0]["trig_source"] = SPCM_TRIGGER.EXT0  # triggering automatically
        self._bandwidth = bandwidth

        # adjusting digitizer parameters according to required resolution
        dig_params[0]["dur_seg"] = 1/bandwidth
        dig_params[0]["n_seg"] = 1

        super().set_fixed_parameters(**fixed_pars)

        # calculating index of the sideband
        dig = self._dig[0]
        iqawg = self._iqawg[0]

        self._iqawg_amplitudes_mul = np.array(iqawg_amplitudes)
        self._iqawg_amplitudes_calib = np.array(iqawg._calibration._if_amplitudes)

        iqawg._calibration._if_amplitudes = self._iqawg_amplitudes_calib*self._iqawg_amplitudes_mul
        if_period = 1/iqawg._calibration._if_frequency
        trigger_every = np.ceil(dig._segment_size/dig.get_sample_rate() / if_period) * if_period  # there are only 1 segment
        iqawg.output_IQ_waves_from_calibration(trigger_sync_every=trigger_every)

        trace_length = dig._segment_size - dig.get_how_many_samples_to_drop_in_front() - dig.get_how_many_samples_to_drop_in_end()
        fft_freqs = np.fft.fftshift(np.fft.fftfreq(trace_length, 1/dig.get_sample_rate()))
        self._sideband_freq_idx = np.argmin(np.abs(fft_freqs - (-iqawg._calibration._if_frequency)))

    def set_swept_parameters(self, start_freq, stop_freq, nop):
        self._freqs_nop = nop
        self._freqs = np.linspace(start_freq, stop_freq, nop)
        self._freqs_range = (start_freq, stop_freq)

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
            from time import sleep
            sleep(0.01)  # wait for microwave source to stabilize
        else:  # 'frequency' is out of "safe calibration zone"
            raise ValueError("frequency is out of range")


    def _recording_iteration(self):
        dig = self._dig[0]
        iqawg = self._iqawg[0]
        data = dig.measure(dig._bufsize).astype(float)
        data_cut = SPCM.extract_useful_data(data, 2, dig._segment_size, dig.get_how_many_samples_to_drop_in_front(),
                                            dig.get_how_many_samples_to_drop_in_end())
        data_cut = data_cut / dig.n_avg / 128 * dig.ch_amplitude
        dataI = data_cut[::2]
        dataQ = data_cut[1::2]
        self.dataI.append(dataI)
        self.dataQ.append(dataQ)

        fft_data = np.fft.fftshift(np.fft.fft(dataI + 1j * dataQ, len(dataI))) / len(dataI)
        S21 = fft_data[self._sideband_freq_idx]

        return S21


class DigVNAResult(MeasurementResult):
    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._lines = [None]*2

    def _prepare_figure(self):
        fig, (ax_phase, ax_amp) = plt.subplots(2, 1, figsize=(17, 8), sharex=True)
        ax_amp.set_ylabel(r"$\left|S_{21}\right|$, dBV")
        ax_amp.set_xlabel("Frequency, [Hz]")
        ax_phase.set_ylabel(r"$\angle S_{21}$, [deg]")
        fig.canvas.set_window_title(self._name)

        return fig, (ax_amp, ax_phase), None

    def _plot(self, data):
        if "data" not in data.keys():
            return

        amps, degs, freqs = self._prepare_data_for_plot(data)
        for idx, (ax, y) in enumerate(zip(self._axes, (amps, degs))):
            if self._lines[idx] is None or not self._dynamic:
                ax.grid()
                ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
                self._lines[idx], = ax.plot(freqs[:self._iter_idx_ready[0]+1], y[:self._iter_idx_ready[0]+1])
                ax.set_xlim(freqs[0], freqs[-1])
            else:
                self._lines[idx].set_data(freqs[:self._iter_idx_ready[0]+1], y[:self._iter_idx_ready[0]+1])
                ax.relim()
                ax.set_xlim(freqs[0], freqs[-1])
                ax.autoscale_view()

    def _prepare_data_for_plot(self, data):
        sdata = data["data"]
        amps = 20*np.log10(np.abs(sdata))
        degs = self._unwrapped_phase(sdata)
        return amps, degs, data["Frequency [Hz]"]

    def _unwrapped_phase(self, sdata):
        try:
            unwrapped_phase = np.unwrap(np.angle(sdata))
            unwrapped_phase[sdata == 0] = 0
            return unwrapped_phase
        except Exception as e:
            print("Exception occured in digVnaResult._unwrapped_phase()", flush=True)
            return np.angle(sdata)







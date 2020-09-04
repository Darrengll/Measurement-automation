from scipy import fftpack as fp
import numpy as np
from scipy import optimize as opt
from scipy.interpolate import interp1d
from drivers.Spectrum_m4x import SPCM_TRIGGER
import matplotlib.pyplot as plt
from .iq_mixer_calibration import IQCalibrationData
import copy

class IQDownconversionCalibrationResult():

    def __init__(self, mixer_id, upconv_cal, ifreq, samplerate,
                 offsets=None,
                 phase=0, r=1):
        """

        Parameters
        ----------
        mixer_id : str
        upconv_cal : IQCalibrationData
        ifreq
        samplerate
        offsets
        phase
        r
        """
        self._mixer_id = mixer_id
        self._upconv_cal = copy.deepcopy(upconv_cal)

        if offsets is None:
            offsets = np.zeros(2)
        # an array of two elements with offsets to I and Q traces in mV
        self.offsets = offsets
        # a phase discrepancy in radians
        self.phase = phase
        # after multiplication of an I trace by r, an amplitude discrepancy
        # disappears
        self.r = r
        # intermittent frequency
        self.ifreq = ifreq
        # sampling rate of a digitizer
        self.samplerate = samplerate

        # influence of the length of all wires in cryostat on the phase of
        # a demodulated signal
        self.cryostat_delay = 0
        self.shift = 0

        # number of samlpes to drop from signal, due to shifting one
        # of the quadratures in process of applying calibration to signal
        self.samples_per_period = int(round(self.samplerate / self.ifreq))

    def get_mixer_parameters(self):
        mix_params = self._upconv_cal.get_mixer_parameters().copy()
        mix_params["mixer_id"] = self._mixer_id
        return mix_params

    def get_radiation_parameters(self):
        return self._upconv_cal.get_radiation_parameters()

    def get_dict(self):
        return self.__dict__

    def update(self, offset_re, offset_im, phase, r):
        self.offsets = [offset_re, offset_im]
        self.phase = phase
        self.r = r

    def set_cryostat_delay(self, delay):
        # delay in seconds times two pi
        self.cryostat_delay = delay

    def set_shift(self, shift):
        # could be LO or IF shift and needed for global phase compensation
        self.shift = shift

    def apply(self, trace):
        return np.exp(-1j * self.cryostat_delay * self.shift) * \
               ((1 + 1j * np.tan(self.phase)) *
                (np.real(trace) - self.offsets[0]) +
                1j / self.r / np.cos(self.phase) *
                (np.imag(trace) - self.offsets[1]))


class IQDownconversion():

    @staticmethod
    def calibrate(calibration_name, mixer_id, trace, ifreq,
                  samplerate):
        """
        Calibration of the IQ trace, measured by the digitizer, so that
        I**2 + Q**2 signal would be constant when I and Q are pure sine and
        cosine.
        This method calibrates simultaneously offsets, phase and amplitude
        discrepancies by minimizing the peaks of the squared signal at the
        intermittent frequency and the double intermittent frequency. Uses
        IQDownconversionCalibrationResult.apply to shift offsets, phase and
        amplitudes.
        Parameters
        ----------
        trace: numpy.ndarray, dtype=complex
            complex I + 1j * Q data trace
        ifreq: float
            intermittent frequency value in Hz
        samplerate:
            the sampling rate of the digitizer

        Returns
        -------
        IQDownconversionCalibrationResult
            Class that represents calibration results
        """
        cal = IQDownconversionCalibrationResult(calibration_name,
                                                mixer_id, ifreq, samplerate,
                                                [0, 0], 0, 1)

        def loss(opr):
            cal.update(*opr)
            tr = cal.apply(trace)
            nfft = fp.next_fast_len(len(trace))
            xf = fp.fftshift(fp.fftfreq(nfft, d=1 / samplerate))
            idx_if = np.argmin(np.abs(xf - ifreq))
            idx_2if = np.argmin(np.abs(xf - 2 * ifreq))
            yf = np.log(np.abs(fp.fftshift(fp.fft(np.abs(tr) ** 2, n=nfft))))
            f1 = np.abs(2 * yf[idx_if] + yf[idx_if - 1] + yf[idx_if + 1])
            f2 = np.abs(2 * yf[idx_2if] + yf[idx_2if - 1] + yf[idx_2if + 1])
            return f1 ** 2 + f2 ** 2

        b1 = np.abs(np.mean(np.real(trace)))
        b2 = np.abs(np.mean(np.imag(trace)))
        bounds = [(-2 * b1, 2 * b1), (-2 * b2, 2 * b2),
                  (-np.pi / 4, np.pi / 4), (0.95, 1.1)]
        cal.update(*opt.shgo(loss, bounds).x)
        return cal

    @staticmethod
    def show_before_and_after(time, trace, trace_cal):
        plt.figure()

        plt.subplot(2, 2, 1)
        plt.title("Before")
        plt.plot(time, np.real(trace))
        plt.plot(time, np.imag(trace))
        plt.plot(time, np.abs(trace))
        plt.tight_layout()
        plt.xlim(0, 100)

        plt.subplot(2, 2, 2)
        plt.title("After")
        plt.plot(time, np.real(trace_cal))
        plt.plot(time, np.imag(trace_cal))
        plt.plot(time, np.abs(trace_cal))
        plt.tight_layout()
        plt.xlim(0, 100)

        plt.subplot(2, 2, 3)
        plt.title("Before")
        nfft = fp.next_fast_len(len(trace))
        xf = fp.fftshift(fp.fftfreq(nfft, d=(time[1] - time[0]) * 1e-9))
        yf = fp.fftshift(fp.fft(np.abs(trace) ** 2, n=nfft))
        plt.plot(xf, 10 * np.log(np.abs(yf)))
        plt.xlim(xf[0], xf[-1])
        plt.grid(True)
        plt.tight_layout()

        plt.subplot(2, 2, 4)
        plt.title("After")
        nfft = fp.next_fast_len(len(trace_cal))
        xf = fp.fftshift(fp.fftfreq(nfft, d=(time[1] - time[0]) * 1e-9))
        yf = fp.fftshift(fp.fft(np.abs(trace_cal) ** 2, n=nfft))
        plt.plot(xf, 10 * np.log(np.abs(yf)))
        plt.xlim(xf[0], xf[-1])
        plt.grid(True)
        plt.tight_layout()

        plt.show()

    @staticmethod
    def show_signal_spectra_before_and_after(trace, trace_cal, sample_rate):
        plt.figure()

        plt.subplot(1, 2, 1)
        plt.title("Before")
        nfft = fp.next_fast_len(len(trace))
        xf = fp.fftshift(fp.fftfreq(nfft, d=1 / sample_rate))
        yf = fp.fftshift(fp.fft(trace, n=nfft))
        plt.plot(xf, 20 * np.log(np.abs(yf)))
        plt.xlim(xf[0], xf[-1])
        plt.grid(True)
        plt.tight_layout()

        plt.subplot(1, 2, 2)
        plt.title("After")
        nfft = fp.next_fast_len(len(trace_cal))
        xf = fp.fftshift(fp.fftfreq(nfft, d=1 / sample_rate))
        yf = fp.fftshift(fp.fft(trace_cal, n=nfft))
        plt.plot(xf, 20 * np.log(np.abs(yf)))
        plt.xlim(xf[0], xf[-1])
        plt.grid(True)
        plt.tight_layout()

        plt.show()

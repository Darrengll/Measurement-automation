from scipy import fftpack as fp
import numpy as np
from scipy import optimize as opt
from drivers.IQAWG import IQAWG
from drivers.Spectrum_m4x import SPCM_TRIGGER, SPCM
import matplotlib.pyplot as plt
from .iq_mixer_calibration import IQCalibrationData
import copy
import lib.data_management as dm


class IQDownconversionCalibrationResult:

    def __init__(self, mixer_id, samplerate, if_frequency, dig_params=None,
                 offsets=None, phase=0, r=1):
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

        if offsets is None:
            offsets = np.zeros(2)
        # an array of two elements with offsets to I and Q traces in mV
        self.offsets = offsets
        # a phase discrepancy in radians
        self.phase = phase
        # after multiplication of an I trace by r, an amplitude discrepancy
        # disappears
        self.r = r
        self._if_frequency = if_frequency
        # digitizer parameters
        # TODO: they has to be in frozenset key in
        #  calibrations storage ".pkl" file
        self._dig_params: dict = dig_params
        self._samplerate : float = samplerate
        # influence of the length of all wires in cryostat on the phase of
        # a demodulated signal
        self.cryostat_delay = 0
        self.shift = 0
        self._IQ_delay_correction = 0

        # number of samlpes to drop from signal, due to shifting one
        # of the quadratures in process of applying calibration to signal
        self.samples_per_period = int(
            round(self._samplerate / self._if_frequency)
        )
        # time values for ADC points
        self._time: np.ndarray = None
        self._trace: np.ndarray = None  # trace collected for calibration
        # corrected trace
        self._trace_cal: np.ndarray = None

    def get_if_frequency(self):
        return self._if_frequency

    def get_dict(self):
        return self.__dict__

    @staticmethod
    def load_dict(d):
        cal = IQDownconversionCalibrationResult("", 1, 1)
        cal.__dict__.update(d)
        return cal

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

    def get_coefficients(self):
        """Apply these coefficient to a trace for calibration"""
        a = np.exp(-1j * self.cryostat_delay * self.shift) \
            * (1 + 1j * np.tan(self.phase))
        b = 1 / self.r / np.cos(self.phase)
        return np.array([[np.real(a), 0, -np.real(a) * self.offsets[0]],
                         [np.imag(a), b, -b * self.offsets[1]]])

    def show_before_and_after(self):
        fig, axs = plt.subplots(2, 2, sharey="row", sharex="row")
        ax_tr_before = axs[0, 0]
        ax_tr_after = axs[0, 1]
        ax_fft2_before = axs[1, 0]
        axs_fft2_after = axs[1, 1]

        ax_tr_before.set_title("Before")
        ax_tr_before.plot(self._time, np.real(self._trace))
        ax_tr_before.plot(self._time, np.imag(self._trace))
        ax_tr_before.plot(self._time, np.abs(self._trace))
        ax_tr_before.set_xlim(0, 100)

        ax_tr_after.set_title("After")
        ax_tr_after.plot(self._time, np.real(self._trace_cal))
        ax_tr_after.plot(self._time, np.imag(self._trace_cal))
        ax_tr_after.plot(self._time, np.abs(self._trace_cal))
        ax_tr_after.set_xlim(0, 100)

        ax_fft2_before.set_title("Before")
        nfft = fp.next_fast_len(len(self._trace))
        xf = fp.fftshift(
            fp.fftfreq(
                nfft,
                d=1e9/self._samplerate
            )
        )
        yf = fp.fftshift(fp.fft(np.abs(self._trace) ** 2, n=nfft))
        ax_fft2_before.plot(xf, 10 * np.log(np.abs(yf)))
        ax_fft2_before.set_xlim(xf[0], xf[-1])
        ax_fft2_before.grid(True)

        axs_fft2_after.set_title("After")
        nfft = fp.next_fast_len(len(self._trace_cal))
        xf = fp.fftshift(
            fp.fftfreq(
                nfft,
                d=1e9/self._samplerate  # ns
            )
        )
        yf = fp.fftshift(fp.fft(np.abs(self._trace_cal) ** 2, n=nfft))
        axs_fft2_after.plot(xf, 10 * np.log(np.abs(yf)))
        axs_fft2_after.set_xlim(xf[0], xf[-1])
        axs_fft2_after.grid(True)

        fig.tight_layout()

        plt.show()

    def show_signal_spectra_before_and_after(self):

        fig, axs = plt.subplots(1, 2, sharey=True)
        ax_fft_before = axs[0]
        ax_fft_after = axs[1]

        ax_fft_before.set_title("Before")
        nfft = fp.next_fast_len(len(self._trace))
        xf = fp.fftshift(fp.fftfreq(nfft, d=1 / self._samplerate))
        yf = fp.fftshift(fp.fft(self._trace, n=nfft))
        ax_fft_before.plot(xf, 20 * np.log(np.abs(yf)))
        ax_fft_before.set_xlim(xf[0], xf[-1])
        ax_fft_before.grid(True)

        ax_fft_after.set_title("After")
        nfft = fp.next_fast_len(len(self._trace_cal))
        xf = fp.fftshift(fp.fftfreq(nfft, d=1 / self._samplerate))
        yf = fp.fftshift(fp.fft(self._trace_cal, n=nfft))
        ax_fft_after.plot(xf, 20 * np.log(np.abs(yf)))
        ax_fft_after.set_xlim(xf[0], xf[-1])
        ax_fft_after.grid(True)

        fig.tight_layout()

        plt.show()


class IQDownconversionCalibrator:

    def __init__(self, iqawg, dig, downconv_mixer_id):
        self._iqawg : IQAWG = iqawg
        self._dig : SPCM = dig
        self._downconv_mixer_id = downconv_mixer_id
        self._trigger_period = None
        # will be set during call of `self.calibrate`
        self._upconv_cal: IQCalibrationData = None
        # last successful calibration will be stored here
        self._last_downconv_cal: IQDownconversionCalibrationResult = None

    def calibrate(self, upconv_cal, initial_guess=None, dig_params=None,
                  trigger_period=1000, amps=(1.0, 1.0)):
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
        initial_guess : dict
            initial guess for parameters.
                "offsets" - I and Q quadratures offsets from zero
                "phase" - phase difference to compensate between I and Q
                channels
                "r" - I and Q quadratures amplitude ratio correction
                multiplier.
        dig_params : dict[str, Any]
            dictionary of parameters for digitizer. See `dig.set_params()`
            for your digitizer driver for parameters available.
        trigger_period : Union[float, int]
            nanoseconds
            period of IQAWG trigger output during continuous wave
            output.
        amps : tuple(float, float)
            pair of IQAWG amplitude multipliers applied to
            `self._upconv_cal._if_amplitudes`

        Returns
        -------
        IQDownconversionCalibrationResult
            Class that represents calibration results
        """
        self._trigger_period = trigger_period
        if_freq = upconv_cal.get_if_frequency()

        # output calibrating signal
        self._iqawg.set_parameters({"calibration": upconv_cal})
        self._iqawg.output_IQ_waves_from_calibration(
            trigger_sync_every=self._trigger_period,
            amp_coeffs=amps
        )

        # set initial parameters for optimization
        if initial_guess is not None:
            cal = IQDownconversionCalibrationResult(
                self._downconv_mixer_id, self._dig.get_sample_rate(),
                if_freq, dig_params, **initial_guess
            )
        else:
            cal = IQDownconversionCalibrationResult(
                self._downconv_mixer_id, self._dig.get_sample_rate(),
                if_freq, dig_params, [0, 0], 0, 1.0
            )

        # set digitizer parameters if necessary
        if dig_params is not None:
            self._dig.set_parameters(dig_params)
        else:
            # TODO: implement and return `_dig.get_parameters()`
            raise Warning("digitizer parameters has to be provided explicitly."
                          " `self._dig.get_parameters()` is not implemented "
                          "yet")
        cal._samplerate = self._dig.get_sample_rate()


        # record trace
        data = self._dig.safe_measure()
        trace = data[0::2] + 1j * data[1::2]
        cal._trace = trace
        dig_samplerate = self._dig.get_sample_rate()
        cal._time = np.linspace(0, len(trace) / dig_samplerate * 1e9,
                                 len(trace), endpoint=False)

        def loss(opr):
            cal.update(*opr)
            tr = cal.apply(trace)
            tr_demod = tr * np.exp(-1j * 2 * np.pi * if_freq * cal._time / 1e9)
            # nfft = fp.next_fast_len(len(trace))
            # xf = fp.fftshift(fp.fftfreq(nfft, d=1 / dig_samplerate))
            # idx_if = np.argmin(np.abs(xf - if_freq))
            # idx_2if = np.argmin(np.abs(xf - 2 * if_freq))
            # yf = np.log10(np.abs(fp.fftshift(fp.fft(np.abs(tr) ** 2, n=nfft))))
            # f1 = np.abs(2 * yf[idx_if] + yf[idx_if - 1] + yf[idx_if + 1])
            # f2 = np.abs(2 * yf[idx_2if] + yf[idx_2if - 1] + yf[idx_2if + 1])
            # return f1 ** 2 + f2 ** 2
            return np.std(tr_demod)**2

        b1 = np.abs(np.mean(np.real(trace)))
        b2 = np.abs(np.mean(np.imag(trace)))
        bounds = [(-2 * b1, 2 * b1), (-2 * b2, 2 * b2),
                  (-np.pi / 4, np.pi / 4), (0.95, 1.1)]
        cal.update(*opt.shgo(loss, bounds).x)
        self._last_downconv_cal = copy.deepcopy(cal)
        # calculate corrected trace and put into result
        cal._trace_cal = cal.apply(trace)
        return cal

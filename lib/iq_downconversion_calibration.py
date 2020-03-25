from scipy import fftpack as fp
import numpy as np
from scipy import optimize as opt
from scipy.interpolate import interp1d
from drivers.Spectrum_m4x import SPCM_TRIGGER

class IQDownconversionCalibrationResult():

    def __init__(self, ifreq, samplerate, offsets=None, phase=0, r=1):
        if offsets is None:
            offsets = np.zeros(2)
        self.offsets = offsets  # an array of two elements with offsets to I and Q traces in mV
        self.phase = phase  # a phase discrepancy in radians
        self.r = r  # after multiplication of an I trace by r, an amplitude discrepancy disappears
        self.ifreq = ifreq  # intermittent frequency
        self.samplerate = samplerate  # sampling rate of a digitizer

    def get_dict(self):
        return self.__dict__


class IQDownconversion():
    @staticmethod
    def measure_and_calibrate(dig, iqawg, amplitude=1):
        dig.trigger_source = SPCM_TRIGGER.EXT0
        ro_cal = iqawg.get_calibration()
        pb = iqawg.get_pulse_builder()

        trace_duration = 500  # ns
        ps = pb.add_sine_pulse(trace_duration, amplitude_mult=amplitude).build()
        iqawg.output_pulse_sequence(ps)

        segment_size = int(trace_duration * 1e-9 * dig.get_sample_rate())
        segment_size -= segment_size % 32
        n_avg = int(1e6)
        ampl = 2500
        data = dig.measure_averaging_mode([1, 2], ampl, 1, segment_size, 32, n_avg) / 128 * ampl / n_avg

        trace = data[0::2] + 1j * data[1::2]
        time = np.linspace(0, len(trace) / dig.get_sample_rate() * 1e9, len(trace), False)
        freq = ro_cal.get_radiation_parameters()["if_frequency"]

        return IQDownconversion.calibrate2(time, trace, freq, dig.get_sample_rate())

    @staticmethod
    def calibrate(time, trace, ifreq, samplerate):
        """
        Calibration of the IQ trace, measured by the digitizer, so that I**2 + Q**2 signal would be constant
        when I and Q are pure sine and cosine.
        Method #1:
        This method first calibrates offsets by minimizing the peak of the
        squared signal at the intermittent frequency, after that it calibrates phase and amplitude discrepancies by
        minimizing the peak at the double intermittent frequency.
        Parameters
        ----------
        time: numpy.ndarray
            time trace is ns, with the length equal to the length of the data trace
        trace: numpy.ndarray, dtype=complex
            complex I + 1j * Q data trace of the measured sinusoidal signal
        ifreq: float
            intermittent frequency value in Hz
        samplerate:
            the sampling rate of the digitizer

        Returns
        -------
        an object of IQDownconversionCalibrationResult class with calibration results
        """
        #     b = signal.firwin(len(trace), 1.5 * ifreq, fs=dig.get_sample_rate())
        #     trace = signal.convolve(trace, b, "same")

        nfft = fp.next_fast_len(len(trace))
        xf = fp.fftshift(fp.fftfreq(nfft, d=1 / samplerate))
        idx_if = np.argmin(np.abs(xf - ifreq))
        idx_2if = np.argmin(np.abs(xf - 2 * ifreq))

        def find_offset(o):
            yf = np.log(np.abs(fp.fftshift(fp.fft(np.abs(trace - o[0] - 1j * o[1]) ** 2, n=nfft))))
            return np.exp(np.abs(2 * yf[idx_if] - yf[idx_if - 1] - np.abs(yf[idx_if + 1])))

        offsets = opt.minimize(find_offset, np.array([0, 0]), method='Nelder-Mead',
                               options={"disp": True}).x

        interp_i = interp1d(time, np.real(trace) - offsets[0], kind="cubic")
        interp_q = interp1d(time, np.imag(trace) - offsets[1], kind="cubic")

        samples_per_period = int(samplerate / ifreq)
        t = time[:-samples_per_period]
        nfft = fp.next_fast_len(len(t))
        xf = fp.fftshift(fp.fftfreq(nfft, d=1 / samplerate))
        idx_2if = np.argmin(np.abs(xf - 2 * ifreq))

        def get_trace(phase, r):
            if phase > 0:
                tr_i = interp_i(t + phase / 2 / np.pi / ifreq * 1e9)
                tr_q = interp_q(t)
            else:
                tr_i = interp_i(t)
                tr_q = interp_q(t - phase / 2 / np.pi / ifreq * 1e9)
            return r * tr_i + 1j * tr_q

        def find_phase_and_amp(pa):
            tr = get_trace(pa[0], pa[1])
            yf = np.log(np.abs(fp.fftshift(fp.fft(np.abs(tr) ** 2, n=nfft))))
            return np.exp(np.abs(2 * yf[idx_2if] - yf[idx_2if - 1] - np.abs(yf[idx_2if + 1])))

        bounds = [(-np.pi / 2, np.pi / 2), (0.95, 1.1)]
        resbrute = opt.brute(find_phase_and_amp, bounds)
        phase = resbrute[0]
        r = resbrute[1]

        return IQDownconversionCalibrationResult(ifreq, samplerate, offsets, phase, r)

    @staticmethod
    def calibrate2(time, trace, ifreq, samplerate):
        """
        Calibration of the IQ trace, measured by the digitizer, so that I**2 + Q**2 signal would be constant
        when I and Q are pure sine and cosine.
        Method #2:
        This method calibrates simultaneously offsets, phase and amplitude
        discrepancies by minimizing the peaks of the squared signal at the intermittent frequency and the double
        intermittent frequency. Uses IQDownconversion.apply_calibration to shift offsets, phase and amplitudes.
        Parameters
        ----------
        time: numpy.ndarray
            time trace is ns, with the length equal to the length of the data trace
        trace: numpy.ndarray, dtype=complex
            complex I + 1j * Q data trace
        ifreq: float
            intermittent frequency value in Hz
        samplerate:
            the sampling rate of the digitizer

        Returns
        -------
        an object of IQDownconversionCalibrationResult class with calibration results
        """

        #     b = signal.firwin(len(trace), 1.5 * ifreq, fs=dig.get_sample_rate())
        #     trace = signal.convolve(trace, b, "same")

        def fun(opr):
            cal = IQDownconversionCalibrationResult(ifreq, samplerate, [opr[0], opr[1]], opr[2], opr[3])
            t, tr = IQDownconversion.apply_calibration(time, trace, cal)
            nfft = fp.next_fast_len(len(t))
            xf = fp.fftshift(fp.fftfreq(nfft, d=1 / samplerate))
            idx_if = np.argmin(np.abs(xf - ifreq))
            idx_2if = np.argmin(np.abs(xf - 2 * ifreq))
            yf = np.log(np.abs(fp.fftshift(fp.fft(np.abs(tr) ** 2, n=nfft))))
            f1 = np.abs(2 * yf[idx_if] - yf[idx_if - 1] - yf[idx_if + 1])
            f2 = np.abs(2 * yf[idx_2if] - yf[idx_2if - 1] - yf[idx_2if + 1])
            return f1 ** 2 + f2 ** 2

        b1 = np.abs(np.mean(np.real(trace)))
        b2 = np.abs(np.mean(np.imag(trace)))
        bounds = [(-2 * b1, 2 * b1), (-2 * b2, 2 * b2), (-np.pi/2, np.pi/2), (0.95, 1.1)]
        o1, o2, phase, r = opt.shgo(fun, bounds).x

        return IQDownconversionCalibrationResult(ifreq, samplerate, np.array([o1, o2]), phase, r)

    @staticmethod
    def apply_calibration(time, trace, calibration):
        """
        Applies calibration results of down-conversion mixer to the measured trace.
        Parameters
        ----------
        time: numpy.ndarray
            time trace is ns, with the length equal to the length of the data trace
        trace: numpy.ndarray, dtype=complex
            complex I + 1j * Q data trace
        calibration: IQDownconversionCalibrationResult
            calibration results
        Returns
        -------
        a tuple with a time trace and a calibrated data trace
        """
        offsets = calibration.offsets
        phase = calibration.phase
        r = calibration.r
        ifreq = calibration.ifreq

        interp_i = interp1d(time, np.real(trace) - offsets[0], kind="cubic")
        interp_q = interp1d(time, np.imag(trace) - offsets[1], kind="cubic")
        samples_per_period = int(calibration.samplerate / ifreq)
        t = time[:-samples_per_period]

        if phase > 0:
            tr_i = interp_i(t + phase / 2 / np.pi / ifreq * 1e9)
            tr_q = interp_q(t)
        else:
            tr_i = interp_i(t)
            tr_q = interp_q(t - phase / 2 / np.pi / ifreq * 1e9)
        return t, r * tr_i + 1j * tr_q

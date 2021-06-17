from math import ceil

import numpy as np
from ipywidgets import widgets
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

import lib.iq_mixer_calibration
from drivers import IQAWG
from lib.data_management import load_IQMX_calibration_database, \
    save_IQMX_calibration
from lib.iq_mixer_calibration import IQCalibrator


class IQVectorGenerator:

    def __init__(self, name, lo, iq_awg: IQAWG, sa, calibration_db_name="IQVG",
                 default_calibration_power=-30, marker_period_divisor=None,
                 slave_iqvgs=None, calibration_step=10e6):
        """

        Parameters
        ----------
        lo
        iq_awg
        sa
        calibration_db_name
        default_calibration_power
        marker_period_divisor: int, ns
            by default, the marker period should be divisible by the if_period
            however, in some cases other divisor may be required, i.e. when
            m3202 is used with PXICLK10 trigger sync mode this divisor
            should be set to 100
        """
        self._name = name
        self._lo = lo
        self._iqawg = iq_awg
        self._sa = sa
        self._cal_db_name = calibration_db_name
        self._default_calibration_power = default_calibration_power
        self._calibration_widget = widgets.HTML()
        self._recalibrate_mixer = False
        self._frequency = 5e9
        self.set_if_frequency(100e6)
        if marker_period_divisor is not None:
            self._marker_period_divisor = marker_period_divisor
        else:
            self._marker_period_divisor = self._if_period
        # for marker period synchronization when iqvgs are on the same AWG
        self._slave_iqvgs = slave_iqvgs if slave_iqvgs is not None else []

        self._power = default_calibration_power
        self._dac_overridden = False
        self._current_cal = None
        self._requested_cal: lib.iq_mixer_calibration.IQCalibrationData = None
        self._cal_db = None

        self._marker_period = None
        self._requested_marker_period = None
        self.set_marker_period(1000)
        self._calibration_initial_guess = {"dc_offsets": np.random.uniform(.03, 0.1, size=2),
                                           "if_amplitudes": (.1, .1),
                                           "if_phase": -np.pi * 0.54}
        self._calibration_step = calibration_step
        self._calibration_test_data = []
        self._load_cal_db()

    def get_calibration_widget(self):
        return self._calibration_widget

    def set_parameters(self, parameters_dict):

        if "power" in parameters_dict:
            self.set_power(parameters_dict["power"])

        if "freq" in parameters_dict:
            self.set_frequency(parameters_dict["freq"])

        if "dac_overridden" in parameters_dict:
            self._dac_overridden = parameters_dict["dac_overridden"]
        else:
            self._dac_overridden = False

    def get_iqawg(self):
        self._iqawg.set_parameters(
            {'calibration': self._current_cal})  # ensure
        return self._iqawg

    def set_if_frequency(self, if_frequency):
        self._if_frequency = if_frequency
        self._if_period = 1 / if_frequency * 1e9  # ns

    def get_if_frequency(self):
        return self._if_frequency

    def set_output_state(self, state):
        self._lo.set_output_state(state)

    def set_frequency(self, freq):
        self._frequency = freq
        self._lo.set_frequency(self._frequency + self._if_frequency)
        self._requested_cal = self.get_calibration(self._frequency,
                                                   self._power)
        self._output_SSB()

    def set_power(self, power):
        if power > self._default_calibration_power + 10:
            raise ValueError("Power can be % dBm max, requested %d dBm" % (
                self._default_calibration_power + 10, power))

        self._power = power
        self._requested_cal = self.get_calibration(self._frequency,
                                                   self._power)
        self._lo.set_power(self._requested_cal.get_lo_power())
        self._output_SSB()

    def get_power(self):
        return self._power

    def set_marker_period(self, marker_period):
        '''
        For some applications there is need to control the length of the interval between triggers
        output by the AWG of the IQVectorGenerator.

        Parameters
        ----------
        marker_period: ns, float
            real trigger period will be recalculated to be not shorter than <marker_period> ns,
            but still divisible by the IF period
        '''
        self._requested_marker_period = marker_period
        correct_marker_period = ceil(
            marker_period / self._marker_period_divisor) * \
                                self._marker_period_divisor

        if correct_marker_period != self._marker_period:
            self._marker_period = correct_marker_period
            if self._requested_cal is not None:
                self._current_cal = None
                self._output_SSB()

        for slave_iqvg in self._slave_iqvgs:
            slave_iqvg.set_marker_period(self._marker_period)

    def _output_SSB(self):
        if self._requested_cal != self._current_cal:
            # print(f"IQVG {self._name}: outputting pulse sequence to update calibration for frequency: {self._frequency/1e9:.4f} GHz"
            #       f", power: {self._power} dBm.")
            self._iqawg.set_parameters({"calibration": self._requested_cal})
            pb = self._iqawg.get_pulse_builder()
            if_freq = self._requested_cal.get_radiation_parameters()[
                "if_frequency"]
            resolution = self._requested_cal.get_radiation_parameters()[
                "waveform_resolution"]
            if_period = 1 / if_freq * 1e9

            if (if_period * 1e9) % resolution != 0:
                print(
                    f"IQVectorGenerator {self._name} warning: IF period is not divisible by "
                    "calibration waveform resolution. Phase coherence will be bad.")

            seq = pb.add_sine_pulse(self._marker_period).build()
            self._iqawg.output_pulse_sequence(seq)

            self._current_cal = self._requested_cal
            # time.sleep(1)

    def _load_cal_db(self):
        self._cal_db = load_IQMX_calibration_database(self._cal_db_name, 0)

    def _around_frequency(self, frequency):
        # return ceil(frequency/self._calibration_step)*self._calibration_step
        return round(frequency / self._calibration_step) * self._calibration_step

    def get_calibration(self, frequency, power):
        frequency = self._around_frequency(frequency)
        # frequency = round(frequency/self._calibration_step)*self._calibration_step

        if self._cal_db is None:
            self._load_cal_db()

        cal = \
            self._cal_db.get(frozenset(dict(lo_power=14,
                                            ssb_power=self._default_calibration_power,
                                            lo_frequency=self._if_frequency + frequency,
                                            if_frequency=self._if_frequency,
                                            waveform_resolution=1,
                                            sideband_to_maintain='left').items()))
        if (cal is None) or self._recalibrate_mixer:
            calibrator = IQCalibrator(self._iqawg, self._sa, self._lo,
                                      self._cal_db_name, 0,
                                      sidebands_to_suppress=6,
                                      output_widget=self._calibration_widget)
            ig = self._calibration_initial_guess
            cal = calibrator.calibrate(
                lo_frequency=frequency + self._if_frequency,
                if_frequency=self._if_frequency,
                lo_power=14,
                ssb_power=self._default_calibration_power,
                waveform_resolution=1,
                iterations=3,
                minimize_iterlimit=100,
                sa_res_bandwidth=300,
                initial_guess=ig)
            save_IQMX_calibration(cal)

            self._load_cal_db()  # make sure to include new calibration into cache
            cal._ssb_power = power
            cal._if_amplitudes = cal._if_amplitudes / np.sqrt(
                10 ** ((self._default_calibration_power - power) / 10))
            # self._calibration_initial_guess["if_amplitudes"] = cal._if_amplitudes
            self._calibration_initial_guess["if_phase"] = cal._if_phase
            return cal
        else:
            cal = cal.copy()
            cal._if_amplitudes = cal._if_amplitudes / np.sqrt(
                10 ** ((self._default_calibration_power - power) / 10))
            return cal

    def calibrate_mixer(self, fstart, fstop, recalibrate=False):
        """
        Performs calibration of the mixer in a frequency range

        Parameters
        ----------
        fstart: float
            start of the frequency range
        fstop : float
            stop of the frequency range
        recalibrate : bool
            Whether or not to calibrate from scratch and override previous
            calibration in this interval.
        """
        fstart = self._around_frequency(fstart)
        fstop = self._around_frequency(fstop)
        self._recalibrate_mixer = recalibrate
        pb = tqdm(np.arange(fstart, fstop + self._calibration_step, self._calibration_step),
                  smoothing=0)
        for frequency in pb:
            pb.set_description("%.3f GHz" % (frequency / 1e9))

            for counter in range(3):
                try:
                    self.set_frequency(frequency)
                    break
                except ValueError:
                    print("Poor calibration at %.3f GHz, retry count "
                          "%d" % (frequency / 1e9, counter))
                    self._calibration_initial_guess["dc_offest"] = \
                        np.random.uniform(.03, 0.1, size=2)
        self._recalibrate_mixer = False

    def test_calibration(self, fstart, fstop, step=1e6,
                         sidebands_to_plot=[-1, 0, 1],
                         remeasure=False):
        """
        Tests the saved calibrations by monitoring all the sidebands throughout
        the specified frequency range
        Parameters
        ----------
        fstart: float, Hz
            start of the frequency range
        fstop: float, Hz
            stop of the frequency range
        step: float, Hz
            step of the scan
        remeasure : bool
            remeasure or just replot the data from the previous run
        """
        sideband_shifts = np.linspace(-3, 3, 7) * self._if_frequency
        freqs = np.arange(fstart, fstop + step, step)

        if remeasure or len(self._calibration_test_data) == 0:
            self._calibration_test_data = []
            for frequency in tqdm(freqs, smoothing=0):
                self.set_frequency(frequency)
                sa_freqs = sideband_shifts + self._frequency
                self._sa.setup_list_sweep(list(sa_freqs), [1000] * 3)
                self._sa.prepare_for_stb()
                self._sa.sweep_single()
                self._sa.wait_for_stb()
                self._calibration_test_data.append(self._sa.get_tracedata())
            self._calibration_test_data = np.array(
                self._calibration_test_data).T

        sidebands_to_plot_idcs = np.array(sidebands_to_plot, dtype=int) + 3
        sideband_shifts = sideband_shifts[sidebands_to_plot_idcs]
        data = self._calibration_test_data[sidebands_to_plot_idcs]
        for row, sideband_shift in zip(data, sideband_shifts):
            plt.plot(freqs, row, label=f"{(sideband_shift / 1e6):.2f} MHz")
        plt.legend()

        self._sa.setup_swept_sa(-self._if_frequency + self._frequency,
                                10 * self._if_frequency,
                                nop=1001, rbw=1e4)
        self._sa.set_continuous()

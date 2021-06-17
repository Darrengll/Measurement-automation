import numpy as np

from .iq_awg import IQAWG, AWGChannel


class IQVectorGenerator:

    def __init__(self, lo, awg, sa, calibration_db_name="IQVG",
                 awg_channels=(1, 2),
                 default_calibration_power=-30):
        """

        Parameters
        ----------
        lo :
        awg :
        sa
        calibration_db_name
        default_calibration_power
        """
        self._lo = lo
        _awg_channels = [AWGChannel(awg, )
        self._iqawg = IQAWG
        self._sa = sa
        self._cal_db_name = calibration_db_name
        self._default_calibration_power = default_calibration_power

        self._recalibrate_mixer = False
        self._frequency = 5e9
        self._if_frequency = 100e6
        self._power = default_calibration_power
        self._dac_overridden = False
        self._current_cal = None
        self._requested_cal = None
        self._cal_db = None
        self._marker_period = 1000


        self._load_cal_db()

    def set_parameters(self, parameters_dict):

        if "power" in parameters_dict:
            self.set_power(parameters_dict["power"])

        if "if_freq" in parameters_dict:
            self.set_frequency(parameters_dict["if_freq"])

        if "dac_overridden" in parameters_dict:
            self._dac_overridden = parameters_dict["dac_overridden"]
        else:
            self._dac_overridden = False

    def get_iqawg(self):
        self._iqawg.set_parameters({'calibration':self._current_cal})  # ensure
        return self._iqawg

    def set_if_frequency(self, if_frequency):
        self._if_frequency = if_frequency

    def get_if_frequency(self):
        return self._if_frequency

    def set_output_state(self, state):
        self._lo.set_output_state(state)

    def set_frequency(self, freq):
        self._frequency = freq
        self._lo.set_frequency(self._frequency + self._if_frequency)
        self._requested_cal = self.get_calibration(self._frequency, self._power)
        self._output_SSB()

    def set_power(self, power):

        if power > self._default_calibration_power + 10:
            raise ValueError("Power can be % dBm max"%(self._default_calibration_power+10))

        self._power = power
        self._requested_cal = self.get_calibration(self._frequency, self._power)
        self._output_SSB()

    def set_marker_repetition_period(self, marker_period):
        '''
        For some applications there is need to control the length of the interval between triggers
        output by the AWG of the IQVectorGenerator.

        Parameters
        ----------
        marker_period: float
            nanoseconds real trigger period will be recalculated to be not
            shorter than <marker_period> ns, but still divisible by the IF
            wave period
        '''
        self._marker_period = marker_period
        if self._requested_cal is not None:
            self._current_cal = None
            self._output_SSB()

    def _output_SSB(self):
        if self._requested_cal != self._current_cal:

            self._iqawg.set_parameters({"calibration": self._requested_cal})
            pb = self._iqawg.get_pulse_builder()
            if_freq = self._requested_cal.get_radiation_parameters()["if_frequency"]
            resolution = self._requested_cal.get_radiation_parameters()["waveform_resolution"]
            if_period = 1 / if_freq * 1e9

            if (if_period * 1e9) % resolution != 0:
                print("IQVectorGenerator warning: IF period is not divisible by "
                      "calibration waveform resolution. Phase coherence will be bad.")

            duration = (self._marker_period // if_period + 1) * if_period
            seq = pb.add_sine_pulse(duration).build()
            self._iqawg.output_pulse_sequence(seq)

            self._current_cal = self._requested_cal

    def _load_cal_db(self):
        self._cal_db = load_IQMX_calibration_database(self._cal_db_name, 0)

    def get_calibration(self, frequency, power):
        frequency = round(round(frequency / 1e9, 2) * 1e9, 2)

        if self._cal_db is None:
            self._load_cal_db()

        cal = \
            self._cal_db.get(frozenset(dict(lo_power=14,
                                  ssb_power=self._default_calibration_power,
                                  lo_frequency=self._if_frequency + frequency,
                                  if_frequency=self._if_frequency,
                                  waveform_resolution=1,
                                  sideband_to_maintain='left').items()))
        if cal is not None and not self._recalibrate_mixer:
            cal = cal.copy()
            cal._if_amplitudes = cal._if_amplitudes / np.sqrt(10**((self._default_calibration_power-power)/10))
            return cal

        calibrator = IQCalibrator(self._iqawg, self._sa, self._lo,
                                  self._cal_db_name, 0, sidebands_to_suppress=6)
        ig = {"dc_offsets": (-0.017, -0.04),
              "if_amplitudes": (.1, .1),
              "if_phase": -np.pi * 0.54}
        cal = calibrator.calibrate(lo_frequency=frequency + self._if_frequency,
                                   if_frequency=self._if_frequency,
                                   lo_power=14,
                                   ssb_power=self._default_calibration_power,
                                   waveform_resolution=1,
                                   iterations=3,
                                   minimize_iterlimit=100,
                                   sa_res_bandwidth=1000,
                                   initial_guess=ig)
        save_IQMX_calibration(cal)

        self._load_cal_db()  # make sure to include new calibration into cache
        cal._ssb_power = power
        cal._if_amplitudes = cal._if_amplitudes / np.sqrt(10 ** ((self._default_calibration_power - power) / 10))
        return cal

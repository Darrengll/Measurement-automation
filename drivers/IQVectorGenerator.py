import numpy

from drivers import IQAWG
from lib.data_management import load_IQMX_calibration_database, save_IQMX_calibration
from lib.iq_mixer_calibration import IQCalibrator


class IQVectorGenerator:

    def __init__(self, lo, iq_awg : IQAWG, sa, calibration_db_name="IQVG"):
        self._lo = lo
        self._iq_awg = iq_awg
        self._sa = sa
        self._cal_db_name = calibration_db_name

        self._frequency = 5e9
        self._power = -20
        self._current_cal = None
        self._requested_cal = self.get_calibration(self._frequency, self._power)
        self._output_SSB()


    def set_frequency(self, freq):
        self._frequency = freq
        self._requested_cal = self.get_calibration(self._frequency, self._power)
        self._output_SSB()

    def set_power(self, power):

        if power > -20:
            raise ValueError("Power can be -20 dBm max")

        self._power = power
        self._requested_cal = self.get_calibration(self._frequency, self._power)
        self._output_SSB()


    def _output_SSB(self):
        if self._requested_cal != self._current_cal:

            self._iq_awg.set_parameters({"calibration": self._requested_cal})
            pb = self._iq_awg.get_pulse_builder()
            if_freq = self._requested_cal.get_radiation_parameters()["if_frequency"]
            resolution = self._requested_cal.get_radiation_parameters()["waveform_resolution"]
            if (1/if_freq) % resolution != 0:
                print("IQVectorGenerator warning: IF period is not divisible by "
                      "calibration waveform resolution. Phase coherence will be bad.")
            seq = pb.add_sine_pulse(10/if_freq*1e9).build()
            self._iq_awg.output_pulse_sequence(seq)

            self._current_cal = self._requested_cal


    def get_calibration(self, frequency, power):
        frequency = round(frequency / 1e9, 2) * 1e9

        if self._current_cal is not None:
            if self._current_cal._ssb_power == power \
                    and self._current_cal._lo_frequency - self._if_frequency == frequency:
                return

        ig = {"dc_offsets": (-0.017, -0.04), \
              "if_amplitudes": (.1, .1), "if_phase": -pi * 0.54}  # initial guess

        db = load_IQMX_calibration_database(self._cal_db_name, 0)
        if db is not None:
            cal = \
                db.get(frozenset(dict(lo_power=14,
                                      ssb_power=self._power,
                                      lo_frequency=self._if_frequency+frequency,
                                      if_frequency=self._if_frequency,
                                      waveform_resolution=1,
                                      sideband_to_maintain = 'left').items()))
            if cal is not None and not self._recalibrate_mixer:
                self._requested_cal = cal
                return

        calibrator = IQCalibrator(self._iqawg, self._sa, self._lo, self._cal_db_name, 0, sidebands_to_suppress=6)
        cal = calibrator.calibrate(lo_frequency=frequency + self._if_frequency,
                                            if_frequency=self._if_frequency,
                                            lo_power=14,
                                            ssb_power=self._power,
                                            waveform_resolution=1,
                                            iterations=3,
                                            minimize_iterlimit=20,
                                            sa_res_bandwidth=500,
                                            initial_guess=ig)
        save_IQMX_calibration(cal)
        return cal

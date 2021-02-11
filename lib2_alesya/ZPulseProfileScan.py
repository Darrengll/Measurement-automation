from lib2.VNATimeResolvedDispersiveMeasurement2D import *


class ZPulseProfileScan(VNATimeResolvedDispersiveMeasurement2D):

    def __init__(self, name, sample_name, **devs_aliases_map):
        super().__init__(name, sample_name, devs_aliases_map)
        self._measurement_result = ZPulseProfileScanResult(name, sample_name)
        self._sequence_generator = \
            IQPulseBuilder.build_z_pulse_profile_scan_sequence


    def set_swept_parameters(self, pi_pulse_delays, excitation_freqs):

        m = None
        if self._fixed_pars["q_awg"][0]["calibration"]._sideband_to_maintain == "left":
            m = 1
        elif self._fixed_pars["q_awg"][0]["calibration"]._sideband_to_maintain == "right":
            m = -1
        if_frequency = self._fixed_pars["q_awg"][0]["calibration"]._if_frequency
        def set_lo_freq(excitation_freq):
            return self._q_lo[0].set_frequency(excitation_freq + m*if_frequency)

        q_if_frequency = self._q_awg[0].get_calibration() \
            .get_radiation_parameters()["if_frequency"]
        swept_pars = {"pi_pulse_delay":
                          (self._set_pi_pulse_delay_and_output,
                           pi_pulse_delays),
                      "excitation_frequency":
                          (set_lo_freq, excitation_freqs)}
        super().set_swept_parameters(**swept_pars)

    def _set_pi_pulse_delay_and_output(self, pi_pulse_delay):
        self._pulse_sequence_parameters["pi_pulse_delay"] = \
            pi_pulse_delay
        super()._output_pulse_sequence()


class ZPulseProfileScanResult(VNATimeResolvedDispersiveMeasurement2DResult):

    def _prepare_data_for_plot(self, data):
        return data["pi_pulse_delay"] / 1e3, \
               data["excitation_frequency"] / 1e9, \
               data["data"].T

    def _annotate_axes(self, axes):
        axes[-1].set_xlabel("$(\pi)$-pulse delay [$\mu$s]")
        axes[-2].set_xlabel("$(\pi)$-pulse delay [$\mu$s]")
        axes[0].set_ylabel("Excitation frequency [GHz]")
        axes[-2].set_ylabel("Excitation frequency [GHz]")

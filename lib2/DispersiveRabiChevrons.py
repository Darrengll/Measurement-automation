from lib2.VNATimeResolvedDispersiveMeasurement2D import *


class DispersiveRabiChevrons(VNATimeResolvedDispersiveMeasurement2D):

    def __init__(self, name, sample_name, **devs_aliases_map):
        super().__init__(name, sample_name, devs_aliases_map)
        self._measurement_result = DispersiveRabiChevronsResult(name, sample_name)
        self._sequence_generator = IQPulseBuilder.build_dispersive_rabi_sequences

    def set_fixed_parameters(self, pulse_sequence_parameters,
                             detect_resonator=True, plot_resonator_fit=True,
                             **dev_params):
        super().set_fixed_parameters(pulse_sequence_parameters,
                                     detect_resonator=detect_resonator,
                                     plot_resonator_fit=plot_resonator_fit,
                                     **dev_params)

    def set_swept_parameters(self, excitation_durations, excitation_freqs):
        q_if_frequency = self._q_awg[0].get_calibration() \
            .get_radiation_parameters()["if_frequency"]
        ssb = self._q_awg[0].get_calibration() \
            .get_radiation_parameters()["sideband_to_maintain"]
        if ssb == "left":
            m = -1
        elif ssb == "right":
            m = 1

        self._measurement_result._if_shift = m * q_if_frequency

        swept_pars = {"excitation_duration": \
                          (self._output_pulse_sequence,
                           excitation_durations),
                      "excitation_frequency":
                          (lambda x: self._exc_iqvg[0].set_frequency(x - self._measurement_result._if_shift),
                           excitation_freqs)}
        super().set_swept_parameters(**swept_pars)

    def _output_pulse_sequence(self, excitation_duration):
        self._pulse_sequence_parameters["excitation_duration"] = \
            excitation_duration
        super()._output_pulse_sequence()


class DispersiveRabiChevronsResult(VNATimeResolvedDispersiveMeasurement2DResult):
    def __init__(self, name, sample_name):
        self._if_shift = None
        super().__init__(name, sample_name)

    def _prepare_data_for_plot(self, data):
        return (data["excitation_frequency"]) / 1e9, \
               data["excitation_duration"] / 1e3, \
               data["data"]

    def _annotate_axes(self, axes):
        axes[0].set_ylabel("Excitation duration [$\mu$s]")
        axes[-2].set_ylabel("Excitation duration [$\mu$s]")
        axes[-1].set_xlabel("Excitation if_freq [GHz]")
        axes[-2].set_xlabel("Excitation if_freq [GHz]")

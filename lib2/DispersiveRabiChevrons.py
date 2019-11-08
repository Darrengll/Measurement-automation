from lib2.VNATimeResolvedDispersiveMeasurement2D import *


class DispersiveRabiChevrons(VNATimeResolvedDispersiveMeasurement2D):

    def __init__(self, name, sample_name, **devs_aliases_map):
        super().__init__(name, sample_name, devs_aliases_map)
        self._measurement_result = DispersiveRabiChevronsResult(name, sample_name)
        self._sequence_generator = IQPulseBuilder.build_dispersive_rabi_sequences

    def set_fixed_parameters(self, pulse_sequence_parameters, **dev_params):
        super().set_fixed_parameters(pulse_sequence_parameters, **dev_params)

    def set_swept_parameters(self, excitation_durations, excitation_freqs):
        q_if_frequency = self._q_awg[0].get_calibration() \
            .get_radiation_parameters()["if_frequency"]
        swept_pars = {"excitation_duration": \
                          (self._output_pulse_sequence,
                           excitation_durations),
                      "excitation_frequency":
                          (lambda x: self._q_lo[0].set_frequency(x + q_if_frequency),
                           excitation_freqs)}
        super().set_swept_parameters(**swept_pars)

    def _output_pulse_sequence(self, excitation_duration):
        self._pulse_sequence_parameters["excitation_duration"] = \
            excitation_duration
        super()._output_pulse_sequence()


class DispersiveRabiChevronsResult(VNATimeResolvedDispersiveMeasurement2DResult):

    def _prepare_data_for_plot(self, data):
        return data["excitation_frequency"] / 1e9, \
               data["excitation_duration"] / 1e3, \
               data["data"]

    def _annotate_axes(self, axes):
        axes[0].set_ylabel("Excitation duration [$\mu$s]")
        axes[-2].set_ylabel("Excitation duration [$\mu$s]")
        axes[-1].set_xlabel("Excitation frequency [GHz]")
        axes[-2].set_xlabel("Excitation frequency [GHz]")

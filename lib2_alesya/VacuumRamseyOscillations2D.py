from lib2.VNATimeResolvedDispersiveMeasurement2D import *


class VacuumRamseyOscillations2D(VNATimeResolvedDispersiveMeasurement2D):

    def __init__(self, name, sample_name, **devs_aliases_map):
        super().__init__(name, sample_name, devs_aliases_map)
        self._measurement_result = VacuumRamseyOscillations2DResult(name,
                                                                    sample_name)
        self._sequence_generator = \
            IQPulseBuilder.build_vacuum_ramsey_oscillations_sequences

    def set_fixed_parameters(self, pulse_sequence_parameters, **dev_params):
        super().set_fixed_parameters(pulse_sequence_parameters, **dev_params)

    def set_swept_parameters(self, z_pulse_offset_voltages, interaction_durations):
        swept_pars = {"z_pulse_offset_voltage":
                          (self._set_z_pulse_offset_voltage,
                           z_pulse_offset_voltages),
                      "interaction_duration":
                          (self._set_interaction_duration_and_output,
                           interaction_durations)}
        super().set_swept_parameters(**swept_pars)

    def _set_z_pulse_offset_voltage(self, z_pulse_offset_voltage):
        self._pulse_sequence_parameters["z_pulse_offset_voltage"] = \
            z_pulse_offset_voltage

    def _set_interaction_duration_and_output(self, interaction_duration):
        self._pulse_sequence_parameters["interaction_duration"] = \
            interaction_duration
        self._output_pulse_sequence()


class VacuumRamseyOscillations2DResult(VNATimeResolvedDispersiveMeasurement2DResult):

    def _prepare_data_for_plot(self, data):
        return data["interaction_duration"] / 1e3, \
               data["z_pulse_offset_voltage"], \
               data["data"]

    def _annotate_axes(self, axes):
        x_label = "Interaction duration [$\mu$s]"
        axes[-1].set_xlabel(x_label)
        axes[-2].set_xlabel(x_label)
        y_label = "Z-pulse offset voltage [V]"
        axes[0].set_ylabel(y_label)
        axes[-2].set_ylabel(y_label)

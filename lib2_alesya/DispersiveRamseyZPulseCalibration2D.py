from matplotlib import pyplot as plt, colorbar
from lib2.VNATimeResolvedDispersiveMeasurement2D import *


class DispersiveRamseyZPulseCalibration2D(VNATimeResolvedDispersiveMeasurement2D):

    def __init__(self, name, sample_name, **devs_aliases_map):
        super().__init__(name, sample_name, devs_aliases_map)
        self._measurement_result = \
            DispersiveRamseyZPulseCalibrationResult2D(name, sample_name)
        self._sequence_generator = IQPulseBuilder.build_z_pulse_ramsey_sequences

    def set_fixed_parameters(self, pulse_sequence_parameters, **dev_params):

        super().set_fixed_parameters(pulse_sequence_parameters, **dev_params)

    def set_swept_parameters(self, z_pulse_durations, z_pulse_offset_voltages):
        q_if_frequency = self._q_awg.get_calibration() \
            .get_radiation_parameters()["if_frequency"]
        swept_pars = {"z_pulse_offset_voltage":
                          (self._set_z_pulse_offset_voltage,
                           z_pulse_offset_voltages),
                      "z_pulse_duration": \
                          (self._set_z_pulse_duration_and_output,
                           z_pulse_durations)}
        super().set_swept_parameters(**swept_pars)

    def _set_z_pulse_offset_voltage(self, z_pulse_offset_voltage):
        self._pulse_sequence_parameters["z_pulse_offset_voltage"] = \
            z_pulse_offset_voltage

    def _set_z_pulse_duration_and_output(self, z_pulse_duration):
        self._pulse_sequence_parameters["z_pulse_duration"] = z_pulse_duration
        super()._output_pulse_sequence()


class DispersiveRamseyZPulseCalibrationResult2D(VNATimeResolvedDispersiveMeasurement2DResult):

    def _prepare_data_for_plot(self, data):
        return data["z_pulse_duration"], \
               data["z_pulse_offset_voltage"], \
               data["data"]

    def _annotate_axes(self, axes):
        axes[0].set_ylabel("z_pulse_offset_voltage [V]")
        axes[-2].set_ylabel("z_pulse_offset_voltage [V]")
        axes[-1].set_xlabel("z_pulse_duration [ns]")
        axes[-2].set_xlabel("z_pulse_duration [ns]")

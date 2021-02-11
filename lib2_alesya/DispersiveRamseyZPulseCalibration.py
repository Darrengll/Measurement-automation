from matplotlib import pyplot as plt, colorbar
from lib2.VNATimeResolvedDispersiveMeasurement1D import *
from lib2.DispersiveRamsey import *


class DispersiveRamseyZPulseCalibration(VNATimeResolvedDispersiveMeasurement1D):

    def __init__(self, name, sample_name, **devs_aliases_map):
        super().__init__(name, sample_name, devs_aliases_map)
        self._measurement_result = \
            DispersiveRamseyResult(name, sample_name)
        self._sequence_generator = IQPulseBuilder.build_z_pulse_ramsey_sequences
        self._swept_parameter_name = "z_pulse_duration"

    def set_swept_parameters(self, z_pulse_durations):
        super().set_swept_parameters(self._swept_parameter_name, z_pulse_durations)


class DispersiveRamseyZPulseCalibrationResult(DispersiveRamseyResult):

    def get_pi_pulse_duration(self):
        return 1 / (self._fit_params[3] / 2 / pi) / 2

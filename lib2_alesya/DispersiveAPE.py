
from lib2.VNATimeResolvedDispersiveMeasurement1D import *

class DispersiveAPE(VNATimeResolvedDispersiveMeasurement1D):

    def __init__(self, name, sample_name, **devs_aliases_map):
        super().__init__(name, sample_name, devs_aliases_map)

        self._measurement_result = DispersiveAPEResult(name, sample_name)
        self._sequence_generator = IQPulseBuilder.build_dispersive_APE_sequences
        self._swept_parameter_name = "ramsey_angle"

    def set_swept_parameters(self, ramsey_angles):
        super().set_swept_parameters(self._swept_parameter_name, ramsey_angles)

class DispersiveAPEResult(VNATimeResolvedDispersiveMeasurement1DResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._x_axis_units = "$\pi$ rad"

    def _theoretical_function(self, ramsey_angle, A, offset, phase_error):
        return A*sin(ramsey_angle+phase_error)+offset

    def _generate_fit_arguments(self, x, data):
        bounds =([0, -10, -pi], [10, 10, pi])
        p0 = [(max(data)-min(data))/2, mean((max(data), min(data))), 0]
        return p0, bounds

    def _prepare_data_for_plot(self, data):
        return data[self._parameter_names[0]], data["data"]

    def _generate_annotation_string(self, opt_params, err):
        return "$\phi_{err} = %.2f\pm%.2f$ deg"%\
            ((opt_params[2]-1/2)*180, err[2]*180)

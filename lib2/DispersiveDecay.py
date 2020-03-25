
from lib2.IQPulseSequence import *
from lib2.VNATimeResolvedDispersiveMeasurement1D import *

class DispersiveDecay(VNATimeResolvedDispersiveMeasurement1D):

    def __init__(self, name, sample_name, **devs_aliases_map):
        super().__init__(name, sample_name, devs_aliases_map)

        self._measurement_result = DispersiveDecayResult(name, sample_name)
        self._sequence_generator = IQPulseBuilder.build_dispersive_decay_sequences
        self._swept_parameter_name = "readout_delay"

    def set_swept_parameters(self, readout_delays):
        super().set_swept_parameters(self._swept_parameter_name, readout_delays)


class DispersiveDecayResult(VNATimeResolvedDispersiveMeasurement1DResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._annotation_v_pos = "top"

    def _model(self, t, A_r, A_i, T_1, offset_r, offset_i):
        return (A_i+A_r*1j)*exp(-1/T_1*t)+offset_r+1j*offset_i

    def get_T_1(self):
        return self._fit_params[2], self._fit_errors[2]

    def _generate_fit_arguments(self, x, data):
        bounds =([-10, -10, 0.1, -10, -10], [10, 10, 100, 10, 10])
        p0 = [ptp(real(data))/2, ptp(imag(data))/2, 1, min(real(data)), min(imag(data))]
        return p0, bounds

    def _generate_annotation_string(self, opt_params, err):
        return "$T_1=%.2f \pm %.2f\mu$s"%(opt_params[2], err[2])

    def get_decay(self):
        return (self._fit_params[2], self._fit_errors[2])

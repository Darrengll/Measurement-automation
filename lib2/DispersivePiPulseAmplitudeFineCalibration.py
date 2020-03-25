from lib2.IQPulseSequence import *
from lib2.VNATimeResolvedDispersiveMeasurement1D import *

class DispersivePiPulseAmplitudeFineCalibration(VNATimeResolvedDispersiveMeasurement1D):
    def __init__(self, name, sample_name, **devs_aliases_map):
        super().__init__(name, sample_name, devs_aliases_map)
        self._measurement_result = \
            DispersivePiPulseAmplitudeFineCalibrationResult(name, sample_name)
        self._sequence_generator = \
            IQPulseBuilder.build_dispersive_pi_half_calibration_sequences
        self._swept_parameter_name = "twice_pi_half_pulses_count"

    def set_fixed_parameters(self, sequence_parameters, **dev_params):
        super().set_fixed_parameters(sequence_parameters, **dev_params)

    def set_swept_parameters(self, calibrated_pulse_pairs_count):
        super().set_swept_parameters(self._swept_parameter_name,
                                     array(calibrated_pulse_pairs_count))


class DispersivePiPulseAmplitudeFineCalibrationResult(VNATimeResolvedDispersiveMeasurement1DResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._x_axis_units = r"calibrated pulse pairs count"

    def get_correct_pi_half_amplitude(self):
        old_amp = self.get_context() \
            .get_pulse_sequence_parameters()["excitation_amplitude"]
        return old_amp / abs(self._fit_params[0]) * (pi / 2)

    def _model(self, twice_pi_half_pulses_count, alpha, T_R, amp_re, amp_im):
        # positive epsilon for under-rotation
        n = twice_pi_half_pulses_count * 2 + 1
        return (amp_re + 1j*amp_im)*(1-cos(n*alpha)*exp(-n/T_R))

    def _generate_fit_arguments(self, x, data):
        bounds = ([0, 0, -10, -10], [pi, 100, 10, 10])
        angle = random.uniform(0, pi, 1)[0]
        p0 = [angle, 10, 0, 0]
        return p0, bounds

    def _prepare_data_for_plot(self, data):
        return data["twice_pi_half_pulses_count"], data["data"]

    def _generate_annotation_string(self, opt_params, err):
        return "$\\varepsilon = %.2f \pm %.2f$ deg\n$T_R$ = %.2f pulses" % (opt_params[0] / pi * 180,
                                                       err[0] / pi * 180, opt_params[1])

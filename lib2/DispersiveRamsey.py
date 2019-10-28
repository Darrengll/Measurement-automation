
from lib2.VNATimeResolvedDispersiveMeasurement1D import *


class DispersiveRamsey(VNATimeResolvedDispersiveMeasurement1D):

    def __init__(self, name, sample_name, plot_update_interval=1,
                 **devs_aliases_map):
        super().__init__(name, sample_name, devs_aliases_map, plot_update_interval)
        self._measurement_result = DispersiveRamseyResult(name, sample_name)
        self._sequence_generator = IQPulseBuilder.build_dispersive_ramsey_sequences
        self._swept_parameter_name = "ramsey_delay"

    def set_swept_parameters(self, ramsey_delays):
        super().set_swept_parameters(self._swept_parameter_name, ramsey_delays)


class DispersiveRamseyResult(VNATimeResolvedDispersiveMeasurement1DResult):

    def _model(self, t, A_r, A_i, T_2_ast, Delta_Omega, offset_r,
               offset_i, phase):
        return (A_r+1j*A_i)*exp(-1/T_2_ast*t)*cos(Delta_Omega*t+phase)\
                                                    +offset_r+1j*offset_i

    def _generate_fit_arguments(self, x, data):
        time_step = x[1]-x[0]
        max_frequency = 1/time_step/2/3
        frequency = random.random(1)*max_frequency
        phase = random.random(1)*2*pi-pi

        bounds =([-10, -10, 0.1, 0*2*pi, -10, -10, -pi],
                        [10, 10, 100, max_frequency*2*pi, 10, 10, pi])
        amp_r, amp_i = ptp(real(data))/2, ptp(imag(data))/2
        p0 = (amp_r, amp_i, 3, frequency, max(real(data))-amp_r,
              max(imag(data))-amp_i, 0)
        return p0, bounds

    def get_basis(self):
        fit = self._fit_params
        A_r, A_i, offset_r, offset_i = fit[0], fit[1], fit[-2], fit[-1]
        ground_state = -A_r+offset_r+1j*(-A_i+offset_i)
        excited_state = A_r+offset_r+1j*(A_i+offset_i)
        return array((ground_state, excited_state))

    def get_ramsey_frequency(self):
        return self._fit_params[3]/2/pi*1e6

    def get_ramsey_decay(self):
        return self._fit_params[2], self._fit_errors[2]

    def _generate_annotation_string(self, opt_params, err):
        return "$T_2^*=%.2f \pm %.2f \mu$s\n$|\Delta\omega/2\pi| = %.3f \pm %.3f$ MHz"%\
            (opt_params[2], err[2], opt_params[3]/2/pi, err[3]/2/pi)

class DispersiveRamseyResultDrift(DispersiveRamseyResult):
    def _model(self, t, A_r, A_i, T_2_ast, Delta_Omega, offset_r, offset_i, phase,
               T_1_ast, exp_offset_r, exp_offset_i):
        value = (A_r+1j*A_i)*exp(-(1/T_2_ast+1/T_1_ast)*t)*cos(Delta_Omega*t+phase) + \
                offset_r+1j*offset_i + (1-np.exp(-1/T_1_ast*t))*(exp_offset_r + 1j*exp_offset_i)
        return value

    def get_ramsey_drift(self):
        return self._fit_params[7], self._fit_errors[7]

    def _generate_annotation_string(self, opt_params, err):
        return super()._generate_annotation_string(opt_params,err) + \
               "\n$Q2 \; T_1^* = {0:.3f} \pm {1:.3f} \mu$s".format(*(self.get_ramsey_drift()))

    def _generate_fit_arguments(self, x, data):
        time_step = x[1]-x[0]
        max_frequency = 1/time_step/2/3
        frequency = random.random(1)*max_frequency
        phase = random.random(1)*2*pi-pi

        bounds =([-10, -10, 0.1, 0*2*pi, -10, -10, -pi, 0.1, -10, -10],
                        [10, 10, 100, max_frequency*2*pi, 10, 10, pi, 100, 10,10])
        amp_r, amp_i = ptp(real(data))/2, ptp(imag(data))/2
        p0 = (amp_r, amp_i, 3, frequency, max(real(data))-amp_r,
              max(imag(data))-amp_i, 0, 3, 0, 0)
        return p0, bounds

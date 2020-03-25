
from lib2.IQPulseSequence import *
from lib2.VNATimeResolvedDispersiveMeasurement1D import *


class DispersiveRabiOscillations(VNATimeResolvedDispersiveMeasurement1D):

    def __init__(self, name, sample_name, plot_update_interval=1,
                 **devs_aliases_map):
        super().__init__(name, sample_name, devs_aliases_map, plot_update_interval)
        self._measurement_result = DispersiveRabiOscillationsResult(name,
                                                                    sample_name)
        self._sequence_generator = IQPulseBuilder.build_dispersive_rabi_sequences
        self._swept_parameter_name = "excitation_duration"

    def set_swept_parameters(self, excitation_durations):
        super().set_swept_parameters(self._swept_parameter_name, excitation_durations)


class DispersiveRabiOscillationsResult(VNATimeResolvedDispersiveMeasurement1DResult):

    def _model(self, t, A_r, A_i, T_R, Omega_R, offset_r, offset_i):
        return -(A_r+1j*A_i)*exp(-1/T_R*t)*cos(Omega_R*t)+offset_r+offset_i*1j

    def _generate_fit_arguments(self, x, data):
        amp_r, amp_i = ptp(real(data))/2, ptp(imag(data))/2
        if abs(max(real(data)) - real(data[0])) < abs(real(data[0])-min(real(data))):
            amp_r = -amp_r
        if abs(max(imag(data)) - imag(data[0])) < abs(imag(data[0])-min(imag(data))):
            amp_i = -amp_i
        offset_r, offset_i = max(real(data))-abs(amp_r), max(imag(data))-abs(amp_i)

        time_step = x[1]-x[0]
        max_frequency = 1/time_step/5
        min_frequency = 0.1
        frequency = random.random(1)*(max_frequency-.1)+.1
        p0 = [amp_r, amp_i, 1, frequency*2*pi, offset_r, offset_i]

        bounds =([-abs(amp_r)*1.5, -abs(amp_i)*1.5, 0.1,
                        min_frequency*2*pi, -10, -10],
                    [abs(amp_r)*1.5, abs(amp_i)*1.5, 100,
                            max_frequency*2*pi, 10, 10])
        return p0, bounds

    def _generate_annotation_string(self, opt_params, err):
        return "$T_R=%.2f \pm %.2f \mu$s\n$\Omega_R/2\pi = %.2f \pm %.2f$ MHz"%\
                (opt_params[2], err[2], opt_params[3]/2/pi, err[3]/2/pi)

    def get_pi_pulse_duration(self):
        return 1/(self._fit_params[3]/2/pi)/2

    def get_T_R(self):
        '''deprecated'''
        return self._fit_params[2], self._fit_errors[2]

    def get_Omega_R(self):
        '''deprecated'''
        return self._fit_params[3], self._fit_errors[3]

    def get_rabi_decay(self):
        return (self._fit_params[2], self._fit_errors[2])

    def get_rabi_frequency(self):
        return (self._fit_params[3], self._fit_errors[3])

    def get_basis(self):
        fit = self._fit_params
        A_r, A_i, offset_r, offset_i = fit[0], fit[1], fit[-2], fit[-1]
        ground_state = -A_r+offset_r+1j*(-A_i+offset_i)
        excited_state = A_r+offset_r+1j*(A_i+offset_i)
        return array((ground_state, excited_state))

    def get_betas(self):
        return [self._fit_params[0] + 1j*self._fit_params[1],  # beta_II
                self._fit_params[4] + 1j * self._fit_params[5]]  # beta_ZI or beta IZ depending on the qubit number in a qubit pair



""" DRIFT class for comparison """
class DispersiveRabiOscillationsResultDrift(DispersiveRabiOscillationsResult):

    def _model(self, t, A_r, A_i, T_R, Omega_R, offset_r, offset_i, phase,
               T_1_ast, exp_offset_r, exp_offset_i):
        value = -(A_r+1j*A_i)*exp(-(1/T_R+1/T_1_ast)*t)*cos(Omega_R*t)+offset_r+offset_i*1j + \
                    (1 - np.exp(-1 / T_1_ast * t)) * (exp_offset_r + 1j * exp_offset_i)
        return value

    def get_rabi_drift(self):
        return self._fit_params[7], self._fit_errors[7]

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

    def _generate_annotation_string(self, opt_params, err):
        return super()._generate_annotation_string(opt_params,err) + \
               "\n$Q2 \; T_1^* = {0:.3f} \pm {1:.3f} \mu$s".format(*(self.get_rabi_drift()))

    def get_betas(self):
        return [self._fit_params[0] + 1j*self._fit_params[1],  # beta_II
                self._fit_params[4] + 1j * self._fit_params[5]]  # beta_ZI or beta IZ depending on the qubit number in a qubit pair

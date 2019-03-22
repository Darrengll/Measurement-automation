from lib2.IQPulseSequence import *
from lib2.VNATimeResolvedDispersiveMeasurement1D import *
from numpy import exp, cos, sin, sqrt, pi, abs

from collections import OrderedDict


class DispersiveRabiOscillationsJoint(VNATimeResolvedDispersiveMeasurement1D):  # TODO Any changes?

    def __init__(self, name, sample_name, plot_update_interval=1,
                 two_qubits=True, easy_fit=False,
                 **devs_aliases_map):
        super().__init__(name, sample_name, devs_aliases_map, plot_update_interval)
        self._sequence_generator = IQPulseBuilder.build_dispersive_rabi_2qubit_sequences
        self._swept_parameter_name = "excitation_duration"
        self._two_qubits = two_qubits
        self._measurement_result = DispersiveRabiOscillationsJointResult(name, sample_name,
                                                                         two_qubits=two_qubits, easy_fit=easy_fit)

    def set_fixed_parameters(self, pulse_sequence_parameters,
                             **dev_params):
        """
        :param dev_params:
            Minimum expected keys and elements expected in each:
                'vna'
                'q_awg': 0,1
                'ro_awg'
        """

        super().set_fixed_parameters(pulse_sequence_parameters,
                                     **dev_params)

    def set_swept_parameters(self, excitation_durations):
        super().set_swept_parameters(self._swept_parameter_name, excitation_durations)


class DispersiveRabiOscillationsJointResult(VNATimeResolvedDispersiveMeasurement1DResult):

    def __init__(self, name, sample_name, two_qubits=True, easy_fit=False):
        self._two_qubits = two_qubits
        self._easy_fit = easy_fit
        super().__init__(name, sample_name)

    @staticmethod
    def rabi_z_av(t, gamma_1, gamma_2, omega_r):
        tau_r = 2 / (gamma_1 + gamma_2)
        omega_r_2 = sqrt(omega_r ** 2 - (1 / tau_r) ** 2)
        return (gamma_1 * gamma_2 + exp(-t / tau_r) * omega_r ** 2 * (
                cos(omega_r_2 * t) + sin(omega_r_2 * t) / (tau_r * omega_r_2))) / (gamma_1 * gamma_2 + omega_r ** 2)

    @staticmethod
    def rabi_x_av(t, gamma_1, gamma_2, omega_r):
        tau_r = 2 / (gamma_1 + gamma_2)
        omega_r_2 = sqrt(omega_r ** 2 - (1 / tau_r) ** 2)
        return (gamma_1 + exp(-t / tau_r) * (gamma_1 * cos(omega_r_2 * t) - sin(omega_r_2 * t) * (
                2 * omega_r ** 2 + gamma_1 * (gamma_2 - gamma_1)) / (2 * omega_r_2))) / (
                       gamma_1 * gamma_2 + omega_r ** 2) * omega_r

    def _model2(self, t,
               b_II_r, b_II_i,
               b_ZI_r, b_ZI_i,
               b_IZ_r, b_IZ_i,
               b_ZZ_r, b_ZZ_i,
               b_XI_r, b_XI_i,
               b_IX_r, b_IX_i,
               b_XX_r, b_XX_i,
               b_XZ_r, b_XZ_i,
               b_ZX_r, b_ZX_i,
               g_1_q1, g_2_q1, omega_r_q1,
               g_1_q2, g_2_q2, omega_r_q2):
        if not self._two_qubits:
            print('Warning: One qubit regime. Fit may suffer.')  # TODO
        value = \
            ((b_II_r + 1j * b_II_i) +
             (b_ZI_r + 1j * b_ZI_i) * self.rabi_z_av(t, g_1_q1, g_2_q1, omega_r_q1) +
             (b_IZ_r + 1j * b_IZ_i) * self.rabi_z_av(t, g_1_q2, g_2_q2, omega_r_q2) +
             (b_ZZ_r + 1j * b_ZZ_i) * self.rabi_z_av(t, g_1_q1, g_2_q1, omega_r_q1) * self.rabi_z_av(t, g_1_q2, g_2_q2,
                                                                                                     omega_r_q2)) \
            if self._easy_fit else \
            ((b_II_r + 1j * b_II_i) +
             (b_XI_r + 1j * b_XI_i) * self.rabi_x_av(t, g_1_q1, g_2_q1, omega_r_q1) +
             (b_ZI_r + 1j * b_ZI_i) * self.rabi_z_av(t, g_1_q1, g_2_q1, omega_r_q1) +
             (b_IX_r + 1j * b_IX_i) * self.rabi_x_av(t, g_1_q2, g_2_q2, omega_r_q2) +
             (b_IZ_r + 1j * b_IZ_i) * self.rabi_z_av(t, g_1_q2, g_2_q2, omega_r_q2) +
             (b_XX_r + 1j * b_XX_i) * self.rabi_x_av(t, g_1_q1, g_2_q1, omega_r_q1) * self.rabi_x_av(t, g_1_q2,
                                                                                                     g_2_q2,
                                                                                                     omega_r_q2) +
             (b_ZX_r + 1j * b_ZX_i) * self.rabi_z_av(t, g_1_q1, g_2_q1, omega_r_q1) * self.rabi_x_av(t, g_1_q2,
                                                                                                     g_2_q2,
                                                                                                     omega_r_q2) +
             (b_XZ_r + 1j * b_XZ_i) * self.rabi_x_av(t, g_1_q1, g_2_q1, omega_r_q1) * self.rabi_z_av(t, g_1_q2,
                                                                                                     g_2_q2,
                                                                                                     omega_r_q2) +
             (b_ZZ_r + 1j * b_ZZ_i) * self.rabi_z_av(t, g_1_q1, g_2_q1, omega_r_q1) * self.rabi_z_av(t, g_1_q2,
                                                                                                     g_2_q2,
                                                                                                     omega_r_q2))
        return value

    def _model(self, t,
               b_II_r, b_II_i,
               b_ZI_r, b_ZI_i,
               b_IZ_r, b_IZ_i,
               b_ZZ_r, b_ZZ_i,
               g_1_q1, g_2_q1, omega_r_q1,
               g_1_q2, g_2_q2, omega_r_q2):
        if not self._two_qubits:
            print('Warning: One qubit regime. Fit may suffer.')  # TODO
        value = \
            ((b_II_r + 1j * b_II_i) +
             (b_ZI_r + 1j * b_ZI_i) * self.rabi_z_av(t, g_1_q1, g_2_q1, omega_r_q1) +
             (b_IZ_r + 1j * b_IZ_i) * self.rabi_z_av(t, g_1_q2, g_2_q2, omega_r_q2) +
             (b_ZZ_r + 1j * b_ZZ_i) * self.rabi_z_av(t, g_1_q1, g_2_q1, omega_r_q1) * self.rabi_z_av(t, g_1_q2, g_2_q2,
                                                                                                     omega_r_q2))
        # TODO bad fitting
        return value

    def _generate_fit_arguments2(self, x, data):  # TODO bounds
        amp_r, amp_i = ptp(real(data)) / 2 + 0.01, ptp(imag(data)) / 2 + 0.01
        time_step = x[1] - x[0]
        max_frequency = 1 / time_step / 2 / 5
        min_frequency = 1e-6
        frequency = random.random() * (max_frequency - min_frequency) + min_frequency
        g_min = min_frequency / 10
        g_max = max_frequency / 10
        g = frequency / 10
        p0 = [amp_r, amp_i,  # b_II
              amp_r, amp_i,  # b_ZI
              amp_r, amp_i,  # b_IZ
              amp_r, amp_i,  # b_ZZ
              0, 0,  # b_XI
              0, 0,  # b_IX
              0, 0,  # b_XX
              0, 0,  # b_XZ
              0, 0,  # b_ZX
              g, g, frequency * 2 * pi,  # Qubit 1, g_1, g_2, omega
              g, g, frequency * 2 * pi]  # Qubit 2, g_1, g_2, omega
        bounds = ([- amp_r * 2, - amp_i * 2,
                   - amp_r * 2, - amp_i * 2,
                   - amp_r * 2, - amp_i * 2,
                   - amp_r * 2, - amp_i * 2,
                   - amp_r * 2, - amp_i * 2,
                   - amp_r * 2, - amp_i * 2,
                   - amp_r * 2, - amp_i * 2,
                   - amp_r * 2, - amp_i * 2,
                   - amp_r * 2, - amp_i * 2,
                   g_min, g_min, min_frequency * 2 * pi,
                   g_min, g_min, min_frequency * 2 * pi],
                  [amp_r * 2, amp_i * 2,
                   amp_r * 2, amp_i * 2,
                   amp_r * 2, amp_i * 2,
                   amp_r * 2, amp_i * 2,
                   amp_r * 2, amp_i * 2,
                   amp_r * 2, amp_i * 2,
                   amp_r * 2, amp_i * 2,
                   amp_r * 2, amp_i * 2,
                   amp_r * 2, amp_i * 2,
                   g_max, g_max, max_frequency * 2 * pi,
                   g_max, g_max, max_frequency * 2 * pi])
        return p0, bounds

    def _generate_fit_arguments(self, x, data):  # TODO bounds
        amp_r, amp_i = ptp(real(data)) / 2 + 0.01, ptp(imag(data)) / 2 + 0.01
        time_step = x[1] - x[0]
        max_frequency = 1 / time_step / 2 / 5
        min_frequency = 1e-6
        frequency = random.random() * (max_frequency - min_frequency) + min_frequency
        g_min = min_frequency / 10
        g_max = max_frequency / 10
        g = frequency / 10
        p0 = [amp_r, amp_i,  # b_II
              amp_r, amp_i,  # b_ZI
              amp_r, amp_i,  # b_IZ
              amp_r, amp_i,  # b_ZZ
              g, g, frequency * 2 * pi,  # Qubit 1, g_1, g_2, omega
              g, g, frequency * 2 * pi]  # Qubit 2, g_1, g_2, omega
        bounds = ([- amp_r * 2, - amp_i * 2,
                   - amp_r * 2, - amp_i * 2,
                   - amp_r * 2, - amp_i * 2,
                   - amp_r * 2, - amp_i * 2,
                   g_min, g_min, min_frequency * 2 * pi,
                   g_min, g_min, min_frequency * 2 * pi],
                  [amp_r * 2, amp_i * 2,
                   amp_r * 2, amp_i * 2,
                   amp_r * 2, amp_i * 2,
                   amp_r * 2, amp_i * 2,
                   g_max, g_max, max_frequency * 2 * pi,
                   g_max, g_max, max_frequency * 2 * pi])
        return p0, bounds

    def _generate_annotation_string(self, opt_params, err):
        # TODO
        return '$T_{q1} = %.2f \pm %.2f \mu$s\n' % (2 / (opt_params[-   6]+opt_params[-5]), err[2]) +\
               '$T_{q2} = %.2f \pm %.2f \mu$s\n' % (2 / (opt_params[-3] + opt_params[-2]), err[2]) +\
               '$\Omega_{q1}/2\pi = %.2f \pm %.2f$ MHz\n' % (opt_params[-4]/2/pi, err[-4]/2/pi) +\
               '$\Omega_{q2}/2\pi = %.2f \pm %.2f$ MHz\n' % (opt_params[-1] / 2 / pi, err[-1] / 2 / pi) + \
               '$\\beta_{II} = %.2f + %.2f i$\n' % (opt_params[0], opt_params[1]) +\
               '$\\beta_{ZI} = %.2f + %.2f i$\n' % (opt_params[2], opt_params[3]) +\
               '$\\beta_{IZ} = %.2f + %.2f i$\n' % (opt_params[4], opt_params[5]) +\
               '$\\beta_{ZZ} = %.2f + %.2f i$\n' % (opt_params[6], opt_params[7])

    def get_betas(self):
        return (self._fit_params[0] + 1j * self._fit_params[1],
                self._fit_params[2] + 1j * self._fit_params[3],
                self._fit_params[4] + 1j * self._fit_params[5],
                self._fit_params[6] + 1j * self._fit_params[7])




"""
CLASS FOR 1 AWG WITH 2 CALIBRATIONS 
"""
from lib2.IQPulseSequence import *
from lib2.VNATimeResolvedDispersiveMeasurement1D import *
from numpy import exp, cos, sin, sqrt, pi


class DispersiveRabiOscillationsJoint2(VNATimeResolvedDispersiveMeasurement1D):  # TODO Any changes?

    def __init__(self, name, sample_name, plot_update_interval=1,
                 two_qubits=True, easy_fit=False,
                 **devs_aliases_map):
        super().__init__(name, sample_name, devs_aliases_map, plot_update_interval)
        self._sequence_generator = IQPulseBuilder.build_dispersive_rabi_2qubit_sequences2
        self._swept_parameter_name = "excitation_duration"
        self._two_qubits = two_qubits
        self._measurement_result = DispersiveRabiOscillationsJointResult2(name, sample_name,
                                                                         two_qubits=two_qubits, easy_fit=easy_fit)

    def set_init_fit_params(self, init_params):
        self._measurement_result._measured_init_params = init_params

    def set_fixed_parameters(self, pulse_sequence_parameters,
                             **dev_params):
        """
        :param dev_params:
            Minimum expected keys and elements expected in each:
                'vna'
                'q_awg': 0,1
                'ro_awg'
        """
        super().set_fixed_parameters(pulse_sequence_parameters,
                                     **dev_params)

    def set_swept_parameters(self, excitation_durations):
        super().set_swept_parameters(self._swept_parameter_name, excitation_durations)


class DispersiveRabiOscillationsJointResult2(VNATimeResolvedDispersiveMeasurement1DResult):

    def __init__(self, name, sample_name, two_qubits=True, easy_fit=False):
        self._two_qubits = two_qubits
        self._easy_fit = easy_fit
        self._measured_init_params = {}
        super().__init__(name, sample_name)

    def _set_init_fit_params(self, init_params):
        self._measured_init_params = init_params

    @staticmethod
    def rabi_z_av(t, gamma_1, gamma_2, omega_r):
        tau_r = 2 / (gamma_1 + gamma_2)
        omega_r_2 = sqrt(omega_r ** 2 - (1 / tau_r) ** 2)
        return (gamma_1 * gamma_2 + exp(-t / tau_r) * omega_r ** 2 * (
                cos(omega_r_2 * t) + sin(omega_r_2 * t) / (tau_r * omega_r_2))) / (gamma_1 * gamma_2 + omega_r ** 2)

    @staticmethod
    def rabi_x_av(t, gamma_1, gamma_2, omega_r):
        tau_r = 2 / (gamma_1 + gamma_2)
        omega_r_2 = sqrt(omega_r ** 2 - (1 / tau_r) ** 2)
        return (gamma_1 + exp(-t / tau_r) * (gamma_1 * cos(omega_r_2 * t) - sin(omega_r_2 * t) * (
                2 * omega_r ** 2 + gamma_1 * (gamma_2 - gamma_1)) / (2 * omega_r_2))) / (
                       gamma_1 * gamma_2 + omega_r ** 2) * omega_r

    def _model2(self, t,
               b_II_r, b_II_i,
               b_ZI_r, b_ZI_i,
               b_IZ_r, b_IZ_i,
               b_ZZ_r, b_ZZ_i,
               b_XI_r, b_XI_i,
               b_IX_r, b_IX_i,
               b_XX_r, b_XX_i,
               b_XZ_r, b_XZ_i,
               b_ZX_r, b_ZX_i,
               g_1_q1, g_2_q1, omega_r_q1,
               g_1_q2, g_2_q2, omega_r_q2):
        if not self._two_qubits:
            print('Warning: One qubit regime. Fit may suffer.')  # TODO
        value = \
            ((b_II_r + 1j * b_II_i) +
             (b_ZI_r + 1j * b_ZI_i) * self.rabi_z_av(t, g_1_q1, g_2_q1, omega_r_q1) +
             (b_IZ_r + 1j * b_IZ_i) * self.rabi_z_av(t, g_1_q2, g_2_q2, omega_r_q2) +
             (b_ZZ_r + 1j * b_ZZ_i) * self.rabi_z_av(t, g_1_q1, g_2_q1, omega_r_q1) * self.rabi_z_av(t, g_1_q2, g_2_q2,
                                                                                                     omega_r_q2)) \
            if self._easy_fit else \
            ((b_II_r + 1j * b_II_i) +
             (b_XI_r + 1j * b_XI_i) * self.rabi_x_av(t, g_1_q1, g_2_q1, omega_r_q1) +
             (b_ZI_r + 1j * b_ZI_i) * self.rabi_z_av(t, g_1_q1, g_2_q1, omega_r_q1) +
             (b_IX_r + 1j * b_IX_i) * self.rabi_x_av(t, g_1_q2, g_2_q2, omega_r_q2) +
             (b_IZ_r + 1j * b_IZ_i) * self.rabi_z_av(t, g_1_q2, g_2_q2, omega_r_q2) +
             (b_XX_r + 1j * b_XX_i) * self.rabi_x_av(t, g_1_q1, g_2_q1, omega_r_q1) * self.rabi_x_av(t, g_1_q2,
                                                                                                     g_2_q2,
                                                                                                     omega_r_q2) +
             (b_ZX_r + 1j * b_ZX_i) * self.rabi_z_av(t, g_1_q1, g_2_q1, omega_r_q1) * self.rabi_x_av(t, g_1_q2,
                                                                                                     g_2_q2,
                                                                                                     omega_r_q2) +
             (b_XZ_r + 1j * b_XZ_i) * self.rabi_x_av(t, g_1_q1, g_2_q1, omega_r_q1) * self.rabi_z_av(t, g_1_q2,
                                                                                                     g_2_q2,
                                                                                                     omega_r_q2) +
             (b_ZZ_r + 1j * b_ZZ_i) * self.rabi_z_av(t, g_1_q1, g_2_q1, omega_r_q1) * self.rabi_z_av(t, g_1_q2,
                                                                                                     g_2_q2,
                                                                                                     omega_r_q2))
        return value

    def _model(self, t,
               b_II_r, b_II_i,
               b_ZI_r, b_ZI_i,
               b_IZ_r, b_IZ_i,
               b_ZZ_r, b_ZZ_i,
               g_1_q1, g_2_q1, omega_r_q1,
               g_1_q2, g_2_q2, omega_r_q2):
        if not self._two_qubits:
            print('Warning: One qubit regime. Fit may suffer.')  # TODO
        value = \
            ((b_II_r + 1j * b_II_i) +
             (b_ZI_r + 1j * b_ZI_i) * self.rabi_z_av(t, g_1_q1, g_2_q1, omega_r_q1) +
             (b_IZ_r + 1j * b_IZ_i) * self.rabi_z_av(t, g_1_q2, g_2_q2, omega_r_q2) +
             (b_ZZ_r + 1j * b_ZZ_i) * self.rabi_z_av(t, g_1_q1, g_2_q1, omega_r_q1) * self.rabi_z_av(t, g_1_q2, g_2_q2,
                                                                                                     omega_r_q2))
        # TODO bad fitting
        return value

    def _generate_fit_arguments2(self, x, data):  # TODO bounds
        amp_r, amp_i = ptp(real(data)) / 2 + 0.01, ptp(imag(data)) / 2 + 0.01
        time_step = x[1] - x[0]
        max_frequency = 1 / time_step / 2 / 5
        min_frequency = 1e-6
        frequency = random.random() * (max_frequency - min_frequency) + min_frequency
        g_min = min_frequency / 10
        g_max = max_frequency / 10
        g = frequency / 10
        p0 = [amp_r, amp_i,  # b_II
              amp_r, amp_i,  # b_ZI
              amp_r, amp_i,  # b_IZ
              amp_r, amp_i,  # b_ZZ
              0, 0,  # b_XI
              0, 0,  # b_IX
              0, 0,  # b_XX
              0, 0,  # b_XZ
              0, 0,  # b_ZX
              g, g, frequency * 2 * pi,  # Qubit 1, g_1, g_2, omega
              g, g, frequency * 2 * pi]  # Qubit 2, g_1, g_2, omega
        bounds = ([- amp_r * 2, - amp_i * 2,
                   - amp_r * 2, - amp_i * 2,
                   - amp_r * 2, - amp_i * 2,
                   - amp_r * 2, - amp_i * 2,
                   - amp_r * 2, - amp_i * 2,
                   - amp_r * 2, - amp_i * 2,
                   - amp_r * 2, - amp_i * 2,
                   - amp_r * 2, - amp_i * 2,
                   - amp_r * 2, - amp_i * 2,
                   g_min, g_min, min_frequency * 2 * pi,
                   g_min, g_min, min_frequency * 2 * pi],
                  [amp_r * 2, amp_i * 2,
                   amp_r * 2, amp_i * 2,
                   amp_r * 2, amp_i * 2,
                   amp_r * 2, amp_i * 2,
                   amp_r * 2, amp_i * 2,
                   amp_r * 2, amp_i * 2,
                   amp_r * 2, amp_i * 2,
                   amp_r * 2, amp_i * 2,
                   amp_r * 2, amp_i * 2,
                   g_max, g_max, max_frequency * 2 * pi,
                   g_max, g_max, max_frequency * 2 * pi])
        return p0, bounds

    def _generate_fit_arguments(self, x, data):  # TODO bounds
        amp_r, amp_i = random.uniform(-1,1)*ptp(real(data)) / 2 + 0.01, random.uniform(-1,1)*ptp(imag(data)) / 2 + 0.01
        time_step = x[1] - x[0]
        max_frequency = 1 / time_step / 2 / 5
        min_frequency = 1e-6
        frequency = random.random() * (max_frequency - min_frequency) + min_frequency
        g_min = min_frequency / 10
        g_max = max_frequency / 10
        g = frequency / 10

        p0_dict = OrderedDict(
            [('b_II_r', amp_r), ( 'b_II_i', amp_i),
            ('b_ZI_r', amp_r), ( 'b_ZI_i', amp_i),
            ('b_IZ_r', amp_r), ( 'b_IZ_i', amp_i),
            ('b_ZZ_r', amp_r), ( 'b_ZZ_i', amp_i),
            ('g_1_q1', g), ('g_2_q1', g), ('omega_r_q1', frequency),
            ('g_1_q2', g), ('g_2_q2', g), ('omega_r_q2', frequency)]
        )

        p0_dict.update(self._measured_init_params)

        p0 = [p0_dict['b_II_r'], p0_dict['b_II_i'],  # b_II
              p0_dict['b_ZI_r'], p0_dict['b_ZI_i'],  # b_ZI
              p0_dict['b_IZ_r'], p0_dict['b_IZ_i'],  # b_IZ
              p0_dict['b_ZZ_r'], p0_dict['b_ZZ_i'],  # b_ZZ
              p0_dict['g_1_q1'], p0_dict['g_2_q1'], p0_dict['omega_r_q1'] * 2 * pi,  # Qubit 1, g_1, g_2, omega
              p0_dict['g_1_q2'], p0_dict['g_2_q2'], p0_dict['omega_r_q2'] * 2 * pi]  # Qubit 2, g_1, g_2, omega

        bounds = ([-10, -10,  # - amp_r * 2, - amp_i * 2,
                   - abs(amp_r * 2), - abs(amp_i * 2),
                   - abs(amp_r * 2), - abs(amp_i * 2),
                   - abs(amp_r * 2), - abs(amp_i * 2),
                   g_min, g_min, min_frequency * 2 * pi,
                   g_min, g_min, min_frequency * 2 * pi],
                  [10,10,  # amp_r * 2, amp_i * 2,
                   abs(amp_r * 2), abs(amp_i * 2),
                   abs(amp_r * 2), abs(amp_i * 2),
                   abs(amp_r * 2), abs(amp_i * 2),
                   g_max, g_max, max_frequency * 2 * pi,
                   g_max, g_max, max_frequency * 2 * pi])

        # restrincting bounds for already measured parameters
        for i, key in enumerate(p0_dict.keys()):
            if key in self._measured_init_params:
                value = p0[i]
                if key.startswith('g'):
                    bounds[0][i] = 0.6 * value
                    bounds[1][i] = 1.4 * value
                else:
                    bounds[0][i] = 0.98*value
                    bounds[1][i] = 1.02*value

        return p0, bounds

    def _generate_annotation_string(self, opt_params, err):
        # TODO
        return '$T_{q1} = %.2f \pm %.2f \mu$s\n' % (2 / (opt_params[-6]+opt_params[-5]), err[2]) +\
               '$T_{q2} = %.2f \pm %.2f \mu$s\n' % (2 / (opt_params[-3] + opt_params[-2]), err[2]) +\
               '$\Omega_{q1}/2\pi = %.2f \pm %.2f$ MHz\n' % (opt_params[-4]/2/pi, err[-4]/2/pi) +\
               '$\Omega_{q2}/2\pi = %.2f \pm %.2f$ MHz\n' % (opt_params[-1] / 2 / pi, err[-1] / 2 / pi) + \
               '$\\beta_{II}-1 = %.4f + %.4f i$\n' % (opt_params[0]-1, opt_params[1]) +\
               '$\\beta_{ZI} = %.4f + %.4f i$\n' % (opt_params[2], opt_params[3]) +\
               '$\\beta_{IZ} = %.4f + %.4f i$\n' % (opt_params[4], opt_params[5]) +\
               '$\\beta_{ZZ} = %.4f + %.4f i$\n' % (opt_params[6], opt_params[7])

    def get_betas(self):
        return (self._fit_params[0] + 1j * self._fit_params[1],
                self._fit_params[2] + 1j * self._fit_params[3],
                self._fit_params[4] + 1j * self._fit_params[5],
                self._fit_params[6] + 1j * self._fit_params[7])



from lib2.IQPulseSequence import *
from lib2.VNATimeResolvedDispersiveMeasurement1D import *


class VacuumRabiOscillations(VNATimeResolvedDispersiveMeasurement1D):

    def __init__(self, name, sample_name,
                 plot_update_interval=1, **devs_aliases_map):
        super().__init__(name, sample_name, devs_aliases_map,
                         plot_update_interval)
        self._measurement_result = VacuumRabiOscillationsResult(name,
                                                                sample_name)
        self._sequence_generator = \
            IQPulseBuilder.build_vacuum_rabi_oscillations_sequences
        self._swept_parameter_name = "interaction_duration"

    def set_swept_parameters(self, interaction_durations):
        super().set_swept_parameters(self._swept_parameter_name, interaction_durations)


class VacuumRabiOscillationsResult(VNATimeResolvedDispersiveMeasurement1DResult):

    def _model(self, t, A_r, A_i, T_R, T_1_ast, Omega_R, phi, offset_r, offset_i,
               exp_offset_r, exp_offset_i):
        return -(A_r + 1j * A_i) * exp(-1 / T_R * t) * (cos(Omega_R * t + phi) + 1) + \
               offset_r + offset_i * 1j + \
               (1 - np.exp(-1 / T_1_ast * t)) * (exp_offset_r + 1j * exp_offset_i)

    def _generate_fit_arguments(self, x, data):
        amp_r, amp_i = ptp(real(data)) / 2, ptp(imag(data)) / 2
        if abs(max(real(data)) - real(data[0])) < abs(real(data[0]) - min(real(data))):
            amp_r = -amp_r
        if abs(max(imag(data)) - imag(data[0])) < abs(imag(data[0]) - min(imag(data))):
            amp_i = -amp_i
        offset_r, offset_i = max(real(data)) - abs(amp_r), max(imag(data)) - abs(amp_i)

        time_step = x[1] - x[0]
        max_frequency = 1 / time_step / 2 / 5
        min_frequency = 0
        frequency = random.random(1) * max_frequency
        p0 = [amp_r, amp_i, 1, 1, frequency * 2 * pi, 0, offset_r,
              offset_i, offset_i, offset_r]

        bounds = ([-abs(amp_r) * 1.5, -abs(amp_i) * 1.5, 0.1, 0.1,
                   min_frequency * 2 * pi, -pi, -10, -10, -10, -10],
                  [abs(amp_r) * 1.5, abs(amp_i) * 1.5, 100, 100,
                   max_frequency * 2 * pi, pi, 10, 10, 10, 10])
        return p0, bounds

    def _generate_annotation_string(self, opt_params, err):
        return "$T_R=%.2f \pm %.2f \mu$s\n$\Omega_R/2\pi = %.2f \pm %.2f$ MHz\n$T_1^*=%.2f \pm %.2f$" % \
               (opt_params[2], err[2],
                opt_params[4], err[4],
                opt_params[3] / 2 / pi, err[3] / 2 / pi,)

    def get_pi_pulse_duration(self):
        return 1 / (self._fit_params[3]) / 2

    def get_basis(self):
        fit = self._fit_params
        A_r, A_i, offset_r, offset_i = fit[0], fit[1], fit[-2], fit[-1]
        ground_state = -A_r + offset_r + 1j * (-A_i + offset_i)
        excited_state = A_r + offset_r + 1j * (A_i + offset_i)
        return array((ground_state, excited_state))

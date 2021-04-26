from matplotlib.gridspec import GridSpec
from numpy import inf
from scipy.optimize import least_squares
from scipy.linalg import inv
from lib2.VNATimeResolvedDispersiveMeasurement import *


class DispersiveAPE(VNATimeResolvedDispersiveMeasurement):

    def __init__(self, name, sample_name, just_ramsey, **devs_aliases_map):
        super().__init__(name, sample_name, devs_aliases_map)

        self._just_ramsey = just_ramsey

        self._measurement_result = DispersiveAPEResult(name,
                                                       sample_name)
        self._sequence_generator = IQPulseBuilder.build_dispersive_APE_sequences

    def set_fixed_parameters(self, pulse_sequence_parameters, just_ramsey=False, **dev_params):

        dev_params['vna'][0]["power"] = dev_params['ro_awg'][0]["calibration"] \
            .get_radiation_parameters()["lo_power"]

        super().set_fixed_parameters(pulse_sequence_parameters, detect_resonator=True,
                                     **dev_params)

    def set_swept_parameters(self, pseudo_I_pulses_counts, ramsey_angles):
        self._pulse_sequence_parameters["max_pseudo_I_pulses_count"] = \
            max(pseudo_I_pulses_counts)
        super().set_swept_parameters(**{"pseudo_I_pulse_count":
                                            (self._set_pseudo_I_pulses_count, pseudo_I_pulses_counts),
                                        "ramsey_angles":
                                            (self._set_ramsey_angle_and_output, ramsey_angles)})

    def _set_pseudo_I_pulses_count(self, pseudo_I_pulses_count):
        if not self._just_ramsey:
            self._pulse_sequence_parameters["pseudo_I_pulses_count"] = \
                pseudo_I_pulses_count
        else:
            self._pulse_sequence_parameters["pseudo_I_pulses_count"] = 0
            self._pulse_sequence_parameters["max_pseudo_I_pulses_count"] = \
                pseudo_I_pulses_count

    def _set_ramsey_angle_and_output(self, ramsey_angle):
        self._pulse_sequence_parameters["ramsey_angle"] = ramsey_angle
        super()._output_pulse_sequence()


class DispersiveAPEResult(VNATimeResolvedDispersiveMeasurementResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._x_axis_units = "$\pi$ rad"
        self._annotation_v_pos = "bottom"
        self._annotation_bbox_props = dict(boxstyle="round", fc="white",
                                           ec="black", lw=1, alpha=0.5)

    def _model(self, ramsey_angle, A_r, A_i, offset_r, offset_i, phase_error):
        return (A_r + 1j * A_i) * cos(phase_error + ramsey_angle) \
               + offset_r + 1j * offset_i

    def _cost_function(self, params, x, data):
        return abs(self._model(x, *params) - data)

    def _generate_fit_arguments(self, x, data):
        amp_r, amp_i = ptp(real(data)) / 2, ptp(imag(data)) / 2
        p0 = (amp_r, amp_i, max(real(data)) - amp_r, max(imag(data)) - amp_i, 0)
        x_scale = absolute(p0[:-1] + (1,))
        bounds = [(-inf, -inf, -inf, -inf, -pi), (inf, inf, inf, inf, pi)]
        return p0, x_scale, bounds

    def _fit_angular_ramsey(self, X, data):
        p0, x_scale, bounds = self._generate_fit_arguments(X, data)
        result = least_squares(self._cost_function, p0, args=(X, data),
                               bounds=bounds, x_scale=absolute(p0[:-1] + (1,)))
        sigma = std(abs(self._model(X, *result.x) - data))
        return result.x, sqrt(diag(sigma ** 2 * inv(result.jac.T.dot(result.jac))))

    def _prepare_data_for_plot(self, data):
        return data[self._parameter_names[0]], \
               data[self._parameter_names[1]], data["data"]

    def _prepare_figure(self):
        fig = plt.figure(figsize=(15, 7))
        gs = GridSpec(2, 2)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[1, 0])
        ax3 = plt.subplot(gs[:, 1])
        axes = (ax1, ax2, ax3)
        plt.tight_layout(pad=3)
        return fig, axes, None

    def _plot(self, axes, caxes):

        data = self.get_data()
        if "data" not in data.keys():
            return

        real_ax, imag_ax, error_plot_ax = axes
        pseudo_I_pulses_counts, ramsey_angles, data = \
            self._prepare_data_for_plot(data)
        for ax in axes:
            ax.reset()
            ax.grid()
        axes[0].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        axes[1].ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

        fit_results = []

        for idx, pseudo_I_pulses_count in enumerate(pseudo_I_pulses_counts):
            Y = data[idx][data[idx] != 0]
            X = ramsey_angles[:len(Y)]
            real_ax.plot(X / pi, real(Y), "C%d" % (idx % 10), ls=":", marker="o",
                         markerfacecolor='none')
            imag_ax.plot(X / pi, imag(Y), "C%d" % (idx % 10), ls=":", marker="o",
                         markerfacecolor='none')
            imag_ax.set_xlabel(r"Ramsey angle [$\pi$ rad]")
            imag_ax.set_ylabel(r"Transmission imaginary part [a.u.]")
            real_ax.set_ylabel(r"Transmission real part [a.u.]")

            if len(X) > 5:
                fit_params, fit_errors = self._fit_angular_ramsey(X, Y)
                fit_results.append((fit_params, fit_errors))
                Y_fit = self._model(ramsey_angles, *fit_params)
                real_ax.plot(ramsey_angles / pi, real(Y_fit), "C%d" % (idx % 10))
                imag_ax.plot(ramsey_angles / pi, imag(Y_fit), "C%d" % (idx % 10))
                for ax in (real_ax, imag_ax):
                    if len(data[idx]) > len(Y):
                        self._annotate_fit_plot(ax, fit_params, fit_errors)

            for ax in axes[:-1]:
                ax.set_xlim(ramsey_angles[0] / pi, ramsey_angles[-1] / pi)

        ready_fits_count = len(fit_results)
        if ready_fits_count > 0:
            fit_results = array(fit_results)
            fit_phase_correction = pi * 0.5 * (1 - sign(fit_results[:, 0, 1]))
            error_plot_ax.errorbar(pseudo_I_pulses_counts[:ready_fits_count],
                                   (fit_phase_correction + fit_results[:, 0, -1]) / pi * 180,
                                   yerr=fit_results[:, 1, -1] / pi * 180, marker='o',
                                   markerfacecolor="none")
        error_plot_ax.set_xlim(pseudo_I_pulses_counts[0], pseudo_I_pulses_counts[-1])
        error_plot_ax.set_xlabel("Pseudo I pulses count")
        error_plot_ax.set_ylabel("Phase error [deg]")
        h_pos = mean(error_plot_ax.get_xlim())
        v_pos = .1 * error_plot_ax.get_ylim()[0] + .9 * error_plot_ax.get_ylim()[1]
        params = self._context.get_pulse_sequence_parameters()
        annotation_string = \
            "Window: %s\n" % params["modulating_window"] + \
            "$\pi/2$ duration: %.2f ns" % params["half_pi_pulse_duration"]
        error_plot_ax.annotate(annotation_string, (h_pos, v_pos),
                               bbox=self._annotation_bbox_props, ha="center")

    def _annotate_fit_plot(self, ax, opt_params, err):
        h_pos = mean(ax.get_xlim())
        v_pos = .9 * ax.get_ylim()[0] + .1 * ax.get_ylim()[1] \
            if self._annotation_v_pos == "bottom" else \
            .1 * ax.get_ylim()[0] + .9 * ax.get_ylim()[1]

        annotation_string = self._generate_annotation_string(opt_params, err)
        ax.annotate(annotation_string, (h_pos, v_pos),
                    bbox=self._annotation_bbox_props, ha="center")

    def _generate_annotation_string(self, opt_params, err):
        return "$\phi_{err} = %.2f\pm%.2f$ deg" % \
               ((opt_params[-1]) / pi * 180, err[-1] * 180)

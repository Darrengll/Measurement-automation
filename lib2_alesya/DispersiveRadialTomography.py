from matplotlib import pyplot as plt, colorbar
from lib2.VNATimeResolvedDispersiveMeasurement2D import *
from scipy.interpolate import interp2d
from lib2.QuantumState import *
import numpy as np
from qutip import qeye, sigmax, sigmay, sigmaz, fidelity, Qobj, expect
from scipy.optimize import least_squares
from IPython.display import clear_output


class DispersiveRadialTomography(VNATimeResolvedDispersiveMeasurement2D):

    def __init__(self, name, sample_name, smoothing_factor=1, **devs_aliases_map):
        super().__init__(name, sample_name, devs_aliases_map)
        self._measurement_result = \
            DispersiveRadialTomographyResult(name, sample_name, smoothing_factor)
        self._sequence_generator = IQPulseBuilder.build_radial_tomography_pulse_sequences

    def set_swept_parameters(self, tomo_phases, tomo_pulse_amplitudes):
        swept_pars = {"tomo_pulse_amplitude":
                          (self._set_exc_ampl,
                           tomo_pulse_amplitudes),
                      "tomo_phase":
                          (self._set_phase_of_drive_and_call_outp_puls_seq,
                           tomo_phases)}
        super().set_swept_parameters(**swept_pars)

    def _set_phase_of_drive_and_call_outp_puls_seq(self, tomo_phase):
        self._pulse_sequence_parameters["tomo_phase"] = tomo_phase
        super()._output_pulse_sequence()

    def _set_exc_ampl(self, tomo_pulse_amplitude):
        self._pulse_sequence_parameters["tomo_pulse_amplitude"] = \
            tomo_pulse_amplitude


class DispersiveRadialTomographyResult(VNATimeResolvedDispersiveMeasurementResult):

    def __init__(self, name, sample_name, smoothing_factor):
        super().__init__(name, sample_name)
        self._pulse_sequence_parameters = self._context \
            .get_pulse_sequence_parameters()
        self._smoothing_factor = smoothing_factor

    def _dm_from_sph_coords(self, r, theta, phi):
        x = r * cos(theta) * cos(phi)
        y = r * cos(theta) * sin(phi)
        z = r * sin(theta)
        return 1 / 2 * (qeye(2) + x * sigmax() + y * sigmay() + z * sigmaz())

    def _model(self, amps, phis, r, theta, phi, A, offset):
        tomo_z_data = []
        rho_0 = self._dm_from_sph_coords(r, theta, phi)
        for amp in amps:
            phi_data = []
            for phi in phis:
                gate = (-1j * amp * pi / 2 * (cos(phi) * sigmax() + sin(phi) * sigmay())).expm()
                rho_1 = gate * rho_0 * gate.dag()
                phi_data.append(expect(sigmaz(), rho_1))
            tomo_z_data.append(phi_data)
        return (array(tomo_z_data) + 1) / 2 * A + offset

    def _cost_function(self, params, amps, phis, data):
        loss = (self._model(amps, phis, *params) - data).ravel()
        # clear_output(wait=True)
        print("\rLoss:", sum(loss ** 2), " params:", params, end="")
        return loss

    def fit_and_plot(self, quadrature):
        converter = imag if quadrature == "imag" else real
        data = self.get_data()
        amplitudes_exp, phases_exp, data_exp = data["tomo_pulse_amplitude"], \
                                               data["tomo_phase"], \
                                               data["data"]
        amplitudes_exp = amplitudes_exp / self \
            ._pulse_sequence_parameters["prep_pulse_pi_amplitude"]

        # data_exp = (converter(data_exp)-converter(data_exp).min())
        # data_exp = data_exp/data_exp.max()

        bounds = [0, -pi / 2, -pi, 0.9, -1], [1, pi / 2, pi, 1.1, 1]
        step_size = max((len(amplitudes_exp), len(phases_exp))) // 10
        prep_pulse_seq = self._pulse_sequence_parameters['prep_pulse']
        expected_state = QuantumState('pulses', prep_pulse_seq)
        expected_state.change_represent('spherical')
        print(expected_state._coords)

        ig = list(expected_state._coords * (np.random.random(1) * (1.1 - 0.9) + 0.9))
        print(ig)
        fit_result = least_squares(self._cost_function, ig + [1, 0],
                                   args=(amplitudes_exp[::step_size], phases_exp[::step_size],
                                         converter(data_exp[::step_size, ::step_size])),
                                   ftol=1e-4, bounds=bounds)
        z = self._model(amplitudes_exp, phases_exp, *fit_result.x)
        expected_state.change_represent('dens_mat')
        fidelya = fidelity(self._dm_from_sph_coords(*fit_result.x[:3]), \
                           Qobj(expected_state._coords))

        x_step = (phases_exp[1] - phases_exp[0])
        X = concatenate((phases_exp - x_step / 2, [phases_exp[-1] + x_step / 2]))
        y_step = (amplitudes_exp[1] - amplitudes_exp[0])
        Y = concatenate((amplitudes_exp, [amplitudes_exp[-1] + y_step]))

        fig, axes = plt.subplots(1, 2, subplot_kw=dict(projection='polar'), figsize=(12, 7))
        plt.tight_layout(pad=4, w_pad=0)
        caxes = []
        Z_data = [z, converter(data_exp)]
        subscripts = ["модель", "эксперимент"]
        for idx, ax in enumerate(axes):
            caxes.append(colorbar.make_axes(ax,
                                            locaion="bottom", orientation="horizontal",
                                            pad=0.1, shrink=0.7, aspect=40)[0])
            ax.text(radians(ax.get_rlabel_position() + 10), \
                    0.13 * abs(ax.get_rmax() - ax.get_rmin()) + ax.get_rmax(), \
                    r"$\Omega$ [$\pi$ рад]",
                    rotation=0, ha='left', va='center')
            ax.set_ylabel(r'$\varphi$ [град]', labelpad=30)
            Z_map = ax.pcolormesh(X, Y, Z_data[idx], cmap="RdBu_r", rasterized=True)
            cb = plt.gcf().colorbar(Z_map, cax=caxes[idx], orientation='horizontal')
            cb.set_label(r"$P_{%s}\left(\left.|e\right>\right)$" % subscripts[idx])
            cb.formatter.set_scientific(False)
            cb.formatter.set_powerlimits((-1, 4))
            cb.update_ticks()
            ax.grid(True)

        plt.gcf().set_size_inches(10, 4)
        # plt.suptitle("Preparation sequence: "+\
        #            str(self._pulse_sequence_parameters["prep_pulse"])+'\n$\mathcal{F}=%.2f\%%$'%(fidelya*100))
        return fit_result, self._dm_from_sph_coords(*fit_result.x[:3]), (fig, axes, caxes)

    def _prepare_figure(self):
        fig, axes = plt.subplots(1, 2, subplot_kw=dict(projection='polar'), figsize=(12, 7))
        plt.tight_layout(pad=4)
        caxes = []
        for ax in axes:
            caxes.append(colorbar.make_axes(ax,
                                            locaion="bottom", orientation="horizontal",
                                            pad=0.1, shrink=0.7, aspect=40)[0])
        return fig, axes, caxes

    def _prepare_data_for_plot(self, data):

        r = data[self._parameter_names[0]]
        theta = data[self._parameter_names[1]]
        s_data = data["data"]

        if self._smoothing_factor > 1:
            f_r = interp2d(theta, r, real(s_data), kind="cubic")
            f_i = interp2d(theta, r, imag(s_data), kind="cubic")
            new_len_r = (len(r) - 1) * self._smoothing_factor + 1
            new_len_theta = (len(theta) - 1) * self._smoothing_factor + 1
            r = linspace(r[0], r[-1], new_len_r)
            theta = linspace(theta[0], theta[-1], new_len_theta)

        theta_step = (theta[1] - theta[0])
        theta = concatenate((theta - theta_step / 2, [theta[-1] + theta_step / 2]))
        r_step = (r[1] - r[0])
        r = concatenate((r, [r[-1] + r_step]))
        r_norm = r / self._pulse_sequence_parameters["prep_pulse_pi_amplitude"]
        if self._smoothing_factor > 1:
            s_data = f_r(theta, r) + 1j * f_i(theta, r)

        return theta, r_norm, s_data

    def _annotate_axes(self, axes):
        """
        Should be implemented in child classes
        """
        pass

    def _plot(self, data):

        axes = self._axes
        caxes = self._caxes
        if "data" not in data.keys():
            return

        r, theta, Z_raw = self._prepare_data_for_plot(data)
        r_2d, theta_2d = meshgrid(r, theta)

        for idx, name in enumerate(['real', 'imag']):
            caxes[idx].clear()
            ax = axes[idx]
            ax.clear()

            Z = self._data_formats[name][0](Z_raw)
            max_Z = max(Z[Z != 0])
            min_Z = min(Z[Z != 0])
            Z_map = ax.pcolormesh(r_2d, theta_2d, Z,
                                  cmap="RdBu_r", vmin=min_Z, vmax=max_Z)
            ax.text(radians(ax.get_rlabel_position() + 10), \
                    0.13 * abs(ax.get_rmax() - ax.get_rmin()) + ax.get_rmax(), \
                    r"Tomography" + "\n" + "pulse [$\pi$ rad]",
                    rotation=0, ha='left', va='center')

            ax.set_ylabel('Tomography phase [deg]', labelpad=30)
            cb = plt.gcf().colorbar(Z_map, cax=caxes[idx], orientation='horizontal')
            cb.set_label(self._data_formats[name][1])
            ax.grid()
            cb.formatter.set_scientific(False)
            cb.formatter.set_powerlimits((-1, 4))
            cb.update_ticks()
        plt.suptitle("Preparation sequence: " + \
                     str(self._pulse_sequence_parameters["prep_pulse"]))
        # self._annotate_axes(axes)
        # plt.tight_layout()

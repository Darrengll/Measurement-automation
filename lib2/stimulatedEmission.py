from scipy import fftpack
import numpy as np
from importlib import reload

from lib2.IQPulseSequence import IQPulseBuilder
import lib2.waveMixing
from lib2.MeasurementResult import MeasurementResult

reload(lib2.waveMixing)
from lib2.waveMixing import PulseMixing


class StimulatedEmission(PulseMixing):

    def __init__(self, name, sample_name, comment, q_lo=None, q_iqawg=None, dig=None):
        super().__init__(name, sample_name, comment, q_lo=q_lo, q_iqawg=q_iqawg, dig=dig)
        self._measurement_result = StimulatedEmissionResult(name, sample_name)
        self._sequence_generator = IQPulseBuilder.build_stimulated_emission_sequence
        self._only_second_pulse_trace = None
        self.data_backup = []

    def sweep_first_pulse_amplitude(self, amplitude_coefficients):
        self._name += "_ampl1"
        swept_pars = {"Pulse 1 amplitude coefficient": (self._set_excitation1_amplitude, amplitude_coefficients)}
        self.set_swept_parameters(**swept_pars)

    def sweep_second_pulse_amplitude(self, amplitude_coefficients):
        self._name += "_ampl2"
        swept_pars = {"Pulse 2 amplitude coefficient": (self._set_excitation_2_amplitude, amplitude_coefficients)}
        self.set_swept_parameters(**swept_pars)

    def _set_excitation_2_amplitude(self, amplitude_coefficient):
        super()._set_excitation2_amplitude(amplitude_coefficient)
        self._output_pulse_sequence()

    def sweep_first_pulse_phase_n_amplitude(self, phases, amplitude1_coefficients):
        self._name += "_phase1_amp_1"
        swept_pars = {"Pulse 1 phase, radians": (self._set_excitation1_phase, phases),
                      "Pulse 1 amplitude coefficient": (self._set_excitation1_amplitude, amplitude1_coefficients)}
        self.set_swept_parameters(**swept_pars)

    def _set_excitation1_phase(self, phase):
        self._pulse_sequence_parameters["phase_shifts"][0] = phase
        # self._output_pulse_sequence()

    def _subtract_second_pulse(self, data_trace):
        if self._only_second_pulse_trace is None:
            self._set_excitation1_amplitude(0)
            self._only_second_pulse_trace = self._measure_one_trace()
        return data_trace - self._only_second_pulse_trace

    def _get_second_pulse_power_density(self):
        self._set_excitation1_amplitude(0)
        self._only_second_pulse_trace = self._measure_one_trace()
        self._only_second_pulse_pd = np.abs(fftpack.fftshift(fftpack.fft(self._only_second_pulse_trace, self._nfft))
                                            / self._nfft) ** 2
        return self._only_second_pulse_pd

    def _recording_iteration(self):
        data = self._measure_one_trace()
        self._measurement_result._iter += 1
        return data

class StimulatedEmissionResult(MeasurementResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._if_freq = None

    def set_parameter_name(self, parameter_name):
        self._parameter_name = parameter_name

    def _prepare_figure(self):
        n_parameters = len(self._parameter_names)
        if n_parameters == 1:
            return self._prepare_figure1D()
        elif n_parameters == 2:
            if self._amps_n_phases_mode:
                return self._prepare_figure2D_amps_n_phases()
            return self._prepare_figure2D_re_n_im()
        else:
            raise NotImplementedError("None or more than 2 swept parameters are set")

    def _prepare_figure1D(self):
        self._last_tr = None
        self._peaks_last_tr = None
        fig = plt.figure(figsize=(12, 8))
        ax_trace = plt.subplot2grid((4, 1), (0, 0), colspan=1, rowspan=1)
        ax_map = plt.subplot2grid((4, 1), (1, 0), colspan=1, rowspan=3)
        plt.tight_layout()
        ax_map.ticklabel_format(axis='x', style='plain', scilimits=(-2, 2))
        ax_map.set_ylabel("Frequency, Hz")
        ax_map.set_xlabel(self._parameter_names[0])
        ax_trace.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_trace.set_xlabel("Frequency, Hz")
        ax_trace.set_ylabel("Fourier[V]")

        ax_map.autoscale_view(True, True, True)
        plt.tight_layout()

        cax, kw = colorbar.make_axes(ax_map, fraction=0.05, anchor=(0.0, 1.0))
        cax.set_title("power, dB")

        return fig, (ax_trace, ax_map), (cax,)

    def _prepare_figure2D(self):
        self._last_tr = None
        self._peaks_last_tr = None
        fig = plt.figure(figsize=(12, 8))
        ax_trace = plt.subplot2grid((4, 1), (0, 0), colspan=1, rowspan=1)
        ax_map = plt.subplot2grid((4, 1), (1, 0), colspan=1, rowspan=3)
        plt.tight_layout()

        ax_trace.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_trace.set_xlabel("Frequency, Hz")
        ax_trace.set_ylabel("power, dB")

        ax_map.ticklabel_format(axis='x', style='plain', scilimits=(-2, 2))
        ax_map.set_ylabel(self._parameter_names[0].upper())
        ax_map.set_xlabel(self._parameter_names[1].upper())
        ax_map.autoscale_view(True, True, True)

        plt.tight_layout()

        cax, kw = colorbar.make_axes(ax_map, fraction=0.05, anchor=(0.0, 1.0))
        cax.set_title("power, dB")

        return fig, (ax_trace, ax_map), (cax,)

    def _prepare_figure2D_amps_n_phases(self):
        self._last_tr = None
        self._peaks_last_tr = None
        fig = plt.figure(figsize=(17, 8))
        ax_trace = plt.subplot2grid((4, 2), (0, 0), colspan=2, rowspan=1)
        ax_map_amps = plt.subplot2grid((4, 2), (1, 0), colspan=1, rowspan=3)
        ax_map_phas = plt.subplot2grid((4, 2), (1, 1), colspan=1, rowspan=3)

        ax_trace.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_trace.set_xlabel("Frequency, Hz")
        ax_trace.set_ylabel("power, dB")

        ax_map_amps.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_map_amps.set_ylabel(self._parameter_names[1].upper())
        ax_map_amps.set_xlabel(self._parameter_names[0].upper())
        ax_map_amps.autoscale_view(True, True, True)
        ax_map_phas.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_map_phas.set_xlabel(self._parameter_names[0].upper())
        ax_map_phas.autoscale_view(True, True, True)
        plt.tight_layout(pad=1, h_pad=2, w_pad=-7)
        cax_amps, kw = colorbar.make_axes(ax_map_amps, aspect=40)
        cax_phas, kw = colorbar.make_axes(ax_map_phas, aspect=40)
        ax_map_amps.set_title("Amplitude, dB", position=(0.5, -0.05))
        ax_map_phas.set_title("Phase, °", position=(0.5, -0.1))
        ax_map_amps.grid(False)
        ax_map_phas.grid(False)
        fig.canvas.set_window_title(self._name)
        return fig, (ax_trace, ax_map_amps, ax_map_phas), (cax_amps, cax_phas)

    def _prepare_figure2D_re_n_im(self):
        self._last_tr = None
        self._peaks_last_tr = None
        fig = plt.figure(figsize=(17, 8))
        ax_trace = plt.subplot2grid((4, 2), (0, 0), colspan=2, rowspan=1)
        ax_map_re = plt.subplot2grid((4, 2), (1, 0), colspan=1, rowspan=3)
        ax_map_im = plt.subplot2grid((4, 2), (1, 1), colspan=1, rowspan=3)

        ax_trace.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_trace.set_xlabel("Frequency, Hz")
        ax_trace.set_ylabel("power, dB")

        ax_map_re.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_map_re.set_ylabel(self._parameter_names[1].upper())
        ax_map_re.set_xlabel(self._parameter_names[0].upper())
        ax_map_re.autoscale_view(True, True, True)
        ax_map_im.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_map_im.set_xlabel(self._parameter_names[0].upper())
        ax_map_im.autoscale_view(True, True, True)
        plt.tight_layout(pad=1, h_pad=2, w_pad=-7)
        cax_re, kw = colorbar.make_axes(ax_map_re, aspect=40)
        cax_im, kw = colorbar.make_axes(ax_map_im, aspect=40)
        ax_map_re.set_title("Real", position=(0.5, -0.05))
        ax_map_im.set_title("Imaginary", position=(0.5, -0.1))
        ax_map_re.grid(False)
        ax_map_im.grid(False)
        fig.canvas.set_window_title(self._name)
        return fig, (ax_trace, ax_map_re, ax_map_im), (cax_re, cax_im)

    def _plot(self, data):
        n_parameters = len(self._parameter_names)
        if n_parameters == 1:
            return self._plot1D(data)
        elif n_parameters == 2:
            if self._amps_n_phases_mode:
                return self._plot2D_amps_n_phases(data)
            return self._plot2D_re_n_im(data)
        else:
            raise NotImplementedError("None or more than 2 swept parameters are set")

    def _plot1D(self, data):
        ax_trace, ax_map = self._axes
        cax = self._caxes[0]
        if "data" not in data.keys():
            return

        XX, YY, Z = self._prepare_data_for_plot1D(data)

        try:
            vmax = np.max(Z[Z != -np.inf])
        except Exception as e:
            print(e)
            print(Z)
            print(Z.shape)
            print(Z[Z != -np.inf])
            print(Z[Z != -np.inf].shape)

        vmin = np.quantile(Z[Z != -np.inf], 0.1)
        extent = [XX[0], XX[-1], YY[0], YY[-1]]
        pow_map = ax_map.imshow(Z.T, origin='lower', cmap="inferno",
                                aspect='auto', vmax=vmax,
                                vmin=vmin, extent=extent)
        cax.cla()
        plt.colorbar(pow_map, cax=cax)
        cax.tick_params(axis='y', right='off', left='on',
                        labelleft='on', labelright='off', labelsize='10')
        last_trace_data = Z[Z != -np.inf][-(len(data["frequency"])):]  # [Z != -np.inf] - flattens the array
        if self._last_tr is not None:
            self._last_tr.remove()
        self._last_tr = ax_trace.plot(data["frequency"], last_trace_data, 'b').pop(0)

        ax_trace.set_ylim([np.min(last_trace_data), np.max(last_trace_data)])

        ax_map.grid(False)
        ax_trace.grid('on')
        ax_trace.axis("tight")

    def _plot2D(self, data):
        ax_trace, ax_map = self._axes
        cax = self._caxes[0]
        if "data" not in data.keys():
            return

        XX, YY, Z, last_trace_y = self._prepare_data_for_plot2D(data)

        vmax = np.max(Z[Z != -np.inf])
        vmin = np.quantile(Z[Z != -np.inf], 0.1)
        step_X = XX[1] - XX[0]
        step_Y = YY[1] - YY[0]
        extent = [XX[0] - 1 / 2 * step_X, XX[-1] + 1 / 2 * step_X,
                  YY[0] - 1 / 2 * step_Y, YY[-1] + 1 / 2 * step_Y]
        pow_map = ax_map.imshow(Z, origin='lower', cmap="inferno",
                                aspect='auto', vmax=vmax,
                                vmin=vmin, extent=extent)
        cax.cla()
        plt.colorbar(pow_map, cax=cax)
        cax.tick_params(axis='y', right='off', left='on',
                        labelleft='on', labelright='off', labelsize='10')

        if self._last_tr is not None:
            self._last_tr.remove()
        self._last_tr = ax_trace.plot(data["frequency"], last_trace_y, 'b').pop(0)

        ax_trace.set_ylim([np.min(last_trace_y), np.max(last_trace_y)])

        ax_map.grid(False)
        ax_trace.grid('on')
        ax_trace.axis("tight")

    def _plot2D_amps_n_phases(self, data):
        ax_trace, ax_map_amps, ax_map_phases = self._axes
        cax_amps = self._caxes[0]
        cax_phases = self._caxes[1]
        if "data" not in data.keys():
            return

        XX, YY, Z_amps, Z_phases, last_trace_y = self._prepare_amps_n_phases_for_plot2D(data)

        amax = np.max(Z_amps[Z_amps != -np.inf])
        amin = np.quantile(Z_amps[Z_amps != -np.inf], 0.1)
        step_X = XX[1] - XX[0]
        step_Y = YY[1] - YY[0]
        extent = [XX[0] - 1 / 2 * step_X, XX[-1] + 1 / 2 * step_X,
                  YY[0] - 1 / 2 * step_Y, YY[-1] + 1 / 2 * step_Y]
        amps_map = ax_map_amps.imshow(Z_amps, origin='lower', cmap="inferno",
                                      aspect='auto', vmax=amax,
                                      vmin=amin, extent=extent)
        cax_amps.cla()
        plt.colorbar(amps_map, cax=cax_amps)
        cax_amps.tick_params(axis='y', right=False, left=True,
                             labelleft=True, labelright=False, labelsize='10')

        phase_map = ax_map_phases.imshow(Z_phases, origin='lower', cmap="twilight_r",
                                         aspect='auto', vmax=180.,
                                         vmin=-180., extent=extent)
        cax_phases.cla()
        plt.colorbar(phase_map, cax=cax_phases)
        cax_phases.tick_params(axis='y', right=False, left=True,
                               labelleft=True, labelright=False, labelsize='10')

        if self._last_tr is not None:
            self._last_tr.remove()
        self._last_tr = ax_trace.plot(data["frequency"], last_trace_y, 'b').pop(0)

        ax_trace.set_ylim([np.min(last_trace_y), np.max(last_trace_y)])

        # ax_map.grid(False)
        ax_trace.grid(True)
        ax_trace.axis("tight")

    def _plot2D_re_n_im(self, data):
        ax_trace, ax_map_re, ax_map_im = self._axes
        cax_re = self._caxes[0]
        cax_im = self._caxes[1]
        if "data" not in data.keys():
            return

        XX, YY, Z_re, Z_im, last_trace_y = self._prepare_re_n_im_for_plot2D(data)

        re_nonempty = Z_re[Z_re != 0]
        im_nonempty = Z_im[Z_im != 0]
        # re_mean = np.mean(re_nonempty)
        # im_mean = np.mean(im_nonempty)
        # re_deviation = np.ptp(re_nonempty)/2
        # im_deviation = np.ptp(im_nonempty)/2
        re_max = max(np.max(re_nonempty), -np.min(re_nonempty))
        im_max = max(np.max(im_nonempty), -np.min(re_nonempty))
        step_X = XX[1] - XX[0]
        step_Y = YY[1] - YY[0]
        extent = [YY[0] - 1 / 2 * step_Y, YY[-1] + 1 / 2 * step_Y,
                  XX[0] - 1 / 2 * step_X, XX[-1] + 1 / 2 * step_X]
        re_map = ax_map_re.imshow(Z_re, origin='lower', cmap="RdBu_r",
                                  aspect='auto', vmax=re_max,
                                  vmin=-re_max, extent=extent)
        cax_re.cla()
        plt.colorbar(re_map, cax=cax_re)
        cax_re.tick_params(axis='y', right=False, left=True,
                           labelleft=True, labelright=False, labelsize='10')

        phase_map = ax_map_im.imshow(Z_im, origin='lower', cmap="RdBu_r",
                                     aspect='auto', vmax=im_max,
                                     vmin=-im_max, extent=extent)
        cax_im.cla()
        plt.colorbar(phase_map, cax=cax_im)
        cax_im.tick_params(axis='y', right=False, left=True,
                           labelleft=True, labelright=False, labelsize='10')

        if self._last_tr is not None:
            self._last_tr.remove()
        self._last_tr = ax_trace.plot(data["frequency"], last_trace_y, 'b').pop(0)

        ax_trace.set_ylim([np.min(last_trace_y), np.max(last_trace_y)])

        # ax_map.grid(False)
        ax_trace.grid(True)
        ax_trace.axis("tight")

    def _prepare_data_for_plot1D(self, data):
        # divide by zero is regularly encountered here
        # due to the fact the during the measurement process
        # data["data"] mostly contain zero values that are repetitively
        # filled with measurement results

        power_data = 20 * np.log10(np.abs(data["data"]) * 1e-3 / np.sqrt(50e-3))

        if self._XX is None and self._YY is None:
            self._XX, self._YY = data[self._parameter_names[0]], data["frequency"]
        return self._XX, self._YY, power_data

    def _prepare_amps_n_phases_for_plot2D(self, data):
        freqs = data["frequency"]
        idx = np.abs(freqs - (self._target_freq_2D)).argmin()

        complex_data = data["data"][:, :, idx].transpose()
        amplitude_data = 20 * np.log10(np.abs(complex_data) * 1e-3 / np.sqrt(50e-3))
        phase_data = np.angle(complex_data) / np.pi * 180
        # last nonzero data of length equal to length of the 'frequency' array
        last_trace_y = data["data"][data["data"] != 0][-len(data["frequency"]):]
        # 1e-3 - convert mV to V
        # sqrt(50e-3) impendance 50 Ohm + convert W to mW
        last_trace_y = 20 * np.log10(np.abs(last_trace_y) * 1e-3 / np.sqrt(50e-3))

        if self._XX is None and self._YY is None:
            self._XX, self._YY = data[self._parameter_names[0]], data[self._parameter_names[1]]
        return self._XX, self._YY, amplitude_data, phase_data, last_trace_y

    def _prepare_re_n_im_for_plot2D(self, data):
        freqs = data["frequency"]
        idx = np.abs(freqs - (self._target_freq_2D)).argmin()

        complex_data = data["data"][:, :, idx]
        idx = np.abs(complex_data).argmax()
        phi = np.angle(complex_data.item(idx))
        phi = np.pi / 2 - phi if phi > 0 else -np.pi / 2 - phi
        re_data = np.real(complex_data * np.exp(1j * 0)).transpose()
        im_data = np.imag(complex_data * np.exp(1j * 0)).transpose()
        # last nonzero data of length equal to length of the 'frequency' array
        last_trace_y = data["data"][data["data"] != 0][-len(data["frequency"]):]
        # 1e-3 - convert mV to V
        # sqrt(50e-3) impendance 50 Ohm + convert W to mW
        last_trace_y = 20 * np.log10(np.abs(last_trace_y) * 1e-3 / np.sqrt(50e-3))

        if self._XX is None and self._YY is None:
            self._XX, self._YY = data[self._parameter_names[1]], data[self._parameter_names[0]]
        return self._XX, self._YY, re_data, im_data, last_trace_y

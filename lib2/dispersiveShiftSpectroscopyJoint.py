from lib2.VNATimeResolvedDispersiveMeasurement1D import *


class DispersiveShiftSpectroscopyJoint(VNATimeResolvedDispersiveMeasurement):

    def __init__(self, name, sample_name, **devs_aliases_map):
        super().__init__(name, sample_name, devs_aliases_map)
        if( len(self._q_awg) == 1 ):  # signle awg for a signle mixer - multiplexing trace sequence generator
            self._sequence_generator = IQPulseBuilder.build_dispersive_shift_joint_sequences_multiplex
        else:  # 2 awg for 2 mixers
            self._sequence_generator = IQPulseBuilder.build_dispersive_shift_joint_sequences

        self._measurement_result = \
            TimeResolvedDispersiveShiftSpectroscopyResult(name, sample_name)

        self._frequencies = None  # list of frequencies readout is measured in
        self._soft_avg = 1

    def set_fixed_parameters(self, pulse_sequence_parameters, **dev_params):
        self._frequencies = np.linspace(*dev_params['vna'][0]["freq_limits"], dev_params["vna"][0]["nop"])

        if ("soft_avg" in dev_params['vna'][0]) and (dev_params['vna'][0]["soft_avg"] > 0):
                self._soft_avg = int(dev_params["vna"][0]["soft_avg"])

        super().set_fixed_parameters(pulse_sequence_parameters, detect_resonator=False,
                                     **dev_params)

    def set_swept_parameters(self, prep_pulses_combinations):
        swept_pars = {"prep_pulses_combination": \
                          (self._output_pulse_sequence,
                           prep_pulses_combinations)}
        super().set_swept_parameters(**swept_pars)

    def _output_pulse_sequence(self, prep_pulses):
        self._pulse_sequence_parameters["prep_pulses"] = prep_pulses
        super()._output_pulse_sequence()

    def _recording_iteration(self):
        vna = self._vna[0]
        q_lo = self._q_lo[0]

        for i in range(self._soft_avg):
            vna.avg_clear();
            vna.prepare_for_stb();
            vna.sweep_single();
            vna.wait_for_stb();

            if i == 0:
                data = vna.get_sdata()
            else:
                data += vna.get_sdata()

            if self._ult_calib:
                q_lo.set_output_state("OFF")
                vna.avg_clear()
                vna.prepare_for_stb()
                vna.sweep_single()
                vna.wait_for_stb()
                bg = vna.get_sdata()
                q_lo.set_output_state("ON")
                if( i == 0 ):
                    data_cal = data - bg
                else:
                    data_cal += data - bg

        if self._ult_calib:
            return data_cal/self._soft_avg
        else:
            return data/self._soft_avg



    def _prepare_measurement_result_data(self, parameter_names, parameters_values):
        measurement_data = \
            super()._prepare_measurement_result_data(parameter_names,
                                                     parameters_values)
        measurement_data["vna_frequency"] = self._frequencies
        return measurement_data


class TimeResolvedDispersiveShiftSpectroscopyResult(VNATimeResolvedDispersiveMeasurementResult):

    def __init__(self, name ,sample_name):
        super().__init__(name, sample_name)

        self._lines = [[None]*4 for i in range(4)]
        self._legend_plotted = False

    def get_basis_betas(self, frequency):
        pass

    def _prepare_data_for_plot(self, data):
        return data["prep_pulses_combination"], \
               data["vna_frequency"] / 1e9, \
               self._remove_delay(data["vna_frequency"], data["data"])

    def _annotate_axes(self, axes):
        axes[0].set_ylabel("VNA frequency [GHz]")
        axes[-2].set_ylabel("VNA frequency [GHz]")

    def _remove_delay(self, frequencies, s_data):
        phases = unwrap(angle(s_data * exp(2 * pi * 1j * 50e-9 * frequencies)))
        k, b = polyfit(frequencies, phases[0], 1)
        phases = phases - k * frequencies - b
        corr_s_data = abs(s_data) * exp(1j * phases)
        corr_s_data[abs(corr_s_data) < 1e-14] = 0
        return corr_s_data

    def _generate_fit_arguments(self):
        """
        Should be implemented in child classes.

        Returns:
        p0: array
            Initial parameters
        scale: tuple
            characteristic scale of the parameters
        bounds: tuple of 2 arrays
            See scipy.optimize.least_squares(...) documentation
        """
        pass

    def _model(self, *params):
        """
        Fit theoretical function. Should be implemented in child classes
        """
        return None

    def _prepare_figure(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 7), sharex=True)
        fig.canvas.set_window_title(self._name)
        axes = ravel(axes)
        return fig, axes, (None, None)

    def _plot(self, data: dict):

        if "data" not in data.keys():
            return

        axes = dict(zip(self._data_formats, self._axes))
        combinations, freqs, data = self._prepare_data_for_plot(data)

        for idx_format, data_format in enumerate(self._data_formats.keys()):
            ax = axes[data_format]
            data_digest = self._data_formats[data_format][0]
            data_name = self._data_formats[data_format][1]

            for prep_idx, prep_pulses in enumerate(combinations):
                if self._lines[idx_format][prep_idx] is None or not self._dynamic:
                    self._lines[idx_format][prep_idx] = ax.plot(freqs,
                                                                data_digest(data[prep_idx]), '.',
                                                                label = str(prep_pulses))[0]

                    ax.set_ylabel(data_name)
                else:
                    self._lines[idx_format][prep_idx].set_xdata(freqs)
                    self._lines[idx_format][prep_idx].set_ydata(data_digest(data[prep_idx]))
                    ax.relim()
                    ax.autoscale_view()


        if self._legend_plotted is False or not self._dynamic:
            self._axes[0].legend()
            self._legend_plotted = True

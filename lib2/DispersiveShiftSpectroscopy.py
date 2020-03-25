from lib2.VNATimeResolvedDispersiveMeasurement2D import *


class TimeResolvedDispersiveShiftSpectroscopy(
    VNATimeResolvedDispersiveMeasurement2D):

    def __init__(self, name, sample_name, **devs_aliases_map):
        devs_aliases_map["q_z_awg"] = None
        super().__init__(name, sample_name, devs_aliases_map)
        self._sequence_generator = IQPulseBuilder.build_dispersive_rabi_sequences
        self._measurement_result = \
            TimeResolvedDispersiveShiftSpectroscopyResult(name, sample_name)

    def set_fixed_parameters(self, pulse_sequence_parameters, **dev_params):
        self._frequencies = linspace(*dev_params['vna'][0]["freq_limits"],
                                     dev_params['vna'][0]["nop"])

        super().set_fixed_parameters(pulse_sequence_parameters, detect_resonator=False,
                                     **dev_params)

    def set_swept_parameters(self, excitation_durations):
        swept_pars = {"excitation_duration": \
                          (self._output_pulse_sequence,
                           excitation_durations)}
        super().set_swept_parameters(**swept_pars)

    def _output_pulse_sequence(self, excitation_duration):
        self._pulse_sequence_parameters["excitation_duration"] = excitation_duration
        super()._output_pulse_sequence()

    def _recording_iteration(self):
        vna = self._vna
        vna.avg_clear();
        vna.prepare_for_stb();
        vna.sweep_single();
        vna.wait_for_stb();
        return vna.get_sdata()

    def _prepare_measurement_result_data(self, parameter_names, parameters_values):
        measurement_data = \
            super()._prepare_measurement_result_data(parameter_names,
                                                     parameters_values)
        measurement_data["vna_frequency"] = self._frequencies
        return measurement_data


class TimeResolvedDispersiveShiftSpectroscopyResult(
    VNATimeResolvedDispersiveMeasurement2DResult):

    def _prepare_data_for_plot(self, data):
        return data["excitation_duration"] / 1e3, \
               data["vna_frequency"] / 1e9, \
               self._remove_delay(data["vna_frequency"], data["data"]).T

    def _annotate_axes(self, axes):
        axes[0].set_ylabel("VNA frequency [GHz]")
        axes[-2].set_ylabel("VNA frequency [GHz]")
        axes[-1].set_xlabel("Excitation duration [$\mu$s]")
        axes[-2].set_xlabel("Excitation duration [$\mu$s]")

    def _remove_delay(self, frequencies, s_data):
        phases = unwrap(angle(s_data * exp(2 * pi * 1j * 50e-9 * frequencies)))
        k, b = polyfit(frequencies, phases[0], 1)
        phases = phases - k * frequencies - b
        corr_s_data = abs(s_data) * exp(1j * phases)
        corr_s_data[abs(corr_s_data) < 1e-14] = 0
        return corr_s_data

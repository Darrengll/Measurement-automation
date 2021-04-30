from matplotlib import pyplot as plt, colorbar
from lib2.VNATimeResolvedDispersiveMeasurement2D import *


class DispersiveRamseyFringes(VNATimeResolvedDispersiveMeasurement2D):

    def __init__(self, name, sample_name, **devs_aliases_map):
        devs_aliases_map["q_z_awg"] = None
        super().__init__(name, sample_name, devs_aliases_map)
        self._measurement_result = \
            DispersiveRamseyFringesResult(name, sample_name)
        self._sequence_generator = IQPulseBuilder.build_dispersive_ramsey_sequences

    def set_fixed_parameters(self, pulse_sequence_parameters, **dev_params):
        super().set_fixed_parameters(pulse_sequence_parameters, **dev_params)

    def set_swept_parameters(self, ramsey_delays, excitation_freqs):
        q_if_frequency = self._q_awg.get_calibration() \
            .get_radiation_parameters()["if_frequency"]
        swept_pars = {"ramsey_delay": \
                          (self._output_pulse_sequence,
                           ramsey_delays),
                      "excitation_frequency":
                          (lambda x: self._q_lo.set_frequency(x + q_if_frequency),
                           excitation_freqs)}
        super().set_swept_parameters(**swept_pars)

    def _output_pulse_sequence(self, ramsey_delay):
        self._pulse_sequence_parameters["ramsey_delay"] = ramsey_delay
        super()._output_pulse_sequence()


class DispersiveRamseyFringesResult(VNATimeResolvedDispersiveMeasurement2DResult):

    def _prepare_data_for_plot(self, data):
        return data["excitation_frequency"] / 1e9, \
               data["ramsey_delay"] / 1e3, \
               data["data"]

    def _annotate_axes(self, axes):
        axes[0].set_ylabel("Ramsey delay [$\mu$s]")
        axes[-2].set_ylabel("Ramsey delay [$\mu$s]")
        axes[-1].set_xlabel("Excitation if_freq [GHz]")
        axes[-2].set_xlabel("Excitation if_freq [GHz]")

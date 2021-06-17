from matplotlib import pyplot as plt, colorbar
from lib2.VNATimeResolvedDispersiveMeasurement2D import *


class ReadoutExcitationShiftCalibration(
    VNATimeResolvedDispersiveMeasurement2D):

    def __init__(self, name, sample_name, **devs_aliases_map):
        super().__init__(name, sample_name, devs_aliases_map)
        self._measurement_result = \
            ReadoutExcitationShiftCalibrationResult(name, sample_name)
        self._sequence_generator = \
            IQPulseBuilder.build_dispersive_rabi_sequences

    def set_fixed_parameters(self, pulse_sequence_parameters, **dev_params):
        super().set_fixed_parameters(pulse_sequence_parameters, **dev_params)

    def set_swept_parameters(self, readout_excitation_shifts,
                             excitation_freqs):
        swept_pars = {"excitation_frequency":
                          (self._exc_iqvg[0].set_frequency,
                           excitation_freqs),
                      "readout_excitation_gap":
                          (self._output_pulse_sequence,
                           readout_excitation_shifts),
                      }
        super().set_swept_parameters(**swept_pars)

    def _output_pulse_sequence(self, readout_excitation_gap):
        self._pulse_sequence_parameters["readout_excitation_gap"] = \
            readout_excitation_gap
        super()._output_pulse_sequence()


class ReadoutExcitationShiftCalibrationResult(
    VNATimeResolvedDispersiveMeasurement2DResult):

    def _prepare_data_for_plot(self, data):
        y = data["excitation_frequency"]
        z = data["data"]
        if y[0] > y[-1]:
            y = y[::-1]
            z = z[::-1, :]

        return data["readout_excitation_gap"], \
               y / 1e9, \
               z

    def _annotate_axes(self, axes):
        axes[0].set_xlabel("Readout-excitation gap [ns]")
        axes[-2].set_xlabel("Readout-excitation gap [ns]")
        axes[-1].set_ylabel("Excitation frequency [GHz]")
        axes[-2].set_ylabel("Excitation frequency [GHz]")

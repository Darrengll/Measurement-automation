from lib2.VNATimeResolvedDispersiveMeasurement1D import *
from lib2.DispersiveRamsey import *
import numpy as np


class Q2cRamsey(VNATimeResolvedDispersiveMeasurement1D):
    """
    TRIGGERING NETWORK MASTER AWG is self._q_awg[0]
    control qubit AWG is self._q_awg[1]
    """
    def __init__(self, name, sample_name, plot_update_interval=1,
                 **devs_aliases_map):
        super().__init__(name, sample_name, devs_aliases_map, plot_update_interval)
        self._measurement_result = Q2cRamseyResult(name, sample_name)
        self._sequence_generator0 = IQPulseBuilder.build_ramsey_comparison_sequences0
        self._sequence_generator1 = IQPulseBuilder.build_ramsey_comparison_sequences1
        self._swept_parameter_name = "ramsey_delay"
        self._sequence_generator = self._sequence_generator_dual
        self._apply_pi_pulse = False

    def _sequence_generator_dual(self, pulse_sequence_parameters, **pbs):
        apply_pi_pulse = self._apply_pi_pulse  # localizing variable
        self._apply_pi_pulse = not self._apply_pi_pulse  # alternating variable for next iteration

        if( apply_pi_pulse is False ):
            return self._sequence_generator0(pulse_sequence_parameters, **pbs)
        elif( apply_pi_pulse is True ):
            return self._sequence_generator1(pulse_sequence_parameters, **pbs)

    def set_swept_parameters(self, ramsey_delays):
        ramsey_delays2 = []
        for i in range(len(ramsey_delays)):
            ramsey_delays2.append(ramsey_delays[i] + 1e-5)
        ramsey_delays = np.array(list(ramsey_delays)+list(ramsey_delays2), dtype=np.float64)
        ramsey_delays.sort()
        super().set_swept_parameters(self._swept_parameter_name, ramsey_delays)


class Q2cRamseyResult(VNATimeResolvedDispersiveMeasurement1DResult):
    def __init__(self, name, sample_name):
        self._DR_zero_result = DispersiveRamseyResult(name, sample_name)
        self._DR_one_result = DispersiveRamseyResultDrift(name, sample_name)
        self._DR_zero_result._parameter_names = ["ramsey_delay"]
        self._DR_one_result._parameter_names = ["ramsey_delay"]
        setattr(self._DR_zero_result, "_dynamic", False)
        setattr(self._DR_one_result, "_dynamic", False)
        super().__init__(name, sample_name)

    def _prepare_figure(self):
        fig, axes = plt.subplots(4, 1, figsize=(15, 7), sharex=True)
        fig.canvas.set_window_title(self._name)
        axes = ravel(axes)

        self._DR_one_result._figure = fig
        self._DR_one_result._axes = axes[0:2]
        self._DR_zero_result._figure = fig
        self._DR_zero_result._axes = axes[2:4]
        return fig, axes, (None, None)

    def _split_data(self, data):
        data1 = {}
        data2 = {}
        for key in data.keys():
            data1[key] = data[key][0::2]
            data2[key] = data[key][1::2]
        return data1, data2

    def _plot(self, data):
        data_zero, data_one = self._split_data(data)
        self._DR_one_result._data = data_one
        self._DR_one_result._plot(data_one)
        self._DR_zero_result._data = data_zero
        self._DR_zero_result._plot(data_zero)
        self._axes[0].set_title("with pi pulse to Q2")
        self._axes[2].set_title("usual ramsey")



class Q2cRamseyMultiplexed(VNATimeResolvedDispersiveMeasurement1D):

    def __init__(self, name, sample_name, plot_update_interval=1,
                 **devs_aliases_map):
        super().__init__(name, sample_name, devs_aliases_map, plot_update_interval)
        self._measurement_result = Q2cRamseyResultMultiplexed(name, sample_name)
        self._sequence_generator0 = IQPulseBuilder.build_ramsey_comparison_sequences0
        self._sequence_generator1 = IQPulseBuilder.build_ramsey_comparison_sequences1_multiplexed
        self._swept_parameter_name = "ramsey_delay"
        self._sequence_generator = self._sequence_generator_dual
        self._apply_pi_pulse = False

    def _sequence_generator_dual(self, pulse_sequence_parameters, **pbs):
        apply_pi_pulse = self._apply_pi_pulse
        self._apply_pi_pulse = not self._apply_pi_pulse
        if( apply_pi_pulse is False ):
            return self._sequence_generator0(pulse_sequence_parameters, **pbs)
        elif( apply_pi_pulse is True ):
            return self._sequence_generator1(pulse_sequence_parameters, **pbs)

    def set_swept_parameters(self, ramsey_delays):
        ramsey_delays2 = []
        for i in range(len(ramsey_delays)):
            ramsey_delays2.append(ramsey_delays[i] + 1e-5)
        ramsey_delays = np.array(list(ramsey_delays)+list(ramsey_delays2), dtype=np.float64)
        ramsey_delays.sort()
        super().set_swept_parameters(self._swept_parameter_name, ramsey_delays)


class Q2cRamseyResultMultiplexed(VNATimeResolvedDispersiveMeasurement1DResult):
    def __init__(self, name, sample_name):
        self._DR_zero_result = DispersiveRamseyResult(name, sample_name)
        self._DR_one_result = DispersiveRamseyResultDrift(name, sample_name)
        self._DR_zero_result._parameter_names = ["ramsey_delay"]
        self._DR_one_result._parameter_names = ["ramsey_delay"]
        setattr(self._DR_zero_result, "_dynamic", False)
        setattr(self._DR_one_result, "_dynamic", False)
        super().__init__(name, sample_name)

    def _prepare_figure(self):
        fig, axes = plt.subplots(4, 1, figsize=(15, 7), sharex=True)
        fig.canvas.set_window_title(self._name)
        axes = ravel(axes)

        self._DR_one_result._figure = fig
        self._DR_one_result._axes = axes[0:2]
        self._DR_zero_result._figure = fig
        self._DR_zero_result._axes = axes[2:4]
        return fig, axes, (None, None)

    def _split_data(self, data):
        data1 = {}
        data2 = {}
        for key in data.keys():
            data1[key] = data[key][0::2]
            data2[key] = data[key][1::2]
        return data1, data2

    def _plot(self, data):
        data_zero, data_one = self._split_data(data)
        self._DR_one_result._data = data_one
        self._DR_one_result._plot(data_one)
        self._DR_zero_result._data = data_zero
        self._DR_zero_result._plot(data_zero)
        self._axes[0].set_title("with pi pulse to Q2")
        self._axes[2].set_title("usual ramsey")

from time import sleep

from numpy import *

from lib2.SingleToneSpectroscopy import SingleToneSpectroscopy
from lib2.fulaut.AnticrossingOracle import AnticrossingOracle

from tqdm import tqdm_notebook

roman_nums = "I II III IV V VI".split(" ")

class ChainInductanceMatrixSpectroscopy:

    def __init__(self, name, sample_name, chain_length,
                 flux_parameter_ranges,
                 flux_parameter_nop,
                 flux_parameter_controllers,
                 resonator_areas, vna):

        self._name = name
        self._sample_name = sample_name
        self._vna = vna
        self._chain_length = chain_length
        self._flux_parameter_ranges = flux_parameter_ranges
        self._flux_parameter_nop = flux_parameter_nop
        self._resonator_areas = resonator_areas
        self._flux_parameter_controllers = flux_parameter_controllers
        self._sts_results = [[[] for _ in range(self._chain_length)]
                                            for _ in range(self._chain_length)]

        self._zero_flux_parameters = [0]*self._chain_length

    def launch(self, transmons = None, biases = None):
        if transmons is None:
            transmons = range(self._chain_length)
        if biases is None:
            biases = range(self._chain_length)

        for primary_transmon_id in tqdm_notebook(transmons, smoothing=0):

            voltages = linspace(*self._flux_parameter_ranges[primary_transmon_id], 101)

            for secondary_transmon_id in tqdm_notebook(biases, smoothing=0):
                self._set_flux_parameters(self._zero_flux_parameters)

                if secondary_transmon_id == primary_transmon_id:
                    continue

                self._sts_results[primary_transmon_id][secondary_transmon_id] = []

                for shift_value in tqdm_notebook(linspace(*self._flux_parameter_ranges[secondary_transmon_id],
                                               self._flux_parameter_nop), smoothing=0):
                    # shift_value = self._flux_parameter_ranges[secondary_transmon_id][shift_value_id]
                    self._flux_parameter_controllers[secondary_transmon_id].set(shift_value)

                    name = "%s-anticrossing-shift-%s@%.2f" % (roman_nums[primary_transmon_id],
                                                              roman_nums[secondary_transmon_id],
                                                              shift_value)
                    STS = SingleToneSpectroscopy(name, self._sample_name,
                                                 plot_update_interval=1, vna=[self._vna])
                    vna_parameters = {"bandwidth": 1000,
                                      "freq_limits": self._resonator_areas[primary_transmon_id],
                                      "nop": 201, "power": -50, "averages": 1}
                    STS.set_fixed_parameters(vna=[vna_parameters])
                    # STS._src[0].set_current(currents[0])
                    sleep(1)
                    STS.set_swept_parameters({'Voltage [V]':
                                              (self._flux_parameter_controllers[primary_transmon_id].set,
                                               voltages)})

                    sts_result = STS.launch()
                    self._sts_results[primary_transmon_id][secondary_transmon_id].append(sts_result)
                    #     sts_result.visualize();
                    self._flux_parameter_controllers[primary_transmon_id].set(0)

    def extract_inductance_matrix(self):
        inductance_matrix = [[0 for _ in range(self._chain_length)] for _ in range(self._chain_length)]

        for primary_transmon_id in range(self._chain_length):
            inductance_matrix[primary_transmon_id][primary_transmon_id] = 1
            for secondary_transmon_id in range(self._chain_length):
                if secondary_transmon_id == primary_transmon_id:
                    continue
                swss = []
                for shift_value_id in [0, 1]:
                    sts_result = self._sts_results[primary_transmon_id][secondary_transmon_id][shift_value_id]

                    ao = AnticrossingOracle("transmon", sts_result, False, True, ["fqmax_below"])
                    res_freq, g, period, sws, max_q_freq, d = ao.launch()[0]
                    swss.append(sws)
                matrix_element = ptp(swss)/ptp(self._flux_parameter_ranges[primary_transmon_id])
                inductance_matrix[primary_transmon_id][secondary_transmon_id] = matrix_element

        return inductance_matrix

    def _set_flux_parameters(self, parameters_values):
        for controller, parameter_value in zip(self._flux_parameter_controllers, parameters_values):
            controller.set(parameter_value)

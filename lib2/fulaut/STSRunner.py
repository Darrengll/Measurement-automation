from drivers.BiasType import BiasType
from lib2.SingleToneSpectroscopy import *
from lib2.fulaut.AnticrossingOracle import *
from lib2.ExperimentParameters import STSRunnerParameters
from loggingserver import LoggingServer
from datetime import datetime


class STSRunner():

    def __init__(self, sample_name, qubit_name, res_freq, res_limits, vna=None, bias_src=None, awgs=None):

        self._sample_name = sample_name
        self._qubit_name = qubit_name
        self._res_freq = res_freq
        self._sts_name = "%s-sts" % qubit_name
        self._scan_area = res_limits
        self._scan_area_width = 1e6 + ptp(res_limits)
        self._scan_area_width_previous = self._scan_area_width
        self._vna = vna
        self._bias_src = bias_src

        self._vna_parameters = STSRunnerParameters().vna_parameters
        self._vna_parameters["power"] = GlobalParameters().readout_power

        self._bias_type = self._bias_src[0].get_bias_type()
        self._bias_parameter_name = BiasType.NAMES[self._bias_type]
        if self._bias_type is BiasType.VOLTAGE:
            self._bias_values = linspace(-1, 1, 101)
        else:
            self._bias_values = linspace(-7e-3, 7e-3, 101)
        self._sts_result = None
        self._launch_datetime = datetime.today()
        self._bias_src[0].set_appropriate_range(max(abs(self._bias_values)))

        self._logger = LoggingServer.getInstance('fulaut')

    def run(self):

        # Check for any prior STS result

        known_results = \
            MeasurementResult.load(self._sample_name,
                                   self._sts_name,
                                   return_all=True)

        if known_results is not None and not STSRunnerParameters().rerun:
            self._sts_result = known_results[-1]
            if hasattr(self._sts_result, "_fit_result"):
                freqs = self._sts_result.get_data()["Frequency [Hz]"]
                self._scan_area_width = ptp(freqs) + 5e6
                self._res_freq = (max(freqs)+min(freqs))/2
                return known_results[-1]._fit_result
        else:
            self._iterate_STS()

        ao = AnticrossingOracle("transmon", self._sts_result,
                                plot=True,
                                fast_res_detect=False,
                                hints = STSRunnerParameters().anticrossing_oracle_hints)
        res_points = ao.get_res_points()
        params, loss = ao.launch()

        self._logger.debug("Error: " + str(loss) + \
                           ", ptp: " + str(ptp(res_points[:, 1]) / 1e6))
        if loss < 0.2 * ptp(res_points[:, 1]) / 1e6:
            self._logger.debug("Success! " + str(params) + " " + str(loss))
            self._sts_result._fit_result = (params, loss)
            print("Saving...", end="")
            self._sts_result.save()
            print("\n")

            return params, loss
        else:
            self._logger.warn("STS fit was unsuccessful")
            self._sts_result._name += "_fit-fail"
            self._sts_result.save()
            raise ValueError("Fit was unsuccessful")

    def _iterate_STS(self):

        counter = 0

        while counter < 10:

            # if counter == 0:
            #     # self._perform_STS_first()
            #     self._perform_STS()
            # else:
            #     self._perform_STS()

            self._perform_STS()

            ao = AnticrossingOracle("transmon", self._sts_result,
                                    plot=True,
                                    fast_res_detect=False,
                                    hints = STSRunnerParameters().anticrossing_oracle_hints)
            res_points = ao.get_res_points()

            self._logger.debug("Scan: " + str(self._scan_area_width / 1e6))
            self._logger.debug("Ptp: " + str(ptp(res_points[:, 1]) / 1e6))
            if 0.1e6 < ptp(res_points[:, 1]) <= 10e6:
                self._scan_area_width = ptp(res_points[:, 1]) * 1.5 + 2e6
                self._res_freq = (max(res_points[:, 1]) + min(res_points[:, 1]))/2
                self._logger.debug("Flux dependence found. Zooming to scan area of: %s",
                                   str(self._scan_area_width))
                # break
            elif ptp(res_points[:, 1]) > 10e6:
                self._logger.debug("Very strong flux dependence found. "
                                   "Probably avoided crossings. Leaving as is..")
                self._res_freq = (max(res_points[:, 1]) + min(res_points[:, 1]))/2
                break
            else:
                self._logger.debug("No dependence found. Trying to zoom in.")
                self._res_freq = (max(res_points[:, 1]) + min(res_points[:, 1]))/2
                self._scan_area_width = self._scan_area_width / 5
                # self._bias_values = self._bias_values*5

            counter += 1

            if counter < 3:
                self._scan_area_width_previous = self._scan_area_width
            elif (self._scan_area_width_previous - 1e6) * 1.1 > (self._scan_area_width - 1e6):
                break
            else:
                self._scan_area_width_previous = self._scan_area_width
            if self._scan_area_width_previous > 16e6:
                break


        # self._vna_parameters["nop"] = 101
        self._perform_STS()
        ao = AnticrossingOracle("transmon",
                                self._sts_result,
                                plot=True,
                                fast_res_detect=False,
                                hints = STSRunnerParameters().anticrossing_oracle_hints)
        period = ao._find_period()

        N_periods = ptp(self._bias_values) / period
        self._logger.debug("Periods: %.2f" % N_periods)

        if N_periods > 1:
            self._bias_values = \
                (self._bias_values - mean(self._bias_values)) / N_periods * 1.5 + mean(self._bias_values)
            self._bias_values = linspace(self._bias_values[0], self._bias_values[-1], 201)

            self._vna_parameters["nop"] = self._vna_parameters["nop"]*2
            self._vna_parameters["bandwidth"] = self._vna_parameters["bandwidth"]*2
            self._perform_STS()
        elif N_periods < 1:
            if max(abs(self._bias_values)) > 1e-3:
                raise ValueError("Flux period is too large!")

            self._logger.debug("Current range too narrow" + str(N_periods))
            self._bias_values = \
                (self._bias_values - mean(self._bias_values)) * 2 + mean(self._bias_values)
            self._perform_STS()

    def _perform_STS(self):

        self._vna_parameters["freq_limits"] = \
            (self._res_freq - self._scan_area_width / 2,
             self._res_freq + self._scan_area_width / 2)

        self._STS = SingleToneSpectroscopy(self._sts_name,
                                           self._sample_name, plot_update_interval=1,
                                           vna=self._vna, src=self._bias_src)

        self._STS.set_fixed_parameters(vna=[self._vna_parameters])
        self._STS.set_swept_parameters({self._bias_parameter_name: \
                                            (self._STS._src[0].set, self._bias_values)})

        self._sts_result = self._STS.launch()

    def _perform_STS_first(self):  # for the case when two resonators are too close

        self._vna_parameters["freq_limits"] = \
            (self._res_freq - self._scan_area_width / 2 - 8e6,
             self._res_freq + self._scan_area_width / 2 - 1e6)

        self._STS = SingleToneSpectroscopy(self._sts_name,
                                           self._sample_name, plot_update_interval=1,
                                           vna=self._vna, src=self._bias_src)

        self._STS.set_fixed_parameters(vna=[self._vna_parameters])
        self._STS.set_swept_parameters({self._bias_parameter_name: \
                                            (self._STS._src[0].set, self._bias_values)})

        self._sts_result = self._STS.launch()


    def get_scan_area(self):
        return (self._res_freq - self._scan_area_width / 2,
                self._res_freq + self._scan_area_width / 2)
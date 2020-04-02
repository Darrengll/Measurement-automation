from lib2.SingleToneSpectroscopy import *
from lib2.fulaut.AnticrossingOracle import *
from lib2.GlobalParameters import GlobalParameters
from loggingserver import LoggingServer
from datetime import datetime


class STSRunner():

    def __init__(self, sample_name, qubit_name, res_freq, vna=None, cur_src=None, awgs=None):

        self._sample_name = sample_name
        self._qubit_name = qubit_name
        self._res_freq = res_freq
        self._sts_name = "%s-sts" % qubit_name
        self._scan_area = 10e6
        self._vna = vna
        self._cur_src = cur_src

        self._vna_power = FulautParameters().sts_runner["power"]

        self._vna_parameters = {"bandwidth": FulautParameters().sts_runner["bandwidth"],
                                "nop": FulautParameters().sts_runner["nop"],
                                "power": self._vna_power,
                                "averages": FulautParameters().sts_runner["averages"]}

        self._currents = linspace(-.1e-3, .1e-3, 101)
        self._sts_result = None
        self._launch_datetime = datetime.today()
        self._cur_src[0].set_appropriate_range(max(abs(self._currents)))

        self._logger = LoggingServer.getInstance('fulaut')

    def run(self):

        # Check if today's anticrossing is present

        known_results = \
            MeasurementResult.load(self._sample_name,
                                   self._sts_name,
                                   date=self._launch_datetime.strftime("%b %d %Y"),
                                   return_all=True)

        if known_results is not None:
            self._sts_result = known_results[-1]
            if hasattr(self._sts_result, "_fit_result"):
                return known_results[-1]._fit_result
        else:
            self._iterate_STS()

        ao = AnticrossingOracle("transmon", self._sts_result,
                                plot=True,
                                fast_res_detect=False,
                                hints = FulautParameters().sts_runner["anticrossing_oracle_hints"])
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
        while (counter < 3):

            self._perform_STS()
            ao = AnticrossingOracle("transmon", self._sts_result,
                                    plot=True,
                                    fast_res_detect=False,
                                    hints = FulautParameters().sts_runner["anticrossing_oracle_hints"])
            res_points = ao.get_res_points()

            self._logger.debug("Scan: " + str(self._scan_area / 1e6))
            self._logger.debug("Ptp: " + str(ptp(res_points[:, 1]) / 1e6))
            if 0.1 * self._scan_area < ptp(res_points[:, 1]) < 0.5 * self._scan_area:
                self._logger.debug("Flux dependence found. Zooming...")
                self._scan_area = max(ptp(res_points[:, 1]) / 0.25, 3e6)
                self._res_freq = mean(res_points[:, 1])
                break
            elif ptp(res_points[:, 1]) > 0.5 * self._scan_area:
                self._logger.debug("Strong flux dependence found. Leaving as is..")
                self._res_freq = mean(res_points[:, 1])
                break
            else:
                self._logger.debug("No dependence found. Trying to zoom in.")
                self._res_freq = mean(res_points[:, 1])
                self._scan_area = self._scan_area / 5
                # self._currents = self._currents*5

            counter += 1

        self._vna_parameters["nop"] = 101
        self._perform_STS()
        ao = AnticrossingOracle("transmon",
                                self._sts_result,
                                plot=True,
                                fast_res_detect=False,
                                hints = FulautParameters().sts_runner["anticrossing_oracle_hints"])
        period = ao._find_period()

        N_periods = ptp(self._currents) / period
        self._logger.debug("Periods: %.2f" % N_periods)

        if N_periods > 1:
            self._currents = \
                (self._currents - mean(self._currents)) / N_periods*1.5 + mean(self._currents)
            self._currents = linspace(self._currents[0], self._currents[-1], 201)

            self._vna_parameters["nop"] = self._vna_parameters["nop"]*2
            self._vna_parameters["bandwidth"] = self._vna_parameters["bandwidth"]*2
            self._perform_STS()
        elif N_periods < 1:
            if max(abs(self._currents)) > 1e-3:
                raise ValueError("Flux period is too large!")

            self._logger.debug("Current range too narrow" + str(N_periods))
            self._currents =\
                (self._currents-mean(self._currents)) * 2 + mean(self._currents)
            self._perform_STS()

    def _perform_STS(self):

        self._vna_parameters["freq_limits"] = \
            (self._res_freq - self._scan_area / 2,
             self._res_freq + self._scan_area / 2)

        self._STS = SingleToneSpectroscopy(self._sts_name,
                                           self._sample_name, plot_update_interval=1,
                                           vna=self._vna, src=self._cur_src)

        self._STS.set_fixed_parameters(vna=[self._vna_parameters])
        self._STS.set_swept_parameters({'Current [A]': \
                                            (self._STS._src[0].set_current, self._currents)})

        self._sts_result = self._STS.launch()

    def get_scan_area(self):
        return (self._res_freq - self._scan_area / 2,
                self._res_freq + self._scan_area / 2)
from lib2.SingleToneSpectroscopy import *
from lib2.fulaut.AnticrossingOracle import *
from lib2.LoggingServer import LoggingServer
from datetime import datetime
from time import sleep

class STSRunner():

    def __init__(self, sample_name, qubit_name, res_freq, vna=None, cur_src=None, awgs=None):

        self._sample_name = sample_name
        self._qubit_name = qubit_name
        self._res_freq = res_freq
        self._sts_name = "%s-sts" % qubit_name
        self._scan_area = 10e6
        self._vna = vna
        self._cur_src = cur_src

        if awgs is not None:
            self._ro_awg = awgs["ro_awg"]
            self._q_awg = awgs["q_awg"]
            self._open_only_readout_mixer()
            self._vna_power = -20
        else:
            self._vna_power = -50

        self._vna_parameters = {"bandwidth": 300,
                                "nop": 201,
                                "power": self._vna_power,
                                "averages": 2,
                                "sweep_type": "LIN"}
        self._currents = linspace(-1e-4, 1e-4, 101)
        #self._currents = linspace(-0.4e-4, 1.2e-4, 101)
        self._sts_result = None
        self._launch_datetime = datetime.today()
        self._cur_src[0].set_appropriate_range(max(abs(self._currents)))
        self._logger = LoggingServer.getInstance()

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

        ao = AnticrossingOracle("transmon", self._sts_result, plot=True)
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
            ao = AnticrossingOracle("transmon", self._sts_result, plot=True)
            res_points = ao.get_res_points()

            self._logger.debug("Scan: " + str(self._scan_area / 1e6))
            self._logger.debug("Ptp: " + str(ptp(res_points[:, 1]) / 1e6))
            if 0.1e6 < ptp(res_points[:, 1]) < 0.5 * self._scan_area:
                self._logger.debug("Flux dependence found. Rescaling y-axis...")
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
        ao = AnticrossingOracle("transmon", self._sts_result, plot=True)
        period = ao._find_period()

        N_periods = ptp(self._currents) / period
        self._logger.debug("Periods: %.2f" % N_periods)

        if N_periods > 1:
            self._currents = \
                (self._currents - mean(self._currents)) / N_periods*1.5 + mean(self._currents)
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

        self._cur_src[0].set_current(self._currents[0])
        sleep(0.5)
        self._sts_result = self._STS.launch()

    def get_scan_area(self):
        return (self._res_freq - self._scan_area / 2,
                self._res_freq + self._scan_area / 2)

    def _open_only_readout_mixer(self):
        self._ro_awg.output_continuous_IQ_waves(frequency=0, amplitudes=(0, 0),
                                                relative_phase=0, offsets=(1, 1),
                                                waveform_resolution=1)
        self._q_awg.output_continuous_IQ_waves(frequency=0, amplitudes=(0, 0),
                                               relative_phase=0, offsets=(0, 0),
                                               waveform_resolution=1)

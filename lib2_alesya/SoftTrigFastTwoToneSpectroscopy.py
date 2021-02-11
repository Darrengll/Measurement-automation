from numpy import *
from lib2.FastTwoToneSpectroscopy import *
from time import sleep


class SoftTrigFastFluxTwoToneSpectroscopy(FastFluxTwoToneSpectroscopy):


    def _recording_iteration(self):

        bandwidth = self._fixed_pars['vna'][0]["bandwidth"]
        point_time = 1 / bandwidth
        min_stable_time = 0.015
        wait_time = max((point_time, min_stable_time))

        self._vna[0].set_trigger_source("BUS")
        self._vna[0].set_stepped_triggered_sweep(True)
        self._vna[0].sweep_continuous()
        self._vna[0].reset_sweep()

        sleep(.5)

        for freq in self._frequencies:
            self._mw_src[0].set_frequency(freq)
            sleep(0.005)
            self._vna[0].send_software_trigger()
            sleep(wait_time + 0.005)

        self._vna[0].sweep_hold()
        self._vna[0].set_trigger_source("INT")

        return self._vna[0].get_sdata()


class SoftTrigFastPowerTwoToneSpectroscopy(FastPowerTwoToneSpectroscopy):

    def _recording_iteration(self):

        bandwidth = self._fixed_pars['vna'][0]["bandwidth"]
        point_time = 1 / bandwidth
        min_stable_time = 0.015
        wait_time = max((point_time, min_stable_time))

        self._vna[0].set_trigger_source("BUS")
        self._vna[0].set_stepped_triggered_sweep(True)
        self._vna[0].sweep_continuous()
        self._vna[0].reset_sweep()

        sleep(.5)

        for freq in self._frequencies:
            if self._interrupted:
                break

            self._mw_src[0].set_frequency(freq)
            sleep(0.005)
            self._vna[0].send_software_trigger()
            sleep(wait_time + 0.005)

        self._vna[0].sweep_hold()
        self._vna[0].set_trigger_source("INT")

        return self._vna[0].get_sdata()
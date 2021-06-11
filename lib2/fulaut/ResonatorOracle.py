from loggingserver import LoggingServer
from scipy import *
import pickle
from scipy.signal import argrelextrema
from matplotlib import gridspec, pyplot as plt
from lib2.ExperimentParameters import ResonatorOracleParameters, GlobalParameters
from lib2.ResonatorDetector import ResonatorDetector
import os

class ResonatorOracle:

    def __init__(self, vna, sample_name, s_param):
        self._logger = LoggingServer.getInstance('fulaut')

        self._vna = vna[0]
        self._sample_name = sample_name
        self._vna.select_S_param(s_param)
        self._area_size = ResonatorOracleParameters().default_scan_area

    def launch(self):

        if not ResonatorOracleParameters().rerun:
            try:
                with open(f"data/{self._sample_name}/resonator_oracle_scan.pkl", "rb") as f:
                    return pickle.load(f)
            except:
                self._logger.debug(f"ResonatorOracle could not load scan areas for {self._sample_name}")

        vna = self._vna
        vna.sweep_hold()
        vna.set_parameters(ResonatorOracleParameters().vna_parameters)

        vna.prepare_for_stb()
        vna.sweep_single()  # triggering the sweep
        vna.wait_for_stb()
        vna.autoscale_all()

        freqs, s_data = self._vna.get_frequencies(), self._vna.get_sdata()
        depth = 0.1
        scan_areas = self.guess_scan_areas(freqs, s_data,
                                           self._area_size, depth)
        peak_number = ResonatorOracleParameters().peak_number
        while len(scan_areas) > peak_number:
            scan_areas = self.guess_scan_areas(freqs, s_data,
                                               self._area_size, depth)
            depth += .2

        plt.figure("Resonator oracle scan")
        gs = gridspec.GridSpec(2, peak_number)
        ax1 = plt.subplot(gs[0, :])
        ax1.plot(freqs / 1e9, 20 * log10(abs(s_data)))

        ax1.set_ylim(min(20 * log10(abs(s_data))) - 5,
                     max(20 * log10(abs(s_data))) + 5)
        ax1.minorticks_on()
        ax1.grid(which="both")
        ax1.set_xlabel("Frequency [GHz]")
        ax1.set_ylabel("$|S_{21}|^2$")

        for scan_area in scan_areas:
            ax1.plot(array(scan_area) / 1e9, ones(2) * min(20 * log10(abs(s_data))), marker="+")

        for j in range(len(scan_areas)):
            vna.set_parameters({"freq_limits": scan_areas[j], "nop": 501})

            vna.prepare_for_stb()
            vna.sweep_single()  # triggering the sweep
            vna.wait_for_stb()
            vna.autoscale_all()

            freqs, s_data = self._vna.get_frequencies(), self._vna.get_sdata()

            rd = ResonatorDetector(freqs, s_data, False, False,
                                   type=GlobalParameters().resonator_type)
            try:
                fr, amp, phase = rd.detect()
                res_width = rd.get_linewidth()
                scan_areas[j] = [fr - res_width*5, fr+res_width*5]
            except ValueError as e:
                print("Couldn't determine linewidth")
            finally:
                axN = plt.subplot(gs[1, j:j+1])
                axN.plot(freqs/ 1e9, 20 * log10(abs(s_data)))
                axN.plot(array(scan_areas[j]) / 1e9, ones(2) * min(20 * log10(abs(s_data))), marker="+")
        plt.gcf().set_size_inches(15, 5)
        plt.tight_layout()

        save_dir = f"data/{self._sample_name}"
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        plt.savefig(f"{save_dir}/resonator_oracle_scan.pdf",
                    bbox_inches="tight")
        with open(f"data/{self._sample_name}/resonator_oracle_scan.pkl", "wb") as f:
            pickle.dump(scan_areas, f)

        return scan_areas

    def guess_scan_areas(self, freqs, s_data, area_size, depth):
        """
        Function to get the approximate positions of the resonator dips
        and return small areas around them
        Parameters:
        -----------
            area_size : double, Hz
                Sets the diams of the returned scan areas

            depth : double
                Count everything deeper than median transmission depth
                near a minimum as a resonator dip. Should be chosen manually
                each time
        Returns:
            scan_areas : list
                A list of tuples each representing an area in frequency
                presumably around the resonator dips
        """
        amps = 20 * log10(abs(s_data))
        window = ResonatorOracleParameters().window
        extrema = argrelextrema(amps, less, order=window)[0]
        deep_minima = []
        for extremum in extrema:
            mean_transmission = median(amps[extremum - window // 2:extremum + window // 2])
            if amps[extremum] < mean_transmission - depth:
                deep_minima.append(freqs[extremum])
        return [(m - area_size / 2, m + area_size / 2) for m in deep_minima]

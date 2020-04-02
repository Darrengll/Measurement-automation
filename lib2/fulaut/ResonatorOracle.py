from scipy import *
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt
from lib2.GlobalParameters import FulautParameters

class ResonatorOracle():

    def __init__(self, vna, s_param, area_size):
        self._vna = vna[0]
        self._vna.select_S_param(s_param)
        self._area_size = area_size

    def launch(self, n_peaks = 8):
        vna = self._vna
        vna.sweep_hold()
        vna.set_parameters(FulautParameters().resonator_oracle)

        vna.prepare_for_stb()
        vna.sweep_single() # triggering the sweep
        vna.wait_for_stb()
        vna.autoscale_all()

        freqs, s_data = self._vna.get_frequencies(), self._vna.get_sdata()
        depth = 0.1
        scan_areas = self.guess_scan_areas(freqs, s_data,
                                            self._area_size, depth)
        while len(scan_areas) > n_peaks:
            scan_areas = self.guess_scan_areas(freqs, s_data,
                                                self._area_size, depth)
            depth+=1

        plt.plot(freqs/1e9, 20*log10(abs(s_data)))

        for scan_area in scan_areas:
            plt.plot(array(scan_area)/1e9, ones(2)*min(20*log10(abs(s_data))), marker="+")

        plt.ylim(min(20*log10(abs(s_data)))-5,
                        max(20*log10(abs(s_data)))+5)
        plt.minorticks_on()
        plt.grid(which="both")
        plt.gcf().set_size_inches(15, 3)
        plt.xlabel("Frequency [GHz]")
        plt.ylabel("$|S_{21}|^2$")

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
        amps = 20*log10(abs(s_data))
        window = 100
        extrema = argrelextrema(amps, less, order=window)[0]
        deep_minima = []
        for extremum in extrema:
            mean_transmission = median(amps[extremum-window//2:extremum+window//2])
            if amps[extremum]<mean_transmission-depth:
                deep_minima.append(freqs[extremum])
        return [(m-area_size/2, m+area_size/2) for m in deep_minima]

"""
Parametric single-tone spectroscopy is perfomed with a Vector Network Analyzer
(VNA) for each parameter value which is set by a specific function that must be
passed to the SingleToneSpectroscopy class when it is created.
"""
from drivers.BiasType import BiasType
from numpy import *
from lib2.SingleToneSpectroscopy import *
from datetime import datetime as dt
from lib2.Measurement import *
from scipy.optimize import curve_fit


class TwoToneSpectroscopyBase(Measurement):

    def __init__(self, name, sample_name,
                 devs_aliases_map, plot_update_interval=5):

        super().__init__(name, sample_name, devs_aliases_map,
                         plot_update_interval)

        self._measurement_result = TwoToneSpectroscopyResult(name, sample_name)
        self._interrupted = False
        self._base_parameter_setter = None
        self._base_parameter_name = None

    def set_zero_output(self):
        self._mw_src[0].set_output_state("OFF")
        self._bias_src[0].set(0)

    def set_fixed_parameters(self, flux_bias=None, detect_resonator=True,
                             **dev_params):

        vna_parameters = dev_params['vna'][0]
        bias_type = self._bias_src[0].get_bias_type()
        self._base_parameter_setter = self._bias_src[0].set
        self._base_parameter_name = BiasType.NAMES[bias_type]
        msg1 = "at %.1e " % (flux_bias) + self._base_parameter_name[-2]  # V
                                                                         # or A

        self._base_parameter_setter(flux_bias)

        if detect_resonator:
            self._mw_src[0].set_output_state("OFF")
            msg = "Detecting a resonator within provided if_freq range of the VNA %s \
                            " % (str(vna_parameters["freq_limits"]))
            print(msg + msg1, flush=True)
            res_freq, res_amp, res_phase = self._detect_resonator(
                vna_parameters, True)
            print(
                "Detected if_freq is %.5f GHz, at %.2f mU and %.2f degrees" % (
                    res_freq / 1e9, res_amp * 1e3, res_phase / pi * 180))
            vna_parameters["freq_limits"] = (res_freq, res_freq)
            self._measurement_result.get_context() \
                .get_equipment()["vna"] = vna_parameters
            self._mw_src[0].set_output_state("ON")

        super().set_fixed_parameters(vna=[vna_parameters],
                                     mw_src=dev_params['mw_src'])

    def _detect_resonator(self, vna_parameters, plot=False):
        vna = self._vna[0]
        parameters = {"nop": vna_parameters["resonator_detection_nop"],
                      "freq_limits": vna_parameters["freq_limits"],
                      "power": vna_parameters["power"],
                      "bandwidth": vna_parameters[
                          "resonator_detection_bandwidth"],
                      "averages": vna_parameters["averages"]}
        vna.set_parameters(parameters)

        result = super()._detect_resonator(plot)
        return result

    def _recording_iteration(self):
        vna = self._vna[0]
        vna.avg_clear()
        vna.prepare_for_stb()
        vna.sweep_single()
        vna.wait_for_stb()
        data = vna.get_sdata()
        return mean(data)

    def _finalize(self):
        pass


class TwoToneSpectroscopyResult(SingleToneSpectroscopyResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._context = ContextBase()
        self._is_finished = False
        self._phase_units = "rad"
        self._annotation_bbox_props = dict(boxstyle="round", fc="white",
                                           ec="black", lw=1, alpha=0.5)

    def _tr_spectrum(self, parameter_value, parameter_value_at_sweet_spot,
                     frequency, period):
        return frequency * sqrt(
            cos((parameter_value - parameter_value_at_sweet_spot) / period))

    def _lorentzian_peak(self, frequency, amplitude, offset, res_frequency,
                         width):
        return amplitude * (0.5 * width) ** 2 / (
                (frequency - res_frequency) ** 2 + (
                0.5 * width) ** 2) + offset

    def _find_peaks(self, freqs, data):
        peaks = []
        for row in data:
            try:
                popt = curve_fit(self._lorentzian_peak,
                                 freqs, row, p0=(ptp(row), median(row),
                                                 freqs[argmax(row)], 10e6))[0]
                peaks.append(popt[2])
            except:
                peaks.append(freqs[argmax(row)])
        return array(peaks)

    def find_transmon_spectrum(self, axes, parameter_limits=(0, -1),
                               format="abs"):
        parameter_name = self._parameter_names[0]
        data = self.get_data()
        x = data[parameter_name][parameter_limits[0]:parameter_limits[1]]
        freqs = data[self._parameter_names[1]]
        Z = data["data"][parameter_limits[0]:parameter_limits[1]]

        if format == "abs":
            Z = abs(Z)
            annotation_ax_idx = 0
        elif format == "angle":
            Z = angle(Z)
            annotation_ax_idx = 1

        y = self._find_peaks(freqs, Z)

        try:
            popt = \
                curve_fit(self._tr_spectrum, x, y,
                          p0=(mean(x), max(y), ptp(x)))[0]
            annotation_string = parameter_name + " sweet spot at: " + self._latex_float(
                popt[0])

            for ax in axes:
                h_pos = mean(ax.get_xlim())
                v_pos = .1 * ax.get_ylim()[0] + .9 * ax.get_ylim()[1]
                ax.plot(x, y / 1e9, ".", color="C2")
                ax.plot(x, self._tr_spectrum(x, *popt) / 1e9)
                ax.plot([popt[0]], [popt[1] / 1e9], "+")

            axes[annotation_ax_idx].annotate(annotation_string, (h_pos, v_pos),
                                             bbox=self._annotation_bbox_props,
                                             ha="center")
            return popt[0], popt[1]
        except Exception as e:
            print("Could not find transmon spectral line" + str(e))

    def _prepare_data_for_plot(self, data):
        s_data = data["data"]
        parameter_list = data[self._parameter_names[0]]
        if parameter_list[0] > parameter_list[-1]:
            parameter_list = parameter_list[::-1]
            s_data = s_data[::-1, :]
        # s_data = self.remove_background('avg_cur')
        return parameter_list, data["Frequency [Hz]"] / 1e9, s_data

    def _prepare_measurement_result_data(self, data):
        return data[self._parameter_names[0]], data["Frequency [Hz]"] / 1e9, \
               data["data"]

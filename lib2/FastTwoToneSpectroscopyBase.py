'''
Parametric single-tone spectroscopy is perfomed with a Vector Network Analyzer
(VNA) for each parameter value which is set by a specific function that must be
passed to the SingleToneSpectroscopy class when it is created.
'''

from numpy import *
from lib2.SingleToneSpectroscopy import *
from datetime import datetime as dt
from lib2.Measurement import *
from scipy.optimize import curve_fit
from enum import Enum


class FluxControlType(Enum):
    CURRENT = 1
    VOLTAGE = 2


class FastTwoToneSpectroscopyBase(Measurement):
    FLUX_CONTROL_PARAM_NAMES = {FluxControlType.CURRENT: ("Current", "A"),
                                FluxControlType.VOLTAGE: ("Voltage", "V")}

    def __init__(self, name, sample_name, flux_control_type,
                 devs_aliases_map, plot_update_interval=5):

        super().__init__(name, sample_name, devs_aliases_map,
                         plot_update_interval)

        self._measurement_result = TwoToneSpectroscopyResult(name, sample_name)
        self._interrupted = False
        self._base_parameter_setter = None  # set function that is used to set sweep parameter
        self._last_resonator_result = None
        self._frequencies = None
        # frequencies are no longer in self._swept_pars
        # sweep over frequencies is provided automatically through
        # trigger settings of the mw_src -> there is no need to call
        # mw_src.set_frequency() setter and cycle through frequency parameters
        # in Measurement._record_data()
        # but it is still swept parameter in some sense.
        # TODO: I propose following solution
        # We should implement the following mechanics:
        # Whenever this happens: some parameters are swept automatically and they are part of the
        # visualization process.
        #  We should just modify Measurement._record_data() in order to
        # be able to determine such situation and skip cycling through this parameters,
        # maybe by some additional parameter like swept_automatically=["par_name1",...,"par_nameN"]
        # or maybe by implementing trigger interface in all device classes that could be triggered by
        # external hardware signal and those who can trigger other devices. This is simple class with 2
        # attributes at most. The deal is to verify a consistency of the idea before starting actual coding

        if flux_control_type is FluxControlType.CURRENT:
            self._base_parameter_setter = self._current_src[0].set_current
        elif flux_control_type is FluxControlType.VOLTAGE:
            self._base_parameter_setter = self._voltage_src[0].set_voltage
        else:
            raise ValueError("Flux parameter type invalid")

        param_name, param_dim = FastTwoToneSpectroscopyBase.FLUX_CONTROL_PARAM_NAMES[flux_control_type]
        self._parameter_name = param_name + " [%s]" % param_dim
        self._info_suffix = "at %.4f " + param_dim

    def set_fixed_parameters(self, flux_control_parameter, detect_resonator = True, **dev_params):

        vna_parameters = dev_params['vna'][0]
        mw_src_parameters = dev_params['mw_src'][0]
        self._frequencies = mw_src_parameters["frequencies"]

        if "ext_trig_channel" in mw_src_parameters.keys():
            # internal adjusted trigger parameters for vna
            vna_parameters["trig_per_point"] = True  # trigger output once per sweep point
            vna_parameters["pos"] = True  # positive edge
            vna_parameters["bef"] = False  # trigger sent before measurement is started

            # internal adjusted trigger parameters for microwave source
            mw_src_parameters["unit"] = "Hz"
            mw_src_parameters["InSweep_trg_src"] = "EXT"
            mw_src_parameters["sweep_trg_src"] = "BUS"

        if flux_control_parameter is not None:
            self._base_parameter_setter(flux_control_parameter)

        
        if detect_resonator:
            self._mw_src[0].set_output_state("OFF")
            msg = "Detecting a resonator within provided frequency range of the VNA %s \
                            " % (str(vna_parameters["freq_limits"]))
            print(msg + self._info_suffix % flux_control_parameter, flush=True)

            res_freq, res_amp, res_phase = self._detect_resonator(vna_parameters, plot=True)
            print("Detected frequency is %.5f GHz, at %.2f mU and %.2f degrees" % (
                res_freq / 1e9, res_amp * 1e3, res_phase / pi * 180))
            vna_parameters["freq_limits"] = (res_freq, res_freq)
            self._measurement_result.get_context() \
                .get_equipment()["vna"] = vna_parameters
            self._mw_src[0].set_output_state("ON")

        super().set_fixed_parameters(vna=dev_params['vna'], mw_src=dev_params['mw_src'])

    def set_swept_parameters(self, **swept_pars):
        setter_function = self._adaptive_setter if self._adaptive else self._base_setter

        for swept_par in swept_pars.keys():  # only 1 parameter here
            swept_pars[swept_par] = setter_function, swept_pars[swept_par][1]

        super().set_swept_parameters(**swept_pars)

    def _base_setter(self, value):
        self._base_parameter_setter(value)
        self._mw_src[0].send_sweep_trigger()  # telling mw_src to be ready to start

    def _adaptive_setter(self, value):
        self._base_parameter_setter(value)
        vna_parameters = self._fixed_pars["vna"][0]
        vna_parameters["freq_limits"] = self._resonator_area

        self._mw_src[0].set_output_state("OFF")
        # print("\rDetecting a resonator within provided frequency range of the VNA %s\
        #            "%(str(vna_parameters["freq_limits"])), flush=True, end="")

        res_result = self._detect_resonator(vna_parameters, plot=False)

        if res_result is None:
            print("Failed to fit resonator, trying to use last successful fit, current = ", value,
                  " A")
            if self._last_resonator_result is None:
                print("no successful fit is present, terminating")
                return None
            else:
                res_freq, res_amp, res_phase = self._last_resonator_result
        else:
            self._last_resonator_result = res_result
            res_freq, res_amp, res_phase = self._last_resonator_result

        # print("\rDetected frequency is %.5f GHz, at %.2f mU and %.2f \
        #            degrees"%(res_freq/1e9, res_amp*1e3, res_phase/pi*180), end="")
        self._mw_src[0].set_output_state("ON")
        vna_parameters["freq_limits"] = (res_freq, res_freq)
        self._vna[0].set_parameters(vna_parameters)  # set new freq_limits
        self._mw_src[0].send_sweep_trigger()  # telling mw_src to be ready to start

    def _prepare_measurement_result_data(self, parameter_names, parameters_values):
        measurement_data = super()._prepare_measurement_result_data(parameter_names, parameters_values)
        measurement_data["Frequency [Hz]"] = self._frequencies
        return measurement_data

    def _detect_resonator(self, vna_parameters, plot=True):

        self._vna[0].set_nop(100)
        self._vna[0].set_freq_limits(*vna_parameters["freq_limits"])
        if "res_find_power" in vna_parameters.keys():
            self._vna[0].set_power(vna_parameters["res_find_power"])
        else:
            self._vna[0].set_power(vna_parameters["power"])
        if "res_find_nop" in vna_parameters.keys():
            self._vna[0].set_nop(vna_parameters["res_find_nop"])
        else:
            self._vna[0].set_nop(vna_parameters["nop"])
        self._vna[0].set_bandwidth(vna_parameters["resonator_detection_bandwidth"])
        self._vna[0].set_averages(vna_parameters["averages"])
        result = super()._detect_resonator(plot)
        self._vna[0].set_parameters(vna_parameters)  # restore initial parameters values
        return result

    def _recording_iteration(self):
        vna = self._vna[0]
        vna.avg_clear()
        vna.prepare_for_stb()
        vna.sweep_single()
        vna.wait_for_stb()
        data = vna.get_sdata()
        return data


class TwoToneSpectroscopyResult(SingleToneSpectroscopyResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._context = ContextBase()
        self._is_finished = False
        self._phase_units = "rad"
        self._annotation_bbox_props = dict(boxstyle="round", fc="white",
                                           ec="black", lw=1, alpha=0.5)

    def _tr_spectrum(self, parameter_value, parameter_value_at_sweet_spot, frequency, period):
        return frequency * sqrt(cos((parameter_value - parameter_value_at_sweet_spot) / period))

    def _lorentzian_peak(self, frequency, amplitude, offset, res_frequency, width):
        return amplitude * (0.5 * width) ** 2 / (
                (frequency - res_frequency) ** 2 + (0.5 * width) ** 2) + offset

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
            popt = curve_fit(self._tr_spectrum, x, y, p0=(mean(x), max(y), ptp(x)))[0]
            annotation_string = parameter_name + " sweet spot at: " + self._latex_float(popt[0])

            for ax in axes:
                h_pos = mean(ax.get_xlim())
                v_pos = .1 * ax.get_ylim()[0] + .9 * ax.get_ylim()[1]
                ax.plot(x, y / 1e9, ".", color="C2")
                ax.plot(x, self._tr_spectrum(x, *popt) / 1e9)
                ax.plot([popt[0]], [popt[1] / 1e9], "+")

            axes[annotation_ax_idx].annotate(annotation_string, (h_pos, v_pos),
                                             bbox=self._annotation_bbox_props, ha="center")
            return popt[0], popt[1]
        except Exception as e:
            print("Could not find transmon spectral line" + str(e))

    # def _prepare_measurement_result_data(self, data):
    #     return data[self._parameter_names[0]], data["Frequency [Hz]"] / 1e9, data["data"]

    def _prepare_data_for_plot(self, data):
        s_data = data["data"]
        parameter_list = data[self._parameter_names[0]]
        return [parameter_list, data["Frequency [Hz]"] / 1e9, s_data]

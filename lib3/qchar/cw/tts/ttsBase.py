"""
Parametric single-tone spectroscopy is perfomed with a Vector Network Analyzer
(VNA) for each parameter value which is set by a specific function that must be
passed to the SingleToneSpectroscopy class when it is created.
"""

# Standard library imports
from enum import Enum
from typing import Union, Dict, Any, List

# Third party imports
import numpy as np
from scipy.optimize import curve_fit

# Local application imports
from lib3.core.contextBase import ContextBase
from lib3.core.measurement import Measurement
from lib3.core._measurement_results.sParMeas2D import SParMeas2D

from lib3.core.compound_devices.iq_awg import AWGChannel
from lib3.core.drivers.yokogawaGS210 import YokogawaGS210
from lib3.core.drivers.agilent_PNA_L import Agilent_PNA_L

from lib3.core.drivers.mw_sources import MwSrcInterface


class FLUX_CONTROL_TYPE(Enum):
    CURRENT = 1
    VOLTAGE = 2


class TTSBase(Measurement):
    FLUX_CONTROL_PARAM_NAMES = {FLUX_CONTROL_TYPE.CURRENT: ("Current", "A"),
                                FLUX_CONTROL_TYPE.VOLTAGE: ("Voltage", "V")}
    SCAN_FREQUENCY_CAPTION = "Scan Frequency, Hz"

    def __init__(self, name, sample_name, flux_control_type,
                 flux_src=None, vna=None, mw_src=None,
                 mw_triggered_by_vna=False,
                 plot_update_interval=5):
        """
        Parameters
        ----------
        name : str
            name of measurement
        sample_name :  str
        flux_control_type : FLUX_CONTROL_TYPE
            whether flux is controlled via voltage or current source.
        flux_src : Union[AWGChannel, YokogawaGS210]
            Devices that can be used as current and/or voltage sources.
            support `set_voltage()` and/or `set_current()`
        vna : Agilent_PNA_L
            vector network analyzer instance
            # TODO add support for compound device DaqAdcVna from `core.compound_devices`
        mw_src : MwSrcInterface
            Unified interface for microwave source. Currently supported by
            SC5502A, N5173B and EXG devices.
        mw_triggered_by_vna : bool
            Whether or not sweep parameter changes is intended to be
            hardware triggered from vna. Measurement scheme should include
            trigger from VNA`s trigger output into sweep
            parameter's  device trigger input.
        plot_update_interval : np.float
            Time between sequential calls to results `_plot()` function in
            case of dynamic visuzalition.
        """
        devs_aliases_map = dict(flux_src=flux_src, vna=vna, mw_src=mw_src)

        # theese attributes will be created and set in call to
        # `super().__init__(...). They declared here explicitly
        # for further convenience
        self._flux_src: List[Union[AWGChannel, YokogawaGS210]] = None
        self._vna: List[Agilent_PNA_L] = None
        self._mw_src: List[MwSrcInterface] = None

        # set proper function to set flux variable, whether it is current or
        # voltage
        if flux_control_type == FLUX_CONTROL_TYPE.CURRENT:
            self._flux_parameter_setter = self._flux_src[0].set_current
        elif flux_control_type == FLUX_CONTROL_TYPE.VOLTAGE:
            self._flux_parameter_setter = self._flux_src[0].set_voltage
        else:
            raise ValueError("Flux parameter type invalid")

        # whether or not microwave source is triggered by vna
        self._mw_triggered_by_vna = mw_triggered_by_vna

        super().__init__(name, sample_name, devs_aliases_map,
                         plot_update_interval)

        self._measurement_result = TTSResultBase(name, sample_name)
        # flag that is used in the main loop of `Measurement` class that
        # indicates whether the process is interrupted by keyboard or
        # anything else
        self._interrupted = False

        # fit results of last resonator fit
        self._last_resonator_result = None

        # tuple of frequencies representing interval where resonator is
        # resided for any `sweep_parameter` value
        # Commonly obtained from STS.
        self._resonator_area = None
        # second tone scan frequencies
        self.scan_frequencies = None

        # function-attribute that is used to set next sweep parameter value
        self._flux_parameter_setter = None
        self._flux_control_type = flux_control_type

        # making nice flux axis caption
        param_name, param_dim = \
            TTSBase.FLUX_CONTROL_PARAM_NAMES[flux_control_type]
        self._flux_format_str = param_name + ", " + param_dim

        # values for real-time string reporting measurement`s progress
        self._info_suffix = "at {:.4f} " + param_dim

    def set_fixed_parameters(self, flux_control_parameter,
                             detect_resonator=True, vna_params_list=None,
                             mw_src_params_list=None):
        """

        Parameters
        ----------
        flux_control_parameter : np.float
            value of the flux control parameter. Expressed in [V] or [A]
            depending on the `self._flux_control_type`
        detect_resonator : bool
            whether or not to detect resonator during the
            `set_fixed_parameters()` call
        vna_params_list : List[Dict[str,Any]]
            Vna parameters as list of dictionaries.
            Only supports list with single entry containing a dictionary.
        mw_src_params_list : List[Dict[str, Any]]
            Microwave source parameters as list of dictionaries.
            Only supports list with single entry containing a dictionary.

        Returns
        -------

        """
        vna_parameters = vna_params_list[0]
        self._resonator_area = vna_parameters["freq_limits"]

        mw_src_parameters = mw_src_params_list[0]
        self.scan_frequencies = mw_src_parameters["frequencies"]

        # if `mw_src` is triggered by `vna` via external trigger input we
        # need to modify their parameters dictionaries accordingly
        if self._mw_triggered_by_vna:
            self._configure_triggered_sweep(
                vna_parameters=vna_parameters,
                mw_src_parameters=mw_src_parameters
            )

        # set initial flux control parameter value if requested
        if flux_control_parameter is not None:
            self._flux_parameter_setter(flux_control_parameter)

        # detects resonator frequency if requested.
        if detect_resonator:
            self._mw_src[0].set_output_state("OFF")
            msg = "Detecting a resonator within provided if_freq range of the VNA %s \
                            " % (str(vna_parameters["freq_limits"]))
            print(msg + self._info_suffix % flux_control_parameter, flush=True)

            self._last_resonator_result = self._detect_resonator(
                vna_parameters, plot=True)
            res_freq, res_amp, res_phase = self._last_resonator_result

            print(
                "Detected if_freq is %.5f GHz, at %.2f mU and %.2f degrees" % (
                    res_freq / 1e9, res_amp * 1e3, res_phase / np.pi * 180))
            vna_parameters["freq_limits"] = (res_freq, res_freq)
            self._measurement_result.get_context() \
                .get_equipment()["vna"] = vna_parameters
            self._mw_src[0].set_output_state("ON")

        super().set_fixed_parameters(vna=vna_parameters,
                                     mw_src=mw_src_parameters)

    def _prepare_measurement_result_data(self, parameter_names,
                                         parameters_values):
        measurement_data = super()._prepare_measurement_result_data(
            parameter_names, parameters_values)
        measurement_data["Frequency [Hz]"] = self.scan_frequencies
        return measurement_data

    def _detect_resonator(self, plot=False, parameters=None, tries_number=3):
        vna_params = {"nop": parameters["resonator_detection_nop"],
                      "freq_limits": parameters["freq_limits"],
                      "power": parameters["power"],
                      "bandwidth": parameters[
                          "resonator_detection_bandwidth"],
                      "averages": parameters["averages"]}
        result = super()._detect_resonator(
            plot=plot, parameters=vna_params,
            tries_number=tries_number)
        return result

    def _recording_iteration(self):
        vna = self._vna[0]
        vna.avg_clear()
        vna.prepare_for_stb()
        vna.sweep_single()
        vna.wait_for_stb()
        data = vna.get_sdata()
        return data

    def get_flux_control_type(self):
        return self._flux_control_type

    def _configure_triggered_sweep(self, vna_parameters, mw_src_parameters):
        """
        NOT TESTED YET
        Configures `vna` trigger output and `mw_src` device trigger input
        Only supports following devices:
            vna - `Agilent_PNA_L`
            mw_src - `EXG` or `N5173B`
        Parameters
        ----------
        vna_parameters : Dict[str, Any]
            Dictionary with parameters of the vna. Dictionary will be
            modified inplace.
        mw_src_parameters : Dict[str, Any]
            Dictionary with parameters of the `mw_src` device. Dictionary
            will be modified inplace.

        Returns
        -------
        None
        """
        # set trigger output parameters for vna
        vna_parameters[
            "trig_per_point"] = True  # trigger output once per sweep point
        vna_parameters["pos"] = True  # positive edge
        # trigger sent before measurement is started
        vna_parameters["bef"] = False

        # set trigger input parameters for microwave source
        mw_src_parameters["unit"] = "Hz"
        mw_src_parameters["InSweep_trg_src"] = "EXT"
        mw_src_parameters["sweep_trg_src"] = "BUS"


class TTSResultBase(SParMeas2D):
    """
    Base class for results of two-tone spectroscopies results vizualization.
    Vertical axis is always scanning frequency. Horizontal axis may change 
    depending on the user's desire:
    - TTS from flux through qubit (mainly from current/voltage of the source).
    - TTS from readout power (AC-Stark effect)
    - TTS from qubit power (Vacuum shift)
    """
    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._context = ContextBase()
        self._is_finished = False
        self._phase_units = "rad"
        self._annotation_bbox_props = dict(boxstyle="round", fc="white",
                                           ec="black", lw=1, alpha=0.5)

    def _tr_spectrum(self, parameter_value, parameter_value_at_sweet_spot,
                     frequency, period):
        return frequency * np.sqrt(
            np.cos((parameter_value - parameter_value_at_sweet_spot) / period))

    def _lorentzian_peak(self, frequency, amplitude, offset, res_frequency,
                         width):
        return amplitude * (0.5 * width) ** 2 / (
                (frequency - res_frequency) ** 2 + (0.5 * width) ** 2) + offset

    def _find_peaks(self, freqs, data):
        peaks = []
        for row in data:
            try:
                popt = curve_fit(self._lorentzian_peak,
                                 freqs, row, p0=(np.ptp(row), np.median(row),
                                                 freqs[np.argmax(row)], 10e6))[0]
                peaks.append(popt[2])
            except:
                peaks.append(freqs[np.argmax(row)])
        return np.array(peaks)

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
            Z = np.angle(Z)
            annotation_ax_idx = 1

        y = self._find_peaks(freqs, Z)

        try:
            popt = \
            curve_fit(self._tr_spectrum, x, y, p0=(np.mean(x), max(y), np.ptp(x)))[0]
            annotation_string = parameter_name + " sweet spot at: " + self._latex_float(
                popt[0])

            for ax in axes:
                h_pos = np.mean(ax.get_xlim())
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

    # def _prepare_measurement_result_data(self, data):
    #     return data[self._parameter_names[0]], data["Frequency [Hz]"] / 1e9, data["data"]

    def _prepare_data_for_plot(self, data):
        s_data = data["data"]
        parameter_list = data[self._parameter_names[0]]
        return [parameter_list, data["Frequency [Hz]"] / 1e9, s_data]

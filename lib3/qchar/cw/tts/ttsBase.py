"""
Parametric single-tone spectroscopy is perfomed with a Vector Network Analyzer
(VNA) for each parameter value which is set by a specific function that must be
passed to the SingleToneSpectroscopy class when it is created.
"""

# Standard library imports
import copy
from enum import Enum
from typing import Union, Dict, Any, List, Tuple

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

from lib3.core.drivers.mw_sources import MwSrcInterface, MwSrcParameters


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
            whether flux is controlled via voltage or bias source.
        flux_src : Union[AWGChannel, YokogawaGS210]
            Devices that can be used as bias and/or voltage sources.
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
        # theese attributes will be created and set in call to
        # `super().__init__(...). They declared here explicitly
        # for further convenience
        self._flux_src: List[Union[AWGChannel, YokogawaGS210]] = None
        self._vna: List[Agilent_PNA_L] = None
        self._vna_pars: Dict[str, Any] = None
        self._mw_src: List[MwSrcInterface] = None
        self._mw_src_pars: MwSrcParameters = None

        devs_aliases_map = dict(flux_src=flux_src, vna=vna, mw_src=mw_src)
        super().__init__(name, sample_name, devs_aliases_map,
                         plot_update_interval)

        # function-attribute that is used to set next sweep parameter value
        # voltage [V] or bias [A]
        self._flux_parameter_setter = None
        self._flux_control_parameter = None
        if flux_control_type == FLUX_CONTROL_TYPE.CURRENT:
            self._flux_parameter_setter = self._flux_src[0].set_current
        elif flux_control_type == FLUX_CONTROL_TYPE.VOLTAGE:
            self._flux_parameter_setter = self._flux_src[0].set_voltage
        else:
            raise ValueError("Flux parameter type invalid")

        # sweep parameter setter function, by default changes flux
        self._set_swept_par = self._flux_parameter_setter

        # whether or not microwave source is triggered by vna
        self._mw_triggered_by_vna = mw_triggered_by_vna

        self._measurement_result = TTSResultBase(name, sample_name)
        self._measurement_result.mw_triggered_by_vna = mw_triggered_by_vna
        # flag that is used in the main loop of `Measurement` class that
        # indicates whether the process is interrupted by keyboard or
        # anything else
        self._interrupted = False

        # fit results of last resonator fit
        self._last_resonator_result = None

        ### ADDITIONAL FIXED PARAMETERS SECTION START ###
        # TODO: declare separate data structure of VNAParams to contain
        #  resonator finding related VNA parmaeters. Add this structure as
        #  argument to `self.set_fixed_parameters`.

        # tuple of frequencies representing interval where resonator is
        # resided for any `sweep_parameter` value
        # Commonly obtained from STS.
        # Received from vna_res_find_parameters["freq_limits"]
        self._res_find_interval: Tuple[np.float64, np.float64] = (0.0, 0.0)
        # how many points in `self._res_find_interval` to use during resonator
        # detection.
        self._res_find_nop: np.int64 = 0
        # Resonator finding bandwidth
        self._res_find_bw: np.int64 = 0
        # Resonator finding averages number
        self._res_find_avgs: np.int64 = 0
        '''
        "RESONATOR_TOOLS" - use resonator tools library to fit resonator
            (default).  `plot` argument used to plot results.
        "MIN" - find absolute minimum and interpret it as resonator
            frequency. `plot` argument value is ignored.
        '''
        self._res_find_method: str = "RESONATOR_TOOLS"
        # second tone scan frequencies
        self.scan_frequencies = None
        ### ADDITIONAL FIXED PARAMETERS SECTION END ###

        self._flux_control_type = flux_control_type

        # making nice flux axis caption
        param_name, param_dim = \
            TTSBase.FLUX_CONTROL_PARAM_NAMES[flux_control_type]
        self._flux_format_str = param_name + ", " + param_dim

        # values for real-time string reporting measurement`s progress
        self._info_suffix = "at {:.4f} " + param_dim

    """ Custom parameters getter/setter functions"""
    def get_flux_control_type(self):
        return self._flux_control_type

    """ Implementation of a base class methods """
    def set_fixed_parameters(self, flux_control_parameter=None,
                             vna_params=None, mw_src_params=None,
                             detect_resonator=True,
                             res_find_method="RESONATOR_TOOLS",
                             plot_resonator=False,
                             res_find_interval=None, res_find_nop=None,
                             res_find_bw=None, res_find_avgs=1):
        """

        Parameters
        ----------
        flux_control_parameter : np.float
            value of the flux control parameter. This value is set during
            this function call. Expressed in [V] or [A]
            depending on the `self._bias_type`
        vna_params : List[Dict[str,Any]]
            Vna parameters as list of dictionaries.
            Only supports list with single entry containing a dictionary.
        mw_src_params : List[MwSrcParameters]
            Microwave source parameters as list of dictionaries.
            Only supports list with single entry containing a dictionary.
        detect_resonator : bool
            whether or not to detect resonator during the
            `set_fixed_parameters()` call
        res_find_method : str
            "RESONATOR_TOOLS" - use resonator tools library to fit resonator
                (default).  `plot` argument used to plot results.
            "MIN" - find absolute minimum and interpret it as resonator
                frequency. `plot` argument value is ignored.
        plot_resonator : bool
            Whether or not to plot resonator fit results.
        res_find_interval : Tuple[np.float64, np.float64]
            Interval that certainly contain resonator response curve
        res_find_nop : np.int64
            Number of equally distributed points to scan in
            `resonator_scan_area`.
        res_find_bw : np.int64
            VNA bandwidth for resonator finding.
        res_find_avgs : np.int64
            VNA averages to use during resonator finding
        """
        # parameters are saved for further internal usage
        self._vna_pars = copy.deepcopy(vna_params[0])
        self._mw_src_pars = copy.deepcopy(mw_src_params[0])
        self._flux_control_parameter = flux_control_parameter
        self._res_find_method = res_find_method

        # TODO: wrap this parameters in VNA parameters structure with
        #  instance name like `res_find_vna_params` supplied to this function.
        #  Resonator monitoring VNA parameters should be supplied as the
        #  same class instance with name like `scan_vna_params`
        self._res_find_interval = res_find_interval
        self._res_find_nop = res_find_nop
        self._res_find_bw = res_find_bw
        self._res_find_avgs = res_find_avgs

        self.scan_frequencies = self._mw_src_pars.get_scan_freqs()

        # if `mw_src` is triggered by `vna` via external trigger input we
        # need to modify their parameters dictionaries accordingly
        if self._mw_triggered_by_vna:
            self._configure_vna_trg()
            self._configure_mw_src_trg()

        # set initial flux control parameter value if requested
        if flux_control_parameter is not None:
            self._flux_parameter_setter(flux_control_parameter)

        # detects resonator frequency if requested.
        if detect_resonator:
            # self explanatory function name
            self._detect_resonator_and_set_vna(
                    plot=plot_resonator, print_info=True
                )

        super().set_fixed_parameters(
            vna=[self._vna_pars], mw_src=[self._mw_src_pars]
        )

    def _prepare_measurement_result_data(self, parameter_names,
                                         parameters_values):
        measurement_data = super()._prepare_measurement_result_data(
            parameter_names, parameters_values)
        if TTSBase.SCAN_FREQUENCY_CAPTION not in measurement_data:
            measurement_data[TTSBase.SCAN_FREQUENCY_CAPTION] = \
                self.scan_frequencies
        return measurement_data

    def _recording_iteration(self):
        vna = self._vna[0]
        # clears VNA averaging data. Next sweep will be
        # started from scratch
        vna.avg_clear()
        # perform single sweep and return data as array of complex numbers
        data = vna.measure_and_get_data()
        return data

    def _finalize(self):
        # set flux to 0, whether it is bias or voltage
        self._flux_parameter_setter(0)
        # turn off microwave sources
        self._mw_src[0].set_output_state("OFF")
        self._vna[0].set_output_state("OFF")

    """ Methods specific for this class """
    def _detect_resonator_and_set_vna(self,
                                      plot=False, print_info=False,
                                      tries_number=3):
        """
        Detects resonator and returns its frequency as well minimal S21 in
        mV and phase at resonance.
        Sets VNA frequency equal to estimated resonator frequency and also
        set VNA ready for microwave source frequency sweep.

        Parameters
        ----------
        plot : bool
            Whether or not to plot result afterwards.
        print_info : bool
            Whether or not to print progress messages
        tries_number : int
            If first try to fit value fails, averages number of data
            sampling will be increased and new resonator search will be
            performed. Limits amount of total resonator searches performed
        """
        if print_info:
            # print progress message
            msg = "Detecting a resonator within provided" + \
                  " frequency interval range of the VNA" + \
                  str(self._res_find_interval)
            print(
                msg + " " + self._info_suffix.format(
                    self._flux_control_parameter
                ),
                flush=True
            )

        # turn off second tone (for clarity)
        self._mw_src[0].set_output_state("OFF")
        # turn on VNA output
        self._vna[0].set_output_state("ON")
        vna_res_scan_params = copy.deepcopy(self._vna_pars)
        vna_res_scan_params["freq_limits"] = self._res_find_interval
        vna_res_scan_params["bandwidth"] = self._res_find_bw
        vna_res_scan_params["averages"] = self._res_find_avgs
        vna_res_scan_params["nop"] = self._res_find_nop

        # prepare VNA to resonator detection
        self._vna[0].set_parameters(vna_res_scan_params)
        result = super()._detect_resonator(
            method=self._res_find_method,
            plot=plot, tries_number=tries_number
        )
        if result is None:
            raise Exception("Failed to fit resonator")
        else:
            self._last_resonator_result = result
        # unwrap result
        res_freq, res_amp, res_phase = result

        if print_info:
            # print progress message
            print(
                "Detected if_freq is "
                "{:.5f} GHz, at {:.2f} mU and {:.2f} degrees".format(
                    res_freq / 1e9, res_amp * 1e3, res_phase / np.pi * 180
                )
            )

        # turn on second tone
        self._mw_src[0].set_output_state("ON")

        # set VNA for microwave source frequency sweep
        self._vna_pars["freq_limits"] = (res_freq, res_freq)
        self._vna[0].set_parameters(self._vna_pars)

    def _configure_vna_trg(self):
        """
        Configures `vna` trigger output
        Only supports following devices:
            vna - `Agilent_PNA_L`
        """
        # set trigger output parameters for vna
        # trigger output once per sweep point
        self._vna_pars["trig_per_point"] = True
        self._vna_pars["pos"] = True  # positive edge
        # trigger sent before measurement is started
        self._vna_pars["bef"] = False

    def _configure_mw_src_trg(self):
        """
        Configure microwave source sweep
        """
        # Number of points in sweep has to be equal the number of points of
        # VNA sweep
        self._mw_src_pars.freq_nop = self._vna_pars["nop"]
        # `mw_src_parameters` is filled by user
        # mode should be "MW_SRC_MODE.SWEEP_FREQ_STEP_LINEAR"
        self._mw_src[0].set_parameters(self._mw_src_pars)

    def _adaptive_setter(self, value):
        self._set_swept_par(value)

        # debug message
        # print("\rDetecting a resonator within provided frequency range of the
        # VNA %s" % (str(vna_res_find_parameters["freq_limits"])), flush=True, end="")

        self._detect_resonator_and_set_vna()

        # debug message
        # print(
        #     "\rDetected if_freq is %.5f GHz, at %.2f mU and %.2f "
        #     "degrees" % (res_freq/1e9, res_amp*1e3, res_phase/np.pi*180),
        #     end=""
        # )

        if self._mw_triggered_by_vna:
            self._mw_src[0].arm_sweep()


class TTSResultBase(SParMeas2D):
    """
    Base class for results of two-tone spectroscopies results vizualization.
    Vertical axis is always scanning frequency. Horizontal axis may change 
    depending on the user's desire:
    - TTS from flux through qubit (mainly from bias/voltage of the source).
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
        self.mw_triggered_by_vna = False

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
                                                 freqs[np.argmax(row)], 10e6))[
                    0]
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
                curve_fit(self._tr_spectrum, x, y,
                          p0=(np.mean(x), max(y), np.ptp(x)))[0]
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
    #     return data[self._parameter_names[0]], data[TTSBase.SCAN_FREQUENCY_CAPTION] / 1e9, data["data"]

    def _prepare_data_for_plot(self, data):
        if not self.mw_triggered_by_vna:
            s_data = data["data"][:, :, 0]
        else:
            s_data = data["data"]
        parameter_list = data[self._parameter_names[0]]
        return [
            parameter_list,
            data[TTSBase.SCAN_FREQUENCY_CAPTION] / 1e9,
            s_data
        ]

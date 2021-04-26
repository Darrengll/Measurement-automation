"""
Paramatric single-tone spectroscopy is perfomed with a Vector Network Analyzer
(VNA) for each parameter value which is set by a specific function that must be
passed to the SingleToneSpectroscopy class when it is created.

Minimal working usage example in Jupyter Notebook:
TODO: add reference to minimal working schematic (link to GDrive is ok).
-------------------------------------
from drivers.Yokogawa_GS200 import Yokogawa_GS210
curr_src = Yokogawa_GS210("GS210_1")
curr_src.set_src_mode_curr()  # set current source mode
curr_src.set_range(1e-3)  # set 1 mA range regime

from drivers.agilent_PNA_L import Agilent_PNA_L
vna = Agilent_PNA_L("PNA-L2")
-------------------------------------
from lib2.SingleToneSpectroscopy import SingleToneSpectroscopy

sts = SingleToneSpectroscopy("STS QOP2 Probe qubit", sample_name, vna=[vna], src=[curr_src])
vna.select_S_param("S21")
vna.set_output_state("ON")
q_freq = 5.2185e9  #Hz
span = 30e6  # Hz
freq_limits = (q_freq - span/2, q_freq + span/2)
currents = np.linspace(-0.1e-3, 0.1e-3, 21)  # Amper
vna_params_q ={
    "bandwidth": 100,  # Hz
    "power": -45,  # dBm
    "averages": 1,
    "nop": 1001,
    "freq_limits": freq_limits
}
sts.set_fixed_parameters(vna_params=[vna_params_q])
sts.sweep_current(currents)
---------------------------------------------------
sts_res = sts.launch()
sts_res.visualize()
sts_res.save()
"""
import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt
from lib2.MeasurementResult import MeasurementResult, ContextBase
from matplotlib import colorbar
from lib2.Measurement import Measurement
from time import sleep

from lib2.directMeasurements.digitizerTimeResolvedDirectMeasurement import DigitizerTimeResolvedDirectMeasurement


class SingleToneSpectroscopy(Measurement):
    """
    Class provides all the necessary methods for single-tone spectroscopy with VNA.

    must-have keywords for constructor:
        vna = list of vector network analyzers classes or internal aliases
        src = list of voltage/current sources classes or internal aliases

        For internal aliases/classes see Measurement._devs_dict dictionary in lib2/Measurement.py
    """

    def __init__(self, name, sample_name, plot_update_interval=5, vna=None,
                 src=None):
        """

        Parameters
        ----------
        name
        sample_name
        plot_update_interval
        vna : list
            vna = list of vector network analyzers classes or internal aliases
        src : list
            src = list of voltage/current sources classes or internal aliases
        """
        self._vna = None  # vector network analyzers list
        self._src = None  # voltage/current sources list
        devs_aliases_map = {"vna": vna, "src": src}
        super().__init__(name, sample_name, devs_aliases_map, plot_update_interval)
        self._measurement_result = SingleToneSpectroscopyResult(name, sample_name)
        self._measurement_result.set_unwrap_phase(True)
        self._frequencies = []

    def set_fixed_parameters(self, vna_params=[]):
        """

        Parameters
        ----------
        vna_params : list[dict]
            list with dictionary of parameters for each `vna`
            from self._vna list

        Returns
        -------
        None
        """
        freq_limits = vna_params[0]["freq_limits"]
        nop = vna_params[0]["nop"]

        self._frequencies = np.linspace(*freq_limits, nop)
        self._vna[0].sweep_hold()
        self._vna[0].set_output_state("ON")
        dev_params = {"vna": vna_params}
        super().set_fixed_parameters(**dev_params)

    def set_swept_parameters(self, swept_parameter):
        """
        SingleToneSpectroscopy only takes one swept parameter in format
        {"parameter_name":(setter, values)}
        """
        super().set_swept_parameters(**swept_parameter)
        par_name = list(swept_parameter.keys())[0]
        par_setter, par_values = swept_parameter[par_name]
        # NOTE: first value of the current/voltage source is set by
        # DISCONTINUOUS jump to this starting value
        par_setter(par_values[0])
        sleep(1)

    def sweep_current(self, currents):
        """
        SingleToneSpectroscopy only takes one swept parameter in format
        {"parameter_name":(setter, values)}
        """
        swept_parameters = {"current, [A]": (self._src[0].set_current,
                                            currents)}
        super().set_swept_parameters(**swept_parameters)
        par_name = list(swept_parameters.keys())[0]
        par_setter, par_values = swept_parameters[par_name]
        # NOTE: first value of the current/voltage source is set by
        # DISCONTINUOUS jump to this starting value
        par_setter(par_values[0])
        sleep(1)

    def _recording_iteration(self):
        vna = self._vna[0]
        vna.avg_clear()
        vna.prepare_for_stb()
        vna.do_set_average(True) #

        vna.sweep_single()

        vna.wait_for_stb()
        return vna.get_sdata()

    def _prepare_measurement_result_data(self, parameter_names, parameters_values):
        measurement_data = super()._prepare_measurement_result_data(parameter_names, parameters_values)
        measurement_data["Frequency [Hz]"] = self._frequencies
        return measurement_data

    def _finalize(self):
        for src in self._src:
            if((hasattr(src, "set_current")) and ( src._visainstrument.query(
                    ":SOUR:FUNC?") == "VOLT\n" )):  # voltage src
                src.set_voltage(0)


class SingleToneSpectroscopyResult(MeasurementResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._context = ContextBase()
        self._is_finished = False
        self._phase_units = "rad"
        self.max_phase = -1
        self.min_phase = 1
        self._plot_limits_fixed = False
        self.max_abs = 1
        self.min_abs = 0
        self._unwrap_phase = False
        self._amps_map = None
        self._phas_map = None
        self._amp_cb = None
        self._phas_cb = None

        self._amps_map = None
        self._phas_map = None
        self._amp_cb = None
        self._phas_cb = None

    def _prepare_figure(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True, sharex=True)
        ax_amps, ax_phas = axes
        ax_amps.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_amps.set_ylabel("Frequency [GHz]")
        xlabel = self._parameter_names[0]
        ax_amps.set_xlabel(xlabel)
        ax_phas.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_phas.set_xlabel(xlabel)
        plt.tight_layout(pad=2, h_pad=-10)
        cax_amps, kw = colorbar.make_axes(ax_amps, aspect=40)
        cax_phas, kw = colorbar.make_axes(ax_phas, aspect=40)
        cax_amps.set_title("$|S_{21}|$", position=(0.5, -0.05))
        cax_phas.set_title("$\\angle S_{21}$\n [%s]" % self._phase_units,
                           position=(0.5, -0.1))
        ax_amps.grid()
        ax_phas.grid()
        fig.canvas.set_window_title(self._name)
        return fig, axes, (cax_amps, cax_phas)

    def set_phase_units(self, units):
        """
        Sets the units of the phase in the plots

        Parameters:
        -----------
        units: "rad" or "deg"
            units in which the phase will be displayed
        """
        if units in ["deg", "rad"]:
            self._phase_units = units
        else:
            print("Phase units invalid")

    def set_unwrap_phase(self, unwrap_phase):
        """
        Set if the phase plot should be unwrapped

        Parameters:
        -----------
        unwrap_phase: boolean
            True or False to control the unwrapping
        """
        self._unwrap_phase = unwrap_phase

    def _plot(self, data):

        ax_amps, ax_phas = self._axes
        cax_amps, cax_phas = self._caxes

        if "data" not in data.keys():
            return

        X, Y, Z = self._prepare_data_for_plot(data)
        phases = abs(np.angle(Z).T)

        phases[Z.T == 0] = 0
        phases = phases if self._phase_units == "rad" else phases * 180 / np.pi

        if self._plot_limits_fixed is False:
            self.max_abs = max(abs(Z)[abs(Z) != 0])
            self.min_abs = min(abs(Z)[abs(Z) != 0])
            self.max_phase = max(phases[phases != 0])
            self.min_phase = min(phases[phases != 0])

        step_x = np.min(np.abs(np.diff(X)))
        step_y = np.min(np.abs(np.diff(Y)))
        extent = [np.min(X) - step_x / 2, np.max(X) + step_x / 2,
                  np.min(Y) - step_y / 2, np.max(Y) + step_y / 2]
        if self._amps_map is None or not self._dynamic:
            self._amps_map = ax_amps.imshow(abs(Z).T, origin='lower', cmap="RdBu_r",
                                            aspect='auto', vmax=self.max_abs, vmin=self.min_abs,
                                            extent=extent, interpolation='none')
            self._amp_cb = plt.colorbar(self._amps_map, cax=cax_amps)
            self._amp_cb.formatter.set_powerlimits((0, 0))
            self._amp_cb.update_ticks()
        else:
            self._amps_map.set_data(abs(Z).T)
            self._amps_map.set_clim(self.min_abs, self.max_abs)

        if self._phas_map is None or not self._dynamic:
            self._phas_map = ax_phas.imshow(phases, origin='lower', aspect='auto',
                                            cmap="RdBu_r", vmin=self.min_phase, vmax=self.max_phase,
                                            extent=extent, interpolation='none')
            self._phas_cb = plt.colorbar(self._phas_map, cax=cax_phas)
        else:
            self._phas_map.set_data(phases)
            self._phas_map.set_clim(self.min_phase, self.max_phase)
            plt.draw()

    def set_plot_range(self, min_abs, max_abs, min_phas=None, max_phas=None):
        self.max_phase = max_phas
        self.min_phase = min_phas
        self.max_abs = max_abs
        self.min_abs = min_abs

    def _prepare_data_for_plot(self, data):
        s_data = self._remove_delay(data["Frequency [Hz]"], data["data"])
        parameter_list = data[self._parameter_names[0]]
        # if parameter_list[0] > parameter_list[-1]:
        #     parameter_list = parameter_list[::-1]
        #     s_data = s_data[::-1, :]
        # s_data = self.remove_background('avg_cur')
        return parameter_list, data["Frequency [Hz]"] / 1e9, s_data

    def remove_delay(self):
        copy = self.copy()
        s_data, frequencies = copy.get_data()["data"], copy.get_data()["Frequency [Hz]"]
        copy.get_data()["data"] = self._remove_delay(frequencies, s_data)
        return copy

    def _remove_delay(self, frequencies, s_data):
        phases = np.unwrap(np.angle(s_data))
        k, b = np.polyfit(frequencies, phases[0], 1)
        phases = phases - k * frequencies - b
        corr_s_data = abs(s_data) * np.exp(1j * phases)
        corr_s_data[abs(corr_s_data) < 1e-14] = 0
        return corr_s_data

    def remove_background(self, direction):
        """
        Remove background

        Parameters:
        -----------
        direction: str
            "avg_cur" for current slice subtraction
            "avg_freq" for frequency slice subtraction

        """
        s_data = self.get_data()["data"]
        len_freq = s_data.shape[1]
        len_cur = s_data.shape[0]
        if direction is "avg_cur":
            avg = np.zeros(len_freq, dtype=complex)
            for j in range(len_freq):
                counter_av = 0
                for i in range(len_cur):
                    if s_data[i, j] != 0:
                        counter_av += 1
                        avg[j] += s_data[i, j]
                avg[j] = avg[j] / counter_av
                s_data[:, j] = s_data[:, j] / avg[j]
        elif direction is "avg_freq":
            avg = np.zeros(len_cur, dtype=complex)
            for j in range(len_cur):
                counter_av = 0
                for i in range(len_freq):
                    if s_data[j, i] != 0:
                        counter_av += 1
                        avg[j] += s_data[j, i]
                avg[j] = avg[j] / counter_av
                s_data[j, :] = s_data[j, :] / avg[j]

        self.get_data()["data"] = s_data
        return s_data


    def __setstate__(self, state):
        self._amps_map = None
        self._phas_map = None
        super().__setstate__(state)

    def __getstate__(self):
        d = super().__getstate__()
        d["_amps_map"] = None
        d["_phas_map"] = None
        d["_amp_cb"] = None
        d["_phas_cb"] = None
        return d


# Not tested yet
class SingleToneSpectroscopy2(DigitizerTimeResolvedDirectMeasurement):
    """
    Class provides all the necessary methods for single-tone spectroscopy with a digitizer,
    IQ-mixers, an AWG and a RF generator.

    must-have keywords for constructor:
        dig = list of digitizer classes or internal aliases
        q_iqawg = list of IQ AWG
        q_lo = lost of RF generators
        src = list of voltage/current sources classes or internal aliases

        For internal aliases/classes see Measurement._devs_dict dictionary in lib2/Measurement.py
    """

    def __init__(self, name, sample_name, plot_update_interval=5, **devs_aliases_map):
        self._src = None  # voltage/current sources list
        super().__init__(name, sample_name, devs_aliases_map, plot_update_interval)
        self._measurement_result = SingleToneSpectroscopyResult(name, sample_name)
        self._frequencies = []
        self._rf_generator_delay = 0
        self._iqawg_sequence = None

    def set_fixed_parameters(self, freq_limits, nop, **dev_params):
        """
        SingleToneSpectroscopy only requires vna parameters in format
        {"bandwidth":int, ...}
        """
        super().set_fixed_parameters(None, dev_params["q_lo"], dev_params["q_iqawg"])
        self._measurement_result.get_context().update({
            "calibration_results": self._q_iqawg[0]._calibration.get_optimization_results(),
            "radiation_parameters": self._q_iqawg[0]._calibration.get_radiation_parameters(),
            "pulse_sequence_parameters": None
        })
        self._frequencies = np.linspace(*freq_limits, nop)
        self._rf_generator_delay = dev_params["q_lo"][0]["frequency_switching_delay"]
        self._Nfft = fftpack.next_fast_len(self._dig._segment_size)
        self._iqawg_sequence = self._q_iqawg.get_pulse_builder().add_sine_pulse(int(self._Nfft / 1e9)).build()

    def set_swept_parameters(self, swept_parameter):
        """
        SingleToneSpectroscopy only takes one swept parameter in format
        {"parameter_name":(setter, values)}
        """
        super().set_swept_parameters(**swept_parameter)
        par_name = list(swept_parameter.keys())[0]
        par_setter, par_values = swept_parameter[par_name]
        par_setter(par_values[0])
        sleep(1)

    def _recording_iteration(self):
        self._q_iqawg.output_pulse_sequence()
        trace = np.empty_like(self._frequencies, dtype=np.complex)
        for i, freq in enumerate(self._frequencies):
            self._q_lo.set_frequency(freq)
            sleep(self._rf_generator_delay)
            trace[i] = self._single_measurement()
        return trace

    def _prepare_measurement_result_data(self, parameter_names, parameters_values):
        measurement_data = super()._prepare_measurement_result_data(parameter_names, parameters_values)
        measurement_data["Frequency [Hz]"] = self._frequencies
        return measurement_data

    def _finalize(self):
        for src in self._src:
            if((hasattr(src, "set_current")) and ( src._visainstrument.ask(":SOUR:FUNC?") == "VOLT\n" )):  # voltage src
                src.set_voltage(0)
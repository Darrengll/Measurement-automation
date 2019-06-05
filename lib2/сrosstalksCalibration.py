from numpy import *
from lib2.Measurement import *
from lib2.MeasurementResult import *
from datetime import datetime as dt
from enum import Enum
from time import sleep


class CrosstalksCalibrationBase(Measurement):


    def __init__(self, name, sample_name, devs_aliases_map, plot_update_interval=5):

        super().__init__(name, sample_name, devs_aliases_map,
                         plot_update_interval)

        self._measurement_result = CrosstalksCalibrationResult(name, sample_name)
        self._interrupted = False
        self._base_parameter_setter = None
        self._last_resonator_result = None


    def set_fixed_parameters(self, detect_resonator = True, bandwidth_factor=1, **dev_params):

        vna_parameters = dev_params['vna'][0]
        mw_src_parameters = dev_params['mw_src'][0]
        self._frequencies = mw_src_parameters["freq_limits"]

        if "ext_trig_channel" in mw_src_parameters.keys():
            # internal adjusted trigger parameters for vna
            vna_parameters["trig_per_point"] = True  # trigger output once per sweep point
            vna_parameters["pos"] = True  # positive edge
            vna_parameters["bef"] = False  # trigger sent before measurement is started

            # internal adjusted trigger parameters for microwave source
            mw_src_parameters["unit"] = "Hz"
            mw_src_parameters["InSweep_trg_src"] = "EXT"
            mw_src_parameters["sweep_trg_src"] = "BUS"

        self._bandwidth_factor = bandwidth_factor

        if detect_resonator:
            self._mw_src[0].set_output_state("OFF")
            msg = "Detecting a resonator within provided frequency range of the VNA %s \
                            " % (str(vna_parameters["freq_limits"]))
            print(msg, flush=True)
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
            swept_pars[swept_par][0] = setter_function

        super().set_swept_parameters(**swept_pars)

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
        self._vna[0].set_bandwidth(vna_parameters["bandwidth"] * self._bandwidth_factor)
        self._vna[0].set_averages(vna_parameters["averages"])
        result = super()._detect_resonator(plot)
        self._vna[0].do_set_power(vna_parameters["power"])
        self._vna[0].do_set_power(vna_parameters["nop"])
        return result

    def _recording_iteration(self):
        vna = self._vna[0]
        vna.avg_clear()
        vna.prepare_for_stb()
        vna.sweep_single()
        vna.wait_for_stb()
        data = vna.get_sdata()
        return data

class CrosstalksCalibration(CrosstalksCalibrationBase):

    def __init__(self, name, sample_name, **devs_aliases_map):
        super().__init__(name, sample_name, devs_aliases_map)

    def set_fixed_parameters(self, bandwidth_factor=10, **dev_params):

        vna_parameters = dev_params['vna'][0]
        mw_src_parameters = dev_params['mw_src'][0]
        self._resonator_area = vna_parameters["freq_limits"]
        self._adaptive = False

        super().set_fixed_parameters(vna=dev_params['vna'], mw_src=dev_params['mw_src'],
                                     detect_resonator= True,
                                     bandwidth_factor=bandwidth_factor)

    def set_swept_parameters(self, first_line_voltages, second_line_voltages):
        setter = self._base_parameter_setter
        swept_pars = {'First_voltage [V]':(setter, first_line_voltages),
        'Second_voltage [V]':(setter, second_line_voltages)}
        super().set_swept_parameters(**swept_pars)


class CrosstalksCalibrationResult(MeasurementResult):

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
        self._annotation_bbox_props = dict(boxstyle="round", fc="white",
                                           ec="black", lw=1, alpha=0.5)

    def _prepare_figure(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True, sharex=True)
        ax_amps, ax_phas = axes
        ax_amps.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_amps.set_ylabel("Second voltage [V]")
        ax_amps.set_xlabel("First voltage [V]")
        ax_phas.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_phas.set_xlabel("First voltage [V]")
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
        phases = abs(angle(Z).T) if not self._unwrap_phase else unwrap(angle(Z)).T
        phases[Z.T == 0] = 0
        phases = phases if self._phase_units == "rad" else phases * 180 / pi

        if self._plot_limits_fixed is False:
            self.max_abs = max(abs(Z)[abs(Z) != 0])
            self.min_abs = min(abs(Z)[abs(Z) != 0])
            self.max_phase = max(phases[phases != 0])
            self.min_phase = min(phases[phases != 0])

        step_x = X[1] - X[0]
        step_y = Y[1] - Y[0]
        extent = [X[0] - step_x / 2, X[-1] + step_x / 2, Y[0] - step_y / 2, Y[-1] + step_y / 2]
        if self._amps_map is None or not self._dynamic:
            self._amps_map = ax_amps.imshow(abs(Z).T, origin='lower', cmap="RdBu_r",
                                            aspect='auto', vmax=self.max_abs, vmin=self.min_abs,
                                            extent=extent)
            self._amp_cb = plt.colorbar(self._amps_map, cax=cax_amps)
            self._amp_cb.formatter.set_powerlimits((0, 0))
            self._amp_cb.update_ticks()
        else:
            self._amps_map.set_data(abs(Z).T)
            self._amps_map.set_clim(self.min_abs, self.max_abs)
            self._amp_cb.set_clim(self.min_abs, self.max_abs)

        if self._phas_map is None or not self._dynamic:
            self._phas_map = ax_phas.imshow(phases, origin='lower', aspect='auto',
                                            cmap="RdBu_r", vmin=self.min_phase, vmax=self.max_phase,
                                            extent=extent)
            self._phas_cb = plt.colorbar(self._phas_map, cax=cax_phas)
        else:
            self._phas_map.set_data(phases)
            self._phas_map.set_clim(self.min_phase, self.max_phase)
            self._phas_cb.set_clim(self.min_phase, self.max_phase)
            plt.draw()


    def set_plot_range(self, min_abs, max_abs, min_phas=None, max_phas=None):
        self.max_phase = max_phas
        self.min_phase = min_phas
        self.max_abs = max_abs
        self.min_abs = min_abs

    def _prepare_data_for_plot(self, data):
        s_data = data["data"]
        current1 = data["First_voltage [V]"]
        current2 = data["Second_voltage [V]"]
        return current1, current2, s_data


    def remove_background(self, direction):
        """
        Remove background

        Parameters:
        -----------
        direction: str
            "h" for horizontal slice subtraction
            "v" for vertical slice subtraction

        """
        s_data = self.get_data()["data"]
        len_freq = s_data.shape[1]
        len_cur = s_data.shape[0]
        if direction is "avg_cur":
            avg = zeros(len_freq, dtype=complex)
            for j in range(len_freq):
                counter_av = 0
                for i in range(len_cur):
                    if s_data[i, j] != 0:
                        counter_av += 1
                        avg[j] += s_data[i, j]
                avg[j] = avg[j] / counter_av
                s_data[:, j] = s_data[:, j] / avg[j]
        elif direction is "avg_freq":
            avg = zeros(len_cur, dtype=complex)
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

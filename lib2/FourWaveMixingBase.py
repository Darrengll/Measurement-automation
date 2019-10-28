from datetime import datetime as dt
from matplotlib import pyplot as plt, colorbar
import numpy as np
from numpy.core.numeric import inf

from lib2.digitizerWithPowerSweepMeasurementBase import DigitizerWithPowerSweepMeasurementBase
from lib2.MeasurementResult import *
from lib2.Measurement import *

class FourWaveMixingBase(DigitizerWithPowerSweepMeasurementBase):
    """
    Class for wave mixing measurements.

    This one must do:
        create Measurement object, set up all devices and take them from the class;
        set up all the parameters
        make measurements:
         -- sweep power/frequency of one/another/both of generators
            and/or central frequency of EXA and measure single trace / list sweep for certain frequencies
         --
    """

    def __init__(self, name, sample_name, **devs_aliases):
        """
        name: name of current measurement
        list_devs_names: {exa_name: default_name, src_plus_name: default_name,
                             src_minus_name: default_name, vna_name: default_name, current_name: default_name}
        sample_name: name of measured sample

        vna and current source is optional

        """
        super().__init__(name, sample_name, FourWaveMixingResult, **devs_aliases)

    def set_fixed_parameters(self, delta, awg_parameters=[], adc_parameters=[], freq_limits=(), lo_parameters=[]):
        """

        Parameters
        ----------
        delta : float
            half of distance between two generators
        awg_parameters
        adc_parameters: dict
            "channels" : [1], # a list of channels to measure
            "ch_amplitude": 200, # amplitude for every channel
            "dur_seg": 100e-6, # duration of a segment in us
            "n_avg": 80000, # number of averages
            "n_seg": 2, # number of segments
            "oversampling_factor": 2, # sample_rate = max_sample_rate / oversampling_factor
            "pretrigger": 32,
        freq_limits
        lo_parameters

        Returns
        -------
        None
        """
        self._delta = delta
        self._measurement_result._delta = delta / 1e6
        super().set_fixed_parameters("CONTINUOUS TWO WAVES FG", awg_parameters, adc_parameters, freq_limits, lo_parameters)

class MollowTrippletMeasurementBase(DigitizerWithPowerSweepMeasurementBase):
    def __init__(self, name, sample_name, **devs_aliases):
        """
        name: name of current measurement
        list_devs_names: {exa_name: default_name, src_plus_name: default_name,
                             src_minus_name: default_name, vna_name: default_name, current_name: default_name}
        sample_name: name of measured sample

        vna and current source is optional

        """
        super().__init__(name, sample_name, MollowTripletResult, **devs_aliases)

    def set_fixed_parameters(self, awg_parameters=[], adc_parameters=[], freq_limits=(), lo_parameters=[]):
        """
         adc_parameters: dictionary
          {
            "channels" :    [1], # a list of channels to measure
            "ch_amplitude":    200, # amplitude for every channel
            "dur_seg":    100e-6, # duration of a segment in us
            "n_avg":    80000, # number of averages
            "n_seg":    2, # number of segments
            "oversampling_factor":    2, # sample_rate = max_sample_rate / oversampling_factor
            "pretrigger": 32,
          }
        """
        super().set_fixed_parameters("CONTINUOUS WAVE", awg_parameters, adc_parameters, freq_limits, lo_parameters)

class FourWaveMixingResult(MeasurementResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        # self._context = ContextBase(comment=input('Enter your comment: '))
        self._context = ContextBase()
        self._is_finished = False
        self._idx = []
        self._midx = []
        self._colors = []
        self._XX = None
        self._YY = None
        self._delta = 0
        self._iter = 0

    def set_parameter_name(self, parameter_name):
        self._parameter_name = parameter_name

    def _prepare_figure(self):
        self._last_tr = None
        self._peaks_last_tr = None
        fig = plt.figure(figsize=(19, 8))
        ax_trace = plt.subplot2grid((4, 8), (0, 0), colspan=4, rowspan=1)
        ax_map = plt.subplot2grid((4, 8), (1, 0), colspan=4, rowspan=3)
        ax_peaks = plt.subplot2grid((4, 8), (0, 4), colspan=4, rowspan=4)
        plt.tight_layout()
        ax_map.ticklabel_format(axis='x', style='plain', scilimits=(-2, 2))
        ax_map.set_ylabel("Frequency, MHz")
        ax_map.set_xlabel(self._parameter_name[0].upper() + self._parameter_name[1:])
        ax_trace.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_trace.set_xlabel("Frequency, MHz")
        ax_trace.set_ylabel("Emission power, dBm")
        ax_peaks.set_xlabel("Input power, dBm")
        ax_peaks.set_ylabel("Emission power, dBm")

        ax_map.autoscale_view(True, True, True)
        plt.tight_layout()

        cax, kw = colorbar.make_axes(ax_map, fraction=0.05, anchor=(0.0, 1.0))
        cax.set_title("$P$,dBm")

        return fig, (ax_trace, ax_map, ax_peaks), (cax,)

    def _plot(self, data):
        ax_trace, ax_map, ax_peaks = self._axes
        cax = self._caxes[0]
        if "data" not in data.keys():
            return

        XX, YY, Z, P_pos, P_neg = self._prepare_data_for_plot(data)

        max_pow = np.max(Z[Z != -inf])
        min_pow = np.min(Z[Z != -inf])
        av_pow = np.average(Z[Z != -inf])
        extent = [XX[0], XX[-1], YY[0], YY[-1]]
        pow_map = ax_map.imshow(Z.T, origin='lower', cmap="inferno",
                                aspect='auto', vmax=max_pow,
                                vmin=av_pow, extent=extent)
        cax.cla()
        plt.colorbar(pow_map, cax=cax)
        cax.tick_params(axis='y', right='off', left='on',
                        labelleft='on', labelright='off', labelsize='10')
        last_trace_data = Z[Z != -inf][-(len(data["frequency"])):]
        if self._last_tr is not None:
            self._last_tr.remove()
        self._last_tr = ax_trace.plot(data["frequency"], last_trace_data, 'b').pop(0)

        N_peaks = len(P_pos[0, :])
        ax_peaks.cla()
        self._peaks_last_tr = [ax_peaks.plot(data["powers at $\\omega_{p}$"],
                                             (P_pos[:, i]+P_neg[:,i])/2, '-', linewidth=1.5) for i in range(N_peaks)]
        self._colors = [ax_peaks.get_lines()[i].get_color() for i in range(N_peaks)]
        # self._peaks_last_tr.append([ax_peaks.plot(data["powers at $\\omega_{p}$"],
        #                                      P_neg[:, i], '-', linewidth=1.5,
        #                                      color=self._colors[i]) for i in range(N_peaks)])

        ax_trace.set_ylim([np.average(last_trace_data), np.max(last_trace_data)])
        P1 = P_pos[:, 1:]
        ax_peaks.set_ylim([P1[P1 != -inf].min(), P1[P1 != -inf].max()])

        ax_map.grid('on')
        ax_trace.grid('on')
        ax_trace.axis("tight")

    def _prepare_data_for_plot(self, data):
        if self._idx == []:
            max_order = 10
            con_eq = self.get_context().get_equipment()
            central_freq = con_eq['iqawg'][0]['calibration'].get_radiation_parameters()['if_frequency'] / 1e6
            freqs = np.array([self._delta * (1 + 2 * i) for i in range(max_order)])  # , 220, 260, 300, 340, 380, 420])
            mfreqs = -freqs
            self._idx = np.searchsorted(data["frequency"], central_freq + freqs)
            self._midx = np.searchsorted(data["frequency"], central_freq + mfreqs)
        power_data = np.real(20 * np.log10(data["data"] * 1e3 / np.sqrt(50e-3)))
        pos_peaks_data = power_data[:, self._idx]
        neg_peaks_data = power_data[:, self._midx]
        if self._XX is None and self._YY is None:
            self._XX, self._YY = data[self._parameter_name], data["frequency"]
        return self._XX, self._YY, power_data, np.array(pos_peaks_data), np.array(neg_peaks_data)

class MollowTripletResult(MeasurementResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        # self._context = ContextBase(comment=input('Enter your comment: '))
        self._context = ContextBase()
        self._is_finished = False
        self._idx = []
        self._midx = []
        self._colors = []
        self._XX = None
        self._YY = None
        self._delta = 0
        self._iter = 0

    def set_parameter_name(self, parameter_name):
        self._parameter_name = parameter_name

    def _prepare_figure(self):
        self._last_tr = None
        fig = plt.figure(figsize=(19, 8))
        ax_trace = plt.subplot2grid((4, 1), (0, 0), colspan=1, rowspan=1)
        ax_map = plt.subplot2grid((4, 1),   (1, 0), colspan=1, rowspan=3)
        plt.tight_layout()
        ax_map.ticklabel_format(axis='x', style='plain', scilimits=(-2, 2))
        ax_map.set_ylabel("Frequency, MHz")
        ax_map.set_xlabel(self._parameter_name[0].upper() + self._parameter_name[1:])
        ax_trace.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_trace.set_xlabel("Frequency, MHz")
        ax_trace.set_ylabel("Emission power, dBm")

        ax_map.autoscale_view(True, True, True)
        plt.tight_layout()

        cax, kw = colorbar.make_axes(ax_map, fraction=0.05, anchor=(0.0, 1.0))
        cax.set_title("$P$,dBm")

        return fig, (ax_trace, ax_map), (cax,)

    def _plot(self, data):
        ax_trace, ax_map = self._axes
        cax = self._caxes[0]
        if "data" not in data.keys():
            return

        XX, YY, Z = self._prepare_data_for_plot(data)

        max_pow = np.max(Z[Z != -inf])
        min_pow = np.min(Z[Z != -inf])
        av_pow = np.average(Z[Z != -inf])
        extent = [XX[0], XX[-1], YY[0], YY[-1]]
        pow_map = ax_map.imshow(Z.T, origin='lower', cmap="inferno",
                                aspect='auto', vmax=max_pow,
                                vmin=av_pow, extent=extent)
        cax.cla()
        plt.colorbar(pow_map, cax=cax)
        cax.tick_params(axis='y', right='off', left='on',
                        labelleft='on', labelright='off', labelsize='10')
        last_trace_data = Z[Z != -inf][-(len(data["frequency"])):]
        if self._last_tr is not None:
            self._last_tr.remove()
        self._last_tr = ax_trace.plot(data["frequency"], last_trace_data, 'b').pop(0)

        ax_trace.set_ylim([np.average(last_trace_data), np.max(last_trace_data)])

        ax_map.grid('on')
        ax_trace.grid('on')
        ax_trace.axis("tight")

    def _prepare_data_for_plot(self, data):
        power_data = np.real(20 * np.log10(data["data"] * 1e3 / np.sqrt(50e-3)))
        if self._XX is None and self._YY is None:
            self._XX, self._YY = data[self._parameter_name], data["frequency"]
        return self._XX, self._YY, power_data
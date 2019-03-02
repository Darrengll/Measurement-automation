from lib2.Measurement import Measurement
from lib2.MeasurementResult import MeasurementResult
from lib2.DispersiveRabiOscillations import DispersiveRabiOscillations

from copy import deepcopy

import numpy as np
from itertools import product
from functools import reduce
from operator import mul
from scipy.optimize import minimize

import matplotlib.pyplot as plt

from datetime import datetime as dt
from threading import Thread
import traceback


from importlib import reload
from . import structures
reload(structures)
from .structures import Snapshot

class DispersiveRabiFromFrequency(Measurement):
    '''
    @brief: class is used to measure qubit lifetimes from the flux/qubit frequency
            displacement from the sweet-spot.

            Measurement setup is the same as for the any other dispersive measurements.
    '''
    def __init__(self, name, sample_name,
                 vna, q_lo, ro_iqawg, q_iqawg, ss_current_or_voltage, ss_freq,
                 tts_result, lowest_ss=True, current_source=None, q_z_awg=None,
                 plot_update_interval=5):
        '''
        @params:
            name: string.
            sample_name: string.
            vna: alias address string or driver class
                vector network analyzer.
            q_lo: alias address string or driver class
                qubit frequency generator for lo input of the mixer.
            ro_iqawg: IQAWG class instance
                    AWG used to control readout pulse generation mixer
            q_iqawg: IQAWG class instance
                    AWG used to control qubit excitation pulse generation mixer
            ss_current_or_voltage: float
                    sweet spot DC current or voltage depending on wether
                    current source or AWG is used to bias qubit flux
            ss_freq: float
                    frequency of the qubit in the sweet-spot of interest
            lowest_ss: bool
                    sign of the second derivative of frequency on flux shift variable
                    if sign is positive, then this is a lower sweet-spot
                        and lower_ss=True
                    if sign is negative -> lower_ss = False

            One of the following DC sources must be provided:
            current_source: alias address string or driver class
                            current source used to tune qubit frequency
            q_z_awg: alias address string or driver class
                     AWG generator that used to tune qubit frequency


            plot_update_interval: float
                                sleep milliseconds between plot updates
        '''
        ## Equipment variables declaration section START ##
        self._vna = None
        self._q_lo = None
        self._current_source = None
        self._q_z_awg = None
        self._ro_awg = None
        self._q_awg = None
        ## Equipment variables declaration section END ##

        # constructor initializes devices from kwargs.keys() with '_' appended
        # keys must coincide with the attributes introduced in
        # "equipment variables declaration section"
        # TODO: set_fixed_parmaeters in DRO class uses q_awg and ro_awg names instead of
        # more proper ro_iqawg and q_iqawg
        devs_aliases_map = {"vna": vna, "q_lo": q_lo, "current_source": current_source,
                            "q_z_awg": q_z_awg, "ro_awg": ro_iqawg, "q_awg": q_iqawg}
        super().__init__(name, sample_name, devs_aliases_map, plot_update_interval)

        ## initializing base class elements with child specific values ##
        self._measurement_result = RabiFromFrequencyResult(name, sample_name, self)

        # last successful two tone spectroscopy result
        # that contains sweet-spot in its area
        # as well as all the qubit frequencies that
        # are going to be measured in this class
        self._tts_result = tts_result
        self._snap = Snapshot(self._tts_result._data)

        ## Initial and current freq(current or voltage) point control START ##
        self._ss_freq = ss_freq
        self._ss_flux_var_value = ss_current_or_voltage
        self._lowest_ss = lowest_ss
        # True if current is used, False if voltage source is used
        self._current_flag = None
        self._flux_var_setter = None

        self._flux_var = None # flux variable value now
        self._last_flux_var = None # last flux variable value

        # constructor arguments consistency test
        if( current_source is not None ):
            self._current_flag = True
            self._flux_var_setter = self._current_source.set_current
        elif( q_z_awg is not None ):
            self._current_flag = False
            self._flux_var_setter = self._q_z_awg.set_voltage
        else:
            print("RabiFromFreq: You must provide one and only one of the following \
                  constructor parameters:\n \
                  current_source or q_z_awg.")
            raise TypeError
        ## Initial and current freq(current or voltage) point control END ##

        # class that is responsible for rabi measurements
        self._DRO = DispersiveRabiOscillations(name, sample_name, **devs_aliases_map)
        # self._DRO.launch().data will be stored in the following list
        self._DRO_results = []

    def plot_tts_connectivity_map(self,rel_threshold=0.5,
                              kernel_x=0.1, kernel_y=0.1,
                              connectivity=8):
        """
        @brief: This function is ought to be called before the
                self.launch() in order to perform visual control
                of the spectrum fitting
        """
        self._snap._connected_components()
        self._snap.visualize_connectivity_map()

    def set_connectivity_component_index(self, cc_index):
        """
        @brief: This function is ought to be called before the
                self.launch() in order to perform visual control
                of the spectrum fitting.
                Function is called right after the call to the
                self.plot_fft_connectivity_map(...)

        :param cc_index: integer
            index that were chosen manually by operator after
            examining self.plot_fft_connectivity_map(...) output
        :return:    interpolation function y(x) that is returned by
                    scipy.interp1d(...)
        """
        self._snap._make_target_component_mask(label_i=cc_index)
        self._snap._interpolate_yx_curve()
        return self._snap._target_y_func

    def set_fixed_parameters(self, vna_parameters, ro_awg_parameters,
                             q_awg_parameters, qubit_frequency, pulse_sequence_parameters,
                             q_z_awg_params=None, plot_resonator_fit=True):
        # TODO: is resonator detection needed here?
        self._DRO.set_fixed_parameters(vna_parameters, ro_awg_parameters,
                                       q_awg_parameters, qubit_frequency, pulse_sequence_parameters,
                                       q_z_awg_params,
                                       detect_resonator=True, plot_resonator_fit=plot_resonator_fit)

    def set_swept_parameters(self, excitation_durations, ss_shifts):
        '''
        @params:
            excitation_durations - list of the rabi excitation pulse durations
            ss_shifts - list of absolute values of the qubit frequency shift from sweet-spot
        '''
        self._DRO.set_swept_parameters(excitation_durations)
        super().set_swept_parameters(ss_shifts=(self._ss_shift_setter, ss_shifts))

    def set_ult_calib(self, ult_calib):
        # TODO: docstring
        self._DRO.set_ult_calib(ult_calib)

    def detect_tts_lines(self, kernel_x=0.1, kernel_y=0.1, rel_threshold=0.5, connectivity=8):
        # TODO: docstring
        self._snap.make_and_visualize_connectivity_map(rel_threshold=rel_threshold,
                                                       kernel_x=kernel_x,kernel_y=kernel_y,
                                                       connectivity=connectivity)

    def select_tts_line(self, label_i):
        # TODO: docstring
        mask = self._snap.select_connectivity_component(label_i)
        self._snap.interpolate_yx_curve()
        return self._snap._target_y_func

    def _ss_shift_setter(self, ss_freq_shift):
        '''
        @brief: sets new flux bias for a qubit to achieve
                qubit frequency = ss_freq +- ss_freq_shift
                '+' or '-' is depending on the qubit freq(flux_bias)
                function behaviour around sweet_spot value
        '''

        qubit_frequency = self._ss_freq + ss_freq_shift

        # finding value of the new flux variable
        f = self._snap._target_y_func

        def f2min(x):
            return (1e9*f(x) - qubit_frequency)**2  # f(x) is returning frequency value in GHz

        ig_x = None
        if( self._last_flux_var is None ):
            ig_x = self._ss_flux_var_value
        else:
            ig_x = self._last_flux_var

        res = minimize(f2min, ig_x, method="L-BFGS-B", bounds=((f.x[0], f.x[-1]),))
        # setting new flux bias
        self._flux_var = res.x[0]
        self._flux_var_setter(self._flux_var)

        fixed_pars = self._DRO._fixed_pars
        pulse_seq_params = self._DRO._measurement_result.get_context().get_pulse_sequence_parameters()
        q_z_awg_params = None if "q_z_awg" not in fixed_pars else fixed_pars["q_z_awg"]

        # TODO: detecting and setting a new resonator point is not optimized.
        # Propose to collect all neccessary code from the call chain of\
        # self._DRO.set_fixed_parameters
        # setting a new qubit_frequency
        self._DRO.set_fixed_parameters(fixed_pars["vna"],
                                       fixed_pars["ro_awg"], fixed_pars["q_awg"],
                                       qubit_frequency, pulse_seq_params,
                                       q_z_awg_params,
                                       detect_resonator=True, plot_resonator_fit=False)
        self._DRO.set_swept_parameters(self._DRO._swept_pars["excitation_duration"][1])

        self._last_flux_var = self._flux_var
        print("new ss shift is setted")
        print("flux var value: {}\nqubit_frequency: {}".format(self._flux_var, qubit_frequency))

    def _recording_iteration(self):
        # _DRO will detect resonator and new qubit frequency current
        # during the call of the setters
        print("starting rabi\n")
        self._rabi_oscillations_record()
        self._DRO_results.append(deepcopy(self._DRO._measurement_result)) # deepcopy of result is stored
        T_R = self._DRO._measurement_result._fit_params[2] # see DispersiveRabiOscillationsResult._model
        T_R_error = self._DRO._measurement_result._fit_errors[2]

        # clearing previous fit results due to the fact, that VNATimeResolvedDispersiveMeasurement1D._fit_complex_curve(..)
        # when this fit_params are not None
        # almost every time tries to use this parameters as the new best initial guess
        # and due to the fact, that the next measurements is performed in entirely different flux point
        # this initial guess vector does not fit the parameter's fit bounds that are generated from
        # the data of the current measurement
        self._DRO._measurement_result._fit_params = None
        self._DRO._measurement_result._fit_errors = None
        return T_R, T_R_error # Rabi decay time and its RMS is stored in self._raw_data


    def _rabi_oscillations_record(self):
        self._DRO._measurement_result.set_start_datetime(dt.now())
        if self._DRO._measurement_result.is_finished():
            print("Starting with a result from a previous launch")
            self._DRO._measurement_result.set_is_finished(False)
        print("Started at: ", self._DRO._measurement_result.get_start_datetime())

        self._DRO._record_data()
        self._DRO._measurement_result.fit(verbose=False)
        print("DRO._record_data finished\n")
        self._DRO._finalize_measurement()
        return self._DRO._measurement_result

class RabiFromFrequencyResult(MeasurementResult):
    def __init__(self, name, sample_name, measurement_class=None):
        super().__init__(name, sample_name, measurement_class)
        self.fig = None
        self._line_scatter = None
        self.ss_shifts = None

    def _prepare_figure(self):
        import matplotlib.pyplot as plt
        self.fig, axes = plt.subplots(3,1)
        ax = axes[0]  # T_R from \nu plot
        ax.set_xlabel( r"$\delta \nu$, MHz")
        ax.set_ylabel( r"$T_R, \; \mu s$")
        ax.grid()
        self._line_scatter, = ax.plot([], [],'r', marker="o",
                                      markerfacecolor='none',)

        # setting x limit for graph
        self.ss_shifts = self._measurement._swept_pars["ss_shifts"][1]
        ax.set_xlim(self.ss_shifts[0]/1e6, self.ss_shifts[-1]/1e6)


        return self.fig, axes, None

    def _plot(self, axes, caxes, dynamic=False):
        '''
        caxes is None
        '''
        if( self._measurement._raw_data is not None ):
            y_data = self._measurement._raw_data[:, 0]
            x = self.ss_shifts[:len(y_data)]

            ax = axes[0]
            ax.clear()
            ax.set_xlabel(r"$\delta \nu$, MHz")
            ax.set_ylabel(r"$T_R, \; \mu s$")
            ax.grid()
            self._line_scatter, = ax.plot(x[y_data > 0]/1e6, y_data[y_data > 0], 'r', marker="o",
                                          markerfacecolor='none', )

            # setting x limit for graph
            ax.set_xlim(self.ss_shifts[0]/1e6, self.ss_shifts[-1]/1e6)

        # plot rabi result
        self._measurement._DRO._measurement_result._plot(axes[1:3], caxes)





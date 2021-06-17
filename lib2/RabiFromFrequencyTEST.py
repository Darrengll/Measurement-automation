from lib2.Measurement import Measurement
from lib2.MeasurementResult import MeasurementResult
from lib2.DispersiveRabiOscillations import DispersiveRabiOscillations
from lib2.DispersiveRamsey import DispersiveRamsey

from copy import deepcopy
import csv

import numpy as np

from scipy.optimize import minimize
from scipy.interpolate import interp1d

from datetime import datetime as dt


from importlib import reload
from . import structures
reload(structures)
from .structures import Snapshot

class DispersiveRabiFromFrequency(Measurement):
    '''
    @brief: class is used to measure qubit lifetimes from the flux/qubit if_freq
            displacement from the sweet-spot.

            Measurement setup is the same as for the any other dispersive measurements.
    '''
    def __init__(self, name, sample_name,
                 ss_current_or_voltage, ss_freq, tts_result,
                 lowest_ss=True, current_source=[None], q_z_awg=[None],
                 plot_update_interval=5,
                 **devs_aliases_map):
        '''
        @params:
            name: string.
            sample_name: string.

            ss_current_or_voltage: float
                    sweet spot DC bias or voltage depending on wether
                    bias source or AWG is used to bias qubit flux
            ss_freq: float
                    if_freq of the qubit in the sweet-spot of interest
            lowest_ss: bool
                    sign of the second derivative of if_freq on flux shift variable
                    if sign is positive, then this is a lower sweet-spot
                        and lower_ss=True
                    if sign is negative -> lower_ss = False

            dev_aliases_map: dict that contains following key:val pairs
                vna: alias address string or driver class
                    vector network analyzer.
                q_lo: alias address string or driver class
                    qubit if_freq generator for lo input of the mixer.
                ro_iqawg: IQAWG class instance
                        AWG used to control readout pulse generation mixer
                q_iqawg: IQAWG class instance
                        AWG used to control qubit excitation pulse generation mixer

            One of the following DC sources must be provided:
            current_source: alias address string or driver class
                            bias source used to tune qubit if_freq
            q_z_awg: alias address string or driver class
                     AWG generator that used to tune qubit if_freq


            plot_update_interval: float
                                sleep milliseconds between plot updates
        '''
        ## Equipment variables declaration section START ##
        self._current_source = None
        self._q_z_awg = None
        self._vna = None
        self._q_lo = None
        self._ro_awg = None
        self._q_awg = None
        ## Equipment variables declaration section END ##

        ## DEBUG
        self._fluxPts_iter_ctr = 0

        # constructor initializes devices from kwargs.keys() with '_' appended
        # keys must coincide with the attributes introduced in
        # "equipment variables declaration section"
        # TODO: set_fixed_parmaeters in DRO class uses q_awg and ro_awg names instead of
        # more proper ro_iqawg and q_iqawg
        devs_aliases_map.update(current_source=current_source, q_z_awg=q_z_awg)
        super().__init__(name, sample_name, devs_aliases_map, plot_update_interval)

        ## initializing base class elements with child specific values ##
        self._measurement_result = RabiFromFrequencyResult(name, sample_name)

        # last successful two tone spectroscopy result
        # that contains sweet-spot in its area
        # as well as all the qubit frequencies that
        # are going to be measured in this class
        self._tts_result = tts_result
        self._snap = Snapshot(self._tts_result._data) # can be used to exctract curves in the future
        self._tts_curves = {} # list of functions that results from scipy.interp1d
        self._current_curve = None # bias tts curve that is chosen by self.set_tts_curve(curve_key) method

        ## Initial and bias freq(bias or voltage) point control START ##
        self._ss_freq = ss_freq
        self._ss_flux_var_value = ss_current_or_voltage
        self._lowest_ss = lowest_ss
        # True if bias is used, False if voltage source is used
        self._current_flag = None
        self._flux_var_setter = None

        self._flux_var = None # flux variable value now
        self._last_flux_var = None # last flux variable value

        # constructor arguments consistency test
        if( current_source is not None ):
            self._current_flag = True
            self._flux_var_setter = self._current_source[0].set_current
        elif( q_z_awg is not None ):
            self._current_flag = False
            self._flux_var_setter = self._q_z_awg[0].set_voltage
        else:
            print("RabiFromFreq: You must provide one and only one of the following \
                  constructor parameters:\n \
                  current_source or q_z_awg.")
            raise TypeError
        ## Initial and bias freq(bias or voltage) point control END ##

        # set_fixed_params args are stored here
        self._fixed_devices_params = {}
        # set_swept_params excitation_durations argument is stored here
        self._basic_excitation_durations = None

        # class that is responsible for rabi measurements
        self._DRO = DispersiveRabiOscillations(name, sample_name, **devs_aliases_map) # Rabi measurement class

        # class that is responsible for Ramsey measurements
        self._DR = DispersiveRamsey(name, sample_name, **devs_aliases_map) # Ramsey measurement class

    def load_curve_from_csv(self, filepath):
        """
        TODO: add description
        """
        with open(filepath, "r") as csv_file:
            rows = list(csv.reader(csv_file))
            header = rows[0]
            curves_N = int(len(header) / 2)

            for curve_idx in range(curves_N):
                curve_name = header[2 * curve_idx]

                x = []
                y = []
                for i in range(0, len(rows) - 2):
                    if (rows[i + 2][2 * curve_idx] != ""):
                        x.append(rows[i + 2][2 * curve_idx])
                        y.append(rows[i + 2][2 * curve_idx + 1])
                    else:
                        break
                # make sure there is no identical 'x' values
                x = np.array(x, dtype=np.float64)
                y = np.array(y, dtype=np.float64)
                unique_idcs = np.unique(x, return_index=True)[1]
                x = x[unique_idcs]
                y = y[unique_idcs]
                    y_from_x_fit = interp1d(x, y, kind="cubic", copy=False, assume_sorted=False, fill_value="extrapolate")
                self._tts_curves[curve_name] = y_from_x_fit

            print("loaded tts_curves from file: " + filepath)
            print("curve labels: ", self._tts_curves.keys())

    def select_tts_curve(self, curve_key):
        self._current_curve = self._tts_curves[curve_key]

    def set_fixed_parameters(self, rabi_sequence_parameters, detect_resonator=False, plot_resonator_fit=False, **devs_params):
        self._fixed_devices_params = devs_params
        self._DRO.set_fixed_parameters(rabi_sequence_parameters,
                                       detect_resonator=detect_resonator, plot_resonator_fit=plot_resonator_fit,
                                       **devs_params)

    def set_swept_parameters(self, rabi_excitation_durations, ss_shifts):
        '''
        @params:
            excitation_durations - list of the rabi excitation pulse durations
            ss_shifts - list of absolute values of the qubit if_freq shift from sweet-spot
        '''
        self._basic_excitation_durations = rabi_excitation_durations
        self._DRO.set_swept_parameters(rabi_excitation_durations)
        ramsey_delays = rabi_excitation_durations
        self._DR.set_swept_parameters(ramsey_delays)
        super().set_swept_parameters(ss_shifts=(self._ss_shift_setter, ss_shifts))

    def set_ult_calib(self, ult_calib):
        # TODO: docstring
        self._DRO.set_ult_calib(ult_calib)
        self._DR.set_ult_calib(ult_calib)

    def _ss_shift_setter(self, ss_freq_shift):
        self._fluxPts_iter_ctr += 1
        '''
        @brief: sets new flux bias for a qubit to achieve
                qubit if_freq = ss_freq +- ss_freq_shift
                '+' or '-' is depending on the qubit freq(flux_bias)
                function behaviour around sweet_spot value
        '''
        if( self._lowest_ss is True ):
            init_qubit_frequency = self._ss_freq + ss_freq_shift
        else:
            init_qubit_frequency = self._ss_freq - ss_freq_shift

        qubit_frequency = init_qubit_frequency

        # finding value of the new flux variable
        f = self._current_curve

        def f2min(x):
            return (f(x) - qubit_frequency)**2  # f(x) is returning if_freq value in Hz


        ig_x = None
        if( self._last_flux_var is None ):
            ig_x = self._ss_flux_var_value
        else:
            ig_x = self._last_flux_var

        res = minimize(f2min, ig_x, method="L-BFGS-B", bounds=((f.x[0], f.x[-1]),))
        # setting new flux bias
        self._flux_var = res.x[0]
        self._flux_var_setter(self._flux_var)
        print("new flux variable: {}".format(self._flux_var), " mA")

        ## Adjusting if_freq to the present spot START ##
        ramsey_freq = 0
        ramsey_shift = 5e6
        ramsey_shift_error = 0.5e6
        iteration_ctr = 0
        # if there is no winner during the choice of the ramsey if_freq side,
        # than we change qubit if_freq by shift in this list
        n_trials = 5
        fail_shift_list = []
        for i in range(n_trials+1):
            m = i % 2
            fail_shift_list.append((2*m-1) * ramsey_shift * (i / (n_trials - 1)))

        trial_ctr = 0
        while( abs(ramsey_freq - ramsey_shift) > ramsey_shift_error ): # waiting for Ramsey to converge
            iteration_ctr += 1
            print( "\npoint number {}  iteration number {} trial number {}".format(self._fluxPts_iter_ctr,
                                                                                 iteration_ctr, trial_ctr+1))
            # gathering last successful device parameters
            dro_fixed_pars = self._DRO._fixed_pars
            pulse_seq_params = self._DRO._measurement_result.get_context().get_pulse_sequence_parameters()

            m = None
            if dro_fixed_pars["q_awg"][0]["calibration"]._sideband_to_maintain == "left":
                m = 1
            elif dro_fixed_pars["q_awg"][0]["calibration"]._sideband_to_maintain == "right":
                m = -1
            dro_fixed_pars["q_lo"][0]["if_freq"] = qubit_frequency + m * dro_fixed_pars["q_awg"][0]["calibration"]._if_frequency

            # finding Rabi pi/2 pulse
            self._DRO.set_fixed_parameters(pulse_seq_params,
                                           detect_resonator=True, plot_resonator_fit=False,
                                           **dro_fixed_pars)
            self._DRO.set_swept_parameters(self._basic_excitation_durations)
            self._rabi_oscillations_record()

            # updating Ramsey pulse sequence
            pi_pulse_duration = self._DRO._measurement_result.get_pi_pulse_duration()*1e3
            basis = self._DRO._measurement_result.get_basis()

            ramsey_pulse_seq_params = deepcopy(pulse_seq_params)
            ramsey_pulse_seq_params.update(half_pi_pulse_duration=pi_pulse_duration / 2)

            # measuring Ramsey if_freq number 1 | (q_freq - ramsey_shift)
            apr_ramsey_freq1 = qubit_frequency - ramsey_shift
            dro_fixed_pars["q_lo"][0]["if_freq"] = apr_ramsey_freq1 + m * dro_fixed_pars["q_awg"][0]["calibration"]._if_frequency
            self._DR.set_fixed_parameters(ramsey_pulse_seq_params,
                                          detect_resonator=True, plot_resonator_fit=False, **dro_fixed_pars)
            self._DR.set_swept_parameters(self._basic_excitation_durations)  # ramsey delays
            self._DR.set_basis(basis)
            self._ramsey_oscillations_record()
            ramsey_freq1 = self._DR._measurement_result.get_ramsey_frequency()
            fit_params1 = self._DR._measurement_result._fit_params
            data1 = self._DR._measurement_result._prepare_data_for_plot(self._DR._measurement_result.get_data())
            # maybe I should calculate fit success index based on the relative residual per point
            residuals1 = self._DR._measurement_result._cost_function(fit_params1, *data1)
            print(ramsey_freq1)

            # measuring Ramsey if_freq number 2 | (q_freq + ramsey_shift)
            apr_ramsey_freq2 = qubit_frequency + ramsey_shift
            dro_fixed_pars["q_lo"][0]["if_freq"] = apr_ramsey_freq2 + m * dro_fixed_pars["q_awg"][0]["calibration"]._if_frequency
            self._DR.set_fixed_parameters(ramsey_pulse_seq_params,
                                          detect_resonator=True, plot_resonator_fit=False, **dro_fixed_pars)
            self._DR.set_swept_parameters(self._basic_excitation_durations)  # ramsey delays
            self._DR.set_basis(basis)
            self._ramsey_oscillations_record()
            ramsey_freq2 = self._DR._measurement_result.get_ramsey_frequency()
            fit_params2 = self._DR._measurement_result._fit_params
            data2 = self._DR._measurement_result._prepare_data_for_plot(self._DR._measurement_result.get_data())
            residuals2 = self._DR._measurement_result._cost_function(fit_params2, *data2)
            print(ramsey_freq2)

            if( (ramsey_freq2 + ramsey_freq1) > (2*ramsey_shift - ramsey_shift_error) and
                    (ramsey_freq1 + ramsey_freq2) < (2*ramsey_shift + ramsey_shift_error) ):
                # if two frequencies are fitted normally and their sum is close
                # to the apriory computed value 2*ramsey_shift
                qubit_frequency = qubit_frequency + ramsey_shift - ramsey_freq2
                ramsey_freq = ramsey_freq2
            else:
                # one of the fit failed
                qubit_frequency = init_qubit_frequency + fail_shift_list[trial_ctr-1]
                trial_ctr += 1


        print("New qubit if_freq: {}", qubit_frequency, flush=True)
        self._last_flux_var = self._flux_var

        print("new ss shift is setted")
        print("flux var value: {}\nqubit_frequency: {}".format(self._flux_var, qubit_frequency))

    def _adjust_freq_with_TTS(self, flux_var, ro_power=None):
        """
        @brief: Function measures single line of TTS in the 'flux_val' point around the
                bias chosen curve y(flux_var) point. The scan is performed with very weak readout freq
                to neglect ACSTark effect. Measured curve is then fitted and maximum corresponding to the qubit
                is exctracted.
        @params:
            flux_var : float
                Flux DC source output value
            ro_power : float (dBm)
                If provided, the readout is set to this value.
                If None -> using readout power settings that corresponds to ro_cal value for
                pulsed measurements.
        @return:
            new_qubit_frequency : float (Hz)
                number that represents the local maximum on the local TTS
        """
        # TODO: consider implementing this function
        raise NotImplementedError

    def _recording_iteration(self):
        # _DRO will detect resonator and new qubit if_freq bias
        # during the call of the setters
        print("starting rabi\n")
        T_R, T_R_error = self._rabi_oscillations_record()
        print("starting ramsey\n")
        T_Ramsey, T_Ramsey_error = self._ramsey_oscillations_record()

        result = [T_R, T_R_error, T_Ramsey, T_Ramsey_error]

        return result  # Pulse measurement decays is stored in self._raw_data

    def _rabi_oscillations_record(self):
        self._measurement_result._now_meas_type = "Rabi"
        # clearing previous fit results due to the fact, that VNATimeResolvedDispersiveMeasurement1D._fit_complex_curve(..)
        # when this fit_params are not None
        # almost every time tries to use this parameters as the new best initial guess
        # and due to the fact, that the next measurements is performed in entirely different flux point
        # this initial guess vector does not fit the parameter's fit bounds that are generated from
        # the data of the bias measurement
        self._DRO._measurement_result._fit_params = None
        self._DRO._measurement_result._fit_errors = None
        # this is due to the fact that first fit of the data
        # in DispersiveRabiOscillations is generating stupid bounds
        # for optimization methods, based on previous measurement data
        # so, this data has to be erased
        self._DRO._measurement_result.set_data({})

        self._measurement_result._DRO_result = self._DRO._measurement_result

        self._DRO._measurement_result.set_start_datetime(dt.now())
        if self._DRO._measurement_result.is_finished():
            print("Starting with a result from a previous launch")
            self._DRO._measurement_result.set_is_finished(False)
        print("Started at: ", self._DRO._measurement_result.get_start_datetime())

        self._DRO._record_data()
        self._DRO._measurement_result.fit(verbose=False)
        print("DRO._record_data finished")

        T_R = self._DRO._measurement_result._fit_params[2]  # see DispersiveRabiOscillationsResult._model
        T_R_error = self._DRO._measurement_result._fit_errors[2]

        self._measurement_result._DRO_results.append(deepcopy(self._DRO._measurement_result))
        return T_R, T_R_error

    def _ramsey_oscillations_record(self):
        self._measurement_result._now_meas_type = "Rabi"
        # clearing previous fit results due to the fact, that VNATimeResolvedDispersiveMeasurement1D._fit_complex_curve(..)
        # when this fit_params are not None
        # almost every time tries to use this parameters as the new best initial guess
        # and due to the fact, that the next measurements is performed in entirely different flux point
        # this initial guess vector does not fit the parameter's fit bounds that are generated from
        # the data of the bias measurement
        self._measurement_result._now_meas_type = "Ramsey"
        self._DR._measurement_result._fit_params = None
        self._DR._measurement_result._fit_errors = None
        # this is due to the fact that first fit of the data
        # in DispersiveRabiOscillations is generating stupid bounds
        # for optimization methods, based on previous measurement data
        # so, this data has to be erased
        # Ramsey oscillations data is erased just for sake of symmetry.
        # Actually there is no need to do this for DispersiveRamsey
        self._DR._measurement_result.set_data({})

        self._measurement_result._DR_result = self._DR._measurement_result

        self._DR._measurement_result.set_start_datetime(dt.now())
        if self._DR._measurement_result.is_finished():
            print("Starting with a result from a previous launch")
            self._DR._measurement_result.set_is_finished(False)
        print("Started at: ", self._DR._measurement_result.get_start_datetime())

        self._DR._record_data()
        self._DR._measurement_result.fit(verbose=False)
        print("DR._record_data finished")

        T_Ramsey, T_Ramsey_error = self._DR._measurement_result.get_ramsey_decay()
        self._measurement_result._DR_results.append(deepcopy(self._DR._measurement_result))
        return T_Ramsey, T_Ramsey_error

    # TODO: NOT WORKING YET. Consider to implement working version or delete the following code
    '''
    self._snap oonnected routines.  
    def plot_tts_connectivity_map(self, rel_threshold=0.5,
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

    def detect_tts_lines(self, kernel_x=0.1, kernel_y=0.1, rel_threshold=0.5, connectivity=8):
        # TODO: docstring
        self._snap.make_and_visualize_connectivity_map(rel_threshold=rel_threshold,
                                                       kernel_x=kernel_x, kernel_y=kernel_y,
                                                       connectivity=connectivity)

    def select_tts_line(self, label_i):
        # TODO: docstring
        mask = self._snap.select_connectivity_component(label_i)
        self._snap.interpolate_yx_curve()
        return self._snap._target_y_func
    '''


class RabiFromFrequencyResult(MeasurementResult):
    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._line_scatter = None
        self.ss_shifts = None

        # results of last oscillation measurement is stored here
        self._DRO_result = None
        self._DR_result = None
        # parameter that shows what is measured at this particular moment
        self._now_meas_type = None  # "Rabi", "Ramsey"


        # self._DRO.launch().data will be stored in the following list
        self._DRO_results = []
        # self._DR.launch().data will be stored in the following list
        self._DR_results = []

    def _prepare_figure(self):
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3,1)
        axes = np.ravel(axes)

        return fig, axes, None

    def _prepare_data_for_plot(self,data):
        return data[self._parameter_names[0]], data["data"][:, 0]

    def _plot(self, data):
        '''
        caxes is None
        '''
        import time

        if( "data" in data.keys()):
            # print(data)
            # time.sleep(2)
            x,y_data = self._prepare_data_for_plot(data)
            xlim = np.array([x[0], x[-1]])/1e6 # convert to MHz
            # crop data that is still to be measured
            y_data = y_data[y_data != 0]
            x = x[:len(y_data)]

            # redrawing axes data
            ax = self._axes[0]
            ax.reset()
            ax.set_xlabel(r"$\delta \nu$, MHz")
            ax.set_ylabel(r"$T_R, \; \mu s$")
            ax.grid()
            self._line_scatter, = ax.plot(x/1e6, y_data, 'r', marker="o",
                                          markerfacecolor='none')
            # setting x limit for graph
            ax.set_xlim(xlim[0], xlim[1])

        ## plot rabi result
        if( (self._DRO_result is not None) and (self._now_meas_type == "Rabi") ):
            # there is no function on the plotting Thread that is called when we are moving from one point to another
            self._DRO_result._axes = self._axes[1:3]
            self._DRO_result._figure = self._figure

            DRO_data = self._DRO_result.get_data() # prepeare bias DRO_data
            # TODO: hotfix by Shamil
            # 'DispersiveRabiOscillationsResult' object has no attribute '_dynamic'
            # when dynamic==True the code in VNATRDM1D skips replotting the data
            # this conflict whith a previously drawn picture e.g. from "Ramsey"
            # when False it replots the whole curve
            setattr(self._DRO_result, "_dynamic", False)
            self._DRO_result._plot(DRO_data)

        elif( (self._DR_result is not None) and (self._now_meas_type == "Ramsey") ):
            # there is no function on the plotting Thread that is called when we are moving from one point to another
            self._DR_result._axes = self._axes[1:3]
            self._DR_result._figure = self._figure

            DR_data = self._DR_result.get_data()  # prepeare bias DRO_data
            # TODO: hotfix by Shamil
            # 'DispersiveRabiOscillationsResult' object has no attribute '_dynamic'
            # when dynamic==True the code in VNATRDM1D skips replotting the data
            # this conflict whith a previously drawn picture e.g. from "Rabi"
            # when False it replots the whole curve
            setattr(self._DR_result, "_dynamic", False)
            self._DR_result._plot(DR_data)





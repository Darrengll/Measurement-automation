
from numpy import *
from scipy.optimize import minimize
from datetime import datetime
from IPython.display import clear_output


class IQCalibrationData():

    def __init__(self, mixer_id, iq_attenuation, lo_frequency, lo_power,
        if_frequency, ssb_power, waveform_resolution, dc_offsets,
        dc_offsets_open, if_offsets, if_amplitudes, if_phase, spectral_values,
        optimization_time, end_date):

        self._mixer_id = mixer_id
        self._iq_attenuation = iq_attenuation
        self._lo_frequency = lo_frequency
        self._if_frequency = if_frequency
        self._lo_power = lo_power
        self._ssb_power = ssb_power
        self._waveform_resolution = waveform_resolution

        self._dc_offsets = dc_offsets
        self._dc_offsets_open = dc_offsets_open
        self._if_offsets = if_offsets
        self._if_amplitudes = if_amplitudes
        self._if_phase = if_phase

        self._spectral_values = spectral_values
        self._optimization_time = optimization_time
        self._end_date = end_date

    def get_optimization_results(self):
        """
        Get the optimal paramters and the resulting spectral component values

        Returns:
            parameters, results: tuple
        """
        return dict(dc_offsets=self._dc_offsets, dc_offset_open=self._dc_offsets_open,
            if_offsets=self._if_offsets, if_amplitudes=self._if_amplitudes,
                if_phase=self._if_phase), self._spectral_values

    def get_radiation_parameters(self):
        return dict(lo_frequency=self._lo_frequency, lo_power=self._lo_power,
            if_frequency=self._if_frequency, ssb_power=self._ssb_power, waveform_resolution=self._waveform_resolution)

    def get_mixer_parameters(self):
        return dict(mixer_id=self._mixer_id, iq_attenuation=self._iq_attenuation)

    def __str__(self):
        return "Calibration data for mixer "+self._mixer_id+\
                "\nMixer parameters: "+str(self.get_mixer_parameters())+\
                "\nRadiation parameters: "+str(self.get_radiation_parameters())+\
                "\nOptimization results: "+str(self.get_optimization_results()[1])+\
                "\nOptimization parameters "+str(self.get_optimization_results()[0])+\
                "\nOptimization time: "+format_time_delta(self._optimization_time)+\
                "\nFinished at: "+str(self._end_date)


class IQCalibrator():

    def __init__(self, awg, sa, lo, mixer_id, iq_attenuation,
                 sideband_to_maintain="left", sidebands_to_suppress=6,
                 optimized_awg_calls = True):
        """
        IQCalibrator is a class that allows you to calibrate automatically an IQ mixer to obtain a Single Sideband (SSB)
        with desired parameters.
        """
        self._awg = awg
        self._sa = sa
        self._lo = lo
        self._mixer_id = mixer_id
        self._iq_attenuation = iq_attenuation
        self.side = sideband_to_maintain
        self._N_sup = sidebands_to_suppress
        self._iterations = 0
        self._optimized_awg_calls = optimized_awg_calls

    def calibrate(self, lo_frequency, if_frequency, lo_power, ssb_power, waveform_resolution=1, initial_guess=None,
                sa_res_bandwidth=500, iterations=5, minimize_iterlimit=20):
        """
        Perform the calibration routine to suppress LO and upper sideband LO+IF
         while maintaining the lower sideband at ssb_power.

        In case of if_frequency equal to zero the DC calibration is performed.
        The ssb_power parameter will be then treated as
        the power of the LO when the mixer is in the open state

        Parameters:
        ----------
        lo_frequency: float
            Frequency of the local oscillator
        if_frequency: float
            Frequency of the awg-generated wavefomrs, i.e. intermediate frequency
        ssb_power: float
            The power which the remaining sideband LO-IF will have after the optimization or
            the power of the LO in the "open" state if if_freq is equal to zero
        waveform_resolution: float, ns
            The resolution in time of the arbitrary waveform representing one period of the continuous wave used in calibration
        initial_guess=None : IQCalibrationData
            It's possible to specify the initial guess by passing the IQCalibrationData object from previous calibrations
        sa_res_bandwisth=500: float
            The bandwidth that spectrum analyser will use during the calibration
        iterations=5: int
            The number of iterations of the cycle {optimize_if_offsets, optimize_if_amolitudes, optimize_if_phase}.
            For the dc offsets interation limit is iterations*minimize_iterlimit
        minimize_iterlimit=20: int
            Iteration limit for the minimize function used in each routine listed above

        Returns:
        iqmx_calibration: IQCalibrationData
            Object containing the parameters and results of the optimization
        """

        def loss_function_dc_offsets(dc_offsets):
            self._awg.output_continuous_IQ_waves(frequency=0,
                amplitudes=(0,0), relative_phase=0, offsets=dc_offsets,
                waveform_resolution=waveform_resolution,
                optimized = self._optimized_awg_calls)
            self._sa.prepare_for_stb();self._sa.sweep_single();self._sa.wait_for_stb()
            data = self._sa.get_tracedata()
            self._iterations += 1
            print("\rDC offsets: ", format_number_list(dc_offsets),
                                    format_number_list(data), self._iterations,
                                    end=", ", flush=True)
            clear_output(wait=True)
            answer =  data[0]
            return answer

        def loss_function_dc_offsets_open(dc_offset_open):

            self._awg.output_continuous_IQ_waves(frequency=0,
                amplitudes=(0,0), relative_phase=0, offsets=(dc_offset_open,)*2,
                waveform_resolution=waveform_resolution,
                optimized = self._optimized_awg_calls)

            self._sa.prepare_for_stb();self._sa.sweep_single();self._sa.wait_for_stb()
            data = self._sa.get_tracedata()

            print("\rDC offsets open: ", format_number_list([dc_offset_open]*2),
                                         format_number_list(data),
                                         end=", ", flush=True)
            clear_output(wait=True)

            if( self.side == "right" ):
                data.reverse()
            answer = abs(data[0]-ssb_power)
            return answer

        def loss_function_if_offsets(if_offsets, args):
            if_amplitudes = args[0]
            phase = args[1]
            self._awg.output_continuous_IQ_waves(frequency=if_frequency,
                amplitudes=if_amplitudes, relative_phase=phase, offsets=if_offsets,
                waveform_resolution=waveform_resolution,
                optimized = self._optimized_awg_calls)
            self._sa.prepare_for_stb();self._sa.sweep_single();self._sa.wait_for_stb()
            data = self._sa.get_tracedata()

            print("\rIF offsets: ", format_number_list(if_offsets),
                                    format_number_list(data),
                                     end="            ", flush=True)
            clear_output(wait=True)

            if( self.side == "right" ):
                data.reverse()
            answer =  10**(data[self._N_sup//2+1]/10)
            return answer

        def loss_function_if_amplitudes(if_amplitudes, args):
            amp1, amp2 = if_amplitudes
            if_offsets = args[0]
            phase = args[1]
            self._awg.output_continuous_IQ_waves(frequency=if_frequency,
                amplitudes=if_amplitudes, relative_phase=phase, offsets=if_offsets,
                waveform_resolution=waveform_resolution,
                optimized = self._optimized_awg_calls)
            self._sa.prepare_for_stb();self._sa.sweep_single();self._sa.wait_for_stb()
            data = self._sa.get_tracedata()

            answer = None
            if abs(abs(amp1)-abs(amp2))<.2:
                loss_value_amp = 0
                for i,psd in enumerate(data):
                    if psd != data[self._N_sup//2]:
                        value = abs((self._N_sup//2-i+.1))**(-1.8)*10**(psd/10)
                        loss_value_amp += value

                answer = 50e3*loss_value_amp \
                        + 10*(abs(ssb_power - data[self._N_sup//2]))
            else:
                answer = 10**(10*abs(abs(amp1)-abs(amp2)))
        #    if( self.side == "right" ):
        #        answer =  data[0] + 10*abs(ssb_power - data[2]) + 0\
        #            if abs(abs(amp1)-abs(amp2))<.2 else 10**(10*abs(abs(amp2)-abs(amp1)))
            print("\rAmplitudes: ", format_number_list(if_amplitudes),
                                    format_number_list(data),
                                    "loss:", answer, end="          ", flush=True)
            clear_output(wait=True)
            return answer

        def loss_function_if_phase(phase, args):
            if_offsets = args[0]
            if_amplitudes = args[1]
            self._awg.output_continuous_IQ_waves(frequency=if_frequency,
                amplitudes=if_amplitudes, relative_phase=phase, offsets=if_offsets,
                waveform_resolution=waveform_resolution,
                optimized = self._optimized_awg_calls)
            self._sa.prepare_for_stb();self._sa.sweep_single();self._sa.wait_for_stb()
            data = self._sa.get_tracedata()

            print("\rPhase: ", "%3.2f"%(phase/pi*180), format_number_list(data), end="             ", flush=True)
            clear_output(wait=True)

            if( self.side == "right" ):
                data.reverse()
            answer =  25e3*sum(array([abs((self._N_sup//2-i+.1))**(-1.8)*10**(psd/10) \
                        for i,psd in enumerate(data) if psd != data[self._N_sup//2]]))\
                         + 10**(abs(ssb_power - data[self._N_sup//2])/10)
            return answer

        def iterate_minimization(prev_results, n=2):

            options = {"maxiter":minimize_iterlimit, "xatol":.1e-3, "fatol":3}
            res_if_offs = minimize(loss_function_if_offsets, prev_results["if_offsets"],
                args=[prev_results["if_amplitudes"], prev_results["if_phase"]],
                method="Nelder-Mead", options=options)
            res_phase = minimize(loss_function_if_phase, prev_results["if_phase"],
                args=[res_if_offs.x, prev_results["if_amplitudes"]],
                method="Nelder-Mead", options=options)
            res_amps = minimize(loss_function_if_amplitudes, prev_results["if_amplitudes"],
                args=[res_if_offs.x, res_phase.x],
                method="Nelder-Mead", options=options)


            results["if_offsets"] = res_if_offs.x
            results["if_amplitudes"] = res_amps.x
            results["if_phase"] = res_phase.x
            if(n-1==0):
                return
            iterate_minimization(results, n-1)

        try:

            start = datetime.now()

            self._lo.set_power(lo_power)
            self._lo.set_frequency(lo_frequency)
            self._lo.set_output_state("ON")

            results = None
            if initial_guess is None:
                results = {"dc_offsets":(1,1), "dc_offset_open":1, "if_offsets":(1,1),
                                "if_amplitudes":(0.5,0.5), "if_phase":pi*0.54}
            else:
                results = initial_guess

            self._sa.setup_list_sweep([lo_frequency], [sa_res_bandwidth])

            for i in range(0, iterations):
                res_dc_offs = minimize(loss_function_dc_offsets, results["dc_offsets"],
                              method="Nelder-Mead", options={"maxiter":minimize_iterlimit,
                              "xatol":.5e-3, "fatol":10})
                self._iterations = 0
                results["dc_offsets"] = res_dc_offs.x

            if if_frequency == 0:
                for i in range(0,iterations):
                    res_dc_offs_open = minimize(loss_function_dc_offsets_open,
                            results["dc_offset_open"],  method="Nelder-Mead",
                            options={"maxiter":minimize_iterlimit,
                                  "xatol":1e-3, "fatol":100})
                    self._iterations = 0
                    results["dc_offset_open"] = res_dc_offs_open.x
                spectral_values = {"dc":res_dc_offs.fun, "dc_open":self._sa.get_tracedata()}
                elapsed_time = (datetime.now() - start).total_seconds()
                return IQCalibrationData(self._mixer_id, self._iq_attenuation,
                    lo_frequency, lo_power, if_frequency, ssb_power, waveform_resolution,
                    results["dc_offsets"], array([results["dc_offset_open"]]*2), None, None, None, spectral_values,
                    elapsed_time, datetime.now())

            else:
                self._sa.setup_list_sweep(list(arange(lo_frequency-self._N_sup//2*if_frequency,\
                                                         lo_frequency+(self._N_sup//2+1)*if_frequency,\
                                                          if_frequency)-if_frequency), [sa_res_bandwidth]*3)
                results["if_offsets"]=res_dc_offs.x
                iterate_minimization(results, iterations)
                spectral_values = {"dc":res_dc_offs.fun, "if":self._sa.get_tracedata()}
                elapsed_time = (datetime.now() - start).total_seconds()
                return IQCalibrationData(self._mixer_id, self._iq_attenuation,
                    lo_frequency, lo_power, if_frequency, ssb_power, waveform_resolution,
                    res_dc_offs.x, None, results["if_offsets"], results["if_amplitudes"],
                    results["if_phase"], spectral_values, elapsed_time, datetime.now())

        except KeyboardInterrupt:
            return results

        finally:
             self._sa.setup_swept_sa(lo_frequency, 10*if_frequency if if_frequency>0 else 1e9, nop=1001, rbw=1e5)
             self._sa.set_continuous()

def format_number_list(number_list):
    formatted_string = "[ "
    for number in number_list:
        formatted_string += "%3.3f "%number
    return formatted_string + "]"

def format_time_delta(delta):
    hours, remainder = divmod(delta, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%s h %s m %s s' % (int(hours), int(minutes), round(seconds, 2))

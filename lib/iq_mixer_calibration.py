from numpy import *
import numpy as np
from scipy.optimize import minimize
from datetime import datetime
from IPython.display import clear_output


class IQCalibrationData():

    def __init__(self, mixer_id, iq_attenuation, lo_frequency, lo_power,
                 if_frequency, sideband_to_maintain, ssb_power,
                 waveform_resolution, dc_offsets, dc_offsets_open,
                 if_offsets, if_amplitudes, if_phase, spectral_values,
                 optimization_time, end_date):

        self._mixer_id = mixer_id
        self._iq_attenuation = iq_attenuation
        self._lo_frequency = lo_frequency
        self._if_frequency = if_frequency
        self._lo_power = lo_power
        self._ssb_power = ssb_power
        self._sideband_to_maintain = sideband_to_maintain
        self._waveform_resolution = waveform_resolution

        sidebands = {"left":    self._lo_frequency - self._if_frequency,
                     "right":   self._lo_frequency + self._if_frequency}
        self._sideband_to_maintain_freq = sidebands[self._sideband_to_maintain]

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
        Get optimal parameters and the resulting spectral component values
        Returns:
            parameters, results: tuple
        """
        return dict(dc_offsets=self._dc_offsets,
                    dc_offset_open=self._dc_offsets_open,
                    if_offsets=self._if_offsets,
                    if_amplitudes=self._if_amplitudes,
                    if_phase=self._if_phase), self._spectral_values

    def get_radiation_parameters(self):
        return dict(lo_frequency=self._lo_frequency, lo_power=self._lo_power,
                    if_frequency=self._if_frequency, ssb_power=self._ssb_power,
                    sideband_to_maintain=self._sideband_to_maintain,
                    waveform_resolution=self._waveform_resolution)

    def get_mixer_parameters(self):
        return dict(mixer_id=self._mixer_id,
                    iq_attenuation=self._iq_attenuation)

    def __str__(self):
        import json
        import datetime
        class Encoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray) or \
                        isinstance(obj, datetime.datetime):
                    return obj.__str__()
                return json.JSONEncoder.default(self, obj)

        def nice_dict(d):
            return json.dumps(d, indent=4, cls=Encoder)

        return (f"Calibration data for mixer {nice_dict(self._mixer_id)}\n"
                f"Mixer parameters: "
                f"{nice_dict(self.get_mixer_parameters())}\n"
                f"Radiation parameters: {nice_dict(self.get_radiation_parameters())}\n"
                f"Optimization results: {nice_dict(self.get_optimization_results()[1])}\n"
                f"Optimization parameters "
                f"{nice_dict(self.get_optimization_results()[0])}\n"
                f"Optimization time: "
                f"{nice_dict(format_time_delta(self._optimization_time))}\n"
                f"Finished at: {nice_dict(self._end_date)}")

    def get_if_frequency(self):
        return self._if_frequency

    def get_lo_frequency(self):
        return self._lo_frequency

    def get_ssb_power(self):
        return self._ssb_power

    def get_lo_power(self):
        return self._lo_power


class IQCalibrator():

    def __init__(self, iqawg, sa, lo, mixer_id, iq_attenuation,
                 sideband_to_maintain="left", sidebands_to_suppress=6,
                 optimized_awg_calls = True):
        """
        Automatically calibrate an IQ mixer to obtain a Single Sideband (SSB)
        with desired parameters.

        iqawg:
            reference to the IQAWG object
        sa:
            reference to the Spectrum Analyzer object
        lo:
            reference to the Local Oscillator object
        mixer_id: str
            name of a mixer
        iq_attenuation:
        sideband_to_maintain : str
            "left","right" - which 1 order sideband to maximize (lo is center)
        sidebands_to_suppress : int
            number of closest sidebands to the 'sideband_to_maintain' that will be accounted for
            and minimized in loss function
        optimized_awg_calls : bool
            TODO: what is this parameter?
        """
        self._iqawg = iqawg
        self._sa = sa
        self._lo = lo
        self._mixer_id = mixer_id
        self._iq_attenuation = iq_attenuation
        self._sideband_to_maintain = sideband_to_maintain
        self._N_sup = sidebands_to_suppress
        self._target_freq_idx = self._N_sup // 2
        self._lo_freq_idx = None
        self._iterations = 0
        self._optimized_awg_calls = optimized_awg_calls

    def calibrate(self, lo_frequency, if_frequency, lo_power, ssb_power,
                  waveform_resolution=1, initial_guess=None,
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
            Frequency of the awg-generated wavefomrs, i.e. intermediate frequency (of I(t) and Q(t) signals)
        lo_power: float
            The power of the local oscillator
        ssb_power: float
            The power which the remaining sideband LO-IF will have after the optimization or
            the power of the LO in the "open" state if if_freq is equal to zero
        waveform_resolution: float, ns
            The resolution in time of the arbitrary waveform representing one period of the continuous wave used in calibration
        initial_guess=None : IQCalibrationData
            It's possible to specify the initial guess by passing the IQCalibrationData object from previous calibrations
        sa_res_bandwidth=500: float
            The bandwidth that spectrum analyser will use during the calibration
        iterations=5: int
            The number of iterations in a cycle {optimize_if_offsets, optimize_if_amplitudes, optimize_if_phase}.
            For the dc offsets iteration limit is iterations*minimize_iterlimit
        minimize_iterlimit=20: int
            Iteration limit for the minimize function used in each routine listed above
        Returns:
        iqmx_calibration: IQCalibrationData
            Object containing the parameters and results of the optimization
        """

        def loss_function_dc_offsets(dc_offsets):
            self._iqawg.output_continuous_IQ_waves(frequency=0,
                amplitudes=(0,0), relative_phase=0, offsets=dc_offsets,
                waveform_resolution=waveform_resolution,
                optimized = self._optimized_awg_calls)
            self._sa.prepare_for_stb()
            self._sa.sweep_single()
            self._sa.wait_for_stb()
            data = self._sa.get_tracedata()
            self._iterations += 1

            answer =  data[0]
            print("\rDC offsets: ", format_number_list(dc_offsets),
                  format_number_list(data), self._iterations,
                  "loss: ", answer,
                  end=", ", flush=True)
            clear_output(wait=True)
            return answer

        def loss_function_dc_offsets_open(dc_offset_open):
            self._iqawg.output_continuous_IQ_waves(frequency=0,
                amplitudes=(0,0), relative_phase=0, offsets=(dc_offset_open,)*2,
                waveform_resolution=waveform_resolution,
                optimized = self._optimized_awg_calls)

            self._sa.prepare_for_stb();
            self._sa.sweep_single();
            self._sa.wait_for_stb()

            data = self._sa.get_tracedata()

            answer = abs(data[0]-ssb_power)
            print("\rDC offsets open: ", format_number_list([dc_offset_open] * 2),
                  format_number_list(data), "loss: ", answer,
                  end=", ", flush=True)
            clear_output(wait=True)
            return answer

        def loss_function_if_offsets(if_offsets, args):
            if_amplitudes = args[0]
            phase = args[1]
            self._iqawg.output_continuous_IQ_waves(frequency=if_frequency,
                amplitudes=if_amplitudes, relative_phase=phase, offsets=if_offsets,
                waveform_resolution=waveform_resolution,
                optimized = self._optimized_awg_calls)
            self._sa.prepare_for_stb();self._sa.sweep_single();self._sa.wait_for_stb()
            data = self._sa.get_tracedata()

            print("\rIF offsets: ", format_number_list(if_offsets),
                                    format_number_list(data),
                                     end="            ", flush=True)
            clear_output(wait=True)

            answer = 10**(data[self._lo_freq_idx]/10)
            return answer

        def loss_function_if_amplitudes(if_amplitudes, args):
            amp1, amp2 = if_amplitudes
            if_offsets = args[0]
            phase = args[1]
            self._iqawg.output_continuous_IQ_waves(frequency=if_frequency,
                amplitudes=if_amplitudes, relative_phase=phase, offsets=if_offsets,
                waveform_resolution=waveform_resolution,
                optimized = self._optimized_awg_calls)
            self._sa.prepare_for_stb();self._sa.sweep_single();self._sa.wait_for_stb()
            data = self._sa.get_tracedata()

            loss_value_amp = 0
            for i,psd in enumerate(data):
                if i != self._target_freq_idx:
                    # value = abs((self._target_freq_idx-i))**(-1.8)*10**(psd/10)
                    value = 10**((psd-ssb_power)/10)
                    loss_value_amp += value

            if_amplitude_difference_loss = (abs(amp1-amp2)*50 if abs(amp1-amp2) > 0.01 else 0)
            answer = loss_value_amp \
                    + 10**(abs(ssb_power - data[self._target_freq_idx])/10)*10 \
                    + if_amplitude_difference_loss

            print("\rAmplitudes: ", format_number_list(if_amplitudes), format_number_list(data),
                  "loss:", answer, "IF IQ amplitude difference loss:", if_amplitude_difference_loss,
                  end="          ", flush=True)
            clear_output(wait=True)
            return answer

        def loss_function_if_phase(phase, args):
            if_offsets = args[0]
            if_amplitudes = args[1]
            self._iqawg.output_continuous_IQ_waves(frequency=if_frequency,
                amplitudes=if_amplitudes, relative_phase=phase, offsets=if_offsets,
                waveform_resolution=waveform_resolution,
                optimized = self._optimized_awg_calls)
            self._sa.prepare_for_stb();self._sa.sweep_single();self._sa.wait_for_stb()
            data = self._sa.get_tracedata()

            answer = 25e3*sum(np.array(
                [abs((self._target_freq_idx-i))**(-1.8)*10**(psd/10) for i, psd in enumerate(data) if i != self._target_freq_idx])
            ) + 10**(abs(ssb_power - data[self._target_freq_idx])/10)

            if self._sideband_to_maintain == "right":
                answer = -data[self._target_freq_idx] + data[self._target_freq_idx-2]
            else:
                answer = -data[self._target_freq_idx] + data[self._target_freq_idx+2]

            print("\rPhase: ", "%3.2f" % (phase / pi * 180), format_number_list(data),
                  "loss:", answer,
                  end="          ", flush=True)
            clear_output(wait=True)

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
                results = {"dc_offsets":(1,1), "dc_offset_open":(1,1), "if_offsets":(1,1),
                                "if_amplitudes":(0.5,0.5), "if_phase":pi*0.54}
            else:
                results = initial_guess

            self._sa.setup_list_sweep([lo_frequency], [sa_res_bandwidth])

            for i in range(0, iterations):
                res_dc_offs = minimize(loss_function_dc_offsets, results["dc_offsets"],
                                       method="Nelder-Mead", options={"maxiter":minimize_iterlimit,
                                                                      "xatol": 0.5e-3, "fatol":10})
                self._iterations = 0
                results["dc_offsets"] = res_dc_offs.x

            if if_frequency == 0:
                for i in range(0,iterations):
                    res_dc_offs_open = minimize(loss_function_dc_offsets_open,
                            results["dc_offset_open"][0],  method="Nelder-Mead",
                            options={"maxiter":minimize_iterlimit,
                                     "xatol": 1e-5, "fatol": 100})
                    self._iterations = 0
                    results["dc_offset_open"] = res_dc_offs_open.x
                spectral_values = {"dc":res_dc_offs.fun, "dc_open":self._sa.get_tracedata()}
                elapsed_time = (datetime.now() - start).total_seconds()
                return IQCalibrationData(self._mixer_id, self._iq_attenuation,
                    lo_frequency, lo_power, if_frequency, self._sideband_to_maintain, ssb_power, waveform_resolution,
                    results["dc_offsets"], array([results["dc_offset_open"]]*2), None, None, None, spectral_values,
                    elapsed_time, datetime.now())

            else:
                freqs = np.arange(lo_frequency - self._target_freq_idx * if_frequency,
                                  lo_frequency + (self._target_freq_idx + 1) * if_frequency,
                                  if_frequency)

                if self._sideband_to_maintain == "right":
                    freqs += if_frequency
                    self._lo_freq_idx = self._target_freq_idx - 1
                elif self._sideband_to_maintain == "left":
                    freqs -= if_frequency
                    self._lo_freq_idx = self._target_freq_idx + 1

                self._sa.setup_list_sweep(list(freqs), [sa_res_bandwidth]*3)

                results["if_offsets"]=res_dc_offs.x
                iterate_minimization(results, iterations)
                spectral_values = {"dc":res_dc_offs.fun, "if":self._sa.get_tracedata()}
                elapsed_time = (datetime.now() - start).total_seconds()
                return IQCalibrationData(self._mixer_id, self._iq_attenuation,
                    lo_frequency, lo_power, if_frequency, self._sideband_to_maintain, ssb_power, waveform_resolution,
                    res_dc_offs.x, None, results["if_offsets"], results["if_amplitudes"],
                    results["if_phase"], spectral_values, elapsed_time, datetime.now())

        except KeyboardInterrupt:
            return results

        finally:
            shift = if_frequency if self._sideband_to_maintain == "right" else -if_frequency
            self._sa.setup_swept_sa(lo_frequency + shift, 10*if_frequency if if_frequency>0 else 1e9, nop=1001, rbw=1e5)
            self._sa.set_continuous()

def format_number_list(number_list):
    formatted_string = f"[ "
    for number in number_list:
        formatted_string += "%3.3f "%number
    return formatted_string + "]"

def format_time_delta(delta):
    hours, remainder = divmod(delta, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%s h %s m %s s' % (int(hours), int(minutes), round(seconds, 2))
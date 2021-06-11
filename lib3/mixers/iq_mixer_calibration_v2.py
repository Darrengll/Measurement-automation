# Standard library imports
import copy
from datetime import datetime
from typing import List

# Third party imports
import numpy as np
from scipy.optimize import minimize
import json
from IPython.display import clear_output

# Local application imports
from lib3.core.drivers.agilent_EXA import Agilent_EXA_N9010A
from lib3.core.drivers.mw_sources import MwSrcInterface
from lib3.core.compound_devices.iq_awg import IQAWG


class IQCalibrationData:

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
        class Encoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray) or \
                        isinstance(obj, datetime):
                    return obj.__str__()
                else:
                    return json.JSONEncoder.default(self, obj)
        return json.dumps(self.toJSON(), indent=4, cls=Encoder)

    def toJSON(self):
        report_dict = {
            "Mixer parameters": self.get_mixer_parameters(),
            "Optimization reulsts":
                self.get_optimization_results(),
            "Radiation parameters":
                self.get_radiation_parameters(),
            "Optimization time": format_time_delta((
                self._optimization_time)),
            "Finished at": self._end_date
        }
        return report_dict

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
                 optimized_awg_calls=True):
        """
        Automatically calibrate an IQ mixer to obtain a Single Sideband (SSB)
        with desired parameters.

        Parameters
        ----------
        iqawg: IQAWG
            reference to the IQAWG object
        sa: Agilent_EXA_N9010A
            reference to the Spectrum Analyzer object
        lo: MwSrcInterface
            reference to the Local Oscillator object
        mixer_id: str
            name of a mixer
        iq_attenuation: float
            redundant parmaeter
        sideband_to_maintain : str
            "left","right" - which 1st order sideband to maximize (lo is
            center)
        sidebands_to_suppress : int
            number of closest sidebands to the 'sideband_to_maintain' that will
            be accounted for and minimized in loss function
        optimized_awg_calls : bool
            TODO: what is this parameter?
        """
        self._iqawg: IQAWG = iqawg
        self._sa: Agilent_EXA_N9010A = sa
        self._lo: MwSrcInterface = lo
        self._mixer_id = mixer_id
        self._iq_attenuation = iq_attenuation
        self._sideband_to_maintain = sideband_to_maintain
        self._N_sup = sidebands_to_suppress
        # Frequencies are indexed from least to largest value
        # starting with 0.
        # And if_freq of interest index is as follows:
        self._target_freq_idx = self._N_sup // 2
        self._lo_freq_idx = None
        self._image_freq_idx = None
        self._iterations = 0
        self._optimized_awg_calls = optimized_awg_calls
        self._phases = {"left": np.pi / 2,
                        "right": -np.pi / 2}

        self._monitoring_frequencies_list: List[float,...] = [0.0]

    def get_spectrum(self):
        self._sa.prepare_for_stb()
        self._sa.set_video_bandwidth(1)
        self._sa.sweep_single()
        self._sa.wait_for_stb()
        return self._sa.get_tracedata()

    def calibrate(self, lo_frequency, if_frequency, lo_power, ssb_power,
                  waveform_resolution=1, initial_guess=None,
                  sa_res_bandwidth=500, sa_vid_bandwidth=100,
                  iterations=5, minimize_iterlimit=20, sa_averages=10):
        """
        Perform the calibration routine to suppress LO and upper sideband
        LO+IF
        while maintaining the lower sideband at ssb_power.
        In case of if_frequency equal to zero the DC calibration is performed.
        The ssb_power parameter will be then treated as
        the power of the LO when the mixer is in the open state

        Parameters
        ----------
        lo_frequency: float
            Frequency of the local oscillator
        if_frequency: float
            Frequency of the awg-generated wavefomrs, i.e. intermediate
            if_freq (of I(t) and Q(t) signals)
        lo_power: float
            The power of the local oscillator
        ssb_power: float
            The power which the remaining sideband LO-IF will have after the
            optimization or the power of the LO in the "open" state if
            if_freq is equal to zero
        waveform_resolution: float, ns
            The resolution in time of the arbitrary waveform representing one
            period of the continuous wave used in calibration
        initial_guess : IQCalibrationData
            It's possible to specify the initial guess by passing the
            IQCalibrationData object from previous calibrations
        sa_res_bandwidth: float
            The bandwidth that spectrum analyser will use during the
            calibration. Default = 500.
        sa_vid_bandwidth : float
            See SA user manual for details (Shamil has no idea what is it)
        iterations: int
            The number of iterations in a cycle {optimize_if_offsets,
            optimize_if_amplitudes, optimize_if_phase}.
            For the dc offsets iteration limit is
            `iterations*minimize_iterlimit`
        minimize_iterlimit: int
            Iteration limit for the minimize function used in each routine
            listed above. Default = 20
        sa_averages : int
            number of averages for spectral analyzer.

        Returns
        ----------
        iqmx_calibration: IQCalibrationData
            Object containing the parameters and results of the optimization

        """

        def loss_function_dc_offsets(dc_offsets):
            self._iqawg.output_continuous_IQ_waves(
                frequency=0, amplitudes=(0, 0),
                relative_phase=0, offsets=dc_offsets,
                waveform_resolution=waveform_resolution,
                optimized=self._optimized_awg_calls)
            data = self.get_spectrum()
            answer = data[0]

            print(f"\rDC offsets: {format_number_list(dc_offsets)} "
                  f"{format_number_list(data)}                  ",
                  flush=True)
            clear_output(wait=True)

            return answer

        def loss_function_dc_offsets_open(dc_offset_open):
            self._iqawg.output_continuous_IQ_waves(
                frequency=0, amplitudes=(0, 0),
                relative_phase=0, offsets=dc_offset_open,
                waveform_resolution=waveform_resolution,
                optimized=self._optimized_awg_calls)
            data = self.get_spectrum()
            answer = abs(data[0]-ssb_power)

            print(f"\rDC offsets open: {format_number_list(dc_offset_open)} "
                  f"{format_number_list(data)}                  ",
                  flush=True)
            clear_output(wait=True)
            return answer

        def loss_function_lo(if_offsets, args):
            if_amplitudes = args[0]
            phase = args[1][0]
            self._iqawg.output_continuous_IQ_waves(
                frequency=if_frequency, amplitudes=if_amplitudes,
                relative_phase=phase, offsets=if_offsets,
                waveform_resolution=waveform_resolution,
                optimized=self._optimized_awg_calls)
            data = self.get_spectrum()

            print(f"\rIF offsets: {format_number_list(if_offsets)}; "
                  f" LO power: {data[self._lo_freq_idx]}        ",
                  # f"{format_number_list(data)}             ",
                  flush=True)
            clear_output(wait=True)

            return data[self._lo_freq_idx]

        def loss_function_image_rejection(imbalance, args):
            r = imbalance[0]
            phi = imbalance[1]
            if_amplitudes = np.array([args[0][0], (1+r) * args[0][1]])
            phase = args[1][0] + phi
            if_offsets = args[2]
            self._iqawg.output_continuous_IQ_waves(
                frequency=if_frequency, amplitudes=if_amplitudes,
                relative_phase=phase, offsets=if_offsets,
                waveform_resolution=waveform_resolution,
                optimized=self._optimized_awg_calls)
            data = self.get_spectrum()
            im_rej = data[self._image_freq_idx] - data[self._target_freq_idx]

            print(f"\rAmplitudes: {format_number_list(if_amplitudes)}; "
                  f"Phase: {phase / np.pi * 180:.3f} °; "
                  f"Image Rejection: {im_rej:.3f} dBc; ",
                  # f"{format_number_list(data)}            ",
                  flush=True)
            clear_output(wait=True)

            return im_rej

        def loss_function_target_power(multiplier, args):
            if_amplitudes = multiplier[0] * args[0]
            phase = args[1][0]
            if_offsets = args[2]
            self._iqawg.output_continuous_IQ_waves(
                frequency=if_frequency, amplitudes=if_amplitudes,
                relative_phase=phase, offsets=if_offsets,
                waveform_resolution=waveform_resolution,
                optimized=self._optimized_awg_calls)
            data = self.get_spectrum()
            deviation = (data[self._target_freq_idx] - ssb_power)

            print(f"\rAmplitudes: {format_number_list(if_amplitudes)}; "
                  f"Sideband Power Deviation: {deviation:.5f} dBc",
                  # f"{format_number_list(data)}            ",
                  flush=True)
            clear_output(wait=True)

            return deviation**2

        def iterate_minimization(prev_results, n=2):
            method_options = {"maxfev": minimize_iterlimit,
                       "xtol": 1e-3,
                       "ftol": 1e-2}

            # res_dc_offs = minimize(loss_function_dc_offsets,
            #                        prev_results["dc_offsets"],
            #                        method="Powell",
            #                        options=method_options)
            # results["dc_offsets"] = res_dc_offs.x
            results['dc_offsets'] = prev_results['dc_offsets']

            res_imbalance = minimize(
                loss_function_image_rejection, np.array([0, 0]),
                args=[prev_results["if_amplitudes"],
                      prev_results["if_phase"],
                      prev_results["if_offsets"]],
                method="Powell", options=method_options)
            results["if_amplitudes"] = np.array([
                prev_results["if_amplitudes"][0],
                prev_results["if_amplitudes"][1] * (1 + res_imbalance.x[0])
            ])
            results["if_phase"] = prev_results["if_phase"] + res_imbalance.x[1]

            res_if_offs = minimize(
                loss_function_lo, prev_results["if_offsets"],
                args=[results["if_amplitudes"],
                      results["if_phase"]],
                method="Powell", options=method_options)
            results["if_offsets"] = res_if_offs.x

            res_mult = minimize(
                loss_function_target_power, np.array([1]),
                args=[results["if_amplitudes"],
                      results["if_phase"],
                      results["if_offsets"]],
                method="Powell", options=method_options)
            results["if_amplitudes"] = res_mult.x * results["if_amplitudes"]

            if n == 1:
                return
            iterate_minimization(results, n-1)

        results = None
        try:
            start = datetime.now()

            self._lo.set_power(lo_power)
            self._lo.set_frequency(lo_frequency)
            self._lo.set_output_state("ON")
            if initial_guess is None:
                results = {
                    "dc_offsets": (0, 0),
                    "dc_offset_open": (0, 0),
                    "if_offsets": (0, 0),
                    "if_amplitudes": (0.1, 0.1),
                    "if_phase": np.array([self._phases[
                        self._sideband_to_maintain]]),
                }
            else:
                results = initial_guess

            # TODO: add averages number
            self._sa.setup_list_sweep([lo_frequency], [sa_res_bandwidth],
                                      [sa_vid_bandwidth])

            options = {"maxfev": minimize_iterlimit,
                       "xtol": 1e-4,
                       "ftol": 0.1}
            res_dc_offs = minimize(loss_function_dc_offsets,
                                   results["dc_offsets"],
                                   method="Powell",
                                   options=options)

            results["dc_offsets"] = res_dc_offs.x

            if if_frequency == 0:
                # SA center if_freq already equals to LO
                for i in range(0, iterations):
                    res_dc_offs_open = minimize(
                        loss_function_dc_offsets_open,
                        results["dc_offset_open"][0],
                        method="Nelder-Mead",
                        options={"maxiter": minimize_iterlimit,
                                 "xatol": 1e-5,
                                 "fatol": 100}
                    )
                    self._iterations = 0
                    results["dc_offset_open"] = res_dc_offs_open.x
                spectral_values = {"dc": res_dc_offs.fun,
                                   "dc_open": self._sa.get_tracedata()}
                elapsed_time = (datetime.now() - start).total_seconds()
                return IQCalibrationData(
                    self._mixer_id, self._iq_attenuation, lo_frequency,
                    lo_power, if_frequency, self._sideband_to_maintain,
                    ssb_power, waveform_resolution, results["dc_offsets"],
                    np.array([results["dc_offset_open"]]*2), None, None,
                    None, spectral_values, elapsed_time, datetime.now())
            else:
                freqs = np.linspace(
                    lo_frequency - self._target_freq_idx * if_frequency,
                    lo_frequency + self._target_freq_idx * if_frequency,
                    2*self._target_freq_idx + 1
                )
                self._monitoring_frequencies_list = copy.deepcopy(freqs)

                if self._sideband_to_maintain == "left":
                    freqs -= if_frequency
                    self._lo_freq_idx = self._target_freq_idx + 1
                    self._image_freq_idx = self._target_freq_idx + 2
                elif self._sideband_to_maintain == "right":
                    freqs += if_frequency
                    self._lo_freq_idx = self._target_freq_idx - 1
                    self._image_freq_idx = self._target_freq_idx - 2

                self._sa.setup_list_sweep(list(freqs), [sa_res_bandwidth]*3,
                                          [sa_vid_bandwidth]*3)

                results["if_offsets"] = res_dc_offs.x
                iterate_minimization(results, iterations)
                spectral_values = {"dc": res_dc_offs.fun,
                                   "if": self._sa.get_tracedata()}
                elapsed_time = (datetime.now() - start).total_seconds()
                return IQCalibrationData(self._mixer_id, self._iq_attenuation,
                                         lo_frequency, lo_power, if_frequency,
                                         self._sideband_to_maintain,
                                         ssb_power, waveform_resolution,
                                         res_dc_offs.x, None,
                                         results["if_offsets"],
                                         results["if_amplitudes"],
                                         results["if_phase"], spectral_values,
                                         elapsed_time, datetime.now())

        except KeyboardInterrupt:
            return results

        finally:
            shift = if_frequency if self._sideband_to_maintain == "right"\
                else -if_frequency
            self._sa.setup_swept_sa(
                lo_frequency + shift,
                10*if_frequency if if_frequency > 0 else 1e9,
                nop=1001, rbw=1e5)
            self._sa.set_continuous()


def format_number_list(number_list):
    formatted_string = "[ "
    for number in number_list:
        formatted_string += "%3.3f " % number
    return formatted_string + "]"


def format_time_delta(delta):
    hours, remainder = divmod(delta, 3600)
    minutes, seconds = divmod(remainder, 60)
    return '%s h %s m %s s' % (int(hours), int(minutes), round(seconds, 2))

from copy import deepcopy

import numpy as np
import tqdm

from lib2.MeasurementResult import MeasurementResult
from lib2.stimulatedEmission import StimulatedEmission
from datetime import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import colorbar
from scipy import signal
import numba


@numba.njit(parallel=True)
def apply_along_axis(data, trace_len):
    temp = np.zeros((trace_len, trace_len),
                    dtype=np.complex128)
    for i in numba.prange(data.shape[0]):
        temp += np.kron(np.conj(data[i]), data[i]) \
            .reshape(trace_len, trace_len)
    return temp

class CorrelatorMeasurement(StimulatedEmission):

    def __init__(self, name, sample_name, comment,
                 q_lo=None, q_iqawg=None, dig=None):
        super().__init__(name, sample_name, comment, q_lo=q_lo,
                         q_iqawg=q_iqawg, dig=dig)
        self._measurement_result = CorrelatorResult(self._name,
                                                            self._sample_name)
        self.avg = None
        self.corr_avg = None
        self.avg_corr = None
        self._iterations_number = 0
        self._segments_number = 1
        self._freq_lims = None  # for digital filtering
        self._conv = None
        self._b = None
        self._temp = None

        self._pause_in_samples_before_next_trigger = 0  # > 80 samples

    def set_fixed_parameters(self, pulse_sequence_parameters,
                             freq_limits=(-90e6, 90e6), delay_correction=0,
                             down_conversion_calibration=None,
                             subtract_pi=False, q_lo_params=None,
                             q_iqawg_params=None, dig_params=None,
                             apply_filter=True, iterations_number=100,
                             do_convert=True):
        """

        Parameters
        ----------
        pulse_sequence_parameters: dict
            single pulse parameters
        freq_limits: tuple of 2 values
        delay_correction: int
         A correction of a digitizer delay given in samples
         For flexibility, it is applied after the measurement. For example,
         when you look at the trace and decide, that the digitizer delay
         should have been different
        down_conversion_calibration: IQDownconversionCalibrationResult
        subtract_pi: bool
            True if you want to make the Furier spectrum clearer by
            subtracting the same trace with pulses shifted by phase pi
        q_lo_params
        q_iqawg_params
        dig_params

        Returns
        -------
        Nothing
        """
        q_lo_params[0]["power"] = q_iqawg_params[0]["calibration"] \
            .get_radiation_parameters()["lo_power"]
        # a snapshot of initial seq pars structure passed into measurement
        self._pulse_sequence_parameters_init = deepcopy(
            pulse_sequence_parameters
        )

        super().set_fixed_parameters(
            pulse_sequence_parameters,
            freq_limits=freq_limits,
            down_conversion_calibration=down_conversion_calibration,
            q_lo_params=q_lo_params,
            q_iqawg_params=q_iqawg_params,
            dig_params=dig_params
        )

        self._delay_correction = delay_correction
        self._freq_lims = freq_limits
        self.apply_filter = apply_filter
        self._do_convert = do_convert

        # longest repetition period is initially set with data from
        # 'pulse_sequence_paramaters'
        self.max_segment_duration = \
            pulse_sequence_parameters["repetition_period"] * \
            pulse_sequence_parameters["periods_per_segment"]
        self._iterations_number = iterations_number
        self._segments_number = dig_params[0]["n_seg"]

        dig = self._dig[0]

        """ Supplying additional arrays to 'self._measurement_result' class """
        meas_data = self._measurement_result.get_data()
        # if_freq is already set in call of 'super()' class method
        # time in nanoseconds
        meas_data["sample_rate"] = dig.get_sample_rate()
        self._measurement_result.sample_rate = dig.get_sample_rate()
        self._measurement_result.set_data(meas_data)

    def dont_sweep(self):
        super().set_swept_parameters(
            **{
                "number": (
                    self._output_pulse_sequence, [False]
                )
            }
        )

    def _output_pulse_sequence(self, zero=False):
        dig = self._dig[0]
        timedelay = self._pulse_sequence_parameters["start_delay"] + \
                    self._pulse_sequence_parameters["digitizer_delay"]
        dig.calc_and_set_trigger_delay(timedelay, include_pretrigger=True)
        self._n_samples_to_drop_by_delay =\
            dig.get_how_many_samples_to_drop_in_front()
        """
            Because readout duration coincides with trigger
        period the very next trigger is missed by digitizer.
            Decreasing segment size by fixed amount > 80
        (e.g. 128) gives enough time to digitizer to catch the very next 
        trigger event.
            Rearm before trigger is quantity you wish to take into account
        see more on rearm timings in digitizer manual
        """
        # readout_start + readout_duration < repetition period - 100 ns
        dig.calc_segment_size(decrease_segment_size_by=self._pause_in_samples_before_next_trigger)
        # not working somehow, but rearming time
        # equals'80 + pretrigger' samples
        # maybe due to the fact that some extra values are sampled at the end
        # of the trace in order to make 'segment_size' in samples to be
        # dividable by 32 as required by digitizer

        self._n_samples_to_drop_in_end =\
            dig.get_how_many_samples_to_drop_in_end()

        dig.setup_current_mode()

        q_pbs = [q_iqawg.get_pulse_builder() for q_iqawg in self._q_iqawg]

        # TODO: 'and (self._q_z_awg[0] is not None)'  hotfix by Shamil (below)
        # I intend to declare all possible device attributes of the measurement class in it's child class definitions.
        # So hasattr(self, "_q_z_awg") is always True
        # due to the fact that I had declared this parameter and initialized it with "[None]" in RabiFromFrequencyTEST.py
        if hasattr(self, '_q_z_awg') and (self._q_z_awg[0] is not None):
            q_z_pbs = [q_z_awg.get_pulse_builder() for q_z_awg in
                       self._q_z_awg]
        else:
            q_z_pbs = [None]

        pbs = {'q_pbs': q_pbs,
               'q_z_pbs': q_z_pbs}

        if not zero:
            seqs = self._sequence_generator(self._pulse_sequence_parameters,
                                            **pbs)
            self.seqs = seqs
        else:
            self._q_iqawg[0].output_zero(
                trigger_sync_every=self._pulse_sequence_parameters["repetition_period"]
            )
            return

        for (seq, q_iqawg) in zip(seqs['q_seqs'], self._q_iqawg):
            # check if output trace length is dividable by awg's output
            # trigger clock period
            # TODO: The following lines are moved to the KeysightM3202A
            #  driver. Should be deleted later from here
            # if seq.get_duration() % \
            #         q_iqawg._channels[0].host_awg.trigger_clock_period != 0:
            #     raise ValueError(
            #         "AWG output duration has to be multiple of the AWG's "
            #         "trigger clock period\n"
            #         f"requested waveform duration: {seq.get_duration()} ns\n"
            #         f"trigger clock period: {q_iqawg._channels[0].host_awg.trigger_clock_period}"
            #     )
            q_iqawg.output_pulse_sequence(seq)
        if 'q_z_seqs' in seqs.keys():
            for (seq, dev) in zip(seqs['q_z_seqs'], self._q_z_awg):
                dev.output_pulse_sequence(seq, asynchronous=False)

    def _record_data(self):
        par_names = self._swept_pars_names

        start_time = dt.now()
        self._measurement_result.set_start_datetime(start_time)

        parameters_values = [self._swept_pars[parameter_name][1]
                             for parameter_name in par_names]

        # This should be implemented in child classes:
        self._raw_data = self._recording_iteration()

        # This may need to be extended in child classes:
        measurement_data = self._prepare_measurement_result_data(par_names,
                                                             parameters_values)
        self._measurement_result.set_data(measurement_data)
        self._measurement_result._iter_idx_ready = [len(parameters_values[0])]

        time_elapsed = dt.now() - start_time
        self._measurement_result.set_recording_time(time_elapsed)
        print(f"\nElapsed time: "
              f"{self._format_time_delta(time_elapsed.total_seconds())}")
        self._finalize()

    def _measure_one_trace(self):
        """
        Function starts digitizer measurement.
        Digitizer assumed already configured and waiting for start trace.

        Returns
        -------
        tuple(np.ndarray, np.ndarray)
            returns pair (time, data) np arrays
            time - real-valued 1D array. If down-conversion calibration is
            applied this array will differ from np.linspace
        """
        dig = self._dig[0]
        dig_data = dig.measure()  # data in mV
        # construct complex valued scalar trace
        data = dig_data[0::2] + 1j * dig_data[1::2]
        '''
        In order to allow digitizer to don't miss very next 
        trigger while the acquisition window is almost equal to 
        the trigger period, acquisition window is shrank
        by 'self.__pause_in_samples_before_trigger' samples.
        In order to obtain data of the desired length the code below adds
        'self.__pause_in_samples_before_trigger' trailing zeros to the end
        of each segment.
        Finally result is flattened in order to perform DFT.
        '''
        dig = self._dig[0]
        # 2D array that will be set to the trace avg value
        # and appended to the end of each segment of the trace
        # scalar average is multiplied by 'np.ones()' of the appropriate 2D
        # shape
        '''commented since  self._pause_in_samples_before_next_trigger is 
        zero'''
        # avgs_to_concat = np.full((dig.n_seg,
        #                           self._pause_in_samples_before_next_trigger),
        #                          np.mean(data))
        data = data.reshape(dig.n_seg, -1)
        # 'slice_stop' variable allows to avoid production of an empty list
        # by slicing operation. Empty slice is obtained if
        # 'slice_stop = -self._n_samples_to_drop_in_end' and equals zero.
        slice_stop = data.shape[-1] - self._n_samples_to_drop_in_end
        # dropping samples from the end to get needed duration
        data = data[:, self._n_samples_to_drop_by_delay:slice_stop]
        # append average to the end of trace to complete overall duration
        # to 'repetition_period' or whatever
        '''commented since  self._pause_in_samples_before_next_trigger is 
        zero '''
        # data = np.hstack((data, avgs_to_concat))

        # for debug purposes, saving raw data
        if self._save_traces:
            self.dataIQ.append(data)

        # Applying mixer down-conversion calibration to data
        time = np.linspace(
            0,
            data.shape[-1] / dig.get_sample_rate() * 1e9,
            data.shape[-1],
            endpoint=False
        )

        if self._down_conversion_calibration is not None:
            data = self._down_conversion_calibration.apply(data)

            if "time_cal" not in self._measurement_result.get_data():
                tmp = self._measurement_result.get_data()
                shift = self._delay_correction
                tmp["time_cal"] = time[shift:self._nfft + shift]
                self._measurement_result.set_data(tmp)

        return time, data

    def _recording_iteration(self):
        for i in tqdm.tqdm_notebook(range(self._iterations_number)):
            # measuring trace
            self._output_pulse_sequence()
            time, data = self._measure_one_trace()
            # measuring only noise
            self._output_pulse_sequence(zero=True)
            time_bg, data_bg = self._measure_one_trace()

            trace_len = data.shape[-1]

            # down-converting to DC
            if self._do_convert:
                if_freq = self._q_iqawg[0].get_calibration().get_if_frequency()
                self._conv = np.exp(-2j * np.pi * if_freq * time / 1e9)
                self._conv = np.resize(self._conv, data.shape)
                data = data * self._conv
                data_bg = data_bg * self._conv
            # filtering excessive frequencies
            if self.apply_filter:
                if self._b is None:
                    if self._do_convert:
                        self._b = signal.firwin(trace_len, self._freq_lims[1],
                                  fs=self._dig[0].get_sample_rate())
                    else:
                        self._b = signal.firwin(len(data), self._freq_lims,
                                  fs=self._dig[0].get_sample_rate(),
                                  pass_zero=(self._freq_lims[0] < 0 <
                                             self._freq_lims[1]))
                    self._b = np.resize(self._b, data.shape)
                data = signal.fftconvolve(self._b, data, mode="same", axes=-1)
                data_bg = signal.fftconvolve(self._b, data_bg, mode="same",
                                             axes=-1)
            # initializing arrays for correlators storage
            if self.avg is None:
                self.avg = np.zeros(trace_len, dtype=np.clongdouble)
                self.corr_avg = np.zeros((trace_len, trace_len),
                                         dtype=np.clongdouble)
                self.avg_corr = np.zeros((trace_len, trace_len),
                                         dtype=np.clongdouble)

            # processing data with trace applied
            avg_corrs = apply_along_axis(data, trace_len)
            # avg_corrs = np.apply_along_axis(
            #     lambda x: np.reshape(
            #         np.kron(np.conj(x), x),
            #         (trace_len, trace_len)
            #     ),
            #     1,  # axis that function is applied along (along traces)
            #     data
            # ).sum(axis=0)
            # processing background data
            avg_bg_corrs = apply_along_axis(data_bg, trace_len)
            # avg_bg_corrs = np.apply_along_axis(
            #     lambda x: np.reshape(
            #         np.kron(np.conj(x), x),
            #         (trace_len, trace_len)
            #     ),
            #     1,  # axis along which function is applied (along traces)
            #     data_bg
            # ).sum(axis=0)
            # <E(t)>
            self.avg += data.sum(axis=0)
            # <E+(t1)><E(t2)>
            self.corr_avg = np.reshape(
                np.kron(np.conj(self.avg), self.avg),
                (trace_len, trace_len)
            )
            # <E+(t1) E(t2)>
            self.avg_corr += avg_corrs - avg_bg_corrs
            # Saving preliminary data in the Measurement Result
            K = (i + 1) * self._segments_number
            self._measurement_result.corr_avg = self.corr_avg.copy() / K**2
            self._measurement_result.avg_corr = self.avg_corr.copy() / K

        # returning the final result
        K = self._iterations_number * self._segments_number
        return self.corr_avg / K**2, self.avg_corr / K

class CorrelatorResult(MeasurementResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._XX = None
        self._YY = None
        self.corr_avg = None
        self.avg_corr = None
        self.sample_rate = 1.25e9

    def set_parameter_name(self, parameter_name):
        self._parameter_name = parameter_name

    def _prepare_figure(self):
        fig, (ax_map_re, ax_map_im) = plt.subplots(nrows=1, ncols=2,
        constrained_layout=True, figsize=(17, 8), sharex=True, sharey=True)

        labelx = "$t_1$, ns"
        labely = "$t_2$, ns"
        ax_map_re.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_map_re.set_ylabel(labely)
        ax_map_re.set_xlabel(labelx)
        ax_map_re.autoscale_view(True, True, True)
        ax_map_im.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_map_im.set_xlabel(labelx)
        ax_map_im.autoscale_view(True, True, True)
        # plt.tight_layout(pad=1, h_pad=2, w_pad=-7)
        cax_re, kw = colorbar.make_axes(ax_map_re, aspect=40)
        cax_im, kw = colorbar.make_axes(ax_map_im, aspect=40)
        ax_map_re.set_title(r"$\langle a^\dagger(t_1)a(t_2) \rangle$")
        # ax_map_im.set_title(r"$\langle a^\dagger(t_1)\rangle \langle a(t_2) "
        #                     r"\rangle$")
        ax_map_im.set_title(r"$\langle a^\dagger(t_1)a(t_2) \rangle$ - "
                            r"$\langle a^\dagger(t_1)\rangle \langle a(t_2) "
                            r"\rangle$")
        ax_map_re.grid(False)
        ax_map_im.grid(False)
        fig.canvas.set_window_title(self._name)
        return fig, (ax_map_re, ax_map_im), (cax_re, cax_im)

    def _plot(self, data):
        # if (self.corr_avg is None) and (self.avg_corr is None):
        if (self.corr_avg is None) or (self.avg_corr is None):
            return
        ax_corr, ax_diff = self._axes
        cax_corr = self._caxes[0]
        cax_diff = self._caxes[1]
        # if self._XX is None:
        #     return

        XX, YY, corr, diff = self._prepare_data_for_plot()

        corr_max = np.max(corr)
        corr_min = np.min(corr)
        diff_max = np.max(diff)
        diff_min = np.min(diff)

        step = XX[1, 0] - XX[0, 0]
        self.extent = (XX[0, 0] - 0.5 * step, XX[0, -1] + 0.5 * step,
                  YY[0, 0] - 0.5 * step, YY[-1, 0] + 0.5 * step)
        if self.extent[2] == self.extent[3]:
            return
        corr_map = ax_corr.imshow(corr, origin='lower', cmap="RdBu_r",
                                      aspect='auto', vmax=corr_max,
                                      vmin=corr_min, extent=self.extent)
        cax_corr.cla()
        plt.colorbar(corr_map, cax=cax_corr)
        cax_corr.tick_params(axis='y', right=False, left=True,
                             labelleft=True, labelright=False, labelsize='10')

        diff_map = ax_diff.imshow(diff, origin='lower', cmap="RdBu_r",
                                         aspect='auto', vmax=diff_max,
                                         vmin=diff_min, extent=self.extent)
        cax_diff.cla()
        plt.colorbar(diff_map, cax=cax_diff)
        cax_diff.tick_params(axis='y', right=False, left=True,
                               labelleft=True, labelright=False,
                               labelsize='10')

    def _prepare_data_for_plot(self):
        time = np.linspace(0,
                           self.avg_corr.shape[0] / self.sample_rate * 1e9,
                           self.avg_corr.shape[0])
        self._XX, self._YY = np.meshgrid(time, time)
        return self._XX, self._YY, np.real(self.avg_corr),\
               np.real(self.avg_corr - self.corr_avg)
                # np.real(self.corr_avg)
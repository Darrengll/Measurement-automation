from lib2.Measurement import Measurement
from lib2.MeasurementResult import MeasurementResult
import numpy as np
from importlib import reload
from drivers.keysightM3202A import KeysightM3202A
import inspect
from scipy import signal
from scipy import fftpack
import matplotlib.pyplot as plt
import multiprocessing as mp
import threading as th
from time import sleep
from datetime import datetime as dt
import os
import psutil

import lib2.IQPulseSequence
reload(lib2.IQPulseSequence)
from lib2.IQPulseSequence import IQPulseBuilder

class MollowTriplet(Measurement):

    def __init__(self, name, sample_name, plot_update_interval=1,
                 q_lo=[], q_iqawg=[], dig=[]):
        # mandatory names for devices in devs_aliases_map
        self._q_iqawg = None
        self._q_lo = None
        self._dig = None
        devs_aliases_map = {"q_lo": q_lo,
                            "q_iqawg": q_iqawg,
                            "dig": dig}
        super().__init__(name, sample_name, devs_aliases_map,
                         plot_update_interval=plot_update_interval)
        self._measurement_result = MollowTripletResult(name, sample_name)

        self._ult_calib = False
        self._n_samples_to_drop_by_dig_delay = 0
        self._n_samples_to_drop_in_end = 0

        # how many averages are performed until data is stored into
        # 2-nd level averaging array
        self._internal_avg = 0
        self._internal_data = None  # numpy array for storing intermediate average data
        self.__internal_data_bg = None  # numpy array for storing intermediate average data of background noise

        # Fourier and measurement parameters
        # see purpose in 'self.set_fixed_parameters()'
        self._freq_limits = None  # tuple with frequency limits
        self._nfft = 0  # number of FFT points
        self._frequencies = None
        # self._frequencies[self._start_idx-1]  < self._freq_limits[0] <= self._frequencies[self._start_idx]
        self._start_idx = None
        # self._frequencies[self._end_idx-1]  < self._freq_limits[1] <= self._frequencies[self._end_idx]
        self._end_idx = None

        self.data_queue = mp.Queue()
        self.fft_queue = mp.Queue()
        self.measuring_flag = False

    def set_fixed_parameters(self, internal_avg=100, freq_limits=(0,50e6), **dev_params):
        """
        :param dev_params:
            Minimum expected keys and elements expected in each:
                'vna': 0
                'q_awg': 0
                'ro_awg': 0

        Parameters
        ----------
        internal_avg : int
            how many averages are performed until data is stored into
            2-nd level averaging array
        digitizer_delay
        """
        self._measurement_result._iter = 0
        super().set_fixed_parameters(**dev_params)
        self.__setup_digitizer()
        self._measurement_result.get_context().update(
            {"calibration_results": self._q_iqawg[0]._calibration.get_optimization_results(),
             "radiation_parameters": self._q_iqawg[0]._calibration.get_radiation_parameters()}
        )

        # how many averages are performed until data is stored into
        # 2-nd level averaging array
        # for now, averages are performed during the "ON" or "OFF" period
        # of the microwave source
        self._internal_avg = int(internal_avg)

        self.__setup_signal_source()

        # Fourier and measurement parameters setup
        self._freq_limits = freq_limits
        trace_len = self._dig[0]._segment_size - self._n_samples_to_drop_in_end - self._n_samples_to_drop_by_dig_delay
        self._nfft = fftpack.helper.next_fast_len(trace_len)
        xf = fftpack.fftshift(fftpack.fftfreq(self._nfft, 1 / self._dig[0].get_sample_rate()))
        self._start_idx = np.searchsorted(xf, self._freq_limits[0])
        self._end_idx = np.searchsorted(xf, self._freq_limits[1])
        self._frequencies = xf[self._start_idx:self._end_idx + 1]
        self._internal_data = np.zeros(self._frequencies.shape[0], dtype=np.float64)
        self._internal_data_bg = np.zeros(self._frequencies.shape[0], dtype=np.float64)

        self._measurement_result._data["frequencies"] = self._frequencies
        self._measurement_result._data["data"] = self._internal_data.copy()

        # Find index of the carrier frequency and store into result
        # This frequency is excluded from y-scaling of the visualization
        self._measurement_result._if_freq_idx = np.argmin(np.abs(self._frequencies+self._q_iqawg[0]._calibration._if_frequency))

        # Array to store temporary data of internal averages (these are not saved to disk and lost forever)
        # self._internal_data = np.zeros(self._frequencies.shape[0], dtype=np.float64)

        # temporary for division testing see 'self.record_iteration'
        # self.__internal_data_bg = np.ones((self._internal_avg, self._frequencies.shape[0]), dtype=np.float64)

    def set_swept_parameters(self, iterations):
        def dummy_setter(parameter):
            # print(parameter)
            pass
        swept_pars= {"iterations": (dummy_setter, range(1, iterations+1))}
        super().set_swept_parameters(**swept_pars)

    def __setup_digitizer(self):
        dig = self._dig[0]

        time_delay = 0
        dig.calc_and_set_trigger_delay(time_delay, include_pretrigger=True)
        self._n_samples_to_drop_by_dig_delay = dig.get_how_many_samples_to_drop_in_front()

        dig.calc_segment_size(extra=self._n_samples_to_drop_by_dig_delay)
        dig.setup_standard_mode()
        self._n_samples_to_drop_in_end = dig.get_how_many_samples_to_drop_in_end()

    def __setup_signal_source(self):
        self._q_lo[0].set_frequency(self._q_iqawg[0]._calibration._lo_frequency)
        self._q_lo[0].set_power(self._q_iqawg[0]._calibration._lo_power)
        self._q_lo[0].set_output_state("ON")

    def set_ult_calib(self, value=False):
        self._ult_calib = value

    """This method will be launched in a new thread of the main process and therefore does not have to be static,
    as it is allowed access to shared memory. It is convenient that this method should stay in the local process, as it
    requires access to measurement equipment and most of of the time does nothing, except waiting for result.
    Threading almost does not speed up the execution."""
    def _measurer(self):
        self.measuring_flag = True
        if self._ult_calib:
            for i in range(self._internal_avg):
                self.turn_signal_on()
                fg = self._single_measurement()
                self.turn_signal_off()
                bg = self._single_measurement()
                self.data_queue.put((fg, bg))
        else:
            self.turn_signal_on()
            for i in range(self._internal_avg):
                fg = self._single_measurement()
                self.data_queue.put((fg,))
        self.data_queue.put(None)
        self.measuring_flag = False

    def _measurer_serial(self):
        self.measuring_flag = True
        if self._ult_calib:
            for i in range(self._internal_avg):
                self.turn_signal_on()
                fg = self._single_measurement()
                self.turn_signal_off()
                bg = self._single_measurement()
                self.data_queue.put((fg, bg))
        else:
            self.turn_signal_on()
            for i in range(self._internal_avg):
                fg = self._single_measurement()
                self.data_queue.put((fg,))
        self.data_queue.put(None)
        self.measuring_flag = False

    """The method has to be static, because it will be sent to another process and has no shared memory
        with the main process, except for the arguments of a function. Creating new processes speeds up the execution"""
    @staticmethod
    def _power_spectrum(data_queue, fft_queue, nfft, start_idx, end_idx, ult_calib, counter, lock):
        p = psutil.Process()
        N = 100
        n = 0
        buff_fg = np.zeros(end_idx - start_idx + 1)
        buff_bg = np.zeros(end_idx - start_idx + 1)
        p.nice(psutil.HIGH_PRIORITY_CLASS)
        if ult_calib:
            for (fg_trace, bg_trace) in iter(data_queue.get, None):
                # fg_pd = np.abs(signal.welch(fg_trace, 1.25e9, nperseg=nfft)[1])
                # bg_pd = np.abs(signal.welch(bg_trace, 1.25e9, nperseg=nfft)[1])
                # if len(buff_fg) != len(fg_pd):
                #     buff_fg = np.zeros(len(fg_pd))
                #     buff_bg = np.zeros(len(bg_pd))
                fg_spectrum = fftpack.fftshift(fftpack.fft(fg_trace, nfft)) / nfft
                bg_spectrum = fftpack.fftshift(fftpack.fft(bg_trace, nfft)) / nfft
                fg_spectrum = fg_spectrum[start_idx:end_idx + 1]
                bg_spectrum = bg_spectrum[start_idx:end_idx + 1]
                buff_fg = np.add(np.abs(fg_spectrum) ** 2, buff_fg)
                buff_bg = np.add(np.abs(bg_spectrum) ** 2, buff_bg)
                # buff_fg = np.add(fg_pd, buff_fg)
                # buff_bg = np.add(bg_pd, buff_bg)
                n += 1
                if n == N:
                    fft_queue.put((buff_fg, buff_bg))
                    buff_fg = np.zeros(end_idx - start_idx + 1)
                    buff_bg = np.zeros(end_idx - start_idx + 1)
                    n = 0
        else:
            for (fg_trace,) in iter(data_queue.get, None):
                fg_spectrum = fftpack.fftshift(fftpack.fft(fg_trace, nfft)) / nfft
                fgsp_cut = fg_spectrum[start_idx:end_idx + 1]
                buff_fg = np.add(np.abs(fgsp_cut) ** 2, buff_fg)
                n += 1
                if n == N:
                    fft_queue.put((buff_fg, ))
                    buff_fg = np.zeros(end_idx - start_idx + 1)
                    n = 0
            if n > 0:
                fft_queue.put((buff_fg))
        data_queue.put(None)
        with lock:
            counter.value -= 1

    """This method will be launched in a new thread of the main process and therefore does not have to be static,
    as it is allowed access to shared memory. It is convenient that this method should stay in the local process, as it
    executes one simple task: adding new data to the all one. Threading almost does not speed up the execution."""
    def _summator(self, counter, lock, done_iterations, lock2):
        is_running = True
        N = 100
        while is_running or not self.fft_queue.empty():
            with lock:
                if counter.value == 0:
                    is_running = False
            if not self.fft_queue.empty():
                if self._ult_calib:
                    power_spectrum_fg, power_spectrum_bg = self.fft_queue.get()
                    self._internal_data = np.add(self._internal_data, power_spectrum_fg)
                    self._internal_data_bg = np.add(self._internal_data_bg, power_spectrum_bg)
                else:
                    power_spectrum_fg, = self.fft_queue.get()
                    self._internal_data = np.add(self._internal_data, power_spectrum_fg)
                with lock2:
                    done_iterations.value += N

    def _record_data(self):
        start_time = self._measurement_result.get_start_datetime()
        number_of_workers = 3 #mp.cpu_count() # PXI CPU has 8 cores

        done_iterations = mp.Value("i", 0)
        lock1 = mp.Lock()
        active_workers_number = mp.Value("i", number_of_workers)
        lock2 = mp.Lock()

        self.measuring_flag = True
        measurer = th.Thread(target=self._measurer)
        measurer.start()
        worker_args = (self.data_queue, self.fft_queue, self._nfft, self._start_idx, self._end_idx, self._ult_calib,
                       active_workers_number, lock2)
        pool = mp.Pool(number_of_workers, MollowTriplet._power_spectrum, worker_args)
        summator = th.Thread(target=self._summator, args=(active_workers_number, lock2, done_iterations, lock1))
        summator.start()

        while summator.is_alive():
            with lock1:
                time_since = (dt.now() - start_time).total_seconds()
                if done_iterations.value > 0:
                    avg_time = time_since / done_iterations.value
                    measurement_data = self._measurement_result.get_data()
                    if self._ult_calib:
                        measurement_data["data"] = self._internal_data.copy() / self._internal_data_bg.copy()
                    else:
                        measurement_data["data"] = self._internal_data.copy()
                    self._measurement_result.set_data(measurement_data)
                else:
                    avg_time = time_since
                time_left = self._format_time_delta(avg_time * (self._internal_avg - done_iterations.value))

                print(f"Time left: {time_left}, iteration number: {done_iterations.value}, "
                      f"measurement queue size: {self.data_queue.qsize()}, fft queue size: {self.fft_queue.qsize()}, "
                      f"average cycle time: {round(avg_time, 2)} s",
                      end="\r", flush=True)

            if self._interrupted:
                self._dig[0].stop_card()
                measurer.join()
                pool.terminate()
                summator.join()
                return
            sleep(5.0)

        measurement_data = self._measurement_result.get_data()
        if self._ult_calib:
            measurement_data["data"] = self._internal_data.copy() / self._internal_data_bg.copy()
        else:
            measurement_data["data"] = self._internal_data.copy() / self._internal_avg
        self._measurement_result.set_data(measurement_data)

        measurer.join()
        summator.join()
        pool.terminate()
        pool.close()

        self._measurement_result.set_recording_time(dt.now() - start_time)
        print("\nElapsed time: %s" % self._format_time_delta((dt.now() - start_time)
                                                             .total_seconds()))
        self._finalize()

    def _single_measurement(self):
        dig = self._dig[0]
        dig_data = dig.measure(dig._bufsize)
        # WTF??
        # dig_data = (2*(dig_data / dig.n_avg + 128) / 255 - 1) * dig.ch_amplitude
        dig_data = dig_data / dig.n_avg / 128 * dig.ch_amplitude
        data_i = dig_data[0::2]
        data_i = data_i[self._n_samples_to_drop_by_dig_delay: -self._n_samples_to_drop_in_end]

        data_q = dig_data[1::2]
        data_q = data_q[self._n_samples_to_drop_by_dig_delay: -self._n_samples_to_drop_in_end]

        return data_i + 1j * data_q

    def _recording_iteration(self):
        """
        Averages are performed during the "ON" or "OFF" period
        of the microwave source. This may be changed in nearest few days.

        Returns
        -------

        """

        if self._ult_calib:
            self.turn_signal_on()
            for i in range(self._internal_avg):
                fg = self._single_measurement()
                self.__internal_data[i] = np.abs(fg)**2

            self.turn_signal_off()
            for i in range(self._internal_avg):
                bg = self._single_measurement()
                # the proper way is substract bg, but here I need to check the last working
                # result that involved division
                self.__internal_data_bg[i] = np.abs(bg)**2
        else:
            self.turn_signal_on()
            for i in range(self._internal_avg):
                fg = self._single_measurement()
                self.__internal_data[i] = np.abs(fg) ** 2

        return np.mean(self.__internal_data, axis=0)/np.mean(self.__internal_data_bg, axis=0)

    def turn_signal_on(self):
        # DC mode
        # v_max_tuple = tuple((awg_channel._host_awg.MAX_OUTPUT_VOLTAGE for awg_channel in self._q_iqawg[0]._channels))
        # self._q_iqawg[0].output_continuous_IQ_waves(0, (0, 0), 0, v_max_tuple, 1)
        self._q_iqawg[0].output_IQ_waves_from_calibration()

    def turn_signal_off(self):
        self._q_iqawg[0].output_zero()


class MollowTripletResult(MeasurementResult):
    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._line = None
        self._iter_idx = 0

        # index of 'if_freq' harmonic to drop in visualization process
        self._if_freq_idx = None

    def _prepare_figure(self):
        fig = plt.figure(figsize=(15, 7))
        fig.canvas.set_window_title(self._name)

        ax = fig.add_subplot(111)
        return fig, (ax,), (None, None)

    def _plot(self, data):
        axes = self._axes
        ax = axes[0]
        if "data" not in data.keys():
            return

        freqs, power = self._prepare_data_for_plot(data)
        noise_power_min = np.min(power)
        noise_power_max = np.max(np.delete(power, self._if_freq_idx))
        # print(noise_power_max, noise_power_min)
        power_scale = np.max([noise_power_max - noise_power_min, 1.0])

        # if this is the first call to '_plot'
        if self._line is None or not self._dynamic:
            self._line, = ax.plot(freqs, power+0.1)
        else:  # line is already initialized
            self._line.set_data(freqs, power)
        ax.set_ylim(noise_power_min - 0.1*power_scale,
                    noise_power_max + 0.1*power_scale)
        ax.autoscale_view()

        # ax.set_ylim(-0.5, 0.5)

        plt.tight_layout(pad=2)
        return (self._line,)

    def _prepare_data_for_plot(self, data):
        tmp_dat = data["data"]  # save intermediate result into variable
        return data["frequencies"], tmp_dat

'''
We are using
Spectrum m4x2212-x4
https://spectrum-instrumentation.com/en/m4x2212-x4

manual:
https://spectrum-instrumentation.com/sites/default/files/download/m4i_m4x_22xx_manual_english.pdf

datasheet:
https://spectrum-instrumentation.com/sites/default/files/download/m4x22_datasheet_english.pdf
'''


import numpy as np
from scipy import fftpack
from scipy.fftpack import fft, fftfreq
import matplotlib.pyplot as plt
from enum import Enum

from drivers.pyspcm import *


class CardError(Exception):
    pass


class CardTimeoutError(TimeoutError):
    pass


class SPCM_MODE(Enum):
    STANDARD = "STANDARD"
    MULTIMODE = "MULTIMODE"
    AVERAGING = "AVERAGE"
    UNDEFINED = "UNDEFINED"


class SPCM_TRIGGER(Enum):
    AUTOTRIG = "AUTOTIG"
    EXT0 = "EXT0"


class SPCM:
    DC, AC = 0, 1
    AVG_ON = False
    MODEL_NAME = "M4X.2212-X4"

    def __init__(self, path):
        self.hCard = spcm_hOpen(create_string_buffer(b'/dev/spcm0'))
        self.__samplerate = 1250000000  # Hz
        self.__oversampling = 1
        self.__antialiasing = 0
        self.__acdc = self.AC

        self._segment_size: int = None
        self._bufsize: int = 0
        self._trigger_mode = SPC_TM_POS
        self._n_samples_to_drop_by_dig_delay: int = 0
        self._n_samples_to_drop_in_end: int = 0

        self.channels: list[int] = []
        self.ch_amplitude: int = 0
        self.dur_seg: int = 0
        self.n_avg: int = 0
        self.n_seg: int = 0
        self.pretrigger: int = 0
        self.mode: SPCM_MODE = SPCM_MODE.STANDARD
        self.trigger_source: SPCM_TRIGGER = SPCM_TRIGGER.AUTOTRIG

    def __del__(self):
        self.close()

    def set_parameters(self, pars_dict):
        if "oversampling_factor" in pars_dict:
            self.set_oversampling_factor(pars_dict["oversampling_factor"])
        if "channels" in pars_dict:
            self.channels = pars_dict["channels"]
        if "ch_amplitude" in pars_dict:
            self.ch_amplitude = pars_dict["ch_amplitude"]
        if "dur_seg" in pars_dict:
            self.dur_seg = pars_dict["dur_seg"]
        if "n_avg" in pars_dict:
            self.n_avg = pars_dict["n_avg"]
        if "n_seg" in pars_dict:
            self.n_seg = pars_dict["n_seg"]
        if "pretrigger" in pars_dict:
            self.pretrigger = pars_dict["pretrigger"]
        if "mode" in pars_dict:
            self.mode = pars_dict["mode"]
        if "trig_source" in pars_dict:
            self.trigger_source = pars_dict["trig_source"]

        # this two variables will be set by the subsequent function call
        self._n_samples_to_drop_by_dig_delay = 0
        self._n_samples_to_drop_in_end = 0

        # calculates and sets
        # segment measurement length in samples
        # based on duration of the segment requested by user
        self.calc_and_set_segment_size(self.dur_seg)

        if self.mode == SPCM_MODE.STANDARD:
            self.setup_standard_mode()
        elif self.mode == SPCM_MODE.MULTIMODE:
            self.setup_multiple_recoding_mode()
        elif self.mode == SPCM_MODE.AVERAGING:
            self.setup_averaging_mode()
        elif self.mode == SPCM_MODE.UNDEFINED:
            # mode was intentionally left underfined so the real mode of
            # operation will be decided later by the measurement class
            pass

    def close(self):
        spcm_vClose(self.hCard)

    def __read_reg_32(self, REG):
        """Read out a 32-bit register and return its value"""
        val = int32()
        spcm_dwGetParam_i32(self.hCard, REG, byref(val))
        return val.value

    def __read_reg_64(self, REG):
        """Read out a 64-bit register and return its value"""
        val = int64()
        spcm_dwGetParam_i64(self.hCard, REG, byref(val))
        return val.value

    def __write_to_reg_32(self, REG, VALUE):
        """Write to a 32-bit register"""
        val = int32(VALUE)
        return spcm_dwSetParam_i32(self.hCard, REG, val)

    def __write_to_reg_64(self, REG, VALUE):
        """Write to a 64-bit register"""
        val = int64(VALUE)
        return spcm_dwSetParam_i64(self.hCard, REG, val)

    def __def_simp_transfer(self, buffer, notifysize=0, offset=0):
        """Define simple transfer"""
        return spcm_dwDefTransfer_i64(self.hCard, SPCM_BUF_DATA, SPCM_DIR_CARDTOPC, int32(notifysize), byref(buffer),
                                      int64(offset), int64(buffer.__len__()))

    def __invalidate_buffer(self):
        """Invalidate the buffer in the digitizer"""
        return spcm_dwInvalidateBuf(self.hCard, SPCM_BUF_DATA)

    def __handle_error(self):
        """Handle an error in the digitizer"""
        error_text = create_string_buffer(ERRORTEXTLEN)
        error_reg = int32()
        error_value = int32()
        err = spcm_dwGetErrorInfo_i32(self.hCard, byref(error_reg), byref(error_value), error_text)
        if err is not ERR_OK:
            raise CardError(error_text.value)

    def __handle_timeout(self, ret):
        """
        This function checks the return value of any low-level driver
        function that waits for card to respond
        and generates an exception in case timeout is exceeded.

        This is the proper way to handle timeout error
        according to content of page 47 of the device manual.

        TODO: test this function

        Parameters
        ----------
        ret : int
            return value of the function that waits for the card state.

        Returns
        -------
        ret : int
            Returns argument in case of success.
            Raises an exception otherwise.

        Examples
        -------
        self.__handle_timeout(
                self.__write_to_reg_32(SPC_M2CMD, M2CMD_CARD_WAITREADY)  # Wait till the card completes the current run
            )
        """
        if ret == ERR_TIMEOUT:
            raise TimeoutError("Execution exceeded the allowed timeout. Reset the Spectrum card")
        return ret

    def set_timeout(self, timeout):
        """
        Set time overflow in the digitizer

        timeout : int
            timeout in ms?
        """
        self.__write_to_reg_32(SPC_TIMEOUT, timeout)

    def setup_SSA(self, memsize, posttrigger_mem):
        """ Setup Standart Single Aquisition mode
            Acquire data immediately and save them in Spectrum memory
            Parameters:
            -----------
            memsize: int, samples
                memory size for a single data acquisition in samples
                memsize = pretrigger_mem + posttrigger_mem
            posttrigger_mem: int, samples
                Posttrigger memory size for a single data acquisition in samples"""
        self.__write_to_reg_32(SPC_CARDMODE, SPC_REC_STD_SINGLE)  # Standard Single acquisition mode
        self.__write_to_reg_64(SPC_MEMSIZE, memsize)  # Set memory size for a single data acquisition
        self.__write_to_reg_32(SPC_POSTTRIGGER, posttrigger_mem)  # Post trigger memory size
        self.AVG_ON = False  # Set the averaging flag off
        self.__handle_error()

    def setup_block_avg_STD(self, memsize, segmentsize, posttrigger, averages):
        """ Setup Standart Single Aquisition mode with the Block Averaging Module
            Acquire data immediately and save them in Spectrum memory"""
        self.__write_to_reg_32(SPC_CARDMODE, SPC_REC_STD_AVERAGE)  # Enables Segment Statistic for standard acquisition
        self.__write_to_reg_32(SPC_AVERAGES, averages)
        self.__write_to_reg_32(SPC_SEGMENTSIZE, segmentsize)
        self.__write_to_reg_32(SPC_POSTTRIGGER,
                               posttrigger)  # Post trigger memory size (pretrigger  = segmentsize - posttrigger)
        self.__write_to_reg_32(SPC_MEMSIZE, memsize)
        self.AVG_ON = True  # Set the averaging flag on
        self.__handle_error()

    def setup_multi_rec_STD(self, memsize, segmentsize, posttrigger):
        """Setup Multiple Reconding Acquisition mode
            Acquires many segments (N = memsize/segmentsize) and saves them in Spectrum memory"""
        self.__write_to_reg_32(SPC_CARDMODE, SPC_REC_STD_MULTI)  # Enables Segment Statistic for standard acquisition
        self.__write_to_reg_32(SPC_MEMSIZE, memsize)
        self.__write_to_reg_32(SPC_SEGMENTSIZE, segmentsize)
        self.__write_to_reg_32(SPC_POSTTRIGGER,
                               posttrigger)  # Post trigger memory size (pretrigger  = segmentsize - posttrigger)
        self.AVG_ON = False
        self.__handle_error()

    def setup_channel(self, channelnum, amplitude):
        """Setup channels of the Digitizer
            Parameters:
            -----------
            channelnum: int
                number of a channel (from 0 to 3)
            amplitude: int, mV
                Amplitude window of a channel is (-amplitude, +amplitude),
                possible values are 200, 500, 1000, 2500 mV"""
        self.__write_to_reg_32(SPC_CHENABLE, CHANNEL0 << channelnum)  # Enable the selected channel
        self.__write_to_reg_32(SPC_AMP0 + 100 * channelnum,
                               amplitude)  # Set the input amplitude (valid values: 200, 500, 1000, 2500 mV)
        self.__write_to_reg_32(SPC_ACDC0 + 100 * channelnum, self.__acdc)  # 0 - DC input, 1 - AC input
        self.__write_to_reg_32(SPC_FILTER0, self.__antialiasing)  # 0 - off, 1 - anti aliasing filter for all channels
        self.__handle_error()

    def setup_channels(self, channels, amplitude):
        """Setup channels of the Digitizer
            Parameters:
            -----------
            channels: int or an array of int
                number of channels (from 0 to 3) to be set
            amplitude: int or array of int, mV
                Amplitude window of a channel is (-amplitude, +amplitude),
                possible values are 200, 500, 1000, 2500 mV"""
        mask = 0b0
        for chan in channels:
            mask |= CHANNEL0 << chan
            self.__write_to_reg_32(SPC_AMP0 + 100 * chan,
                                   amplitude)  # Set the input amplitude (valid values: 200, 500, 1000, 2500 mV)
            self.__write_to_reg_32(SPC_ACDC0 + 100 * chan, self.__acdc)  # 0 - DC input, 1 - AC input
        self.__write_to_reg_32(SPC_FILTER0, self.__antialiasing)  # 0 - off, 1 - anti aliasing filter for all channels
        self.__write_to_reg_32(SPC_CHENABLE, mask)
        self.__handle_error()

    def set_antialiasing(self, on):
        self.__antialiasing = 1 if on else 0

    def is_set_antialiasing(self):
        return True if self.__antialiasing > 0 else False

    def set_ACDC(self, acdc):
        self.__acdc = acdc

    def is_set_AC_or_DC(self):
        return self.__acdc

    def get_maximal_trigger_delay(self):
        return {
            "Maximal delay": self.__read_reg_64(SPC_TRIG_AVAILDELAY),
            "Delay step": self.__read_reg_32(SPC_TRIG_AVAILDELAY_STEP)
        }

    def set_trigger_delay(self, delay_in_samples):
        self.__write_to_reg_64(SPC_TRIG_DELAY, int(delay_in_samples))
        self.__handle_error()

    def calc_and_set_trigger_delay(self, timedelay, include_pretrigger=True):
        """ !!! Must be called after the oversampling factor is set
            Calculates the trigger delay and sets it

            Returns how many samples to cut from the beginning of the measured trace for the timedelay
            to be exact
        """
        delay = int(timedelay * 1e-9 * self.get_sample_rate())
        if include_pretrigger:
            delay += self.pretrigger
        self._n_samples_to_drop_by_dig_delay = delay % 32  # number of samples to drop from waveform in software
        delay -= self._n_samples_to_drop_by_dig_delay  # digitizer delay must be dividable by 32
        self.set_trigger_delay(delay)

    def calc_and_set_segment_size(self, dur_seg=0, extra=0, samples_drop=0):
        """
        !!! Must be called after the oversampling factor is set but before the mode is chosen
            set_parameters() handles this issue
        """
        if dur_seg == 0:
            dur_seg = self.dur_seg

        self._segment_size = int(dur_seg * self.get_sample_rate()) + extra - samples_drop
        self._n_samples_to_drop_in_end = 32 - self._segment_size % 32
        self._segment_size += self._n_samples_to_drop_in_end  # completing requested signal length to the multiple of 32
        self._bufsize = self.n_seg * self._segment_size * 4 * len(self.channels)

    def get_how_many_samples_to_drop_in_front(self):
        return self._n_samples_to_drop_by_dig_delay

    def get_how_many_samples_to_drop_in_end(self):
        return self._n_samples_to_drop_in_end

    def setup_internal_clock(self):
        self.__write_to_reg_32(SPC_CLOCKMODE, SPC_CM_INTPLL)

    def setup_pxi_clock(self):
        self.__write_to_reg_32(SPC_CLOCKMODE, SPC_CM_PXIREFCLOCK)

    def set_oversampling_factor(self, factor):
        """ Set oversampling factor.
            Sample rate = Max. sample rate / Oversampling factor
            Parameters:
            -----------
            factor: int
                Oversampling factor
                Possible values: 1, 2, 4, 8, 16, ..., 262144
        """
        allowed = [2 ** n for n in range(0, 19)]
        if factor in allowed:
            self.__oversampling = factor
            self.__samplerate = self.__read_reg_32(SPC_PCISAMPLERATE) // self.__oversampling
        else:
            raise ValueError("This oversampling factor is not supported. Allowed factors are 1, 2, 4, 8, ..., 262144")

    def setup_sample_rate(self):
        self.__write_to_reg_32(SPC_SAMPLERATE, self.__samplerate)
        # setsamplerate = self.__read_reg_32(SPC_SAMPLERATE)
        # oversamplingfactor = self.__read_reg_32(SPC_OVERSAMPLINGFACTOR)
        # print("Sample rate is set to %d Hz\nOversampling factor is %d" % (setsamplerate, oversamplingfactor))

    def get_sample_rate(self):
        """
        Returns
        -------
        samplerate : float
            samplerate in Hz
        """
        return self.__samplerate

    def set_trigger_mode(self, mode):
        self._trigger_mode = mode

    def setup_ext0_trigger(self):
        self.__write_to_reg_32(SPC_TRIG_EXT0_LEVEL0, 1000)  # 0-level is < 1000 mV
        self.__write_to_reg_32(SPC_TRIG_EXT0_LEVEL1, 1200)  # 1-level is > 2000 mV
        self.__write_to_reg_32(SPC_TRIG_EXT0_MODE,
                               self._trigger_mode)  # trigger on the rising edge (voltage crosses 0-level barrier)
        self.__write_to_reg_32(SPC_TRIG_ORMASK, SPC_TMASK_EXT0)  # Enable the external triggel

    def setup_pxi_trigger(self):
        self.__write_to_reg_32(SPC_PXITRG1_MODE, SPCM_PXITRGMODE_IN)
        self.__write_to_reg_32(SPC_TRIG_ORMASK, SPC_TMASK_PXI1)  # Enable the external triggel

    def setup_auto_trigger(self):
        self.__write_to_reg_32(SPC_TRIG_ORMASK, SPC_TMASK_SOFTWARE)  # Trigger the card immediately after start

    def start_card(self):
        """Start the card execution"""
        self.__write_to_reg_32(SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER)
        self.__handle_error()

    def wait_for_card(self):
        """Wait until the card completes the current run"""
        self.set_timeout(0)
        try:
            self.__handle_timeout(
                self.__write_to_reg_32(SPC_M2CMD, M2CMD_CARD_WAITREADY)  # Wait till the card completes the current run
            )
        except KeyboardInterrupt:
            self.stop_card()
        finally:
            self.__handle_error()

    def wait_for_trigger(self):
        """Wait until the first trigger"""
        self.set_timeout(1000)
        self.__handle_timeout(
            self.__write_to_reg_32(SPC_M2CMD, M2CMD_CARD_WAITTRIGGER)
        )
        self.__handle_error()

    def stop_card(self):
        """Stop the card execution"""
        self.__write_to_reg_32(SPC_M2CMD, M2CMD_CARD_STOP)

    def reset_card(self):
        """Stop the configured parameters of the card to default"""
        self.__write_to_reg_32(SPC_M2CMD, M2CMD_CARD_RESET)

    def get_trigger_counter(self):
        """Get the trigger counter value"""
        cnt = self.__read_reg_64(SPC_TRIGGERCOUNTER)
        self.__handle_error()
        return cnt

    def obtain_data(self, bufsize):
        pcData = (int8 * bufsize)()
        res = self.__def_simp_transfer(pcData)  # define Card -> PC transfer buffer
        if res is not 0:
            print("Error: %d" % res)
            return None
        self.__write_to_reg_32(SPC_M2CMD,
                               M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)  # Start the transfer and wait till it's completed
        self.__write_to_reg_32(SPC_M2CMD, M2CMD_DATA_STOPDMA)  # Explicitly stop DMA transfer
        self.__invalidate_buffer()  # Invalidate the buffer
        if self.AVG_ON:
            return np.frombuffer(pcData, dtype=np.int32)
        else:
            return np.frombuffer(pcData, dtype=np.int8)

    def setup_standard_mode(self, channels=None, ampl=None, memsize=None, pretrigger=None):
        if channels is None:
            channels = self.channels
        if ampl is None:
            ampl = self.ch_amplitude
        if memsize is None:
            memsize = self._segment_size
        if pretrigger is None:
            pretrigger = self.pretrigger

        # if function was not invoked from 'set_parameters'
        self.mode = SPCM_MODE.STANDARD

        self.n_seg = 1
        self.n_avg = 1

        if type(channels) is int:
            channels = [channels]

        posttrigger_mem = memsize - pretrigger  # memory allocated for a signal after the trigger
        self.setup_trigger_source()
        self.setup_pxi_clock()
        self.setup_sample_rate()
        self.setup_channels(channels, ampl)
        self.setup_SSA(memsize, posttrigger_mem)  # measure the signal and save in the card memory

    def setup_multiple_recoding_mode(self, channels=None, ampl=None, num_segments=None, segment_size=None, pretrigger=None):
        if channels is None:
            channels = self.channels
        if ampl is None:
            ampl = self.ch_amplitude
        if num_segments is None:
            num_segments = self.n_seg
        if segment_size is None:
            segment_size = self._segment_size
        if pretrigger is None:
            pretrigger = self.pretrigger

        # if function was not invoked from 'set_parameters'
        self.mode = SPCM_MODE.MULTIMODE

        posttrigger_mem = segment_size - pretrigger
        memsize = num_segments * segment_size

        # card driver does not throw error on exceeding the segment size
        # see manual p.153
        max_seg_size = self.__read_reg_64(SPC_PCIMEMSIZE) // 2 // len(channels)
        if segment_size > max_seg_size:
            raise CardError(f"Segment size {segment_size} exceeds maximal "
                            f"segment size {max_seg_size} for {len(channels)} channels")

        if segment_size % 32 > 0:
            raise CardError(f"Segment size must be a multiple of 32")

        self.setup_multi_rec_STD(memsize, segment_size, posttrigger_mem)
        self.setup_channels(channels, ampl)
        self.setup_internal_clock()
        self.setup_trigger_source()
        self.setup_sample_rate()

    def setup_averaging_mode(self, channels=None, ampl=None, num_segments=None, segment_size=None, pretrigger=None,
                             num_averages=None):
        if channels is None:
            channels = self.channels
        if ampl is None:
            ampl = self.ch_amplitude
        if num_segments is None:
            num_segments = self.n_seg
        if segment_size is None:
            segment_size = self._segment_size
        if pretrigger is None:
            pretrigger = self.pretrigger
        if num_averages is None:
            num_averages = self.n_avg
        # if function was not invoked from 'set_parameters'
        self.mode = SPCM_MODE.AVERAGING

        posttrigger_mem = segment_size - pretrigger
        memsize = num_segments * segment_size

        # card driver does not throw error on exceeding the segment size
        # see manual p.153
        max_seg_size = int(64 * 1024 / len(channels))
        if segment_size > max_seg_size:
            raise CardError(f"Segment size {segment_size} exceeds maximal "
                            f"segment size {max_seg_size} for {len(channels)} channels")
        if segment_size % 32 > 0:
            raise CardError(f"Segment size must be a multiple of 32")

        self.setup_block_avg_STD(memsize, segment_size, posttrigger_mem, num_averages)
        self.setup_channels(channels, ampl)
        self.setup_pxi_clock()
        self.setup_trigger_source()
        self.setup_sample_rate()

    def set_trigger_source(self, trig_source):
        """
        You may use this function as an alternative to
        'self.set_parameters'

        Parameters
        ----------
        trig_source : str
        "EXT" - ext0 from front panel
        "AUTO" - autotrigger

        Returns
        -------
        None
        """
        if trig_source == "EXT":
            self.trigger_source = SPCM_TRIGGER.EXT0
        elif trig_source == "AUTO":
            self.trigger_source = SPCM_TRIGGER.AUTOTRIG

    def setup_trigger_source(self):
        if self.trigger_source == SPCM_TRIGGER.AUTOTRIG:
            self.setup_auto_trigger()
        elif self.trigger_source == SPCM_TRIGGER.EXT0:
            self.setup_ext0_trigger()

    def measure(self, bufsize):
        """

        Parameters
        ----------
        bufsize

        Returns
        -------
        data : np.ndarray
        """
        self.start_card()
        self.wait_for_card()  # wait till the end of a measurement
        data = self.obtain_data(bufsize)  # download data from the card
        return data

    def measure_standard_mode(self, channels, ampl, memsize, pretrigger):
        if type(channels) is int:
            N = 1
            channels = [channels]
        else:
            N = len(channels)
        self.setup_standard_mode(channels, ampl, memsize, pretrigger)
        return self.measure(memsize * N)

    def measure_averaging_mode(self, channels, ampl, num_segments, segment_size, pretrigger, num_averages):
        if type(channels) is int:
            N = 1
            channels = [channels]
        else:
            N = len(channels)

        # num_pulses = num_segments * num_averages
        self.setup_averaging_mode(channels, ampl, num_segments, segment_size, pretrigger, num_averages)
        data = self.measure(num_segments * segment_size * 4 * N)
        # print("Number of triggers detected is %d out of %d" % (self.get_trigger_counter(), num_pulses))
        return data

    def measure_multiple_recording_mode(self, channels, ampl, num_segments, segment_size, pretrigger):
        if type(channels) is int:
            N = 1
            channels = [channels]
        else:
            N = len(channels)

        self.setup_multiple_recoding_mode(channels, ampl, num_segments, segment_size, pretrigger)
        data = self.measure(num_segments * segment_size * N)
        return data

    @staticmethod
    def amps_to_dbm(amps):
        # SHAMIL: are you sure it is not '... / 50 / 1e-3)' instead of '... / 50 * 1e-3'?
        return 10 * np.log10((amps / 1e3) ** 2 / 50 * 1e-3)

    @staticmethod
    def plot_spectrum(channels, data, freq_from, freq_until, samplerate):
        if type(channels) is int:
            N = 1
            channels = [channels]
        else:
            N = len(channels)

        fig = plt.figure(figsize=(9, 7))
        freq = samplerate / 1e6
        ymax = 0

        nfft = fftpack.helper.next_fast_len(int(len(data) / N))
        xf = np.fft.fftfreq(nfft, 1. / samplerate) / 1e6
        startfrom = np.searchsorted(xf[:int(nfft / 2) - 1], freq_from)
        endin = np.searchsorted(xf[:int(nfft / 2) - 1], freq_until)

        for i in range(0, N):
            ydata = data[i::N]
            yf = np.abs(np.fft.fft(ydata, nfft, norm="ortho")) * 2 / nfft / np.sqrt(50 * 1e-3)
            plt.plot(xf[startfrom:endin], 20 * np.log10(yf[startfrom:endin]))
            ymax = max(max(yf[2:]), ymax)

        plt.grid(True)
        plt.xlim(freq_from, freq_until)
        #         plt.ylim(-ymax * 0.1, 1.1 * ymax)
        plt.xlabel("freq, MHz")
        plt.ylabel("Power, dBm")
        legends = ["CH" + str(ch) for ch in channels]
        plt.legend(legends)
        plt.show()

    @staticmethod
    def plot_signal(channels, data, ampl, time_from, time_until, samplerate, num_averages=1):
        if type(channels) is int:
            N = 1
            channels = [channels]
        else:
            N = len(channels)

        fig = plt.figure(figsize=(10, 7))
        start_x = int(time_from / 1e9 * samplerate)
        end_x = int(time_until / 1e9 * samplerate)

        xdata = np.arange(0, len(data) / N) / samplerate

        for i in range(0, N):
            ydata = data[i::N] / 128 * ampl / num_averages
            plt.plot(xdata[start_x:end_x], ydata[start_x:end_x])

        legends = ["CH" + str(ch) for ch in channels]
        plt.ylabel("Voltage, mV")
        plt.xlabel("time, $\mu$s")
        plt.xlim(time_from / 1e9, time_until / 1e9)
        plt.ylim(-ampl, +ampl)
        plt.legend(legends)
        plt.grid(True)
        plt.show()

    def show_channel(self, channels, ampl, time_from, time_until):
        """Plots data recorded on channels
            Parameters:
            -----------
            channels: int or an array of int
                number of channels (from 0 to 3) to be set
            ampl: int or array of int, mV
                Amplitude window of a channel is (-amplitude, +amplitude),
                possible values are 200, 500, 1000, 2500 mV
            time_from: int, us (microseconds)
                where should time window of the plot begin
            time_until: int, us
                where should time window of the plot end
                defines the total length of a measured segment
        """
        freq = self.get_sample_rate() / 1e6  # MHz
        end_x = int(time_until * freq)
        memsize = end_x + 32 - end_x % 32  # total memory for a single acquisiton
        data = self.measure_standard_mode(channels, ampl, memsize, 32)
        SPCM.plot_signal(channels, data, ampl, time_from, time_until, self.get_sample_rate())

    def show_averaged_channel(self, channels, ampl, n_seg, time_from, time_until, pretrigger, num_averages=10000):
        """
        Plots averaged data recorded on channels
            Parameters:
            -----------
            channels: int or an array of int
                number of channels (from 0 to 3) to be set
            ampl: int or array of int, mV
                Amplitude window of a channel is (-amplitude, +amplitude),
                possible values are 200, 500, 1000, 2500 mV
            time_from: int, ns
                where should time window of the plot begin
            time_until: int, ns
                where should time window of the plot end
                defines the total length of a measured segment
            num_averages: int, default 10 000
                number of averages from 4 to 16 000 000
        """
        if type(channels) is int:
            N = 1
            channels = [channels]
        else:
            N = len(channels)

        freq = self.get_sample_rate()  # Hz
        segment_size_optimal = int(time_until / 1e9 * freq)
        segment_size = segment_size_optimal + 32 - segment_size_optimal % 32  # completing requested signal length to the multiple of 32
        data = self.measure_averaging_mode(channels, ampl, n_seg, segment_size, pretrigger, num_averages)

        SPCM.plot_signal(channels, data, ampl, time_from, time_until, self.get_sample_rate(), num_averages)
        return data

    def show_spectrum(self, channels, ampl, memsize, pretrigger, freq_from, freq_until):
        data = self.measure_standard_mode(channels, ampl, memsize, pretrigger)
        SPCM.plot_spectrum(channels, data, freq_from, freq_until, self.get_sample_rate())

    def show_averaged_spectrum(self, channels, ampl, n_seg, segment_time_length, pretrigger, freq_from, freq_until,
                               num_averages=10000):
        """Shows averaged data recorded on channels
            Parameters:
            -----------
            channels: int or an array of int
                number of channels (from 0 to 3) to be set
            ampl: int or array of int, mV
                Amplitude window of a channel is (-amplitude, +amplitude),
                possible values are 200, 500, 1000, 2500 mV
            n_seg: int
                Number of segments to be measured
            segment_time_length: float, seconds
                Duration of one segment in seconds
                from 64 to 64k with a step 32
            pretrigger: int, samples
                Size of a pretrigger
                from 32 to 64k-32 with a step 32
            freq_from: float, MHz
                where should the frequency window of the plot begin
            freq_until: float, MHz
                where should the frequency window of the plot end
            num_averages: int, default 10 000
                number of averages from 4 to 16 000 000
        """
        if type(channels) is int:
            N = 1
            channels = [channels]
        else:
            N = len(channels)
        segment_size_optimal = int(segment_time_length * self.get_sample_rate())
        segment_size = segment_size_optimal + 32 - segment_size_optimal % 32
        data = self.measure_averaging_mode(channels, ampl, n_seg, segment_size, pretrigger, num_averages)

        # deleting extra samples from segments
        a = np.arange(N * segment_size_optimal, len(data), N * segment_size)
        b = np.concatenate([a + i for i in range(0, N * (segment_size - segment_size_optimal))])
        data_cut = np.delete(data, b) * ampl / 128 / num_averages
        SPCM.plot_spectrum(channels, data_cut, freq_from, freq_until, self.get_sample_rate())
        return data_cut

    @staticmethod
    def extract_useful_data(data, n_channels, segment_size,
                            samples_per_segment_to_cut_at_beginning, samples_per_segment_to_cut_at_end):
        every = n_channels * segment_size
        # Generation of an array of indexes to cut at the beginning
        if samples_per_segment_to_cut_at_beginning > 0:
            first_idxs = np.arange(0, len(data), every, dtype=np.int)
            # to cut data from every channel
            times_at_beginning = n_channels * samples_per_segment_to_cut_at_beginning
            idxs_at_beginning = np.concatenate([first_idxs + i for i in range(0, times_at_beginning)])
        else:
            idxs_at_beginning = np.array([])
        # Generation of an array of indexes to cut at the end
        if samples_per_segment_to_cut_at_end > 0:
            end = (segment_size - samples_per_segment_to_cut_at_end) * n_channels
            first_idxs = np.arange(end, len(data), every)
            times_at_end = n_channels * samples_per_segment_to_cut_at_end
            idxs_at_end = np.concatenate([first_idxs + i for i in range(0, times_at_end)])
        else:
            idxs_at_end = np.array([])
        # An array of all indexes to be cut
        idxs_to_cut = np.concatenate((idxs_at_beginning, idxs_at_end))
        # Execution
        return np.delete(data, idxs_to_cut)

"""
product link:
https://www.keysight.com/en/pd-2747446-pn-M3202A/pxie-arbitrary-waveform-generator-1-gs-s-14-bit-400-mhz?cc=RU&lc=en

user's guide for Keysight SD1 version 2.x
https://literature.cdn.keysight.com/litweb/pdf/M3201-90001.pdf?id=2787170

user's guide for Keysight SD1 version 3.x
https://literature.cdn.keysight.com/litweb/pdf/M3XXX-90003.pdf?id=3120777
"""
import numpy as np
from scipy.interpolate import interp1d

try:
    from drivers.keysightSD1 import SD_AOU, SD_Wave
    from drivers.keysightSD1 import SD_TriggerModes, SD_TriggerExternalSources, \
        SD_TriggerBehaviors, SD_TriggerDirections
    from drivers.keysightSD1 import SD_WaveformTypes, SD_Waveshapes, \
        SD_MarkerModes, SD_SyncModes, SD_Error
    from drivers.keysightSD1 import SD_ModulationTypes, SD_Compatibility
except OSError:
    pass  # we are not on the measurement PC

class KeysightM3202A:
    MAX_OUTPUT_VOLTAGE = 1.5  # V
    VOLTAGE_RESOLUTION_BITS = 12
    MIN_SAMPLE_PERIOD = 1  # ns

    def __init__(self, awg_alias, slot,
                 chassis=0, allow_unmatched_waveforms=True):
        '''

        Parameters
        ----------
        slot: int
            Slot where the m3202 is physically installed, written on the chassis or
            can be found in the Windows Device Manager
        chassis: int
            If several chassis are mounted, will be used to specify in which one the
            AWG is installed
        allow_unmatched_waveforms: bool
            If true, the waveforms on different channels will be allowed to have different durations.
            This may be not safe when we share the AWG channel pairs between two virtual IQ devices since
            it may lead to leftover sequences and thus m3202-state-dependent behaviours of the virtual
            IQ devices (e.g. we can't set the repetition rate for IQAWG(ch1, ch2) higher than already set in
            IQAWG(ch3,ch4) because its trailing sequence will remain too long, and will have to manually clear ch3,ch4
            to work around this, see also output_arbitraty_waveform())
            Therefore, as for the Tektronix5014c device which does not allow different-duration waveforms, setting
            this parameter to False will ensure that all waveforms that do not match the most recently loaded waveform
            are deleted. This way the user will always get what he requests though will have to make sure he loads
            the waveforms of same duration. Fortunately, this is the case for all PulseSequences that we usually use and
            for synchronized channels, in general.
        '''
        self.alias = awg_alias
        self.module = SD_AOU()
        self.module_id = self.module.openWithSlotCompatibility(
            "M3202A", chassis, slot, compatibility=SD_Compatibility.LEGACY)
        self._handle_error(self.module_id)

        self.waveforms = [None] * 4
        # store `waveform_number` parameter, see manual for details
        self.waveform_ids = [-1] * 4
        # in case of AM or FM
        self.waveshape_types = [SD_Waveshapes.AOU_AWG] * 4
        self.repetition_frequencies = [None] * 4
        self.output_voltages = [None] * 4
        # deviation gains `G` for modulated signals, see manual for details
        self.deviation_gains = [0.0] * 4
        self.trigger_modes = [SD_TriggerModes.AUTOTRIG] * 4
        # from front panel, can be also from PXI_n triggering bus
        self.trigger_ext_sources = [
                                       SD_TriggerExternalSources.TRIGGER_EXTERN] * 4
        # rising edge by default
        self.trigger_behaviours = [SD_TriggerBehaviors.TRIGGER_RISE] * 4
        self.trigger_output = True
        # see self.trigger_output_config()
        self.trigger_sync_mode = SD_SyncModes.SYNC_CLK10

        self.sync_mode = 1  # 0 is for CLKsys (1 GHz)
                            # 1 is for the 10  MHz PXI clock

        self.synchronized_channels = None
        self._source_channels_group = []  # channels that are the source of the waveforms for dependent group
        self._dependent_channels_group = []  # channels which waveforms repeats corresponding waveforms
        # from _source_channels_group
        self._update_dependent_on_start = False

        self._prescaler = 0  # from 0 to 4095 p.23 user manual
        self.trigger_length = 100  # ns

        self._allow_unmatched_waveforms = allow_unmatched_waveforms

        self.reset()  # clear internal memory and AWG queues according to p.67 of the user guide

    def __del__(self):
        if self.module.isOpen():
            self.reset()
            self.module.close()
            del self.module

    def get_voltage_range(self):
        return [-1.5, 1.5]  # see specification

    def _handle_error(self, ret_val):
        if ret_val < 0:
            print(ret_val, SD_Error.getErrorMessage(ret_val))
            raise Exception

    def reset(self, channels=None):
        if channels is None:
            channels = [1, 2, 3, 4]

        for channel in channels:
            # stop modulations
            self.stop_modulation(channel)
            # clear queues, theres no command to clear memory for a specific channel
            # however, leaving internal memory as is
            # is not a problem: it will inevitably be overwritten by following
            # requests to the AWG via module.reloadWaveform...)
            ret = self.module.AWGflush(channel - 1)
            self._handle_error(ret)

            # clearing software variables
            self.waveforms[channel - 1] = None
            self.waveform_ids[channel - 1] = None
            self.repetition_frequencies[channel - 1] = None
            self.waveshape_types[channel - 1] = SD_Waveshapes.AOU_AWG  # default

        if channels == [1, 2, 3, 4]:
            # clear ALL: internal memory and AWG queues
            ret = self.module.waveformFlush()
            self._handle_error(ret)

    def synchronize_channels(self, *channels):
        self.synchronized_channels = channels

    def unsynchronize_channels(self):
        self.synchronized_channels = None

    def set_trigger(self, trigger_string: str = "CONT", channel: int = -1):
        """
        trigger_string : string
           'EXT' - external trigger on the front panel is used as a trigger trace source
           'CONT' - continious output  <---- DEFAULT setting
        channel : int
            1,2,3,4 - channel number
        """
        if (channel == -1) or (channel in self.synchronized_channels):
            channels_to_config = self.synchronized_channels
        else:
            channels_to_config = [channel]

        for channel in channels_to_config:
            if trigger_string == "EXT":  # front panel
                # for each 'cycle' (see 'cycle' definition in docs)
                self.trigger_modes[channel - 1] = SD_TriggerModes.EXTTRIG_CYCLE
                ret = self.module.AWGtriggerExternalConfig(
                    channel - 1,
                    self.trigger_ext_sources[channel - 1],  # front panel only
                    # on rising edge by default
                    self.trigger_behaviours[channel - 1],
                    # sync with PXI 10 MHz clock
                    SD_SyncModes.SYNC_CLK10
                )
                self._handle_error(ret)
            elif trigger_string == "CONT":
                self.trigger_modes[channel - 1] = SD_TriggerModes.AUTOTRIG

    def trigger_output_config(self, trig_mode="ON", channel=-1,
                              trig_length=100, sync = "PXICLK10"):
        """
        Manipulates the output trigger.
        If channel is not supplied, trigger output is set for the first channel
        from synchronized channel group.
            To set synchronized channels, use self.synchronize_channels(...) method. If
        self.synchronized_channels were not set, then the default value is [] and no
        trigger output would be configured.

        Parameters
        ----------
        trig_mode : str
            "ON" - output trigger for 'channel' or the first channel in its synchronized group
            "OFF" - no output
        channel : int
            Channel number to set trigger for. Starting from 1.
            if channel is in synchronized group than trigger is configured for the first channel in group
        trig_length : int
            trigger duration in ns
            trigger duration resolution is 10 ns
        sync: str
            Which clock is used to sample the trigger, "PXICLK10" for the
            PXI clock or "CLK100" for the internal clock, manual at p.31, p.126
            For the PXICLK10 waveform durations must be multiples of 100 ns
            ohterwise 100 ns jitter will occur. For CLK100, somehow,
            there is no jitter for arbitrary durations (tested 1.5 MHz rep
            rate with a period of 666.67 ns)
        """

        if sync == "PXICLK10":
            self.trigger_sync_mode = SD_SyncModes.SYNC_CLK10
        elif sync == "CLK100":
            self.trigger_sync_mode = SD_SyncModes.SYNC_NONE
        else:
            raise ValueError(f'Sync mode {sync} invalid. Can be "PXICLK10" '
                             f'or "CLK100"')

        if trig_mode == "ON":
            self.trigger_output = True
            self.trigger_length = trig_length
        elif trig_mode == "OFF":
            self.trigger_output = False
        else:
            raise ValueError(
                "trig_mode argument can be only 'ON' or 'OFF' ")

        # if channel is equal to -1 or is stored in
        # `self.synchronized_channels`,
        # then output trigger is configured for all synchronized channels
        if (channel == - 1) or (channel in self.synchronized_channels):
            channels_to_config = self.synchronized_channels
        else:
            channels_to_config = [channel]

        for chan in channels_to_config:
            if self.trigger_output:  #
                # configuring trigger IO as output
                self.module.triggerIOconfig(SD_TriggerDirections.AOU_TRG_OUT)
                # here was changed to PXI trigger output
                # adding marker to the specified channel
                trgPXImask = 0b0
                trgIOmask = 0b1
                self.module.AWGqueueMarkerConfig(
                    chan - 1, SD_MarkerModes.EVERY_CYCLE,
                    trgPXImask, trgIOmask, 1,
                    # trigger sync with internal CLKsys by default
                    syncMode=self.trigger_sync_mode,
                    # trigger length (100a.u. x 10ns => 1000 ns trigger length)
                    length=int(trig_length / 10),
                    delay=0
                )
                #  enabling trigger for the first channel from group is
                # enough to produce output marker
                break
            else:  # disable trigger for all channels form group
                self.module.triggerIOconfig(SD_TriggerDirections.AOU_TRG_OUT)
                self.module.triggerIOwrite(0,
                                           SD_SyncModes.SYNC_NONE)  # make sure the output is zero
                # here was changed to PXI trigger output
                # deleting marker to the specified channel
                trgPXImask = 0b0
                trgIOmask = 0b1
                self.module.AWGqueueMarkerConfig(chan - 1,
                                                 SD_MarkerModes.DISABLED,
                                                 trgPXImask, trgIOmask, 1,
                                                 syncMode=SD_SyncModes.SYNC_NONE,
                                                 # trigger sync with internal 100
                                                 # MHz clock for the 'off'
                                                 # state
                                                 length=100,
                                                 # trigger length (100a.u. x 10ns => 1000 ns trigger length)
                                                 delay=0)
                self.module.triggerIOconfig(SD_TriggerDirections.AOU_TRG_IN)

    def output_arbitrary_waveform(self, waveform, frequency, channel):
        """
        Prepare and output an arbitrary waveform repeated at some repetition_rate

        Parameters:
        -----------
        waveform: array
            ADC levels, in Volts. max( abs(waveform) ) < 1.5 V
        if_freq: float, Hz
            if_freq at which the waveform will be repeated
        channel: 1,2,3,4
            channel which will output the waveform

        NOTE_1: waveform length must be a multiple of 10, but you do not need to
        provide an array that complies with such condition. Note only that this array will be copied 10
        times before being loaded to board RAM in order to satisfy the datasheet condition above.
        See user guide, p.140
        and datasheet, p.8

        NOTE_2: Sampling speed depends on the SD_WaveformType value of the SD_Wave().
        see datasheet for more details

        NOTE_3: waveform's last point must coincide with the waveforms first point.
        Shamil: I assume there is no need to pass np.sin(2*np.pi*np.linspace(0,1,N_pts+1)) here
        because this array basically includes the boundary point twice, which values are already
        assumed to be equal to each other when specifying waveform if_freq.
        It is rather more rational to  pass np.sin(2*np.pi*np.linspace(0,1,N_pts+1)[:-1])
        Also I'd prefer to pass function explicitly as lambda or something like that
        and only then generate points inside this class method. 25.04.2019

        NOTE_4: I suggest we use embeded function generators with amplitude modulation
        in our AWG solution, rather than generating sinus from scratch and using DAC only.
        This is necessary in order to improve if_freq accuracy and stability.
        """
        # check if pulse sequence length meets the requirement imposed by a
        # trigger sampling
        waveform_duration = np.round(1 / frequency * 1e9)
        if self.trigger_sync_mode == SD_SyncModes.SYNC_CLK10:
            if waveform_duration % 100 != 0:
                raise ValueError("Duration of the waveform must be a "
                  "multiple of 100 ns, because the "
                  "bias trigger is synchronized with "
                  "PXI clock and therefore sampled with "
                  "10 MHz. Not following this requirement "
                  "leads to uncertainty of the trigger "
                  "pulse position relative to the waveform")

        # stopping AWG so the changes will take place according to the documentation
        # (not neccessary but a good practice)
        self.stop_AWG(channel)
        self.stop_modulation(channel)

        def clear_unmatched_waveforms():
            # Checks if other channels' waveforms have matching length,
            # otherwise clear them (similar to Tektronix implementation)
            # This helps avoiding situations when 2 channels are inadvertently still
            # outputting very long sequences while other 2 are short (e.g. switching from
            # time-resolved measurements to single-tone spectroscopy when only 2 channels are
            # controlled by the VNA and set to continuous IF sine and the other 2 remain
            # in the pulsed mode -- STS class doesn't control them)
            # Alternative solution is to explicitly control every device in every class,
            # however I think we should make the low-level drivers smart enough to avoid that
            to_clear = []
            for idx, existing_freq in enumerate(self.repetition_frequencies):
                if idx != channel - 1:
                    if existing_freq is not None:
                        if (existing_freq - frequency) != 0:
                            to_clear.append(idx + 1)
            self.reset(to_clear)

        if not self._allow_unmatched_waveforms:
            clear_unmatched_waveforms()

        # loading a waveform to internal RAM and putting waveform into the channel's AWG queue
        self.load_waveform_to_channel(waveform, frequency, channel)
        # starting operation
        self.start_AWG(channel)
        # self.reset_phase()

    def output_continuous_wave_old(self, frequency, amplitude, phase, offset,
                                   waveform_resolution,
                                   channel, asynchronous=False):
        n_points = np.around(
            1 / frequency / waveform_resolution * 1e9) + 1 if frequency != 0 else 3
        waveform = amplitude * np.sin(
            2 * np.pi * np.linspace(0, 1, n_points) + phase) + offset
        self.output_arbitrary_waveform(waveform, frequency, channel,
                                       asynchronous=asynchronous)

    def reset_phase(self):
        self.module.channelPhaseResetMultiple(
            sum([1 << (chan - 1) for chan in self.synchronized_channels]))

    def output_continuous_wave(self, frequency, amplitude, phase, offset,
                               waveform_resolution,
                               channel, asynchronous=False,
                               trigger_sync_every=None):
        """

        Parameters
        ----------
        frequency
        amplitude
        phase
        offset
        waveform_resolution
        channel : int
            Channel to operate with. Numbering starts from 1.
        asynchronous
        trigger_sync_every : int
            period of trigger trace in ns
            trigger is synchronized with the continuous wave.

        Returns
        -------

        """
        self.stop_AWG(channel)  # stop output from this channel
        # disable all modulation types for this channel
        self.stop_modulation(channel)
        # setup embedded function generator to produce
        # sine wave with parameters
        self.setup_fg_sine(frequency, amplitude, phase, offset, channel)

        # output trigger has to be generated
        if trigger_sync_every is not None:
            # add zero modulation with trigger output at start
            arr = np.zeros(int(
                np.around(trigger_sync_every * self.get_sample_rate() / 1e9)))
            self._load_array_into_AWG(arr, channel)
            self.setup_modulation_amp(channel, 0)
            self.trigger_output_config(
                trig_mode="ON", channel=channel,
                trig_length=self.trigger_length
            )

        # resetting phase for synchronization of multiple carrier
        # signals from internal Function Generator
        self.module.channelPhaseResetMultiple(
            sum([1 << (chan - 1) for chan in self.synchronized_channels])
        )
        self.start_AWG(channel)

    def setup_fg_sine(self, frequency, amplitude, phase, offset, channel):
        if frequency > 0:
            self.waveshape_types[channel - 1] = SD_Waveshapes.AOU_SINUSOIDAL
        else:
            self.waveshape_types[channel - 1] = SD_Waveshapes.AOU_DC
        self.module.channelWaveShape(channel - 1,
                                     self.waveshape_types[channel - 1])
        self.output_voltages[channel - 1] = amplitude
        self.module.channelAmplitude(channel - 1,
                                     self.output_voltages[channel - 1])
        self.module.channelFrequency(channel - 1, frequency)
        self.module.channelPhase(channel - 1, phase / np.pi * 180)
        self.module.channelOffset(channel - 1, offset)

    def get_sample_rate(self):
        """
        Returns
        -------
        sample_rate : int
            Returns bias AWG's sample rate in Hz.
        """
        if self._prescaler == 0:
            fs = int(1e9)
        elif self._prescaler == 1:
            fs = int(2e8)
        elif self._prescaler > 1:
            fs = int(100 / self._prescaler * 1e6)
        return fs

    def get_sample_period(self):
        """
        Returns
        -------
        sample_period : np.float
            Sample period in nanoseconds
        """
        if self._prescaler == 0:
            dt = 1  # ns
        elif self._prescaler == 1:
            dt = 5  # ns
        elif self._prescaler > 1:
            dt = 10 * self._prescaler  # ns

        return dt

    def setup_amplitude_modulation(self, channel, waveform_id, array,
                                   deviationGain, prescaler):
        self.load_modulating_waveform(array, waveform_id)
        self.queue_waveform(channel, waveform_id, prescaler)
        self.setup_modulation_amp(channel, deviationGain)

    def load_modulating_waveform(self, waveform_array_normalized, wave_id):
        wave = SD_Wave()
        wave.newFromArrayDouble(SD_WaveformTypes.WAVE_ANALOG,
                                waveform_array_normalized)
        paddingMode = 0  # add zeros at the end if waveform length is smaller than
        ret = self.module.waveformLoad(wave, wave_id, paddingMode)
        if (ret == SD_Error.INVALID_OBJECTID):
            # probably, such wave_id already exists
            print("INVALID_OBJECTID")
            ret = self.module.waveformReLoad(wave, wave_id)
        self._handle_error(ret)

    def queue_waveform(self, channel, wave_id, prescaler):
        cycles = 0  # Zero specifies infinite cycles
        startDelay = 0
        ret = self.module.AWGqueueWaveform(channel - 1, wave_id,
                                           self.trigger_modes[channel - 1],
                                           startDelay, cycles, prescaler)
        self._handle_error(ret)

    def stop_modulation(self, channel):
        '''
        Sets that channel out of any of the modulation modes (Amp, Freq or IQ)
        and removes any phase and, most importantly, DC offset
        :param channel:
        :return:
        '''
        self.deviation_gains[channel - 1] = 0
        self.module.modulationAmplitudeConfig(channel - 1,
                                              SD_ModulationTypes.AOU_MOD_OFF,
                                              self.deviation_gains[
                                                  channel - 1])
        self.module.modulationAngleConfig(channel - 1,
                                          SD_ModulationTypes.AOU_MOD_OFF,
                                          self.deviation_gains[channel - 1])
        self.module.modulationIQconfig(channel - 1, self.deviation_gains[
            channel - 1])  # disable IQ modulation
        self.module.channelOffset(channel - 1, 0)
        # setting output to sample from channel queue
        self.waveshape_types[channel - 1] = SD_Waveshapes.AOU_AWG
        self.module.channelWaveShape(channel - 1,
                                     self.waveshape_types[channel - 1])
        self.module.channelPhase(channel - 1, 0)
        self.module.channelOffset(channel - 1, 0)
        # not calling channelFrequency and channelAmplitude as
        # the prescaler will control the sample rate
        # and thus the repetition if_freq in non-modulated mode,
        # and channelAmplitude will be always called there

    def change_amplitude_of_carrier_signal(self, amplitude, channel,
                                           ampl_coef=1):
        self.output_voltages[channel - 1] = amplitude * ampl_coef
        self.module.channelAmplitude(channel - 1,
                                     self.output_voltages[channel - 1])

    def setup_modulation_amp(self, channel, deviation_gain):
        """

        Parameters
        ----------
        channel : int
            number of channel to be configured. Starts from 1.
        deviation_gain : float
            coefficient 'G' for in formula for AM:
                f(t) = (A + G AWG(t)) Cos(w t + \phi)
                where AWG(t) - normalized waveform from AWG RAM.

        Returns
        -------
        None
        """
        self.deviation_gains[channel - 1] = deviation_gain
        self.module.modulationAmplitudeConfig(channel - 1,
                                              SD_ModulationTypes.AOU_MOD_AM,
                                              deviation_gain)

    def load_waveform_to_channel(self, waveform, frequency, channel):
        if np.max(np.abs(waveform)) > 1.5:
            raise ValueError("Trace maximal amplitude is exceeding AWG range: (-1.5 ; 1.5) volts")

        # number of points
        if (frequency > 1e9):
            raise ValueError("if_freq is exceeding AWG sampling rate: 1 GHz")

        duration_initial = 1 / frequency * 1e9 if frequency != 0 else 10.0  # float, ns

        if not np.allclose(duration_initial, 1/self.get_sample_rate() * 1e9 * len(waveform)):
            # interpolating input waveform to the next step
            # that rescales waveform to fit repetition frequency
            interpolation_method = "cubic" if frequency != 0 else "linear"
            old_x = np.linspace(0, duration_initial, len(waveform) + 1)
            interpolating_function = interp1d(old_x, np.concatenate((waveform, [waveform[0]])),
                              kind=interpolation_method)

            # in order to satisfy NOTE_1 we simply make 10 subsequent waveforms
            # but to provide if_freq accuracy, we are sampling from
            # interval 1000 times wider then the original, and we are extending
            # interpolation function domain using its periodicity
            # duration = duration_initial*1e4 if duration_initial < 1e2 else 1e6  # here it is

            duration = duration_initial
            new_x = np.linspace(0, duration, int(
                np.round(duration / self.get_sample_rate() * 1e9)), endpoint=False)

            # converting domain values in the function domain
            new_x_converted = np.remainder(new_x, duration_initial)
            waveform = interpolating_function(new_x_converted)  # obtaining new waveform values

        normalization = np.max(np.abs(waveform))

        if self.waveshape_types[channel - 1] == SD_Waveshapes.AOU_AWG:
            self.output_voltages[channel - 1] = normalization

        # normalize waveform to (-1,1) interval
        if normalization != 0:
            waveform /= normalization
        else:
            # all points are equal to zero
            pass

        self.repetition_frequencies[channel - 1] = frequency
        self._load_array_into_AWG(waveform, channel)

    def _load_array_into_AWG(self, waveform_array_normalized, channel):
        """

        Parameters
        ----------
        waveform_array_normalized
        channel : int
            Channel number starting from 1.

        Returns
        -------

        """
        from copy import deepcopy
        waveform_array_normalized = deepcopy(waveform_array_normalized)
        # only float 16 is supported by AWG (it is actually 12 bit AWG), so
        # if you want to guess what is actually outputted by AWG
        # you should properly convert this to 12 bit numbers
        reload = False
        if self.waveform_ids[channel-1] is not None:
            #  we have already loaded a waveform to the AWG
            #  the API allows to reload a waveform using the same
            #  memory space of the on-board RAM; however, the length
            #  of the new waveform should be smaller than or equal to
            #  the length of the old one; if that condition is satisfied,
            #  we will call waveformReLoad. TODO: In the opposite case, ideally
            #  a full memory flush has to be done and all waveforms reloaded for
            #  all channels
            if len(self.waveforms[channel-1]) >= len(waveform_array_normalized):
                reload = True

        self.waveforms[channel - 1] = waveform_array_normalized

        # creating SD_Wave() object from keysight API
        wave = SD_Wave()
        wave.newFromArrayDouble(SD_WaveformTypes.WAVE_ANALOG,
                                waveform_array_normalized)
        waveform_number = channel - 1
        self.waveform_ids[channel - 1] = waveform_number

        # setting function generation waveshape type parameters
        # OFF, direct AWG, SINUSOIDAL, TRIANGULAR and more
        # see 'SD_Waveshapes' class for complete details
        ret = self.module.channelWaveShape(channel - 1,
                                           self.waveshape_types[channel - 1])
        self._handle_error(ret)

        # clear channel queue
        self.module.AWGflush(channel - 1)

        # load waveform to board RAM
        if not reload:
            ret = self.module.waveformLoad(wave, waveform_number)
            if ret == SD_Error.INVALID_OBJECTID or ret == SD_Error.INVALID_OPERATION:
                self._handle_error(ret)
        else:
            ret = self.module.waveformReLoad(wave, waveform_number)
            self._handle_error(ret)

        # clear channel queue
        self.module.AWGflush(channel - 1)

        # put waveform as the first and only member of the
        # channel's AWG queue
        ret = self.module.AWGqueueWaveform(channel - 1, waveform_number,
                                           self.trigger_modes[channel - 1],
                                           # default trigger mode is "CONT"
                                           0,  # 0 ns starting delay
                                           0,  # 0 - means infinite
                                           0)  # prescaler is 1 (sampling freq is 1 GHz)
        self._prescaler = 0
        self._handle_error(ret)

        # set amplitude in volts
        ret = self.module.channelAmplitude(channel - 1,
                                           self.output_voltages[channel - 1])
        self._handle_error(ret)

    def start_AWG(self, channel):
        """

        Parameters
        ----------
        channel : int
            Number of channel to start. Start from 1.

        Returns
        -------

        """
        if self._update_dependent_on_start:
            self._load_dependent_channels()

        if (not self.synchronized_channels) or \
                (channel not in self.synchronized_channels):
            # if it is single channel
            self.module.AWGqueueSyncMode(channel - 1, syncMode=self.trigger_sync_mode)
            ret = self.module.AWGstart(channel - 1)
            self._handle_error(ret)
        elif channel in self.synchronized_channels:
            # if channel belongs to one of the synchronized groups
            channels_mask = 0
            for chan in self.synchronized_channels:
                channels_mask += 1 << (chan - 1)
                self.module.AWGqueueSyncMode(chan - 1, syncMode=self.sync_mode)
            ret = self.module.AWGstartMultiple(channels_mask)
            self._handle_error(ret)
        else:
            raise NotImplementedError(
                "Check channel number conditions on argument"
                " provided: channel={}".format(channel)
            )

        return ret

    def stop_AWG(self, channel):
        if (not self.synchronized_channels) or (
                channel not in self.synchronized_channels):
            ret = self.module.AWGstop(channel - 1)
            self._handle_error(ret)
        elif channel in self.synchronized_channels:
            channels_mask = sum(
                [1 << (chan - 1) for chan in self.synchronized_channels])
            ret = self.module.AWGstopMultiple(channels_mask)
            # print("{:b}".format(channels_mask))
            self._handle_error(ret)
        else:
            raise NotImplementedError(
                "Check channel number conditions on argument provided: channel={}".format(
                    channel))

    def setup_channel_duplicate_groups(self, _source_channels_group,
                                       _dependent_channels_group):
        """
        Set channels groups that are going to be duplicated
        Parameters
        ----------
        _source_channels_group : channels that are the source of the waveforms for dependent group
        _dependent_channels_group : channels which waveforms repeats corresponding waveforms from _source_channels_group

        Returns : None
        -------
        """
        self._source_channels_group = _source_channels_group
        self._dependent_channels_group = _dependent_channels_group
        self._update_dependent_on_start = True  # copy source channels waveform to dependent channels during each call of self.start_AWG(...) method

    def reset_duplicate_groups(self):
        self._source_channels_group = []
        self._dependent_channels_group = []
        self._update_dependent_on_start = False

    def _load_dependent_channels(self):
        """
        Duplicate output from channels with numbers in self._source_channels_group to
        corresponding channels from self._dependent_channels_group
        Assuming len(group2) >= len(group1). Excessive channels from dependent group (group2) is not affected

        Parameters
        ----------
        group1 : list
        group2 : list

        Returns : None
        -------
        """
        for source_channel_idx, (source_chan, dependent_chan) in enumerate(
                zip(self._source_channels_group,
                    self._dependent_channels_group)):
            if (self.waveforms[source_chan - 1] is not None):
                self.repetition_frequencies[dependent_chan - 1] = \
                    self.repetition_frequencies[source_chan - 1]
                self.output_voltages[dependent_chan - 1] = \
                    self.output_voltages[source_chan - 1]
                self._load_array_into_AWG(self.waveforms[source_chan - 1],
                                          dependent_chan)
                # print(self.waveforms[source_chan-1]*self.output_voltages[source_chan-1],self.output_voltages[dependent_chan-1]*self.waveforms[source_chan-1])

    def get_prescaler(self):
        return self._prescaler

    def plot_waveforms(self, voltage_output=False):
        import matplotlib.pyplot as plt
        plt.figure()
        for i, waveform in enumerate(self.waveforms):
            if (waveform is not None):
                mult = self.output_voltages[i] if voltage_output else 1
                plt.plot(waveform * mult, label="CH" + str(i + 1))
        plt.legend()

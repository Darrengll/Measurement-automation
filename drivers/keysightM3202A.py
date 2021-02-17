"""
product link:
https://www.keysight.com/en/pd-2747446-pn-M3202A/pxie-arbitrary-waveform-generator-1-gs-s-14-bit-400-mhz?cc=RU&lc=en

user's guide
https://literature.cdn.keysight.com/litweb/pdf/M3201-90001.pdf?id=2787170
"""

from drivers.instrument import Instrument

from drivers.keysightSD1 import SD_AOU, SD_Wave
from drivers.keysightSD1 import SD_TriggerModes, SD_TriggerExternalSources, \
    SD_TriggerBehaviors, SD_TriggerDirections
from drivers.keysightSD1 import SD_WaveformTypes, SD_Waveshapes, \
    SD_MarkerModes, SD_SyncModes, SD_Error
from drivers.keysightSD1 import SD_ModulationTypes, SD_Compatibility

import numpy as np
from scipy.interpolate import interp1d


class KeysightError(Exception):
    pass


class KeysightM3202A(Instrument):
    MAX_OUTPUT_VOLTAGE = 1.5  # V

    def __init__(self, name, slot, chassis=0):
        '''

        Parameters
        ----------
        name : redundant parameter for `Instrument` class
        slot
        chassis
        '''
        super().__init__(name, tags=['physical'])
        self.mask = 0
        self.module = SD_AOU()
        self.module_id = self.module.openWithSlotCompatibility("M3202A",
                                                               chassis, slot,
                                                               compatibility=SD_Compatibility.LEGACY)

        # Shamil a.k.a. 'BATYA' code here
        self.waveforms = [None] * 4
        # store `waveform_number` parameter, see manual for details
        self.waveform_ids = [-1] * 4
        self.waveshape_types = [
                                   SD_Waveshapes.AOU_AWG] * 4  # in case of AM or FM
        self.repetition_frequencies = [None] * 4
        self.output_voltages = [None] * 4
        # deviation gains `G` for modulated signals, see manual for details
        self.deviation_gains = [0.0] * 4
        self.trigger_modes = [SD_TriggerModes.AUTOTRIG] * 4
        self.trigger_ext_sources = [
                                       SD_TriggerExternalSources.TRIGGER_EXTERN] * 4  # from front panel, can be also from PXI_n triggering bus
        self.trigger_behaviours = [
                                      SD_TriggerBehaviors.TRIGGER_RISE] * 4  # rising edge by default
        self.trigger_output = True
        # default and only option at MIPT is synchronizing with PXI 10 MHz
        # clock
        self.trigger_sync_mode = SD_SyncModes.SYNC_CLK10
        # as written in manual at p.31 in case of CLK10 trigger sync
        # trigger can be outputted only at the 10 MHz PXI clock cycles
        self.trigger_clock_period = 100  # ns
        self.synchronized_channels = []
        self.sync_mode = SD_SyncModes.SYNC_CLK10  # 0 - PXI 10 MHz ; 1 - CLKSYS 1 GHz

        self._source_channels_group = []  # channels that are the source of the waveforms for dependent group
        self._dependent_channels_group = []  # channels which waveforms repeats corresponding waveforms from _source_channels_group
        self._update_dependent_on_start = False

        self._prescaler = 0
        self.trigger_length = 100  # ns

        self.clear()  # clear internal memory and AWG queues according to p.67 of the user guide

    def _handle_error(self, ret_val):
        if ret_val < 0:
            print(ret_val, SD_Error.getErrorMessage(ret_val))
            raise Exception

    def clear(self):
        # clear internal memory and AWG queues
        ret = self.module.waveformFlush()

        # stop all modulations
        for channel in [1, 2, 3, 4]:
            self.stop_modulation(channel)
        self._handle_error(ret)

    def synchronize_channels(self, *channels):
        self.synchronized_channels = channels

    def unsynchronize_channels(self):
        self.synchronized_channels = []

    def set_trigger(self, trigger_string: str = "CONT", channel: int = -1):
        """
        trigger_string : string
           'EXT' - external trigger on the front panel is used as a trigger signal source
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
                ret = self.module.AWGtriggerExternalConfig(channel - 1,
                                                           self.trigger_ext_sources[
                                                               channel - 1],
                                                           # front panel only
                                                           self.trigger_behaviours[
                                                               channel - 1],
                                                           # on rising edge by default
                                                           SD_SyncModes.SYNC_CLK10)  # sync with internal 100 MHz clock
                self._handle_error(ret)
            elif trigger_string == "CONT":
                self.trigger_modes[channel - 1] = SD_TriggerModes.AUTOTRIG

    def trigger_output_config(self, trig_mode="ON", channel=-1,
                              trig_length=1000):
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
        """

        if trig_mode == "ON":
            self.trigger_output = True
            self.trigger_length = trig_length
        elif trig_mode == "OFF":
            self.trigger_output = False
        else:
            raise NotImplementedError(
                "trig_mode argument can be only 'ON' or 'OFF' ")

        # if channel is equal to -1, then output trigger is configured for all synchronized channels
        if (channel == - 1) or (channel in self.synchronized_channels):
            channels_to_config = self.synchronized_channels
        else:
            channels_to_config = [channel]

        for chan in channels_to_config:
            if self.trigger_output:  # enable trigger for the first channel from group
                # configuring trigger IO as output
                self.module.triggerIOconfig(SD_TriggerDirections.AOU_TRG_OUT)
                # here was changed to PXI trigger output
                # adding marker to the specified channel
                trgPXImask = 0b0
                trgIOmask = 0b1
                self.module.AWGqueueMarkerConfig(chan - 1,
                                                 SD_MarkerModes.EVERY_CYCLE,
                                                 trgPXImask, trgIOmask, 1,
                                                 syncMode=self.sync_mode,
                                                 # trigger sync with internal CLKsys
                                                 length=int(trig_length / 10),
                                                 # trigger length (100a.u. x 10ns => 1000 ns trigger length)
                                                 delay=0)
                break  # first channel from group is enough to produce output marker
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
                                                 syncMode=self.sync_mode,
                                                 # trigger synch with internal CLKsys
                                                 length=100,
                                                 # trigger length (100a.u. x 10ns => 1000 ns trigger length)
                                                 delay=0)
                self.module.triggerIOconfig(SD_TriggerDirections.AOU_TRG_IN)

    def output_arbitrary_waveform(self, waveform, frequency, channel,
                                  asynchronous=False):
        """
        Prepare and output an arbitrary waveform repeated at some repetition_rate

        Parameters:
        -----------
        waveform: array
            ADC levels, in Volts. max( abs(waveform) ) < 1.5 V
        frequency: float, Hz
            frequency at which the waveform will be repeated
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
        assumed to be equal to each other when specifying waveform frequency.
        It is rather more rational to  pass np.sin(2*np.pi*np.linspace(0,1,N_pts+1)[:-1])
        Also I'd prefer to pass function explicitly as lambda or something like that
        and only then generate points inside this class method. 25.04.2019

        NOTE_4: I suggest we use embeded function generators with amplitude modulation
        in our AWG solution, rather than generating sinus from scratch and using DAC only.
        This is necessary in order to improve frequency accuracy and stability.
        """
        # check if pulse sequence length meets the requirement imposed by a
        # trigger sampling
        if self.trigger_sync_mode is SD_SyncModes.SYNC_CLK10:
            waveform_duration = int(len(waveform) / self.get_sample_rate()
                                    * 1e9)  # ns
            if waveform_duration % 100 != 0:
                raise KeysightError(f"Duration of the waveform must be a "
                                    f"multiple of 100 ns, because the "
                                    f"current trigger is synchronized with "
                                    f"PXI clock and therefore sampled with "
                                    f"10 MHz. Not following this requirement "
                                    f"leads to uncertainty of the trigger "
                                    f"pulse position relative to the waveform")

        # stopping AWG so the changes will take place according to the documentation
        # (not neccessary but a good practice)
        self.stop_AWG(channel)
        # loading a waveform to internal RAM and putting waveform into the channel's AWG queue
        self.load_waveform_to_channel(waveform, frequency, channel)
        # starting operation
        self.start_AWG(channel)

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
            period of trigger signal in ns
            trigger is synchronized with the continuous wave.

        Returns
        -------

        """
        self.stop_AWG(channel)  # stop output from this channel
        self.stop_modulation(
            channel)  # disable all modulation types for this channel
        # setup embedded function generator to produce sine wave with parameters
        self.setup_fg_sine(frequency, amplitude, phase, offset, channel)

        # output trigger has to be generated
        if trigger_sync_every is not None:
            # add zero modulation with trigger output at start
            arr = np.zeros(int(
                np.around(trigger_sync_every * self.get_sample_rate() / 1e9)))
            self._load_array_into_AWG(arr, channel)
            self.setup_modulation_amp(channel, 0)
            self.trigger_output_config("ON", channel,
                                       trig_length=self.trigger_length)

        # resetting phase for synchronization of multiple carrier signals from internal Function Generator
        self.module.channelPhaseResetMultiple(
            sum([1 << (chan - 1) for chan in self.synchronized_channels]))
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
        if self._prescaler == 0:
            fs = int(1e9)
        elif self._prescaler == 1:
            fs = int(2e8)
        elif self._prescaler > 1:
            fs = int(100 // self._prescaler * 1e6)
        return fs

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
        if np.max(np.abs(waveform)) >= 1.5:
            raise Exception(
                "signal maximal amplitude is exceeding AWG range: (-1.5 ; 1.5) volts")

        # number of points
        if (frequency > 1e9):
            raise Exception("frequency is exceeding AWG sampling rate: 1 GHz")

        duration_initial = 1 / frequency * 1e9 if frequency != 0 else 10.0  # float
        # interpolating input waveform to the next step
        # that rescales waveform to fit frequency
        interpolation_method = "cubic" if frequency != 0 else "linear"
        old_x = np.linspace(0, duration_initial, len(waveform) + 1)
        f_wave = interp1d(old_x, np.concatenate((waveform, [waveform[0]])),
                          kind=interpolation_method)

        # in order to satisfy NOTE_1 we simply make 10 subsequent waveforms
        # but to provide frequency accuracy, we are sampling from
        # interval 1000 times wider then the original, and we are extending
        # interpolation function domain using its periodicity
        # duration = duration_initial*1e4 if duration_initial < 1e2 else 1e6  # here it is

        duration = duration_initial
        new_x = np.linspace(0, duration, int(
            np.round(duration / self.get_sample_rate() * 1e9)), endpoint=False)

        # converting domain values in the function domain
        new_x_converted = np.remainder(new_x, duration_initial)
        waveform_array = f_wave(
            new_x_converted)  # obtaining new waveform walues

        normalization = np.max(np.abs(waveform_array))

        if (self.waveshape_types[channel - 1] == SD_Waveshapes.AOU_AWG):
            self.output_voltages[channel - 1] = normalization

        waveform_array /= normalization  # normalize waveform to (-1,1) interval
        self.repetition_frequencies[channel - 1] = frequency
        self._load_array_into_AWG(waveform_array, channel)

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
        ret = self.module.waveformLoad(wave, waveform_number)
        if (ret == SD_Error.INVALID_OBJECTID):
            # probably, such wave_id already exists
            ret = self.module.waveformReLoad(wave, waveform_number)

        self._handle_error(ret)

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
        if (self._update_dependent_on_start):
            self._load_dependent_channels()

        if ((not self.synchronized_channels) or (
                channel not in self.synchronized_channels)):
            # if it is single channel
            self.module.AWGqueueSyncMode(channel - 1, syncMode=self.sync_mode)
            ret = self.module.AWGstart(channel - 1)
            self._handle_error(ret)
        elif (channel in self.synchronized_channels):
            # if channel belongs to one of the synchronized groups
            channels_mask = 0
            for chan in self.synchronized_channels:
                channels_mask += 1 << (chan - 1)
                self.module.AWGqueueSyncMode(chan - 1, syncMode=self.sync_mode)
            ret = self.module.AWGstartMultiple(channels_mask)
            self._handle_error(ret)
        else:
            raise NotImplementedError(
                "Check channel number conditions on argument provided: channel={}".format(
                    channel))

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
        self._prescaler

    def plot_waveforms(self, voltage_output=False):
        import matplotlib.pyplot as plt
        plt.figure()
        for i, waveform in enumerate(self.waveforms):
            if (waveform is not None):
                mult = self.output_voltages[i] if voltage_output else 1
                plt.plot(waveform * mult, label="CH" + str(i + 1))
        plt.legend()

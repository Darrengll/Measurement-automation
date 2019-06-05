from drivers.instrument import Instrument

import sys
sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')

import keysightSD1
from keysightSD1 import SD_TriggerModes, SD_TriggerExternalSources, SD_TriggerBehaviors, SD_TriggerDirections
from keysightSD1 import SD_WaveformTypes, SD_Waveshapes, SD_MarkerModes, SD_SyncModes, SD_Error

import numpy as np
from scipy.interpolate import interp1d

class KeysightM3202A(Instrument):

    def __init__(self, name, slot, chassis=0):

        super().__init__(name, tags=['physical'])
        self.mask = 0
        self.module = keysightSD1.SD_AOU()
        self.module_id = self.module.openWithSlotCompatibility("M3202A", chassis, slot,
                                                               compatibility=keysightSD1.SD_Compatibility.LEGACY)
        self.clear()  # clear internal memory and AWG queues
        self.amplitudes = [0.0] * 4
        self.offsets = [0.0] * 4
        self.clock = None

        # Shamil a.k.a. 'BATYA' code here
        self.waveforms = [None] * 4
        self.waveform_ids = [-1] * 4
        self.output_voltages = [None] * 4
        self.trigger_modes = [SD_TriggerModes.AUTOTRIG] * 4
        self.trigger_ext_sources = [SD_TriggerExternalSources.TRIGGER_EXTERN] * 4  # from front panel
        self.trigger_behaviours = [SD_TriggerBehaviors.TRIGGER_FALL] * 4
        self.trigger_output = False
        self.synchronized_channels = None

    def _handle_error(self, ret_val):
        if( ret_val < 0 ):
            print(ret_val)
            raise Exception

    def clear(self):
        # clear internal memory and AWG queues
        ret = self.module.waveformFlush()
        self._handle_error(ret)

    def synchronize_channels(self, *channels):
        self.synchronized_channels = channels

    def unsychronize_channels(self):
        self.synchronized_channels = None

    def set_trigger(self, trigger_string: str="CONT", channel: int=1,
                    trigger_source: str="FRONT_PANEL"):
        """
        trigger_string : string
           'EXT' - external trigger on the front panel is used as trigger signal source
           'OUT' - device trigger input is a source of the starting trigger
           'CONT' - continious output  <---- DEFAULT setting
        channel : int
            1,2,3,4 - channel number
        trigger_source : string
            TODO implement trigger source if needed
            "FRONT_PANEL" - trigger from front panel
            "PXI_n" - PXI hardware trigger number 'n'
        """
        nAWG = channel
        if trigger_string == "EXT":  # front panel
            # for each 'cycle' (see 'cycle' definition in docs)
            self.trigger_modes[channel-1] = SD_TriggerModes.EXTTRIG_CYCLE
            ret = self.module.AWGtriggerExternalConfig(nAWG, self.trigger_ext_sources[channel],
                                                       self.trigger_behaviours[channel],
                                                       SD_SyncModes.SYNC_NONE)
            self._handle_error(ret)
        elif trigger_string == "CONT":
            self.trigger_modes[channel-1] = SD_TriggerModes.AUTOTRIG

    def trigger_output_config(self, trig_mode="ON", channel=-1):
        """
        Manipulates output trigger for AWG 86110A
        Parameters
        ----------
        trig_mode : str
            "ON" - output syncAB for channel 1
            "OFF" - no output
        """
        if trig_mode == "ON":
            self.trigger_output = True
        elif trig_mode == "OFF":
            self.trigger_output = False
        else:
            raise NotImplementedError("trig_mode argument can be only 'ON' or 'OFF' ")

        # if channel is equal to -1, then output trigger is disabled for all synchronized channels
        if( channel == - 1 ):
            channels_to_config = self.synchronized_channels
        else:
            channels_to_config = [channel]

        for chan in channels_to_config:
            if self.trigger_output:
                # configuring trigger IO as output
                self.module.triggerIOconfig(SD_TriggerDirections.AOU_TRG_OUT)

                # adding marker to the specified channel
                self.module.AWGqueueMarkerConfig(chan - 1, SD_MarkerModes.EVERY_CYCLE,
                                                 0, 1, 1, syncMode=0,  # trigger synch with internal CLKsys
                                                 length=100,  # trigger length (100a.u. x 10ns => 1000 ns trigger length)
                                                 delay=0)
                break  # first channel from group is enough to produce output marker
            else:
                self.module.triggerIOconfig(SD_TriggerDirections.AOU_TRG_OUT)
                # deleting marker to the specified channel
                self.module.AWGqueueMarkerConfig(chan - 1, SD_MarkerModes.DISABLED,
                                                 0, 1, 1, syncMode=0,  # trigger synch with internal CLKsys
                                                 length=100,  # trigger length (100a.u. x 10ns => 1000 ns trigger length)
                                                 delay=0)

    def output_arbitrary_waveform(self, waveform, frequency, channel, asynchronous=False):
        """
        Prepare and output an arbitrary waveform repeated at some repetition_rate

        Parameters:
        -----------
        waveform: array
            ADC levels, in Volts. max( abs(waveform) ) < 1.5 V
        repetition_rate: float, Hz
            frequency at which the waveform will be repeated
        channel: 1,2,3,4
            channel which will output the waveform

        NOTE_1: waveform length must be multiple of 10, but you do not need to
        provide array that comply with such condiition. Note only that this array will be copied 10
        times before loading to board RAM in order to satisfy datasheet condition above.
        See user guide, p.140
        and datasheet, p.8

        NOTE_2: Sampling speed depends on the SD_WaveformType value of the SD_Wave().
        see datasheet for more details

        NOTE_3: waveform's last point must coincide with the waveforms first point.
        Shamil: I assume there is no need to pass np.sin(2*np.pi*np.linspace(0,1,N_pts+1)) here
        because this array basically includes the boundary point twice, which values are automatically
        should be assumed equal by specifying waveform frequency.
        It is rather more rational to  pass np.sin(2*np.pi*np.linspace(0,1,N_pts+1)[:-1])
        Also I'd prefer to pass function explicitly as lambda or something like that
        and only then generate points inside this class method. 25.04.2019

        NOTE_4: I suggest we rather use embeded function generators with amplitude modulation
        in our AWG solution, rather than generating sinus from scratch and using DAC only.
        This is necessary in order to improve frequency accuracy and stability.
        """
        # loading waveform to internal RAM and putting waveform into the channel's AWG queue
        self.load_waveform_to_channel(waveform, channel, frequency)

        # start operation
        ret = self.start_AWG(channel)
        self._handle_error(ret)

    def output_continuous_wave(self, frequency, amplitude, phase, offset, waveform_resolution,
                               channel, asynchronous=False):
#         self.module.channelAmplitude(channel, amplitude)
#         self.module.channelFrequency(channel, frequency)
#         self.module.channelPhase(channel, phase / np.pi * 180)
#         self.module.channelOffset(channel, offset)
#         self.module.channelWaveShape(channel, keysightSD1.SD_Waveshapes.AOU_SINUSOIDAL)
        
#         self.module.waveformFlush()
#         self.module.AWGflush(channel)
        
#         ret = self.start_AWG(channel)
#         self._handle_error(ret)
        n_points = np.around(1 / frequency / waveform_resolution * 1e9) + 1 if frequency != 0 else 3
        waveform = amplitude * np.sin(2 * np.pi * np.linspace(0, 1, n_points) + phase) + offset
        self.output_arbitrary_waveform(waveform, frequency, channel, asynchronous=asynchronous)

    def load_waveform_to_channel(self, waveform, channel, frequency):
        waveform = np.array(waveform, dtype=np.float16)

        if np.max(np.abs(waveform)) >= 1.5:
            raise Exception("signal maximaxl amplitude is exceeding AWG range: (-1.5 ; 1.5) volts")

        # number of points
        if (frequency > 1e9):
            raise Exception("frequency is exceeding AWG sampling rate: 1 GHz")

        duration_initial = 1 / frequency * 1e9 if frequency != 0 else 10.0  # float
        # interpolating input waveform to the next step
        # that rescales waveform to fit frequency
        interpolation_method = "cubic" if frequency != 0 else "linear"
        old_x = np.linspace(0, duration_initial, len(waveform))
        f_wave = interp1d(old_x, waveform, kind=interpolation_method)

        # in order to satisfy NOTE_1 we simply make 10 subsequent waveforms
        # but to provide frequency accuracy, we are sampling from
        # interval 1000 times wider then the original, and we are extending
        # interpolation function domain using its periodicity
        # duration = duration_initial*1e4 if duration_initial < 1e2 else 1e6  # here it is

        duration = duration_initial
        new_x = np.arange(0, duration, 1.0)

        # converting domain values in the function domain
        new_x_converted = np.remainder(new_x, duration_initial)
        waveform_accurate = f_wave(new_x_converted)  # obtaining new waveform walues
        wave_amp = np.max(np.abs(waveform_accurate))
        waveform_accurate /= wave_amp  # normalizing waveform to [-1; 1] interval

        self.waveforms[channel - 1] = waveform_accurate
        self.waveform_ids[channel - 1] = channel

        # NOT WORKING SOMEHOW
        # ret = self.module.AWGfromArray(nAWG, self.trigger_modes[channel-1], 0, 1, 0,
        #                                SD_WaveformTypes.WAVE_ANALOG,
        #                                waveform_rescaled)
        # self._handle_error(ret)

        # creating SD_Wave() object from keysight API
        wave = keysightSD1.SD_Wave()
        wave.newFromArrayDouble(SD_WaveformTypes.WAVE_ANALOG, waveform_accurate)
        wave_id = channel - 1

        # setting generation parameters (see user's guide)
        ret = self.module.channelWaveShape(channel - 1, SD_Waveshapes.AOU_AWG)
        self._handle_error(ret)

        # load waveform to board RAM
        ret = self.module.waveformLoad(wave, wave_id)
        if (ret == SD_Error.INVALID_OBJECTID):
            # probably, such wave_id already exists
            ret = self.module.waveformReLoad(wave, wave_id)
        self._handle_error(ret)

        # clear AWG channel queue
        ret = self.module.AWGflush(channel - 1)
        self._handle_error(ret)

        # put waveform as the first and only member of the
        # channel's AWG queue
        ret = self.module.AWGqueueWaveform(channel - 1, wave_id, self.trigger_modes[channel - 1],
                                           # default trigger mode is "CONT"
                                           0, 0, 0)  # 0 - means infinite
        self._handle_error(ret)

        # set amplitude in volts
        ret = self.module.channelAmplitude(channel - 1, wave_amp)
        self._handle_error(ret)
        self.output_voltages[channel - 1] = wave_amp

    def start_AWG(self, channel):
        # stop all AWG channels
        if ((self.synchronized_channels is None) or
                (channel not in self.synchronized_channels)):
            ret = self.module.AWGstop(channel-1)
            self._handle_error(ret)
        elif (channel in self.synchronized_channels):
            channels_mask = sum([1 << (chan - 1) for chan in self.synchronized_channels])
            ret = self.module.AWGstopMultiple(channels_mask)
            self._handle_error(ret)
        else:
            raise NotImplementedError("none of the conditions is true")

        # clear all markers of the selected device
        for chan in [1, 2, 3, 4]:
            self.module.AWGqueueMarkerConfig(chan-1, SD_MarkerModes.DISABLED,
                                             0, 1, 1, 0, 10, 0)

        self.trigger_output_config("ON" if self.trigger_output else "OFF", channel)

        # explicitly specifying waveform synchronization clock
        self.module.AWGqueueSyncMode(channel - 1, syncMode=0)  # synch with internal CLKsys

        # start single or multiple channels
        if ((self.synchronized_channels is None) or
                (channel not in self.synchronized_channels)):
            ret = self.module.AWGstart(channel-1)
        elif channel in self.synchronized_channels:
            channels_mask = sum([1 << (chan - 1) for chan in self.synchronized_channels])
            ret = self.module.AWGstartMultiple(channels_mask)

        return ret


# from qsweepy.instrument_drivers._Keysight_M3202A.simple_sync import *

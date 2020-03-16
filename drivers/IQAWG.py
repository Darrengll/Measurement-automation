    # keysightAWG.py
# Gleb Fedorov <vdrhc@gmail.com>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA


from numpy import *
from lib2.IQPulseSequence import *
import drivers.keysightSD1 as keysightSD1
from drivers.keysightM3202A import KeysightM3202A
# there are functions that are not universal and work only with M3202A
from drivers.keysightAWG import KeysightAWG


class AWGChannel():

    def __init__(self, host_awg, channel_number):

        self._host_awg = host_awg
        self._channel_number = channel_number

    def output_arbitrary_waveform(self, waveform, frequency, asynchronous=False):

        self._host_awg.output_arbitrary_waveform(waveform, frequency,
                                                 self._channel_number, asynchronous=asynchronous)

    def output_continuous_wave(self, frequency, amplitude, phase, offset, waveform_resolution, asynchronous=False,
                               trigger_sync_every=None):
        self._host_awg.output_continuous_wave(frequency, amplitude, phase,
                                              offset, waveform_resolution, self._channel_number, asynchronous=asynchronous,
                                              trigger_sync_every=trigger_sync_every)

class CalibratedAWG():

    def __init__(self, channel: AWGChannel):
        self._channel = channel
        # requires to be setted before pulse builder object is instantiated
        self._calibration = None

    def set_parameters(self, parameters):
        """
        Sets various parameters from a dictionary

        Parameters:
        -----------
        parameteres: dict {"param_name":param_value, ...}
        """
        par_names = ["calibration"]
        for par_name in par_names:
            if par_name in parameters.keys():
                setattr(self, "_"+par_name, parameters[par_name])


    def get_calibration(self):
        return self._calibration


    def get_pulse_builder(self):
        """
        Returns a PulseBuilder instance using the calibration loaded before
        """
        return PulseBuilder(self._calibration)

    def output_pulse_sequence(self, pulse_sequence, asynchronous=False):
        """
        Load and output given PulseSequence.

        Parameters:
        -----------
        pulse_sequence: PulseSequence instance
        """
        frequency = 1/pulse_sequence.get_duration()*1e9
        self._channel.output_arbitrary_waveform(pulse_sequence\
                        .get_waveform(), frequency, asynchronous=asynchronous)

class IQAWG():
    def __init__(self, channel_I: AWGChannel, channel_Q: AWGChannel, triggered=False):
        self._channels = [channel_I, channel_Q]
        self.MAX_OUTPUT_VOLTAGE = channel_I._host_awg.MAX_OUTPUT_VOLTAGE
        self._triggered = triggered
        self._calibration: IQCalibrationData = None  # TODO: BUG CAN BE HERE (SHAMIL 23.04.2019)

    def set_parameters(self, parameters):
        """
        Sets various parameters from a dictionary

        Parameters:
        -----------
        parameteres: dict {"param_name":param_value, ...}
        """
        par_names = ["calibration"]
        for par_name in par_names:
            if par_name in parameters.keys():
                setattr(self, "_"+par_name, parameters[par_name])

    def get_calibration(self):
        return self._calibration

    # def set_channel_coupling(self, state):
    #     '''
    #     Assuming that user knows what he is doing here. Make sure your channels
    #     are synchronized!
    #     '''
    #     pass

    def get_pulse_builder(self):
        """
        Returns an IQPulseBuilder instance using the IQ calibration loaded before
        """
        return IQPulseBuilder(self._calibration)

    def output_continuous_IQ_waves(self, frequency, amplitudes, relative_phase,
        offsets, waveform_resolution, optimized=True):
        """
        Prepare and output a sine wave of the form: y = A*sin(2*pi*frequency + phase) + offset
        on both of the I and Q channels
        Parameters:
        -----------
        frequency: float, Hz
            frequency of the output waves
        amplitudes: float, V
            amplitude of the output waves
        phase: float
            relative phase in radians of the output waves
        offsets: float, V
            voltage offset of the waveforms
        waveform_resolution: float, ns
            resolution in time of the arbitrary waveform representing one period
            of the wave
        channel: 1 or 2
            channel which will output the wave
        optimized: boolean
            first channel will be called with asynchronous = True if optimized is True
        """
        self._output_continuous_wave(frequency, amplitudes[0], relative_phase,
            offsets[0], waveform_resolution, 1, asynchronous = optimized)
        self._output_continuous_wave(frequency, amplitudes[1], 0,
            offsets[1], waveform_resolution, 2, asynchronous=False)

    def output_IQ_waves_from_calibration(self, optimized=True, trigger_sync_every=None):
        """

        Parameters
        ----------
        optimized
        trigger_sync_every

        Returns
        -------

        Notes
        -------
        What is actually happening, when there are two channels that are in the
        same 'synchronization group' (term from keysightM3202A.py)?
        Both channels are modulated by amplitude with zero modulation coefficient
        Every call to 'output_continuous_wave' with some number 'trigger_sync_every'
        will setup trigger on starting the first channel from the channel group.
        So the first channel will setup output and mark itself as trigger source.
        The first channel will setup its own output yet still mark the first channel as trigger source.
        The last is due to the fact that channel №1 is usually first in the 'synchronization group' of the
        awg device.
        """
        cal = self._calibration
        self._output_continuous_wave(cal._if_frequency, cal._if_amplitudes[0],
                                     cal._if_phase[0], cal._if_offsets[0],
                                     cal._waveform_resolution, 1, asynchronous=optimized,
                                     trigger_sync_every=trigger_sync_every)
        self._output_continuous_wave(cal._if_frequency, cal._if_amplitudes[1],
                                     0, cal._if_offsets[1],
                                     cal._waveform_resolution, 2, asynchronous=False,
                                     trigger_sync_every=trigger_sync_every)

    def output_zero(self, trigger_every_period=False, repetition_period_ns=None):
        cal = self._calibration
        awg = self._channels[0]._host_awg
        chanI = self._channels[0]._channel_number
        chanQ = self._channels[1]._channel_number
        awg.synchronize_channels(chanI, chanQ)
        if trigger_every_period:
            # 100 ns trigger length after every 'start' of the playing
            awg.trigger_output_config(trig_mode="ON", trig_length=100)
        else:
            # turns trigger of
            awg.trigger_output_config(trig_mode="OFF")
        waveform = np.zeros(int(repetition_period_ns/awg.get_sample_rate()*1e9))
        self._channels[0].output_arbitrary_waveform(waveform, 1/repetition_period_ns*1e9)
        self._channels[1].output_arbitrary_waveform(waveform, 1/repetition_period_ns*1e9)

    def output_continuous_two_freq_IQ_waves(self, dfreq, ampl_coefs=(2, 2)):
        fs = self._channels[0]._host_awg.get_sample_rate()  # Hz
        N = fs / dfreq
        array_mod = cos(linspace(0, 2 * pi, N, endpoint=False))
        self._channels[0]._host_awg.synchronize_channels(*[channel._channel_number for channel in self._channels])
        self.output_modulated_IQ_waves(array_mod, self._channels[0]._host_awg._prescaler, ampl_coefs)

    def change_amplitudes_of_cont_IQ_waves(self, ampl_coef):
        cal = self._calibration
        awg = self._channels[0]._host_awg
        awg.change_amplitude_of_carrier_signal(cal._if_amplitudes[0], self._channels[0]._channel_number, ampl_coef)
        awg.change_amplitude_of_carrier_signal(cal._if_amplitudes[1], self._channels[1]._channel_number, ampl_coef)

    def update_modulation_coefficient_of_IQ_waves(self, modulation_amp):
        """
        Parameters
        ----------
        modulation_amp : float
            amplitude of the modulation signal
            G * AWG(t) * carrier_signal(t)
            G = 'modulation_amp'
            AWG(t) - normalized such that max(abs(AWG(t))) = 1

        Returns
        -------
        None
        """
        awg = self._channels[0]._host_awg
        chanI = self._channels[0]._channel_number
        chanQ = self._channels[1]._channel_number
        awg.setup_modulation_amp(chanI, modulation_amp)
        awg.setup_modulation_amp(chanQ, modulation_amp)

    def setup_AM_and_carrier_from_calibration(self, calibration=None, amp_coeffs=(1, 1)):
        """
        This function tells awg that it's output will be modulated and setups
        carrier sine signal parameters based on calibration parameters.
        This function is used primarly by time domain measurement classes that utilize sine
        function generator feature of the AWG. This function is called by them during
        'set_fixed_parameters'.

        Parameters
        ---------
        modulationAmp : float
            WARNING. THIS IS SPECIFIC FOR M3202A.
            AM Modulation
            Output(t) = G * AWG(t) * cos(2 * pi * f * t + phi)
            'modulation_amp' = G

        calibration : IQCalibrationData
            Overwrites 'self._calibration' if provided
        amp_coeffs : tuple[float]
            Multipliers for carrier signal amplitude
            Used to tune the power of the output signal
            that is roughly proportional to the this coefficients

        Returns
        -------
        None
        """
        if calibration is not None:
            self._calibration = calibration
        elif self._calibration is not None:
            cal = self._calibration
        else:  # no calibration found
            raise ValueError("no calibration provided")

        if( self._channels[0]._host_awg is self._channels[1]._host_awg ):
            awg = self._channels[0]._host_awg
        else:
            raise NotImplementedError("Two channels are in different AWG. "
                                      "'setup_carrier_from_calibration' is not implemented")

        # getting AWG's channel numbers
        chanI = self._channels[0]._channel_number
        chanQ = self._channels[1]._channel_number

        # stop AWG
        awg.stop_AWG(chanI)

        awg.synchronize_channels(chanI, chanQ)  # verify that both channels are started simultaneously

        # tell AWG that amplitude modulation is chosen
        awg.setup_modulation_amp(chanI, cal._if_amplitudes[0] * amp_coeffs[0])
        awg.setup_modulation_amp(chanQ, cal._if_amplitudes[1] * amp_coeffs[1])
        # setup carrier signal according to calibration
        awg.setup_fg_sine(cal._if_frequency, 0,
                          cal._if_phase[0], cal._if_offsets[0], chanI)
        awg.setup_fg_sine(cal._if_frequency, 0,
                          0, cal._if_offsets[1], chanQ)

    def output_modulated_IQ_waves(self, array_mod, prescaler=0, ampl_coeffs=(1, 1)):
        """
        WARNING: THIS FUNCTION IS FOR M3202A ONLY

        AM Modulation
        -------------
        Output(t) = G * AWG(t)) * cos(2 * pi * f * t + phi)
        G = 'modulation_amp'

        Example:
        --------
        for required signal s(t) = A * cos(2*pi*f*t) * cos(2*pi*dfreq*t)
        Modulation should be
        A + G * AWG(t) = A * cos(2*pi*dfreq*t)
        G * AWG(t) = A * (cos(2*pi*dfreq*t) - 1)
        The array must be normalized, i.e. max(abs(AWG(t))) = 1
        AWG(t) = (cos(2*pi*dfreq*t) - 1) / 2, where division by 2 is for normalization
        G = 2 * A
        modulationCoeff = 2
        """
        cal = self._calibration
        awg = self._channels[0]._host_awg
        chanI = self._channels[0]._channel_number
        chanQ = self._channels[1]._channel_number
        awg.synchronize_channels(chanI, chanQ)

        awg.stop_AWG(chanI)
        awg.clear()
        awg.setup_fg_sine(cal._if_frequency, 0,
                          cal._if_phase[0], cal._if_offsets[0], chanI)
        awg.setup_fg_sine(cal._if_frequency, 0,
                          0, cal._if_offsets[1], chanQ)
        waveform_id = 3
        awg.load_modulating_waveform(array_mod, waveform_id)

        # single waveform from RAM is used to modulate both channelss
        awg.queue_waveform(chanI, waveform_id, prescaler)
        awg.queue_waveform(chanQ, waveform_id, prescaler)
        deviationGainI = cal._if_amplitudes[0] * ampl_coeffs[0]
        deviationGainQ = cal._if_amplitudes[1] * ampl_coeffs[1]
        awg.module.modulationAmplitudeConfig(chanI-1, keysightSD1.SD_ModulationTypes.AOU_MOD_AM, deviationGainI)
        awg.module.modulationAmplitudeConfig(chanQ-1, keysightSD1.SD_ModulationTypes.AOU_MOD_AM, deviationGainQ)
        awg.start_AWG(chanI)

    def stop_modulated_IQ_waves(self):
        awg = self._channels[0]._host_awg
        chanI = self._channels[0]._channel_number
        chanQ = self._channels[1]._channel_number
        awg.stop_modulation(chanI)
        awg.stop_modulation(chanQ)

    def _output_continuous_wave(self, frequency, amplitude, phase, offset,
            waveform_resolution, channel, asynchronous=False, trigger_sync_every=None):
        """
        Prepare and output a sine wave of the form: y = A*sin(2*pi*frequency + phase) + offset

        Parameters:
        -----------
        frequency: float, Hz
            frequency of the output wave
        amplitude: float, V
            amplitude of the output wave
        phase: float
            phase in radians of the iutput wave
        offset: float, V
            voltage offset of the waveform
        waveform_resolution: float, ns
            resolution in time of the arbitrary waveform representing one period
            of the wave
        channel: 1 or 2
            channel which will output the wave
        """
        self._channels[channel-1].output_continuous_wave(frequency, amplitude, phase, offset,
                                                         waveform_resolution, asynchronous=asynchronous,
                                                         trigger_sync_every=trigger_sync_every)

    def output_pulse_sequence(self, pulse_sequence, asynchronous=False):
        """
        Load and output given IQPulseSequence.

        Parameters:
        -----------
        pulse_sequence: IQPulseSequence instance
        """
        resolution = pulse_sequence.get_waveform_resolution()
        length = len(pulse_sequence.get_I_waveform())
        if self._triggered:
            # this is made if 2 AWG is triggering another one and has the same signal period
            duration = pulse_sequence.get_duration() - 1000 * resolution
            end_idx = length - 1000
        else:
            duration = pulse_sequence.get_duration()
            end_idx = length

        frequency = 1 / duration * 1e9
        self._channels[0].output_arbitrary_waveform(pulse_sequence \
                                                    .get_I_waveform()[:end_idx], frequency,
                                                    asynchronous=True)
        self._channels[1].output_arbitrary_waveform(pulse_sequence
                                                    .get_Q_waveform()[:end_idx], frequency,
                                                    asynchronous=asynchronous)

class IQAWG_Multiplexed(IQAWG):
    """
        (generate 2 qubit frequencies pulses with single lo and 2 channeled AWG)
        uses 2 different callibrations to generate pulses for q1 & q2 resp.
    """

    def set_parameters(self, parameters):
        """
        Sets various parameters from a dictionary

        Parameters:
        -----------
        parameteres: dict {"param_name":param_value, ...}
        """
        par_names = ["calibration","calibration2"]
        for par_name in par_names:
            if par_name in parameters.keys():
                setattr(self, "_"+par_name, parameters[par_name])

    def get_pulse_builder(self):
        """
        Returns an IQPulseBuilder instance using the IQ calibration loaded before
        """
        return [IQPulseBuilder(self._calibration), IQPulseBuilder(self._calibration2)]

    def output_continuous_IQ_waves(self, frequency, amplitudes, relative_phase,
        offsets, waveform_resolution, optimized = True):
        """
        Prepare and output a sine wave of the form: y = A*sin(2*pi*frequency + phase) + offset
        on both of the I and Q channels
        Parameters:
        -----------
        frequency: float, Hz
            frequency of the output waves
        amplitudes: float, V
            amplitude of the output waves
        phase: float
            relative phase in radians of the iutput waves
        offsets: float, V
            voltage offset of the waveforms
        waveform_resolution: float, ns
            resolution in time of the arbitrary waveform representing one period
            of the wave
        channel: 1 or 2
            channel which will output the wave
        optimized: boolean
            first channel will be called with asynchronous = True if optimized is True
        """
        self._output_continuous_wave(frequency, amplitudes[0], relative_phase,
            offsets[0], waveform_resolution, 1, asynchronous = optimized)
        self._output_continuous_wave(frequency, amplitudes[1], 0,
            offsets[1], waveform_resolution, 2, asynchronous = False)

    def _output_continuous_wave(self, frequency, amplitude, phase, offset,
            waveform_resolution, channel, asynchronous):
        """
        Prepare and output a sine wave of the form: y = A*sin(2*pi*frequency + phase) + offset

        Parameters:
        -----------
        frequency: float, Hz
            frequency of the output wave
        amplitude: float, V
            amplitude of the output wave
        phase: float
            phase in radians of the iutput wave
        offset: float, V
            voltage offset of the waveform
        waveform_resolution: float, ns
            resolution in time of the arbitrary waveform representing one period
            of the wave
        channel: 1 or 2
            channel which will output the wave
        """

        N_points = 1/frequency/waveform_resolution*1e9+1 if frequency !=0 else 3
        waveform = amplitude*sin(2*pi*linspace(0,1,N_points)+phase) + offset
        self._channels[channel-1].output_arbitrary_waveform(waveform, frequency,
                                                            asynchronous=asynchronous)

    def output_pulse_sequence(self, pulse_sequence, asynchronous=False):
        """
        Load and output given IQPulseSequence.

        Parameters:
        -----------
        pulse_sequence: IQPulseSequence instance
        """
        resolution = pulse_sequence.get_waveform_resolution()
        length = len(pulse_sequence.get_I_waveform())
        if self._triggered:
            duration = pulse_sequence.get_duration() - 1000 * resolution
            end_idx = length - 1000
        else:
            duration = pulse_sequence.get_duration()
            end_idx = length

        frequency = 1 / duration * 1e9
        self._channels[0].output_arbitrary_waveform(pulse_sequence \
                                                    .get_I_waveform()[:end_idx], frequency,
                                                    asynchronous=True)
        self._channels[1].output_arbitrary_waveform(pulse_sequence
                                                    .get_Q_waveform()[:end_idx], frequency,
                                                    asynchronous=asynchronous)
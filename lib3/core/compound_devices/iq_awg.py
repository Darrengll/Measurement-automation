""" IN DEVELOPMENT. NOT USED ANYWHERE."""
from lib3.pulses.pulse_sequence import PulseBuilder
from lib3.pulses.iq_pulse_sequence import IQPulseBuilder, IQPulseSequence
from lib3.mixers.data_structures import HetIQCalibration, CalibrationSingleUp

from drivers.keysightM3202A import KeysightM3202A
from drivers.keysightAWG import KeysightAWG

import numpy as np

from typing import Union


class AWGChannel:
    def __init__(self, host_awg, channel_number):
        """
        Parameters
        ----------
        host_awg : KeysightM3202A
            AWG instance class.
        channel_number : int
            number of channel starting from 0.

        Notes
        ---------
        Only `KeysightM3202A` is currently supported
        """
        self.host_awg = host_awg
        self.awg_channel_number = channel_number

    def output_arbitrary_waveform(self, waveform, repetition_frequency):
        self.host_awg.output_arbitrary_waveform(
            waveform, repetition_frequency, self.awg_channel_number
        )

    def output_signal(self, signal, rep_freq):
        if isinstance(signal, IQPulseSequence):
            pass  # TODO implement to output both IQPulseSequence and iterables

    def output_continuous_wave(self, carrier_frequency, amplitude,
                               phase, offset, waveform_resolution,
                               trigger_sync_every=None):
        self.host_awg.output_continuous_wave(
            carrier_frequency, amplitude, phase,
            offset, waveform_resolution,
            self.awg_channel_number,
            trigger_sync_every=trigger_sync_every
        )


class CalibratedAWGChannel(AWGChannel):
    """
    Extension of AWGChannel class with upconversion calibration.
    Mostly unused. Pending to delete.
    """
    def __init__(self, host_awg, channel_number, calibration=None):
        """

        Parameters
        ----------
        host_awg : Union[KeysightM3202A, KeysightAWG]
            AWG instance class.
        channel_number : int
            number of channel starting from 0.
        calibration : Optional[CalibrationSingleUp]
            Upconversion calibration
            Note: easily extended to include downconversion calibration.
        """
        super().__init__(host_awg, channel_number)
        # requires to be setted before pulse builder object is instantiated
        self._calibration = CalibrationSingleUp()

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


class IQAWG:
    def __init__(self, channel_i, channel_q, calibration=None,
                 triggered=False):
        """
        Parameters
        ----------
        channel_i : AWGChannel
        channel_q : AWGChannel
        calibration : HetIQCalibration
        triggered : bool
        """
        self._awg_channels = [channel_i, channel_q]
        self._channel_i = channel_i
        self._channel_q = channel_q
        # `host_awg` has to be the same
        if channel_i.host_awg == channel_q.host_awg:
            self.host_awg = channel_i.host_awg
        else:
            raise NotImplementedError(
                "Channels with different `host_awg` values are "
                "not supported"
            )
        self.MAX_OUTPUT_VOLTAGE = self.host_awg.MAX_OUTPUT_VOLTAGE
        self.VOLTAGE_RESOLUTION_BITS = self.host_awg.VOLTAGE_RESOLUTION_BITS
        self.MIN_SAMPLE_PERIOD = self.host_awg.MIN_SAMPLE_PERIOD

        self._calibration = calibration
        self._triggered = triggered

    def reset_host_awgs(self):
        for awg_channel in self._awg_channels:
            awg_channel.host_awg.reset()

    def set_calibration(self, calibration):
        self._calibration = calibration

    def get_calibration(self):
        return self._calibration

    def get_sample_rate(self):
        """
        Returns
        -------
        sample_rate : int
            Returns current AWG's sample rate in Hz.
        """
        self.host_awg.get_sample_rate()

    def get_sample_period(self):
        """
        Returns
        -------
        sample_period : np.float
            Sample period in nanoseconds
        """
        return self.host_awg.get_sample_period()

    def get_pulse_builder(self):
        """
        Returns an IQPulseBuilder instance using the IQ calibration loaded before
        """
        return IQPulseBuilder(self._calibration)

    def output_iq_signal(self, s):
        """
        Outputs IQ signal `s` represented as complex valued 1D array in Volts
        with sampling rate equal to default IQAWG sample rate.
        Signal is repeated infinitely by hardware.

        Parameters
        ----------
        s : Union[IQPulseSequence,np.ndarray, list]
            Complex valued 1D array. Real part interpreted as signal for I
            channel. Image part as Q channel accordingly.

        Returns : None
        -------
        """
        rep_frequency = 1/(len(s)*self.get_sample_period())
        self._channel_i.output_arbitrary_waveform(np.real(s), rep_frequency)
        self._channel_q.output_arbitrary_waveform(np.imag(s), rep_frequency)

    def output_zero(self, trigger_sync_every=None):
        """

        Parameters
        ----------
        trigger_sync_every

        Returns
        -------

        """
        cal = self._calibration
        awg = self._awg_channels[0].host_awg
        chanI = self._awg_channels[0].awg_channel_number
        chanQ = self._awg_channels[1].awg_channel_number
        awg.synchronize_channels(chanI, chanQ)
        if trigger_sync_every is None:
            # turns trigger off
            awg.trigger_output_config(trig_mode="OFF")
        else:
            # 100 ns trigger length after every 'start' of the playing
            awg.trigger_output_config(trig_mode="ON",
                                      trig_length=trigger_sync_every)
        waveform0 = np.zeros(
            int(trigger_sync_every / awg.get_sample_rate() * 1e9)) \
                    + cal._dc_offsets[0]
        waveform1 = np.zeros(
            int(trigger_sync_every / awg.get_sample_rate() * 1e9)) \
                    + cal._dc_offsets[1]
        self._channels[0].output_arbitrary_waveform(
            waveform0, 1 / trigger_sync_every * 1e9)
        self._channels[1].output_arbitrary_waveform(
            waveform1, 1 / trigger_sync_every * 1e9)

    def output_pulse_sequence(self, pulse_sequence):
        """
        Load and output given IQPulseSequence.

        Parameters:
        -----------
        pulse_sequence: IQPulseSequence
        """
        sample_period = 1/self.get_sample_period()  # sample ate in GHz
        length = pulse_sequence.get_length()
        if self._triggered:
            # this is made if 2 AWG is triggering another one and has
            # the same trace period.
            # Signal is cutted at the end for 100 nanoseconds
            cut_ns = 100
            duration = pulse_sequence.get_duration() - cut_ns
            end_idx = length - np.ceil(cut_ns*sample_rate)
        else:
            duration = pulse_sequence.get_duration()
            end_idx = length

        frequency = 1 / duration * 1e9
        self._channel_i.output_arbitrary_waveform(
            pulse_sequence.get_I_waveform()[:end_idx], frequency
        )
        self._channel_q.output_arbitrary_waveform(
            pulse_sequence.get_Q_waveform()[:end_idx], frequency)

# KeysightAWG.py
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


from drivers.instrument import Instrument
from numpy import *
import visa
import types
import time
import logging
from lib2.IQPulseSequence import *

from enum import Enum


class WaveformType(Enum):
    pulse = "PULS"
    square = "SQUA"
    ramp = "RAMP"
    noise = "NOIS"
    dc = "DC"
    sine = "SIN"
    arbitrary = "USER"


class KeysightAWG(Instrument):

    def __init__(self, address):
        """Create a default Keysight AWG instrument"""
        Instrument.__init__(self, 'AWG', tags=['physical'])
        self._address = address
        rm = visa.ResourceManager()
        self._visainstrument = rm.open_resource(self._address)

        self._visainstrument.write(":DIG:TRAN:INT 1")

        self.add_parameter('outp1',
                           flags=Instrument.FLAG_GETSET, units='', type=int)

        self.add_parameter('outp2',
                           flags=Instrument.FLAG_GETSET, units='', type=int)

        self.add_parameter('outp1_compl',
                           flags=Instrument.FLAG_GETSET, units='', type=int)

        self.add_parameter('outp2_compl',
                           flags=Instrument.FLAG_GETSET, units='', type=int)

        self.add_parameter('2nd_delay',
                           flags=Instrument.FLAG_GETSET, units='', type=float)

        self.add_parameter('1st_delay',
                           flags=Instrument.FLAG_GETSET, units='', type=float)

        self.add_parameter('2nd_width',
                           flags=Instrument.FLAG_GETSET, units='', type=float)

        self.add_parameter('1st_width',
                           flags=Instrument.FLAG_GETSET, units='', type=float)

        self.add_function("apply_waveform")

        # High-level functions

    def output_arbitrary_waveform(self, waveform, repetition_rate, channel, asynchronous=False):
        """
        Prepare and output an arbitrary waveform repeated at some repetition_rate

        Parameters:
        -----------
        waveform: array
            ADC levels, in Volts
        repetition_rate: foat, Hz
            frequency at which the waveform will be repeated
        channel: 1 or 2
            channel which will output the waveform
        """
        waveform = array(waveform)
        if len(set((waveform * 8191).astype(int))) == 1:
            # Crest data out of range KOSTYL FUCK YOU KEYSIGHT look carefully.
            waveform = waveform[:3]
        self.load_arbitrary_waveform_to_volatile_memory(waveform[:-1], channel)
        self.prepare_waveform(WaveformType.arbitrary, repetition_rate, 2, 0, channel)
        self.set_output(channel, 1)

    def output_continuous_wave(self, frequency=100e6, amplitude=0.1, phase=0, offset=0, waveform_resolution=1,
                               channel=1):
        """
        Prepare and output a sine wave of the form: y = A*sin(2*pi*frequency + phase) + offset

        Parameters:
        -----------
        frequency: float
            frequency of the output wave
        amplitude: float
            amplitude of the output wave
        phase: float
            phase in radians of the iutput wave
        offset: float
            voltage offset of the waveform
        waveform_resolution: float, ns
            resolution in time of the arbitrary waveform representing one period of the wave
        chanel:1 or 2
            channel which witll output the wave
        """

        N_points = 1 / frequency / waveform_resolution * 1e9 + 1 if frequency != 0 else 3
        waveform = amplitude * sin(2 * pi * linspace(0, 1, N_points) + phase) + offset
        self.output_arbitrary_waveform(waveform, frequency, channel)

        # Basic low-level functions

    def set_channel_coupling(self, state):
        self._visainstrument.write(":TRAC:CHAN1:%s" % ("ON" if state == True else "OFF"))
        self._visainstrument.write(":TRAC:CHAN2:%s" % ("ON" if state == True else "OFF"))

    def apply_waveform(self, waveform, freq, amp, offset, channel=1):
        """
        Set one of the pre-loaded waveforms as output and output it.
        This function will turn on both + and - outputs of the channel. If you
        don't want this behaviour, you may use the prepare_waveform fucntion.

        Parameters:
        -----------
        waveform: KeysightAWG.WaveformType
            one of the supported types of the waveform
        freq: float
            frequency of the applied waveform, i.e. 1000 or 1e3
        amp: float
            amplitude of the applied signal
        offset: float
            dc-offset added to the signal
        channel = 1: int
            channel which will be set to ON and used as output, 1 or 2

        """
        self._visainstrument.write("*OPC")
        self._visainstrument.write(":APPL%i:%s %f, %f, %f" % (channel, waveform.value,
                                                              freq, amp, offset))
        self._visainstrument.query("*OPC?")

    def prepare_waveform(self, waveform, freq, amp, offset, channel=1, blocking=True):
        """
        Set one of the pre-loaded waveforms as output, but do not output anything.

        Parameters:
        -----------
        waveform: KeysightAWG.WaveformType
            one of the supported types of the waveform
        freq: float
            frequency of the applied waveform, i.e. 1000 or 1e3
        amp: float
            amplitude of the applied signal
        offset: float
            dc-offset added to the signal
        channel = 1: int
            channel which will be set to ON and used as output, 1 or 2

        """
        self._visainstrument.write("*OPC")
        self._visainstrument.write(":FUNC{0} {1}; :FREQ{0} {2}; :VOLT{0} {3};\
                :VOLT{0}:OFFS {4}".format(channel, waveform.value, freq, amp, offset))
        self._visainstrument.write("*OPC?")

    def list_arbitrary_waveforms(self, channel=1):
        """
        Get all waveform names currently loaded in the permanent memory of the
        specified channel.

        Parameters:
        -----------
            channel=1: 1 or 2
                The channel for shich the waveforms will be listed

        """
        return self._visainstrument.query(":DATA%d:CAT?" % channel) \
            .replace('"', "").replace('\n', "").split(",")

    def select_arbitary_waveform(self, waveform_name, channel=1):
        """
        Select one of the seven built-in arbitrary waveforms,
        one of the four userdefined waveforms, or the waveform currently
        downloaded to volatile memory.
        This command does not output the selected arbitrary waveform.
        Use the apply_waveform command with WaveformType.arbitrary
        to output the selected waveform.

        Parameters:
        -----------
        waveform_name: string
            one of the waveform names returned by
            list_arbitrary_waveforms method
        channel=1: int
            channel for which the waveform will be selected, 1 or 2

        """
        if waveform_name in self.list_arbitrary_waveforms():
            self._visainstrument.write(":FUNC%i:USER %s" % (channel, waveform_name))
        else:
            raise ValueError(
                "No such waveform, check available waveforms with list_arbitrary_waveforms method")

    def get_arbitary_waveform(self, channel=1):
        """
        Get the name of the currently selected arbitrary waveform.

        Parameters:
        -----------
        channel=1: int
            channel for which the waveform name will aquired, 1 or 2

        """
        return self._visainstrument.query(":FUNC%i:USER?" % channel)

    def load_arbitrary_waveform_to_volatile_memory(self, waveform_array, channel=1):
        """
        Load an arbitrary waveform as an array into volatile memory.
        It then will be available in select_arbitrary_waveform method.

        The actual timescale and amplitude of the waveform will be defined by
        its frequency and amplitude as specified in apply_waveform method.

        Parameters:
        -----------
        waveform_array: ndarray
            an array of floats within the range [-1,1], of length
            131072 at maximum; if the range is too large the array
            will be normalized
        channel : 1 or 2
            channel index where the waveform will be stored

        """
        waveform_array = around(waveform_array * 8191).astype(int)
        self._visainstrument.write("*OPC")
        # self._visainstrument.write(":DATA%d VOLATILE, "%channel+array_string)
        self._visainstrument.write_binary_values(":DATA%d:DAC VOLATILE," % channel,
                                                 waveform_array, "h", True)
        self._visainstrument.query("*OPC?")

        """Output switches"""

    def set_output(self, channel, status):
        """
        Control the output on a channel

        Parameters:
        -----------
        channel: 1 or 2
            channel to switch output for
        status: int
            1 for ON and 0 for OFF

        """
        self._visainstrument.write("OUTP%i %i" % (channel, status))

    def do_set_outp1(self, status):
        """
        Turn first output channnel on and off.

        Parameters:
        -----------
        status: int
            1 for ON and 0 for OFF

        """
        self._visainstrument.write("OUTP1 %i" % status)

    def do_get_outp1(self):
        """Check if first output channnel is turned on"""
        return self._visainstrument.query("OUTP1?")

    def do_set_outp2(self, status):
        """
        Turn second output channnel on and off.

        Parameters:
        -----------
        status: int
            1 for ON and 0 for OFF

        """
        self._visainstrument.write("OUTP2 %i" % status)

    def do_get_outp2(self):
        """Check if second output channnel is turned on"""
        return self._visainstrument.query("OUTP2?")

    def do_set_outp1_compl(self, status):
        """
        Turn first output complement channnel on and off.

        Parameters:
        -----------
        status: int
            1 for ON and 0 for OFF

        """
        self._visainstrument.write("OUTP1:COMP %i" % status)

    def do_get_outp1_compl(self):
        """Check if first output complement channnel is turned on"""
        return self._visainstrument.query("OUTP1:COMP?")

    def do_set_outp2_compl(self, status):
        """
        Turn second output complement channnel on and off.

        Parameters:
        -----------
        status: int
            1 for ON and 0 for OFF

        """
        self._visainstrument.write("OUTP2:COMP %i" % status)

    def do_get_outp2_compl(self):
        """Check if second output complement channel is turned on"""
        return self._visainstrument.query("OUTP2:COMP?")

    def do_set_2nd_delay(self, delay):
        """
        Set a delay to the second puls.

        Parameters:
        -----------
        channel: int (1,2)
        delay: float in ns

        """
        return self._visainstrument.write("PULS:DEL2 %.1fNS" % delay)

    def do_get_2nd_delay(self):
        """
        Get a delay from 2nd channel
        """
        return self._visainstrument.query("PULS:DEL2?")

    def do_set_1st_delay(self, delay):
        """
        Set a delay to the first puls.

        Parameters:
        -----------
        channel: int (1,2)
        delay: float in ns

        """
        return self._visainstrument.write("PULS:DEL1 %.1fNS" % delay)

    def do_get_1st_delay(self):
        """
        Get a delay from 2nd channel
        """
        return self._visainstrument.query("PULS:DEL1?")

    def do_set_2nd_width(self, width):
        """
        Set a width to the second puls.

        Parameters:
        -----------
        channel: int (1,2)
        delay: float in ns

        """
        return self._visainstrument.write("FUNC2:PULS:WIDT %.1fNS" % width)

    def do_get_2nd_width(self):
        """
        Get a delay from 2nd channel
        """
        return self._visainstrument.query("FUNC2:PULS:WIDT?")

    def do_set_1st_width(self, width):
        """
        Set a width to the first puls.

        Parameters:
        -----------
        channel: int (1,2)
        delay: float in ns

        """
        return self._visainstrument.write("FUNC1:PULS:WIDT %.1fNS" % width)

    def do_get_1st_width(self):
        """
        Get a delay from 2nd channel
        """
        return self._visainstrument.query("FUNC1:PULS:WIDT?")

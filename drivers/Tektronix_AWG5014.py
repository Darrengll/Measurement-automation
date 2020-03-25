# Tektronix_AWG5014.py class, to perform the communication between the Wrapper and the device
# Pieter de Groot <pieterdegroot@gmail.com>, 2008
# Martijn Schaafsma <qtlab@mcschaafsma.nl>, 2008
# Guenevere Prawiroatmodjo <guen@vvtp.tudelft.nl>, 2009
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
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
import visa
import types
import logging
import numpy as np
import struct
from itertools import chain


class Tektronix_AWG5014(Instrument):
    """
    This is the python driver for the Tektronix AWG5014
    Arbitrary Waveform Generator

    Usage:
    Initialize with
    <name> = instruments.create('name', 'Tektronix_AWG5014', address='<GPIB address>',
        reset=<bool>, nop=<int>)

    think about:    clock, waveform length

    TODO:
    1) Get All
    2) Remove test_send??
    3) Add docstrings
    4) Add 4-channel compatibility
    """

    MAX_OUTPUT_VOLTAGE = 1

    def __init__(self, address, reset=False, clock=1e9, nop=1000, marker_enob=10):
        """
        Initializes the AWG520.

        Input:
            name (string)    : name of the instrument
            address (string) : GPIB address
            reset (bool)     : resets to default values, default=false
            nop (int)  : sets the number of datapoints

        Output:
            None
        """
        logging.debug(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, "Tektronix_AWG5014", tags=['physical'])

        self._address = address
        self._visainstrument = visa.ResourceManager() \
            .open_resource(self._address, read_termination="\n",
                           write_termination="\n")
        self._values = {}
        self._values['files'] = {}
        self._clock = clock
        self._nop = nop
        self._waveforms = [None] * 4
        self._amplitudes = [None] * 4
        self._markers = [None] * 8
        self._clear_all_waveforms()
        self._marker_enob = marker_enob
        self._marker_voltages = [{} for _ in range(8)]
        for i in range(1, 9):
            self.get_marker_voltages(i)
        for i in range(1, 5):
            self.set_amplitude(2, i)

    def output_continuous_wave(self, frequency, amplitude, phase, offset, waveform_resolution,
                               channel, asynchronous=False, trigger_sync_every=None):
        n_points = np.around(1 / frequency / waveform_resolution * 1e9) + 1 if frequency != 0 else 3
        waveform = amplitude * np.sin(2 * np.pi * np.linspace(0, 1, n_points) + phase) + offset
        self.output_arbitrary_waveform(waveform, frequency, channel, asynchronous=asynchronous)

    def output_arbitrary_waveform(self, waveform, repetition_rate,
                                  channel, asynchronous=True):
        """
        Prepare and output an arbitrary waveform repeated at some repetition_rate

        Parameters:
        -----------
        waveform: array
            ADC levels, in Volts
        repetition_rate: foat, Hz
            frequency at which the waveform will be repeated
        channel: int
            1..4 for DACs or -1..-8 for corresponding marker outputs, two for one
            DAC
        """

        def clear_unmatched_waveforms(channel):
            # Checks if other channels' waveforms have matching length,
            # otherwise clear them
            for idx, existing_waveform in enumerate(self._waveforms):
                if (idx != channel - 1):
                    if self._waveforms[idx] is not None:
                        if (len(existing_waveform) != len(waveform)):
                            self._waveforms[idx] = None
                            self._markers[idx * 2] = None
                            self._markers[idx * 2 + 1] = None
                            self._clear_waveform(idx + 1)

        waveform = np.array(waveform)

        if channel in [1, 2, 3, 4]:
            clear_unmatched_waveforms(channel)
            self._waveforms[channel - 1] = waveform

            norm = self._amplitudes[channel - 1] / 2
            self.set_waveform(waveform / norm, repetition_rate, channel)
            self.set_output(1, channel)
            self.run()
            if not asynchronous:
                self._visainstrument.query("*OPC?")

        else:  # we have a waveform for a marker
            marker_waveform = waveform
            marker_id = -channel
            host_channel = marker_id // 2 + marker_id % 2

            marker_low = np.min(marker_waveform)
            marker_high = np.max(marker_waveform)
            if marker_low == marker_high:
                if marker_low <= 0:
                    marker_waveform[:] = 0
                    self.set_marker_voltages(marker_id, marker_low, 1)
                else:
                    marker_waveform[:] = 1
                    self.set_marker_voltages(marker_id, -1, marker_high)
            else:
                # marker_mid = (marker_low+marker_high)/2
                # marker_copy = marker_waveform.copy()
                # marker_copy[marker_waveform>marker_mid] = 1
                # marker_copy[marker_waveform<=marker_mid] = 0
                # marker_waveform = marker_copy
                ptp = marker_high - marker_low
                marker_waveform = (marker_waveform - marker_low) / ptp
                marker_waveform = \
                    self._pwm_signal(marker_waveform, self._marker_enob)

                self.set_marker_voltages(marker_id, marker_low, marker_high)

            self._markers[marker_id - 1] = marker_waveform

            if not asynchronous:
                # Use existing or create a zero waveform for the host channel
                # and output both host channel and it's marker

                clear_unmatched_waveforms(host_channel)

                host_channel_waveform = self._waveforms[host_channel - 1]
                if host_channel_waveform is not None \
                        and len(host_channel_waveform) == len(marker_waveform):
                    host_channel_waveform = self._waveforms[host_channel - 1]
                else:
                    host_channel_waveform = np.zeros((len(marker_waveform)))
                norm = self._amplitudes[channel - 1] / 2
                self.set_waveform(host_channel_waveform /norm, \
                                  repetition_rate, host_channel)
                self.set_output(1, host_channel)
                self.run()
                self._visainstrument.query("*OPC?")

    def _clear_waveform(self, channel):
        self._visainstrument.write('WLIST:WAVeform:DELETE "CH%iWFM"' % channel)

    def _clear_all_waveforms(self):
        self._visainstrument.write('WLIST:WAVeform:DELETE ALL')

    def run(self):
        self._visainstrument.write('AWGC:RUN:IMM')

    def load_waveform(self, channel, filename, drive='C:', path='\\'):
        self._visainstrument.write('SOUR%s:FUNC:USER "%s/%s","%s"' % \
                                   (channel, path, filename, drive))

    def set_waveform(self, waveform, repetition_rate, channel):

        w = np.array(waveform, dtype=np.float)
        m1 = self._markers[(channel - 1) * 2]
        m2 = self._markers[(channel - 1) * 2 + 1]

        if m1 is None:
            m1 = np.zeros(len(w), dtype=np.int)
            self.set_marker_voltages((channel - 1) * 2 + 1, 0, 1)
        elif len(m1) != len(w):
            if len(np.unique(m1)) == 1:
                # match marker length and value and do not change its high/low
                m1 = np.ones(len(w), dtype=np.int) * m1[0]
            else:
                m1 = np.zeros(len(w), dtype=np.int)
                self.set_marker_voltages((channel - 1) * 2 + 1, 0, 1)

        if m2 is None:
            m2 = np.zeros(len(w), dtype=np.int)
            self.set_marker_voltages((channel - 1) * 2 + 2, 0, 1)
        elif len(m2) != len(w):
            if len(np.unique(m2)) == 1:
                # match marker length and value and do not change its high/low
                m2 = np.ones(len(w), dtype=np.int) * m2[0]
            else:
                m2 = np.zeros(len(w), dtype=np.int)
                self.set_marker_voltages((channel - 1) * 2 + 2, 0, 1)

        m1 = np.array(m1, dtype=np.int)
        m2 = np.array(m2, dtype=np.int)

        filename = 'test_ch{0}.wfm'.format(channel)

        self.send_waveform(w[:-1], m1[:-1], m2[:-1], filename,
                           repetition_rate * len(w[:-1]))
        self.load_waveform(channel, filename)
        # self.do_set_filename(filename, channel=channel)

    # Send waveform to the device
    def send_waveform(self, w, m1, m2, filename, clock):
        """
        Sends a complete waveform. All parameters need to be specified.
        See also: resend_waveform()

        Input:
            w (float[nop]) : waveform
            m1 (int[nop])  : marker1
            m2 (int[nop])  : marker2
            filename (string)    : filename
            clock (int)          : frequency (Hz)

        Output:
            None
        """
        logging.debug(__name__ + ' : Sending waveform %s to instrument' % filename)
        # Check for errors
        dim = len(w)

        if (not ((len(w) == len(m1)) and ((len(m1) == len(m2))))):
            raise ValueError(('Dimension mishmatch: markers (%d, %d)' \
                              + 'and waveform (%d)') % (len(m1), len(m2), len(w)))

        self._values['files'][filename] = {}
        self._values['files'][filename]['w'] = w
        self._values['files'][filename]['m1'] = m1
        self._values['files'][filename]['m2'] = m2
        self._values['files'][filename]['clock'] = clock
        self._values['files'][filename]['nop'] = len(w)

        m = m1 + np.multiply(m2, 2)
        ws = bytes()
        for i in range(0, len(w)):
            ws = ws + struct.pack('<fB', w[i], int(m[i]))

        s1 = str.encode('MMEM:DATA "%s",' % filename)
        s3 = str.encode('MAGIC 1000\n')
        s5 = ws
        s6 = str.encode('CLOCK %.10e' % clock)
        s4 = str.encode('#' + str(len(str(len(s5)))) + str(len(s5)))

        lenlen = str(len(str(len(s6) + len(s5) + len(s4) + len(s3))))
        s2 = str.encode('#' + lenlen + str(len(s6) + len(s5) + len(s4) + len(s3)))

        mes = s1 + s2 + s3 + s4 + s5 + s6

        self._visainstrument.write_raw(mes + b"\n")

    def do_get_waveform(self, channel):
        return self._waveforms[channel - 1]

    def do_set_filename(self, name, channel):
        """
        Specifies which file has to be set on which channel
        Make sure the file exists, and the nop and clock of the file
        matches the instrument settings.

        If file doesn't exist an error is raised, if the nop doesn't match
        the command is neglected

        Input:
            name (string) : filename of uploaded file
            channel (int) : 1 or 2, the number of the designated channel

        Output:
            None
        """
        logging.debug(__name__ + ' : Try to set %s on channel %s' % (name, channel))
        exists = False
        self._visainstrument.write('MMEM:IMP "%s", "%s", WFM' % (name, name))
        self._visainstrument.write('SOURCE%s:WAVEFORM "%s"' % (channel, name))
        if name in self._values['files']:
            exists = True
            logging.debug(__name__ + ' : File exists in local memory')
            self._values['recent_channel_%s' % channel] = self._values['files'][name]
            self._values['recent_channel_%s' % channel]['filename'] = name
        else:
            logging.debug(__name__ + ' : File does not exist in memory, \
            reading from instrument')
            lijst = self._visainstrument.ask('MMEM:CAT? "C:"')
            bool = False
            bestand = ""
            for i in range(len(lijst)):
                if (lijst[i] == '"'):
                    bool = True
                elif (lijst[i] == ','):
                    bool = False
                    if (bestand == name): exists = True
                    bestand = ""
                elif bool:
                    bestand = bestand + lijst[i]
        if exists:
            self._visainstrument.write('SOUR%s:FUNC:USER "%s","C:"' % (channel, name))
        else:
            logging.error(__name__ + ' : Invalid filename specified %s' % name)

    def set_output(self, state, channel):
        if (state == 1):
            self._visainstrument.write('OUTP%s:STAT ON' % channel)
        if (state == 0):
            self._visainstrument.write('OUTP%s:STAT OFF' % channel)

    def get_output(self, channel):
        return self._visainstrument.ask('OUTP%s:STAT?' % channel)

    def set_repetition_period(self, repetition_period):
        self.repetition_period = repetition_period
        self.set_nop(int(repetition_period * self.get_clock()))

    def get_repetition_period(self, repetition_period):
        return self.get_numpoint() / self.get_clock()

    def get_clock(self):
        return self._clock

    def set_clock(self, clock):
        self._clock = clock
        self._visainstrument.write('SOUR:FREQ %f' % clock)

    def get_amplitude(self, channel):
        self._amplitudes[channel - 1] = \
            float(self._visainstrument.ask('SOUR%s:VOLT:AMPL?' % channel))
        return self._amplitudes[channel - 1]

    def set_amplitude(self, amp, channel):
        self._amplitudes[channel - 1] = amp
        self._visainstrument.write('SOUR%s:VOLT:AMPL %.6f' % (channel, amp))

    def get_offset(self, channel):
        return float(self._visainstrument.ask('SOUR%s:VOLT:LEV:IMM:OFFS?' % channel))

    def set_offset(self, offset, channel):
        self._visainstrument.write('SOUR%s:VOLT:LEV:IMM:OFFS %.6f' % (channel, offset))

    def get_marker_voltages(self, marker_id):
        channel_id = marker_id // 2 + marker_id % 2
        marker_sub_id = (marker_id - 1) % 2 + 1

        command = 'SOUR%s:MARK%s:VOLT:' \
                  % (channel_id, marker_sub_id)
        low, high = float(self._visainstrument.query(command + "LOW?")), \
                    float(self._visainstrument.query(command + "HIGH?"))

        self._marker_voltages[marker_id - 1]["low"] = low
        self._marker_voltages[marker_id - 1]["high"] = high
        return low, high

    def set_marker_voltages(self, marker_id, low=0, high=1):

        channel_id = marker_id // 2 + marker_id % 2
        marker_sub_id = (marker_id - 1) % 2 + 1

        command_low = 'SOUR%s:MARK%s:VOLT:LOW %.3f' \
                      % (channel_id, marker_sub_id, low)
        command_high = 'SOUR%s:MARK%s:VOLT:HIGH %.3f' \
                       % (channel_id, marker_sub_id, high)

        if self._marker_voltages[marker_id - 1]["low"] != low:
            self._marker_voltages[marker_id - 1]["low"] = low
            self._visainstrument.write(command_low)

        if self._marker_voltages[marker_id - 1]["high"] != high:
            self._marker_voltages[marker_id - 1]["high"] = high
            self._visainstrument.write(command_high)

    def set_marker_enob(self, enob):
        self._marker_enob = enob

    #  Ask for string with filenames
    def get_filenames(self):
        logging.debug(__name__ + ' : Read filenames from instrument')
        return self._visainstrument.ask('MMEM:CAT? "MAIN"')

    def _pwm_signal(self, X, enob):

        def find_bit_num(signal_level, enob):
            for bit_num in range(0, enob + 1):
                if bit_num / enob <= signal_level < (bit_num + 1) / enob:
                    if avg_signal_level > (2 * bit_num + 1) / enob / 2:
                        bit_num += 1
                    pixel = [0] * (enob - bit_num) + [1] * (bit_num)
                    return pixel

        Y = []

        for i in range(0, len(X) - enob - 1, enob):
            avg_signal_level = np.mean(X[i:i + enob])
            Y.append(find_bit_num(avg_signal_level, enob))

        residual_enob = len(X) - len(Y) * enob
        if residual_enob > 0:
            avg_signal_level = np.mean(X[len(Y) * enob:len(Y) * enob + residual_enob - 1])
            Y.append(find_bit_num(avg_signal_level, residual_enob))

        return np.array(list(chain(*Y)))

    def resend_waveform(self, channel, w=[], m1=[], m2=[], clock=[]):
        """
        Resends the last sent waveform for the designated channel
        Overwrites only the parameters specified

        Input: (mandatory)
            channel (int) : 1, 2, 3 or 4, the number of the designated channel

        Input: (optional)
            w (float[nop]) : waveform
            m1 (int[nop])  : marker1
            m2 (int[nop])  : marker2
            clock (int) : frequency

        Output:
            None
        """
        filename = self._values['recent_channel_%s' % channel]['filename']
        logging.debug(__name__ + ' : Resending %s to channel %s' % (filename, channel))

        if (w == []):
            w = self._values['recent_channel_%s' % channel]['w']
        if (m1 == []):
            m1 = self._values['recent_channel_%s' % channel]['m1']
        if (m2 == []):
            m2 = self._values['recent_channel_%s' % channel]['m2']
        if (clock == []):
            clock = self._values['recent_channel_%s' % channel]['clock']

        if not ((len(w) == self._nop) and (len(m1) == self._nop) and (len(m2) == self._nop)):
            logging.error(__name__ + ' : one (or more) lengths of waveforms do not match with nop')

        self.send_waveform(w, m1, m2, filename, clock)
        self.do_set_filename(filename, channel)

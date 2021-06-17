# Keysight_DSO-X2014.py
# Gleb Fedorov <vdrhtc@gmail.com>

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
from enum import Enum

class Channel(Enum):
    ONE = "CHAN1"
    TWO = "CHAN2"
    THREE = "CHAN3"
    FOUR = "CHAN4"

class Keysight_DSOX2014(Instrument):
    """
    This is the python driver for the Keysight DSO-X 2014

    Usage:
    Initialise with
    <name> = instruments.create(address='<VISA address>')
    """
    def __init__(self, address):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.WARNING)

        Instrument.__init__(self, "", tags=['physical'])

        self._address = address
        rm = visa.ResourceManager()
        self._visainstrument = rm.open_resource(self._address)

        self.add_parameter('averages', type=int,
            flags=Instrument.FLAG_GETSET,
            minval=1, maxval=65536)

        self.add_parameter('nop', type=int,
            flags=Instrument.FLAG_GETSET,
            minval=2, maxval=65536)

        self._visainstrument.write(":WAV:FORMat WORD")
        self._vsteps = 65536 # the discretization for the WORD format
        self._visainstrument.write(":WAV:UNSIGNED 0")
        self._visainstrument.write(":TIMebase:REFerence LEFT")

        Channel.ALL = [Channel.ONE, Channel.TWO, Channel.THREE, Channel.FOUR]
        self.set_offset(0, *Channel.ALL)

    def digitize(self):
        """
        Initiates a measurement, blocks until measurement finished.
        """
        self._visainstrument.timeout = 100e3
        self._visainstrument.write(":DIGitize")
        self._visainstrument.query("*OPC?")
        self._visainstrument.timeout = 1000

    def get_data(self, *channels):
        """
        Get the data points from the specified channel.
        The channel must be displayed on the screen of the DSO for this function
        to work!

        Parameters:
        -----------
        channels: one or several Keysight_DSOX2014.Channel objects
            Channel enum value, i.e. Channel.ONE

        Returns:
        --------
        times: numpy.array
            Time axis for the waveforms, channel independent
        data: numpy.array or tuple of numpy.arrays
            Depending on the value of the channel argument may return data for one
            selected channel or data for all channels
        Example:
        >>> dso.set_averages(100)
        >>> dso.digitize() # acquire data for all channels, blocks until finished
        >>> channel1data, channel2data = dso.get_data(Channel.ONE, Channel.TWO)

        """
        data = []
        for channel in channels:
            preamble = self.get_preamble(channel)
            data.append(self.apply_preamble(self.get_data_raw(channel), preamble))
        return self.get_times(preamble), data if len(data)>1 else data[0]

    def get_data_raw(self, *channels):
        """
        To convert these values to the actual scale use the PREamble:
        Structure:
        FORMAT
        TYPE
        POINTS - number of data points transferred.
        COUNT - 1 and is always 1.
        XINCREMENT - time difference between data points.
        XORIGIN - always the first data point in memory.
        XREFERENCE - specifies the data point associated with x-origin.
        YINCREMENT - voltage diff between data points.
                    #Vsteps =
                    65536 (if FROMAT = WORD)
                    256 (if FORMAT = BYTE)
        YORIGIN - value is the voltage at center screen.
        YREFERENCE - specifies the data point where y-origin occurs
        """
        raw_data = []
        for channel in channels:
            raw_data.append(np.array(self._visainstrument.query_binary_values(":WAV:SOURce "+channel.value+"; :WAV:DATA?", "h", True)))
        return raw_data if len(raw_data)>1 else raw_data[0]

    def get_preamble(self, *channels):
        """
        Returns the preamble value for the specified channel(s)

        Parameters:
        -----------
        channels: one or several Keysight_DSOX2014.Channel objects
            Channel enum value, i.e. Channel.ONE

        Returns:
        preambles: list of dictionary objects
            Preamble for the channel(s)
        """
        preambles = []
        for channel in channels:
            channel_preamble = {}
            params = self._visainstrument.query(":WAV:SOURce "+channel.value+";:WAV:PREamble?").split(",")
            channel_preamble["nop"] = int(params[2])
            channel_preamble["xincrement"] = float(params[4])
            channel_preamble["xorigin"] = float(params[5])
            channel_preamble["xreferecnce"] = int(params[6])
            channel_preamble["yincrement"] = float(params[7])
            channel_preamble["yorigin"] = float(params[8])
            channel_preamble["yreference"] = int(params[9])
            preambles.append(channel_preamble)
        return preambles if len(preambles)>1 else preambles[0]

    def apply_preamble(self, raw_data, preamble):
        """
        Get scaled Y data from the raw data and the preamble
        """
        return raw_data*preamble["yincrement"]

        #### Getters and setters

    def get_times(self, preamble=None):
        """
        Get the time points of the X axis
        """
        if preamble is None:
            preamble = self.get_preamble(Channel.ONE)

        total_time = preamble["nop"]*preamble["xincrement"]
        return np.linspace(0, total_time, preamble["nop"])

    def set_time_range(self, time_range):
        """
        Sets the full-scale horizontal time for the oscilloscope.
        The range is 10 times the bias time-per-division setting

        Parameters:
        -----------
        time_range: float
            The new range for the time axis, in seconds
        """
        self._visainstrument.write(":TIMebase:RANGe %.2e"%time_range)

    def get_time_range(self):
        """
        Returns the full-scale horizontal time in seconds for the oscilloscope.
        The range is 10 times the bias time-per-division setting
        """
        return float(self._visainstrument.query(":TIMebase:RANGe?"))

    def get_time_offset(self):
        """
        Returns the time offset from the trigger event in seconds
        """
        return float(self._visainstrument.query(":TIMebase:POSition?"))

    def set_time_offset(self, time_offset):
        """
        Set the time offset from the trigger event in seconds
        """
        self._visainstrument.write(":TIMebase:POSition %.2e"%time_offset)
        return self.get_time_offset()

    def do_set_nop(self, nop):
        """
        Set the number of data points to be acquired from DSO

        Parameters:
        ----------
        nop: int
            Channel number of points, from 100 to 1000 in NORMal mode
        """
        self._visainstrument.write(":WAV:POINts "+str(nop))

    def do_get_nop(self):
        """
        Get the number of points for the specified channels
        """
        return int(self._visainstrument.query(":WAV:POINts?"))

    def do_set_averages(self, averages):
        """
        Set the number of averages to perform.

        Paramaters:
        -----------
        averages: int, from 1 to 65536
                number of averages that will be performed
        """
        if averages == 1:
            self._visainstrument.write(":ACQuire:TYPE NORMal")
        else:
            self._visainstrument.write(":ACQuire:TYPE AVERage; :ACQuire:COUNt "+str(averages))

    def do_get_averages(self):
        """
        Get the number of averages that the oscilloscope is going to perform.
        """
        if self._visainstrument.query(":ACQuire:TYPE?") == "NORMAL":
            return 1
        else:
            return self._visainstrument.query(":ACQuire:COUNt?")

    def set_offset(self, offset, *channels):
        """
        Set offset in volts for the specified channel(s)

        Parameters:
        -----------
        offset: float
            offset in volts
        channels: Keysight_DSOX2014.Channel objects
            Channel enum value, i.e. Channel.ONE
        """
        for channel in channels:
            self._visainstrument.write("%s:OFFSet %f"%(channel.value, offset))

    def get_offset(self, *channels):
        """
        Get offset of the specified channel(s)

        Paramaters:
        -----------
        channels: Keysight_DSOX2014.Channel objects
            Channel enum value, i.e. Channel.ONE
        """
        offsets = []
        for channel in channels:
            offsets.append(self._visainstrument.query("%s:OFFSet?"%channel.value))
        return offsets if len(offsets)>1 else offsets[0]

    def get_error(self):
        return self._visainstrument.query(":SYST:ERR?")

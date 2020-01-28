# KeysightAWG.py
# Gleb Fedorov <vdrhc@gmail.com>
# Alexander Korenkov <soyer94_44@mail.ru>
# Alexey Dmitriev <dmitrmipt@gmail.com>
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
import visa
import types
import time
import logging
import numpy
import sys
import math
import serial

import matplotlib.pyplot as plt 
from ctypes import (Union, Array, c_uint8, c_float, cdll, CDLL)
from enum import Enum
import crcmod
crc8_func = crcmod.mkCrcFun(0x131, initCrc=0x00, xorOut=0x00)

def format_e(n):
    a = '%e' % n
    return a.split('e')[0].rstrip('0').rstrip('.') + 'e' + a.split('e')[1]


	
class NCS(Instrument):
    '''
    The driver for the Yokogawa_GS210. Default operation regime is CURRENT source.

        CURRENT SOURCE
        Source Range    Range Generated     Resolution      Max. Load Voltage
        1 mA            ±1.20000 mA         10 nA           ±30 V
        10 mA           ±12.0000 mA         100 nA          ±30 V
        100 mA          ±120.000 mA         1 μA            ±30 V
        200 mA          ±200.000 mA         1 μA            ±30 V

        VOLTAGE SOURCE
        Source Range    Range Generated     Resolution      Max. Load Current
        10 mV           ±12.0000 mV         100 nV          --------
        100 mV          ±120.000 mV         1 μV            --------
        1 V             ±1.20000 V          10 μV           ±200 mA
        10 V            ±12.0000 V          100 μV          ±200 mA
        30 V            ±32.000 V           1 mV            ±200 mA
    '''

    current_ranges_supported = [.001, .01, .1, .2]           #possible current ranges supported by current source
    voltage_ranges_supported = [.01, .1, 1, 10, 30]



    def __init__(self, address, volt_compliance = 1, current_compliance = .001):
        '''Create a default NCS object as a current source'''
        Instrument.__init__(self, 'NCS', tags=['physical'])
        self._address = address
        #rm = visa.ResourceManager()
        #self._visainstrument = rm.open_resource(self._address)
        self.ser = serial.Serial(port = 'COM3', baudrate=115200, parity=serial.PARITY_NONE,stopbits=serial.STOPBITS_ONE,bytesize=serial.EIGHTBITS,timeout=None)
        current_range = (-50e-3, 50e-3)
        voltage_range = (-32, 32)

        self._mincurrent = -10e-3
        self._maxcurrent =  10e-3

        self._minvoltage = -1e-6
        self._maxvoltage =  1e-6

        self.add_parameter('current', flags = Instrument.FLAG_GETSET,
        units = 'A', type = float, minval = current_range[0], maxval = current_range[1])

        self.add_parameter('current_compliance', flags = Instrument.FLAG_GETSET,
        units = 'V', type = float, minval = current_range[0], maxval = current_range[1])

        self.add_parameter('voltage', flags = Instrument.FLAG_GETSET,
        units = 'V', type = float, minval = voltage_range[0], maxval = voltage_range[1])

        self.add_parameter('voltage_compliance', flags = Instrument.FLAG_GETSET,
        units = 'V', type = float, minval = voltage_range[0], maxval = voltage_range[1])

        self.add_parameter('status',
        flags = Instrument.FLAG_GETSET, units = '', type = int)

        self.add_parameter('range',
        flags = Instrument.FLAG_GETSET, units = '', type = float)

#        self.add_function("get_id")

        self.add_function("clear")
        self.add_function("set_src_mode_volt")


        temp_cmd_output = [0] * 9
        temp_cmd_output[0] = 0xC0 | 1  # 1 - current source address, as default 1
        temp_cmd_output[1] = 0x70 | 1
        temp_cmd_output[2] = 0x70  # ch1
        temp_cmd_output[3] = 0x70  # ch2
        temp_cmd_output[4] = 0x70  # ch3
        temp_cmd_output[5] = 0x70  # ch4
        temp_cmd_output[6] = 0x70  # ch5
        temp_cmd_output[7] = 0  # reserved
        temp_cmd_output[8] = 0  # reserved
        temp_cmd_output[1 + 1] = 0x70 | 1
        temp_b = bytearray(temp_cmd_output)
        temp_crc = crc8_func(temp_b).to_bytes(1, byteorder='big')
        temp_b += temp_crc
        self.ser.write(temp_b)


        temp_cmd_output = [0] * 9
        temp_cmd_output[0] = 0xC0 | 1  # 1 - current source address, as default 1
        temp_cmd_output[1] = 0xA0 | 1
        temp_cmd_output[2] = 0xA0  # ch1
        temp_cmd_output[3] = 0xA0  # ch2
        temp_cmd_output[4] = 0xA0  # ch3
        temp_cmd_output[5] = 0xA0  # ch4
        temp_cmd_output[6] = 0xA0  # ch5
        temp_cmd_output[7] = 0  # reserved
        temp_cmd_output[8] = 0  # reserved
        temp_cmd_output[1 + 1] = 0xA0 | 1
        temp_z = bytearray(temp_cmd_output)
        temp_crc = crc8_func(temp_z).to_bytes(1, byteorder='big')
        temp_z += temp_crc
        self.ser.write(temp_z)


#        self._visainstrument.write(":SOUR:FUNC CURR")

    #        self.set_voltage_compliance(volt_compliance)
        self.set_current(0)
 #       self.set_status(1)

    class uint8_array(Array):
        _type_ = c_uint8
        _length_ = 4
    class f_type(Union):
        _fields_ = ("float", c_float), ("u8", uint8_array)

    def float_to4bytes(float_data):
        temp_data = f_type()
        temp_data.float = float_data
        return temp_data.u8[:]
    def byteArray_toFloat(data_array, offset = 0):
        temp_data = f_type()
        temp_data.u8[:] = (data_array[0+offset],data_array[1+offset],data_array[2+offset],data_array[3+offset])
        return temp_data.float


    def get_id(self):
        '''Get basic info on device'''
        return "NCS_1"
    #range = 0 - '1uA'... range = 5 - '50mA'
    def ncs_output_enable(self, state=0, channel=1):
        temp_cmd_output = [0] * 9
        temp_cmd_output[0] = 0xC0 | 1  # 1 - current source address, as default 1
        temp_cmd_output[1] = 0x70 | channel
        temp_cmd_output[2] = 0x70  # ch1
        temp_cmd_output[3] = 0x70  # ch2
        temp_cmd_output[4] = 0x70  # ch3
        temp_cmd_output[5] = 0x70  # ch4
        temp_cmd_output[6] = 0x70  # ch5
        temp_cmd_output[7] = 0  # reserved
        temp_cmd_output[8] = 0  # reserved
        temp_cmd_output[channel + 1] = 0x70 | state
        temp_b = bytearray(temp_cmd_output)
        temp_crc = crc8_func(temp_b).to_bytes(1, byteorder='big')
        temp_b += temp_crc
        return temp_b


    def ncs_channel_enable(self, state=0, channel=1):
        temp_cmd_output = [0] * 9
        temp_cmd_output[0] = 0xC0 | 1  # 1 - current source address, as default 1
        temp_cmd_output[1] = 0xA0 | channel
        temp_cmd_output[2] = 0xA0  # ch1
        temp_cmd_output[3] = 0xA0  # ch2
        temp_cmd_output[4] = 0xA0  # ch3
        temp_cmd_output[5] = 0xA0  # ch4
        temp_cmd_output[6] = 0xA0  # ch5
        temp_cmd_output[7] = 0  # reserved
        temp_cmd_output[8] = 0  # reserved
        temp_cmd_output[channel + 2] = 0xA0 | state
        temp_v = bytearray(temp_cmd_output)
        temp_crc = crc8_func(temp_v).to_bytes(1, byteorder='big')
        temp_v += temp_crc
        return temp_v

    def ncs_set_range(self, range = 0, channel = 1):
        temp_cmd_output = [0]*9
        temp_cmd_output[0] = 0xC0 | 1 #1 - current source address, as default 1
        temp_cmd_output[1] = 0x60 | channel
        temp_cmd_output[2] = 0x60 #ch1
        temp_cmd_output[3] = 0x60 #ch2
        temp_cmd_output[4] = 0x60 #ch3
        temp_cmd_output[5] = 0x60 #ch4
        temp_cmd_output[6] = 0x60 #ch5
        temp_cmd_output[7] = 0 #reserved
        temp_cmd_output[8] = 0 #reserved
        temp_cmd_output[channel + 2] = 0x60 | range
        temp_b = bytearray(temp_cmd_output)
        temp_crc = crc8_func(temp_b).to_bytes(1, byteorder = 'big')
        temp_b += temp_crc

        return self.temp_b

    def ncs_set_output_mode(self, mode = 0, channel = 1):
        temp_cmd_output = [0]*9
        temp_cmd_output[0] = 0xC0 | 1 #1 - current source address, as default 1
        temp_cmd_output[1] = 0x30 | channel
        temp_cmd_output[2] = 0x30 #ch1
        temp_cmd_output[3] = 0x30 #ch2
        temp_cmd_output[4] = 0x30 #ch3
        temp_cmd_output[5] = 0x30 #ch4
        temp_cmd_output[6] = 0x30 #ch5
        temp_cmd_output[7] = 0 #reserved
        temp_cmd_output[8] = 0 #reserved
        temp_cmd_output[channel + 1] = 0x30 | mode
        temp_b = bytearray(temp_cmd_output)
        temp_crc = crc8_func(temp_b).to_bytes(1, byteorder = 'big')
        temp_b += temp_crc
        return(temp_b)

    def ncs_set_current(self, current = 0.0, channel = 1):
        temp_cmd_output = [0]*9
        temp_cmd_output[0] = 0xC0 | 1 #1 - current source address, as default 1
        temp_cmd_output[1] = 0xD0 | channel
        temp_current = float_to4bytes(current)
        temp_cmd_output[2] = temp_current[0]
        temp_cmd_output[3] = temp_current[1]
        temp_cmd_output[4] = temp_current[2]
        temp_cmd_output[5] = temp_current[3]
        temp_cmd_output[6] = 0
        temp_cmd_output[7] = 0
        temp_cmd_output[8] = 0
        temp_b = [0]
        temp_b = bytearray(temp_cmd_output)
        temp_crc = crc8_func(temp_b).to_bytes(1, byteorder = 'big')
        temp_b += temp_crc
        return(temp_b)
    def do_set_current(self, current):
        '''Set current'''
        if (self._mincurrent <= current <= self._maxcurrent):

            self.ser.write(ncs_set_current(current, 1))
            time.sleep(0.1)
            sys.stdout.flush()
        else:
            print("Error: current limits,",(self._mincurrent, self._maxcurrent)," exceeded.")

			
				
    def do_get_current(self):
        '''Get current'''
        return float(0)

    def do_set_voltage(self, voltage):
        '''Set voltage'''
        if (self._visainstrument.ask(":SOUR:FUNC?") == "CURR\n"):
            print("Tough luck, mode is current source, cannot get voltage.")
            return False
        else:
            if (self._minvoltage < voltage < self._maxvoltage):
                self._visainstrument.write("SOUR:LEVEL %e"%voltage)
                print("Voltage set",format_e(voltage), "V")
            else:
                print("Error: voltage limits exceeded.")

    def do_get_voltage(self):
        '''Get voltage'''
        if (self._visainstrument.ask(":SOUR:FUNC?") == "CURR\n"):
            print("Tough luck, mode is current source, cannot get voltage.")
            return False
        return float(self._visainstrument.ask("SOUR:LEVEL?"))

    def do_set_status(self, status):
        '''
        Turn output on and off

        Parameters:
        -----------
            status: 0 or 1
                0 for off, 1 for on
        '''
        self._visainstrument.write("OUTP "+("ON" if status==1 else "OFF"))

    def do_get_status(self):
        '''Check if output is turned on'''
        return "Off" #self._visainstrument.query("OUTP?")

    def do_get_voltage_compliance(self):
        '''Get compliance voltage'''
        return 5.0 #float(self._visainstrument.ask("SOUR:PROT:VOLT?"))

    def do_set_voltage_compliance(self, compliance):
        '''Set compliance voltage'''
        #if (self._visainstrument.ask(":SOUR:FUNC?") == "VOLT\n"):
        #    print("Tough luck, mode is voltage source, cannot set voltage compliance.")
        #    return False
        #self._visainstrument.write("SOUR:PROT:VOLT %e"%compliance)
        pass
		
    def do_get_current_compliance(self):
        '''Get compliance voltage'''
        return 5.0e-3 #float(self._visainstrument.ask("SOUR:PROT:CURR?"))

    def do_set_current_compliance(self, compliance):
        '''Set compliance current'''
        #if (self._visainstrument.ask(":SOUR:FUNC?") == "CURR\n"):
        #    print("Tough luck, mode is current source, cannot set current compliance.")
        #    return False
        #self._visainstrument.write("SOUR:PROT:CURR %e"%compliance)
        pass
    def do_get_range(self):
        '''Get current range in A'''
        currange = .001 #self._visainstrument.ask("SOUR:RANG?")[:-1]
        return float(currange)

    def do_set_range(self, maxval):
        '''Set current range in A'''
        #if (self._visainstrument.ask(":SOUR:FUNC?") == "CURR\n"):
        if not (maxval in self.current_ranges_supported):
            print("Given current range is invalid. Please enter valid current range in !!!Amperes!!!\nValid ranges are (in A): {0}".format(self.current_ranges_supported))
            return False
        else:
            if (maxval < 1e-6):
                self.ser.write(ncs_set_range(0, channel = 1))
            if (maxval > 1e-6 and maxval < 10e-6):
                self.ser.write(ncs_set_range(1, channel = 1))
            if (maxval > 10e-6 and maxval < 100e-6):
                self.ser.write(ncs_set_range(2, channel = 1))
            if (maxval > 100e-6 and maxval < 1e-3):
                self.ser.write(ncs_set_range(3, channel = 1))
            if (maxval > 1e-3 and maxval < 10e-3):
                self.ser.write(ncs_set_range(4, channel = 1))
            if (maxval > 10e-3 and maxval < 50e-3):
                self.ser.write(ncs_set_range(5, channel = 1))
        #if(self._visainstrument.ask(":SOUR:FUNC?") == "VOLT\n"):
            #if not (maxval in self.voltage_ranges_supported):
               # print("Given voltage range is invalid. Please enter valid voltage range in !!!Volts!!!\nValid ranges are (in A): {0}".format(self.voltage_ranges_supported))
               # return False
            #else:
               # self._visainstrument.write("SOUR:RANG %e"%maxval)

    def set_appropriate_range(self, maxcurrent=1E-3, mincurrent=-1E-3):
        '''Detect which range includes limits and set it'''

        required_current = max(abs(maxcurrent), abs(mincurrent))
        for current_range in self.current_ranges_supported:
            if current_range >= required_current:
                self.set_range(current_range)
                return True
            if(current_range == self.current_ranges_supported[-1]):
                print("Current is too big, can't handle it!")
                return False


    def set_src_mode_volt(self, current_compliance = .001):
        '''
        Changes mode from current to voltage source, compliance current is given as an argument

        Returns:
            True if the mode was changed, False otherwise
        '''
        #if (self._visainstrument.ask(":SOUR:FUNC?") == "VOLT\n"):
        #   return False
        #else:
            #self._visainstrument.write(":SOUR:FUNC VOLT")
            #self.set_current_compliance(current_compliance)
        return True

    def set_src_mode_curr(self, voltage_compliance = 1):
        '''
        Changes mode from voltage to current source, compliance voltage is given as an argument

        Returns:
            True if the mode was changed, False otherwise
        '''
        #if (self._visainstrument.ask(":SOUR:FUNC?") == "CURR\n"):
        #    return False
        #else:
        #    self._visainstrument.write(":SOUR:FUNC CURR")
        #    self.set_voltage_compliance(voltage_compliance)
        return True

    def set_current_limits(self, mincurrent = -1E-3, maxcurrent = 1E-3):
        ''' Sets a limits within the range if needed for safe sweeping'''
        if (True == True):#(self._visainstrument.ask(":SOUR:FUNC?") == "CURR\n"):
            if mincurrent >= -1.2*self.get_range():
                   self._mincurrent = mincurrent
            else:
                print("Too low mincurrent asked to set.")
            if maxcurrent <= 1.2*self.get_range():
                self._maxcurrent = maxcurrent
            else:
                print("Too high maxcurrent asked to set.")
        #else:
        #    print("Go in current mode first.")

    def set_voltage_limits(self, minvoltage = -1E-3, maxvoltage = 1E-3):
        ''' Sets a voltage limits within the range if needed for safe sweeping'''
        if (True == True): #(self._visainstrument.ask(":SOUR:FUNC?") == "VOLT\n"):
            if minvoltage >= -1*self.get_range():
                   self._minvoltage = minvoltage
            else:
                print("Too low minvoltage asked to set.")
            if maxvoltage <= self.get_range():
                self._maxvoltage = maxvoltage
            else:
                print("Too high maxvoltage asked to set.")
        else:
            print("Go in voltage mode first.")



    def clear(self):
        """
        Clear the event register, extended event register, and error queue.
        """
        #self._visainstrument.write("*CLS")
        pass



    # TODO:
    #       Дописать переключение режимов CURRENT/VOLTAGE
    #       Дописать недостающие команды и параметры в кострукторе для напряжения ( add_parameter('voltage', ...) и операции do_set_voltage и иже с ним)

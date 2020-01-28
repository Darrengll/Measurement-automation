
from drivers.instrument import Instrument
import visa
import types
import time
import logging
import numpy
import sys
import serial
from ctypes import (Union, Array, c_uint8, c_float, cdll, CDLL)
from enum import Enum
import crcmod



class uint8_array(Array):
    _type_ = c_uint8
    _length_ = 4
class f_type(Union):
    _fields_ = ("float", c_float), ("char", uint8_array)


def format_e(n):
    a = '%e' % n
    return a.split('e')[0].rstrip('0').rstrip('.') + 'e' + a.split('e')[1]

class NCS513(Instrument):
    """
    The driver for the NCS513. 

        CURRENT SOURCE
        Source Range    Range Generated     Resolution      Max. Load Voltage
        1 uA            ±1.00000 uA         			10 pA           ±24 V
        10 uA          ±10.0000 uA         			100 pA         ±24 V
        100 uA        ±100.000 mA         			1 nA            ±24 V
        1 mA           ±1.00000 mA         			10 nA           ±24 V
		10mA		  ±10.0000 mA					100nA		   ±24V
		50mA         ±50.0000mA				 	1uA			   ±24V

    """

    current_ranges_supported = [.000001, .000001, .0001, .001, 0.01, 0.05]           #possible current ranges supported by current source
    voltage_ranges_supported = [.01, .1, 1, 10, 30]



    def __init__(self, com_name, volt_compliance = None, current_compliance = .001):
        """Create a default Yokogawa_GS210 object as a current source"""
        Instrument.__init__(self, 'NCS513', tags=['physical'])
        self._address = com_name
        #rm = visa.ResourceManager()
        #self._visainstrument = rm.open_resource(self._address)
        self.com_name = com_name
        self.session_open = 0
        self.serial_device = serial.Serial()
        self.serial_device.port = com_name
        self.serial_device.baudrate = 115200
        self.serial_device.timeout = 5	
		
        self.serial_device.open()
		
        self.current_value = 0
        self.active_channel = 1
		
        current_range = (-50e-3, 50e-3)
        voltage_range = (-24, 24)

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

        self.add_function("get_id")

        #self.add_function("clear")
        #self.add_function("set_src_mode_volt")

       # self._visainstrument.write(":SOUR:FUNC CURR")

        #self.set_voltage_compliance(volt_compliance)
        #self.set_current(0)
        #self.set_status(1)

        
    def close(self):
        self.serial_device.close()
        self.session_open = 0
    def open(self):
        if self.session_open == 0:
            try:
                self.serial_device.open()
                self.session_open = 1
            except serial.SerialException:
                print("com port not open")
        else:
            print("device already initialize")
    def write(self, data_bytearray):
        self.serial_device.write(data_bytearray)
    def read(self, bytes_num):
        return (self.serial_device.read(bytes_num))
		
    def crc8(self, data_in):
        """
        this function evaluate crc
        poly: 0x0131
        
        """
        crc8_func = crcmod.mkCrcFun(0x131, initCrc=0x00, xorOut=0x00)
        return (crc8_func(bytearray(data_in)))
    
    def formated_data(self, data_in):
        temp_data = data_in 
        temp_data.append(self.crc8(data_in))
        return (bytearray(temp_data))
    def enable_channel(self, ch_num = 1):
        temp_cmd = [0]*9
        temp_cmd[0] = 0xC1
        temp_cmd[1] = 0xA0 | ch_num
        temp_cmd[2] = 0xA0
        temp_cmd[3] = 0xA0
        temp_cmd[4] = 0xA0
        temp_cmd[5] = 0xA0
        temp_cmd[6] = 0xA0
        temp_cmd[ch_num+1] = 0xA1
        temp_cmd[7] = 0xA0
        temp_cmd[8] = 0xA0
        ftemp = []
        ftemp = self.formated_data(temp_cmd)
        self.write(ftemp)
        return (self.read(10))
    
    def disable_channel(self, ch_num = 1):
        temp_cmd = [0]*9
        temp_cmd[0] = 0xC1
        temp_cmd[1] = 0xA0 | ch_num
        temp_cmd[2] = 0xA0
        temp_cmd[3] = 0xA0
        temp_cmd[4] = 0xA0
        temp_cmd[5] = 0xA0
        temp_cmd[6] = 0xA0
        temp_cmd[ch_num+1] = 0xA0
        temp_cmd[7] = 0xA0
        temp_cmd[8] = 0xA0
        ftemp = []
        ftemp = self.formated_data(temp_cmd)
        self.write(ftemp)
        return(self.read(10))  
    def set_channel_state(self, ch_num, state):
        """control supply delivery of the analog channel"""
        temp_cmd = [0]*9
        temp_cmd[0] = 0xC1
        temp_cmd[1] = 0xA0 | ch_num
        temp_cmd[2] = 0xA0
        temp_cmd[3] = 0xA0
        temp_cmd[4] = 0xA0
        temp_cmd[5] = 0xA0
        temp_cmd[6] = 0xA0
        if state == 0:
            temp_cmd[ch_num+1] = 0xA0
        if state == 1:    
            temp_cmd[ch_num+1] = 0xA1
        temp_cmd[7] = 0xA0
        temp_cmd[8] = 0xA0
        ftemp = []
        ftemp = self.formated_data(temp_cmd)
        self.write(ftemp)
        return (self.read(10))        
    def set_range_backend(self, ch_num = 1, range_value = '1uA'):
        temp_cmd = [0]*9
        temp_cmd[0] = 0xC1
        temp_cmd[1] = 0x60 | ch_num
        temp_cmd[2] = 0x60
        temp_cmd[3] = 0x60
        temp_cmd[4] = 0x60
        temp_cmd[5] = 0x60
        temp_cmd[6] = 0x60
        if range_value == '1uA':
            temp_cmd[ch_num+1] = 0x60
        if range_value == '10uA':
            temp_cmd[ch_num+1] = 0x61
        if range_value == '100uA':
            temp_cmd[ch_num+1] = 0x62
        if range_value == '1mA':
            temp_cmd[ch_num+1] = 0x63
        if range_value == '10mA':
            temp_cmd[ch_num+1] = 0x64
        if range_value == '100mA':
            temp_cmd[ch_num+1] = 0x65
        temp_cmd[7] = 0x60
        temp_cmd[8] = 0x60
        ftemp = []
        ftemp = self.formated_data(temp_cmd)
        self.write(ftemp)
        return(self.read(10))
    def set_out_mode(self, ch_num = 1, out_mode = 'INNER_LO_OUTER_LO'):
        """
        set relay configuration
        available regimes:
        INNER_LO_OUTER_LO | INNER_LO_OUTER_GND | INNER_LOGND_OUTER_LOGND | 
        INNER_GUARD_OUTER_LO | INNER_GUARD_OUTER_LOGND | INNER_LO_OUTER_NC |
        INNER_NC_OUTER_LO
        """
        temp_cmd = [0]*9
        temp_cmd[0] = 0xC1
        temp_cmd[1] = 0x30 | ch_num
        temp_cmd[2] = 0x30
        temp_cmd[3] = 0x30
        temp_cmd[4] = 0x30
        temp_cmd[5] = 0x30
        temp_cmd[6] = 0x30
        if out_mode == 'INNER_LO_OUTER_LO':
            temp_cmd[ch_num+1] = 0x31
        if out_mode == 'INNER_LO_OUTER_GND':
            temp_cmd[ch_num+1] = 0x32
        if out_mode == 'INNER_LOGND_OUTER_LOGND':
            temp_cmd[ch_num+1] = 0x33
        if out_mode == 'INNER_GUARD_OUTER_LO':
            temp_cmd[ch_num+1] = 0x34
        if out_mode == 'INNER_GUARD_OUTER_LOGND':
            temp_cmd[ch_num+1] = 0x35
        if out_mode == 'INNER_LO_OUTER_NC':
            temp_cmd[ch_num+1] = 0x36
        if out_mode == 'INNER_NC_OUTER_LO':
            temp_cmd[ch_num+1] = 0x37
        temp_cmd[7] = 0x30
        temp_cmd[8] = 0x30
        ftemp = []
        ftemp = self.formated_data(temp_cmd)
        self.write(ftemp)
        return(self.read(10))
    def set_voltage_compliance(self, ch_num = 1, voltage_compliance_value = 10):
        """ no hardware support"""
        pass
    def get_voltage_compliance(self):
        temp_cmd = [0xC1, 0xB0, 0xBF, 0xBF, 0xBF, 0xBF, 0xBF, 0xBF, 0xBF]
        self.write(self.formated_data(temp_cmd))
        temp_list = []
        temp_list = list(self.read(10))
        return([temp_list[2], temp_list[3], temp_list[4], temp_list[5], temp_list[6]])
    def set_output_state(self, ch_num = 1, state = 0):
        temp_cmd = [0]*9
        temp_cmd[0] = 0xC1
        temp_cmd[1] = 0x70 | ch_num
        temp_cmd[2] = 0x70
        temp_cmd[3] = 0x70
        temp_cmd[4] = 0x70
        temp_cmd[5] = 0x70
        temp_cmd[6] = 0x70
        if state == 0:
            temp_cmd[ch_num+1] = 0x70
        if state == 1:
            temp_cmd[ch_num+1] = 0x71
        temp_cmd[7] = 0x70
        temp_cmd[8] = 0x70
        ftemp = []
        ftemp = self.formated_data(temp_cmd)
        self.write(ftemp)
        return (self.read(10))
    def get_output_state(self, ch_num):
        """no realized, need fix mcu programm to implement """
        pass
    def float_to4bytes(self, float_data):
        temp_data = f_type()
        temp_data.float = float_data
        return temp_data.char[:]
    def byteArray_toFloat(self, data_array, offset = 0):
        temp_data = f_type()
        temp_data.char[:] = (data_array[0+offset],data_array[1+offset],data_array[2+offset],data_array[3+offset])
        return temp_data.float
        
    def set_current_backend(self, ch_num = 1, current_value = 0.0):
        temp_float_as_4bytes = []*4
        temp_float_as_4bytes = self.float_to4bytes(current_value)
        temp_cmd = [0]*9
        temp_cmd[0] = 0xC1
        temp_cmd[1] = 0xD0 | ch_num
        temp_cmd[2] = temp_float_as_4bytes[0]
        temp_cmd[3] = temp_float_as_4bytes[1]
        temp_cmd[4] = temp_float_as_4bytes[2]
        temp_cmd[5] = temp_float_as_4bytes[3]
        temp_cmd[6] = 0xD0
        temp_cmd[7] = 0xD0
        temp_cmd[8] = 0xD0
        ftemp = []
        ftemp = self.formated_data(temp_cmd)
        self.write(ftemp)
        return (self.read(10))
    def set_active_channel(self, ch_num):
        self.active_channel = ch_num
        
        
        
    def get_id(self):
        """Get basic info on device"""
        return 'NCS513_v2'#self._visainstrument.ask("*IDN?")

    def do_set_current(self, current):
        """Set current"""
        self.current_value = current
        temp_float_as_4bytes = []*4
        temp_float_as_4bytes = self.float_to4bytes(current)
        temp_cmd = [0]*9
        temp_cmd[0] = 0xC1
        temp_cmd[1] = 0xD0 | self.active_channel
        temp_cmd[2] = temp_float_as_4bytes[0]
        temp_cmd[3] = temp_float_as_4bytes[1]
        temp_cmd[4] = temp_float_as_4bytes[2]
        temp_cmd[5] = temp_float_as_4bytes[3]
        temp_cmd[6] = 0xD0
        temp_cmd[7] = 0xD0
        temp_cmd[8] = 0xD0
        ftemp = []
        ftemp = self.formated_data(temp_cmd)
        self.write(ftemp)

    def do_get_current(self):
        """Get current"""
        return float(self.current_value)

    def do_set_voltage(self, voltage):
        """Set voltage"""
        pass

    def do_get_voltage(self):
        """Get voltage"""

        return float(0)

    def do_set_status(self, status):
        """
        Turn output on and off

        Parameters:
        -----------
            status: 0 or 1
                0 for off, 1 for on
        """
        pass
		# replace by comparator status

    def do_get_status(self):
        """Check if output is turned on"""
        pass

    def do_get_voltage_compliance(self):
        """Get compliance voltage"""
        pass

    def do_set_voltage_compliance(self, compliance):
        """Set compliance voltage"""
        pass

    def do_get_current_compliance(self):
        """Get compliance voltage"""
        pass

    def do_set_current_compliance(self, compliance):
        """Set compliance current"""
        pass

    def do_get_range(self):
        """Get current range in A"""
        
        return float(self.current_range)

    def do_set_range(self, maxval):
        """Set current range in A"""
		
        if maxval < 1e-6:
            self.current_range = 1e-6
            self.set_range_backend(self.active_channel, '1uA')
        if maxval >1e-6 and maxval < 10e-6:
            self.current_range = 10e-6
            self.set_range_backend(self.active_channel, '10uA')
        if maxval >10e-3 and maxval < 100e-3:
            self.current_range = 100e-6
            self.set_range_backend(self.active_channel, '100uA')			
        if maxval >100e-6 and maxval < 1e-3:
            self.current_range = 1e-3
            self.set_range_backend(self.active_channel, '1mA')		
        if maxval >1e-3 and maxval < 10e-3:
            self.current_range = 10e-3
            self.set_range_backend(self.active_channel, '10mA')			
        if maxval >10e-3 and maxval < 50e-3:
            self.current_range = 50e-3
            self.set_range_backend(self.active_channel, '50mA')		
        if maxval > 50e-3:
            self.current_range = 50e-3
            self.set_range_backend(self.active_channel, '50mA')
            print("Range limits error")
		
		
    def set_appropriate_range(self, maxcurrent=1E-3, mincurrent=-1E-3):
        """Detect which range includes limits and set it"""

        pass


    def set_src_mode_volt(self, current_compliance = .001):
        """
        Changes mode from current to voltage source, compliance current is given as an argument

        Returns:
            True if the mode was changed, False otherwise
        """
        pass

    def set_src_mode_curr(self, voltage_compliance = 1):
        """
        Changes mode from voltage to current source, compliance voltage is given as an argument

        Returns:
            True if the mode was changed, False otherwise
        """
        pass

    def set_current_limits(self, mincurrent = -1E-3, maxcurrent = 1E-3):
        """ Sets a limits within the range if needed for safe sweeping"""
        pass

    def set_voltage_limits(self, minvoltage = -1E-3, maxvoltage = 1E-3):
        """ Sets a voltage limits within the range if needed for safe sweeping"""
        pass



    def clear(self):
        """
        Clear the event register, extended event register, and error queue.
        """
        pass


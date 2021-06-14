"""
User manual for the device:
http://downloads.signalcore.com/SC5502A_Manual.pdf

"""
import ctypes
from ctypes import WinDLL, create_string_buffer
from ctypes import c_char_p, c_uint, c_ulong, c_float, c_ulonglong, \
    c_uint8, c_bool
from ctypes import POINTER, byref, cast

MAXDEVICES = 128
MAXDESCRIPTORSIZE = 9

class SC5502A():

    def __init__(self, idx=0, master=True):
        self._idx = idx

        self.search()
        self.open()

        # REFERENCE_MODE(0x15)-This register sets the  behavior of the
        # reference clock  section
        if master:
            self._ext_ref_lock = 0   # Bit  0  enables  (1) or disables(0) the
            # device  to  phase-lock  to  an  external  source
            self._ext_ref_output = 1   # Bit  1  enables (1) or  disables  (0)
            # the  output  reference  signal,
            self._ext_ref_100Mhz = 0   # Bit 2 selects whether the output
            # reference
            # signalis 10 MHz (0) or  100  MHz  (1)
            self._pxi_10MHz_ref_output = 0   # Bit 3 enables (1) or disables
            # (0)
            # the PXI10 MHz clock
            self.set_ext_reference_lock(True)
        else:
            self._ext_ref_lock = 1   # Bit  0  enables  (1) or disables(0) the
            # device  to  phase-lock  to  an  external  source
            self._ext_ref_output = 1   # Bit  1  enables (1) or  disables  (0)
            # the  output  reference  signal,
            self._ext_ref_100Mhz = 0   # Bit 2 selects whether the output
            # reference
            # signalis 10 MHz (0) or  100  MHz  (1)
            self._pxi_10MHz_ref_output = 0   # Bit 3 enables (1) or disables
            # (0)
            # the PXI10 MHz clock
            self.set_ext_reference_lock(True)

    # 1101
    # 1000
    def search(self):

        self._lib = WinDLL(
            r"C:\Program Files\SignalCore\SC5502A\api\c\lib\x64\sc5502a.dll"
        )


        buffers = [create_string_buffer(MAXDESCRIPTORSIZE + 1) for bid in range(MAXDEVICES)]
        buffer_pointer_array = (c_char_p * MAXDEVICES)()
        for device in range(MAXDEVICES):
            buffer_pointer_array[device] = cast(buffers[device], c_char_p)
        buffer_pointer_array_p = cast(buffer_pointer_array, POINTER(c_char_p))

        devices_number = c_uint()
        self._handle = c_ulong()
        found = self._lib.sc5502a_SearchDevices(buffer_pointer_array_p, byref(devices_number))
        if not found:
            print('Found sc5502a device with it\'s pxi address {}'.format(str(buffer_pointer_array_p[0])))
        else:
            msg = 'Failed to find any device'
            raise RuntimeError(msg)

        self._device_ids = buffer_pointer_array_p

    def open(self):
        open = self._lib.sc5502a_OpenDevice(self._device_ids[self._idx],
                                            byref(self._handle))
        if open:
            msg = f'Failed to connect to the instrument with pxi address ' \
                  f'{(self._buffer_pointer_array_p[self._handle.value - 1])}' \
                  f' and handle {self._handle}'
            raise RuntimeError(msg)

    def close(self):
        close = self._lib.sc5502a_CloseDevice(self._handle)
        if close:
            msg = 'Failed to close the instrument with handle {}'.format(self._handle)
            raise RuntimeError(msg)

    def set_frequency(self, freq):
        setFreq = self._lib.sc5502a_SetFrequency(self._handle, c_ulonglong(int(freq)))
        if setFreq:
            msg = 'Failed to set frequency on the instrument with handle {}'.format(self._handle)
            raise RuntimeError(msg)

    def set_power(self, power):
        setPower = self._lib.sc5502a_SetPowerLevel(self._handle, c_float(power))
        if setPower:
            msg = 'Failed to set power level on the instrument with handle {}'.format(self._handle)
            raise RuntimeError(msg)

    def set_parameters(self, parameters_dict):

        if "frequency" in parameters_dict.keys():
            setFreq = self._lib.sc5502a_SetFrequency(self._handle, c_ulonglong(int(parameters_dict["frequency"])))
            if setFreq:
                msg = 'Failed to set frequency on the instrument with handle {}'.format(self._handle)
                raise RuntimeError(msg)

        if "power" in parameters_dict.keys():
            setPower = self._lib.sc5502a_SetPowerLevel(self._handle, c_float(parameters_dict['power']))
            if setPower:
                msg = 'Failed to set power level on the instrument with handle {}'.format(self._handle)
                raise RuntimeError(msg)

        if "frequencies" in parameters_dict.keys():
            pass  # we just ignore this option, be careful, there's no support for a list sweep in this source

    def send_sweep_trigger(self):
        pass  # stub method

    def getTemperature(self, temperature):
        getTemperature = self._lib.sc5502a_GetTemperature(self._handle, byref(temperature))
        if getTemperature:
            msg = 'Failed to get temperature on the instrument with handle {}'.format(self._handle)
            raise RuntimeError(msg)

    def set_output_state(self, output_state):
        if output_state == "OFF":
            set_output = self._lib.sc5502a_SetRfOutput(self._handle, c_bool(0))
        else:
            set_output = self._lib.sc5502a_SetRfOutput(self._handle, c_bool(1))

        if set_output:
            msg = 'Failed to set output state on the instrument with handle {}'.format(self._handle)
            raise RuntimeError(msg)

    def set_reference_clock_output(self, state):

        #if state is True:
        #     val = c_uint8(1)
        # elif state is False:
        #     val = c_uint8(0)
        # else:
        #     raise ValueError("state can be either True of False")

        self._ext_ref_output = state
        self._lib.sc5502a_SetClockReference(self._handle,
                                            self._ext_ref_lock,
                                            self._ext_ref_output,
                                            self._ext_ref_100Mhz,
                                            self._pxi_10MHz_ref_output)
        # 1101
        # 1000
    def set_ext_reference_lock(self, state):

        #if state is True:
        #    val = c_uint8(1)
        #elif state is False:
        #   val = c_uint8(0)
        #else:
         #   raise ValueError("state can be either True of False")

        self._ext_ref_lock = state
        self._lib.sc5502a_SetClockReference(self._handle,
                                            self._ext_ref_lock,
                                            self._ext_ref_output,
                                            self._ext_ref_100Mhz,
                                            self._pxi_10MHz_ref_output)

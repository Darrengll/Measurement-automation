import ctypes
from ctypes import *

MAXDEVICES = 128
MAXDESCRIPTORSIZE = 9


class SignalCore_5502a():


    def __init__(self):

        self.search()
        self.open()

        self._ext_ref_lock = 1
        self._ext_ref_output = 1
        self._ext_ref_100Mhz = 0
        self._pxi_10MHz_ref_output = 0

        self.set_ext_reference_lock(True)

    def search(self):

        self._lib = WinDLL('C:/Program Files/SignalCore/SC5502A/api/c/lib/x64/sc5502a.dll')

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
        open = self._lib.sc5502a_OpenDevice(self._device_ids[0], byref(self._handle))
        if open:
            msg = 'Failed to connect to the instrument with pxi address {} and handle {}'.format(
                str(self._device_ids[self._handle.value - 1]), self._handle)
            raise RuntimeError(msg)

    def close(self):
        close = self._lib.sc5502a_CloseDevice(self._handle)
        if close:
            msg = 'Failed to close the instrument with handle {}'.format(self._handle)
            raise RuntimeError(msg)

    def set_frequency(self, freq):
        self._lib.sc5502a_SetFrequency(self._handle, c_ulonglong(int(freq)))
        setFreq = self._lib.sc5502a_SetFrequency(self._handle, c_ulonglong(int(freq))) # ensure complete operation
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
            self.set_frequency(parameters_dict["frequency"])

        if "power" in parameters_dict.keys():
            self.set_power(parameters_dict["power"])

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

        if state is True:
            val = c_uint8(1)
        elif state is False:
            val = c_uint8(0)
        else:
            raise ValueError("state can be either True of False")

        self._ext_ref_output = state
        self._lib.sc5502a_SetClockReference(self._handle,
                                            self._ext_ref_lock,
                                            self._ext_ref_output,
                                            self._ext_ref_100Mhz,
                                            self._pxi_10MHz_ref_output)

    def set_ext_reference_lock(self, state):

        if state is True:
            val = c_uint8(1)
        elif state is False:
            val = c_uint8(0)
        else:
            raise ValueError("state can be either True of False")

        self._ext_ref_lock = state
        self._lib.sc5502a_SetClockReference(self._handle,
                                            self._ext_ref_lock,
                                            self._ext_ref_output,
                                            self._ext_ref_100Mhz,
                                            self._pxi_10MHz_ref_output)

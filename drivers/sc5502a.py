"""
User manual for the device:
http://downloads.signalcore.com/SC5502A_Manual.pdf

"""
import ctypes
import time
from ctypes import WinDLL, create_string_buffer
from ctypes import c_char_p, c_uint, c_ulong, c_float, c_ulonglong, \
    c_uint8, c_bool
from ctypes import POINTER, byref, cast

from loggingserver import LoggingServer

from log.LogName import LogName

MAXDEVICES = 128
MAXDESCRIPTORSIZE = 9


class SC5502A():

    def __init__(self, idx=0):
        self._idx = idx
        self.search()
        self.open()

        self._ext_ref_lock = 1  # Bit  0  enables  (1) or disables(0) the
        # device  to  phase-lock  to  an  external  source
        self._ext_ref_output = 1  # Bit  1  enables (1) or  disables  (0)
        # the  output  reference  signal
        self._ext_ref_100Mhz = 0  # Bit 2 selects whether the output
        # reference signal is 10 MHz (0) or  100  MHz  (1)
        self._pxi_10MHz_ref_output = 0  # Bit 3 enables (1) or disables
        # (0) the PXI10 MHz clock forwarding to the front pane
        self.setup_reference_lock()

        self._power = None
        self._frequency = None
        self._output_state = None
        self._logger = LoggingServer.getInstance(LogName.NAME)

    def search(self):

        self._lib = WinDLL(
            r"C:\Program Files\SignalCore\SC5502A\api\c\lib\x64\sc5502a.dll"
        )

        buffers = [create_string_buffer(MAXDESCRIPTORSIZE + 1) for bid in
                   range(MAXDEVICES)]
        buffer_pointer_array = (c_char_p * MAXDEVICES)()
        for device in range(MAXDEVICES):
            buffer_pointer_array[device] = cast(buffers[device], c_char_p)
        buffer_pointer_array_p = cast(buffer_pointer_array, POINTER(c_char_p))

        devices_number = c_uint()
        self._handle = c_ulong()
        op_status = self._lib.sc5502a_SearchDevices(buffer_pointer_array_p,
                                                    byref(devices_number))
        if not op_status:
            dev_address = str(buffer_pointer_array_p[self._idx])
            if len(dev_address) > 0:
                self._address = dev_address
            else:
                msg = f'Failed to find the device with idx {self._idx}'
                raise RuntimeError(msg)
        else:
            msg = 'Device search failed'
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
            msg = 'Failed to close the instrument with handle {}'.format(
                self._handle)
            raise RuntimeError(msg)

    def set_frequency(self, freq):
        self._frequency = freq
        # self.set_output_state("OFF")
        # time.sleep(0.01)
        setFreq = self._lib.sc5502a_SetFrequency(self._handle,
                                                 c_ulonglong(int(freq)))
        # self.set_output_state("ON")
        # time.sleep(0.01)

        if setFreq:
            msg = 'Failed to set freq on the instrument with handle {}'.format(
                self._handle)
            raise RuntimeError(msg)

    def set_fast_tune(self, value, step = "1 Hz"):
        '''

        Parameters
        ----------
        value: bool
        step: "1 Hz", "25 kHz", "1 MHz"
        '''
        self._fast_tune = value
        codes = {"1 Hz": 2, "25 kHz": 1, "1 MHz": 0}
        setMode = self._lib.sc5502a_SetSynthesizerMode(self._handle, c_uint8(int(value)), c_uint(int(codes[step])))
        if setMode:
            msg = f'Failed to set fast tune mode on the instrument with handle {self._handle}, mode {value}, step {step}'
            raise RuntimeError(msg)

    def set_power(self, power):
        if power > 10:
            # self._logger.debug("SignalCore: clipped power to max 10 dBm, see manual")
            power = 10
        self._power = power
        setPower = self._lib.sc5502a_SetPowerLevel(self._handle,
                                                   c_float(power))
        if setPower:
            msg = 'Failed to set power level on the instrument with handle {}'.format(
                self._handle)
            raise RuntimeError(msg)

    def get_power(self):
        return self._power

    def get_frequency(self):
        return self._frequency

    def get_output_state(self):
        return self._output_state

    def set_parameters(self, parameters_dict):
        if "freq" in parameters_dict.keys():
            self.set_frequency(parameters_dict["freq"])

        if "power" in parameters_dict.keys():
            self.set_power(parameters_dict["power"])

        if "frequencies" in parameters_dict.keys():
            pass  # we just ignore this option, be careful, there's no support for a list sweep in this source

    def send_sweep_trigger(self):
        pass  # stub method

    def getTemperature(self, temperature):
        getTemperature = self._lib.sc5502a_GetTemperature(self._handle,
                                                          byref(temperature))
        if getTemperature:
            msg = 'Failed to get temperature on the instrument with handle {}'.format(
                self._handle)
            raise RuntimeError(msg)

    def set_output_state(self, output_state):
        self._output_state = output_state
        if output_state == "OFF":
            set_output = self._lib.sc5502a_SetRfOutput(self._handle, c_bool(0))
        else:
            set_output = self._lib.sc5502a_SetRfOutput(self._handle, c_bool(1))

        if set_output:
            msg = 'Failed to set output state on the instrument with handle {}'.format(
                self._handle)
            raise RuntimeError(msg)

    def set_reference_clock_output(self, state):
        self._ext_ref_output = state
        self._lib.sc5502a_SetClockReference(self._handle,
                                            self._ext_ref_lock,
                                            self._ext_ref_output,
                                            self._ext_ref_100Mhz,
                                            self._pxi_10MHz_ref_output)

    def setup_reference_lock(self):
        self._lib.sc5502a_SetClockReference(self._handle,
                                            self._ext_ref_lock,
                                            self._ext_ref_output,
                                            self._ext_ref_100Mhz,
                                            self._pxi_10MHz_ref_output)

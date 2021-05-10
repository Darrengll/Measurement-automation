"""
User manual for the device:
http://downloads.signalcore.com/sc5502x_3x/SC5502A_Manual.pdf

"""
# Standard library imports
# ------------------------

# Third party imports
from ctypes import WinDLL, create_string_buffer
from ctypes import c_char_p, c_uint, c_ulong, c_float, c_ulonglong, \
    c_uint8, c_bool
from ctypes import POINTER, byref, cast

# Local application imports
from . import MwSrcInterface


MAXDEVICES = 128
MAXDESCRIPTORSIZE = 9


class SC5502A(MwSrcInterface):
    def __init__(self, idx=0, master=True):
        super().__init__(name="sc5502A_" + str(idx))
        self._idx = idx  # device index you wish to open
        self._visa_handle = c_uint()  # handle to the device
        self._device_ids = None

        # TODO: hardcoded path to API library dll
        self._lib = WinDLL(
            r"C:\Program Files\SignalCore\SC5502A\api\c\lib\x64\sc5502a.dll"
        )

        self.search()
        self.open()

        # REFERENCE_MODE(0x15)-This register sets the  behavior of the
        # reference clock  section
        if master:
            self._ext_ref_lock = 1   # Bit  0  enables  (1) or disables(0) the
            # device  to  phase-lock  to  an  external  source
            self._ext_ref_output = 1   # Bit  1  enables (1) or  disables  (0)
            # the  output  reference  trace,
            self._ext_ref_100Mhz = 0   # Bit 2 selects whether the output
            # reference
            # signalis 10 MHz (0) or  100  MHz  (1)
            self._pxi_10MHz_ref_output = 1   # Bit 3 enables (1) or disables
            # (0)
            # the PXI10 MHz clock
            self.set_ext_reference_lock(True)
        else:
            self._ext_ref_lock = 1   # Bit  0  enables  (1) or disables(0) the
            # device  to  phase-lock  to  an  external  source
            self._ext_ref_output = 0   # Bit  1  enables (1) or  disables  (0)
            # the  output  reference  trace,
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
        # array of pointers to the begginings of the devices descriptors
        visa_handles_p = (c_char_p * MAXDEVICES)()
        # create descriptor strings and
        # filling array of pointer to strings with pointers
        # to descriptor strings created
        for i in range(MAXDEVICES):
            _temp_str = create_string_buffer(MAXDESCRIPTORSIZE+1)
            visa_handles_p[i] = cast(_temp_str, c_char_p)

        # number of devices discovered will be placed here
        devices_n = c_uint()

        # searching discovered devices
        status = self._lib.sc5502a_SearchDevices(
            byref(visa_handles_p), byref(devices_n)
        )
        self._visas_handle_p = visa_handles_p
        if devices_n.value > 0:
            print("found {} devices".format(devices_n.value))
            print("device to be found: ", self._idx)
            print('Found sc5502a device with it\'s pxi address {}'.format(
                str(visa_handles_p[self._idx]))
            )
        else:
            msg = 'Failed to find any device'
            raise RuntimeError(msg)

        self._device_ids = visa_handles_p

    def open(self):
        opened = self._lib.sc5502a_OpenDevice(
            self._device_ids[self._idx], byref(self._visa_handle)
        )
        if opened:
            msg = f'Failed to connect to the instrument with pxi address ' \
                  f'{(self._visas_handle_p[self._visa_handle.value - 1])}' \
                  f' and handle {self._visa_handle}'
            raise RuntimeError(msg)

    def close(self):
        close = self._lib.sc5502a_CloseDevice(self._visa_handle)
        if close:
            msg = 'Failed to close the instrument with handle {}'.format(self._visa_handle)
            raise RuntimeError(msg)

    def set_frequency(self, freq):
        setFreq = self._lib.sc5502a_SetFrequency(self._visa_handle, c_ulonglong(int(freq)))
        if setFreq:
            msg = 'Failed to set if_freq on the instrument with handle {}'.format(self._visa_handle)
            raise RuntimeError(msg)

    def set_power(self, power):
        setPower = self._lib.sc5502a_SetPowerLevel(self._visa_handle, c_float(power))
        if setPower:
            msg = 'Failed to set power level on the instrument with handle {}'.format(self._visa_handle)
            raise RuntimeError(msg)

    def set_parameters(self, parameters_dict):
        if "if_freq" in parameters_dict.keys():
            setFreq = self._lib.sc5502a_SetFrequency(self._visa_handle, c_ulonglong(int(parameters_dict["if_freq"])))
            if setFreq:
                msg = 'Failed to set if_freq on the instrument with handle {}'.format(self._visa_handle)
                raise RuntimeError(msg)

        if "power" in parameters_dict.keys():
            setPower = self._lib.sc5502a_SetPowerLevel(self._visa_handle, c_float(parameters_dict['power']))
            if setPower:
                msg = 'Failed to set power level on the instrument with handle {}'.format(self._visa_handle)
                raise RuntimeError(msg)

        if "frequencies" in parameters_dict.keys():
            pass  # we just ignore this option, be careful, there's no support for a list sweep in this source

    def send_sweep_trigger(self):
        pass  # stub method

    def getTemperature(self, temperature):
        getTemperature = self._lib.sc5502a_GetTemperature(self._visa_handle, byref(temperature))
        if getTemperature:
            msg = 'Failed to get temperature on the instrument with handle {}'.format(self._visa_handle)
            raise RuntimeError(msg)

    def set_output_state(self, output_state):
        if output_state == "OFF":
            set_output = self._lib.sc5502a_SetRfOutput(self._visa_handle, c_bool(0))
        else:
            set_output = self._lib.sc5502a_SetRfOutput(self._visa_handle, c_bool(1))

        if set_output:
            msg = 'Failed to set output state on the instrument with handle {}'.format(self._visa_handle)
            raise RuntimeError(msg)

    def set_reference_clock_output(self, state):

        #if state is True:
        #     val = c_uint8(1)
        # elif state is False:
        #     val = c_uint8(0)
        # else:
        #     raise ValueError("state can be either True of False")

        self._ext_ref_output = state
        self._lib.sc5502a_SetClockReference(self._visa_handle,
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
        self._lib.sc5502a_SetClockReference(self._visa_handle,
                                            self._ext_ref_lock,
                                            self._ext_ref_output,
                                            self._ext_ref_100Mhz,
                                            self._pxi_10MHz_ref_output)

    def set_frequency_sweep(self, frequencies=None, power=None,
                            insweep_step_trg_src=None, sweep_trg_src=None,
                            arm_trigger_src=None):
        raise NotImplemented("this device does not have such functionalyty")

    def get_parameters(self):
        raise NotImplementedError("This method not implemented yet")

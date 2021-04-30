import visa
import types
import logging
from time import sleep
import numpy as np

from .mw_src_data_structures import

class MXG:
    def __init__(self, address):
        self._address = address
        rm = visa.ResourceManager()
        self._visainstrument = rm.open_resource(self._address)

        self.nop = None
        self.ext_trig_channel = None
        self.InSweep_trg_src = None
        self.sweep_trg_src = None

    def read(self):
        return self._visainstrument.read()

    def write(self, msg):
        return self._visainstrument.write(msg)

    def query(self, msg):
        return self._visainstrument.query(msg)

    def get_parameters(self):
        """
        Returns a dictionary containing if_freq and power currently used
        by the device
        """
        return {"power": self.get_power(), "if_freq": self.get_frequency()}

    def set_parameters(self, ):
        """

        Parameters
        ----------
        parameters_dict

        Returns
        -------

        """
        if mode ==
        if power is not None:
            self.set_power(power)
        if freq is not None:
            self.set_frequency(freq)
        if sweep_trg_src is not None:
            self.set_sweep_trg_src(sweep_trg_src)
            self.set_step_sweep()
            self.set_trig_type_single()
            self.sweep_cont_trig()
        else:
            self.set_single_point()
        if frequencies is not None:
            self.set_freq_limits((frequencies[0], frequencies[-1]))
            self.set_nop(len(frequencies))

        if "InSweep_trg_src" in keys:
            self.set_InSweep_trg_src(parameters_dict["InSweep_trg_src"])
        if "ext_trig_channel" in keys:
            self.set_ext_trig_channel(parameters_dict["ext_trig_channel"])\


    def use_internal_clock(self, is_clock_internal):
        if is_clock_internal:
            self.write(":SOURce:ROSCillator:SOURce:AUTO OFF")
        else:
            self.write(":SOURce:ROSCillator:SOURce:AUTO ON")

    def set_output_state(self, output_state):
        """
        "ON" of "OFF"
        """
        self.write(":OUTput:STATe "+output_state)

    def get_output_state(self):
        return self.query(":OUTput:STATe?")

    def set_freq_mode_fixed(self):
        self.write(":SOURce:FREQuency:MODE FIXed")

    def set_power_mode_fixed(self):
        self.write(":SOURce:POWer:MODE FIXed")

    def set_frequency(self, freq):
        self.write(":SOURce:FREQuency:CW {0}HZ".format(freq))

    def get_frequency(self):
        bla = self.query(":SOURce:FREQuency:CW?")
        try:
            output = float(bla)
        except:
            print("Error in get_freq(): value returned: {0}".format(bla))
            output = -1.0
        return output

    def set_power(self, power_dBm):
        if (power_dBm >= -130) & (power_dBm <= 19):
            self.write(":SOURce:POWer {0}DBM".format(power_dBm))
        else:
            print("Error: power must be between -130 and 19 dBm")

    def get_power(self):
        bla = self.query(":SOURce:POWer?")
        try:
            output = float(bla)
        except:
            print("Error in get_power(): value returned: {0}".format(bla))
            output = -1.0
        return output

    # def set_frequency_sweep(self):

    def set_ext_trig_channel(self, ext_trig_channel):
        self.write(":LIST:TRIG:EXT:SOUR %s" % (ext_trig_channel))  # choose external trigger channel

    def get_ext_trig_channel(self):
        raise NotImplemented

    def set_freq_sweep(self):
        # LIST, CW OR FIXED
        # (CW and FIXED is the same and refers to the fixed if_freq)
        self.write(":FREQuency:MODE LIST")

    def set_single_point(self):
        self.write(":FREQuency:MODE CW")

    def attn_hold_off(self):
        self.write(":SOUR:POW:ATT:AUTO ON")

    def set_step_sweep(self):
        # STEP - interval and number of pts | LIST - list ought to be loaded
        self.write(":LIST:TYPE STEP")

    def set_freq_limits(self, freq_limits):
        self.write(":FREQuency:STARt %f%s" % (freq_limits[0], "Hz")) # TODO: rename
        self.write(":FREQuency:STOP %f%s" % (freq_limits[-1], "Hz"))

    def set_nop(self, nop):
        self.write(":SWEep:POINts %i" % (nop))

    def get_nop(self):
        raise NotImplemented

    def set_InSweep_trg_src(self, InSweep_trg_src):
        # sweep event trigger source
        # (BUS is equivalent to GPIB source "*TRG" trace)
        self.write(":LIST:TRIG:SOUR %s" % (InSweep_trg_src))

    def get_InSweep_trg_src(self):
        raise NotImplemented

    def send_sweep_trigger(self):
        # starting trigger
        self.write("*TRG")

    def set_sweep_trg_src(self, sweep_trg_src):
        # This command sets the sweep trigger source for a list or step sweep.
        self.write(":TRIG:SOUR %s" % (sweep_trg_src))

    def do_get_sweep_trg_src(self):
        raise NotImplemented

    def sweep_cont_trig(self):
        # This command selects either a continuous or single list or step sweep.
        # Execution of this command does not affect a sweep in progress.
        self.write(":INITiate:CONT ON")

    def set_trig_type_single(self):
        self.write(":TRIG:TYPE SINGLE")


class EXG(MXG):

    def set_power(self, power_dBm):
        if (power_dBm >= -20) & (power_dBm <= 19):
            self.write(":SOURce:POWer {0}DBM".format(power_dBm))
        else:
            print("Error: power must be between -20 and 19 dBm")

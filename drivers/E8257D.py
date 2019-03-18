import drivers.instr as instr

from drivers.instrument import Instrument
import visa
import types
import logging
from time import sleep
import numpy


class MXG(Instrument):

    def __init__(self, address):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.WARNING)

        Instrument.__init__(self, "", tags=['physical'])

        self._address = address
        rm = visa.ResourceManager()
        self._visainstrument = rm.open_resource(self._address)  # no term_chars for GPIB!!!!!

        self.set_output_state("ON")

        self.add_parameter('nop', type=int,
                           flags=Instrument.FLAG_GETSET, minval=1, maxval=10 ** 6,
                           tags=['sweep'])

        # external triggering parameters
        self.add_parameter('ext_trig_channel', type=bytes,
                           flags=Instrument.FLAG_GETSET)

        self.add_parameter('InSweep_trg_src', type=bytes,
                           flags=Instrument.FLAG_GETSET)

        self.add_parameter('sweep_trg_src', type=bytes,
                           flags=Instrument.FLAG_GETSET)

    def read(self):
        return self._visainstrument.read()

    def write(self, msg):
        return self._visainstrument.write(msg)

    def query(self, msg):
        return self._visainstrument.query(msg)

    def get_parameters(self):
        """
        Returns a dictionary containing frequency and power currently used
        by the device
        """
        return {"power":self.get_power(), "frequency":self.get_frequency()}

    def set_parameters(self, parameters_dict):
        """
        Method allowing to set all of the VNA parameters at once (bandwidth, nop,
        power, averages and freq_limits)
        """
        keys = parameters_dict.keys()
        if "power" in keys:
            self.set_power(parameters_dict["power"])
        if "frequency" in keys:
            self.set_frequency(parameters_dict["frequency"])

        # trigger stuff by Elena
        if "sweep_trg_src" in keys:
            self.set_freq_sweep()

        if "frequencies" in keys:
            freqs = parameters_dict["frequencies"]
            self.set_freq_limits((freqs[0], freqs[-1]))
            self.set_nop(len(freqs))

        if "sweep_trg_src" in keys:
            self.set_sweep_type()
            self.set_trig_type_single()
        else:
            self.set_single_point()

        if "sweep_trg_src" in keys:
            self.set_sweep_trg_src(parameters_dict["sweep_trg_src"])

        if "sweep_trg_src" in keys:
            self.sweep_cont_trig()

        if "InSweep_trg_src" in keys:
            self.set_InSweep_trg_src(parameters_dict["InSweep_trg_src"])
        if "ext_trig_channel" in keys:
            self.set_ext_trig_channel(parameters_dict["ext_trig_channel"])



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
        bla = 0  # self.read(":SOURce:FREQuency:CW?")
        try:
            output = float(bla)
        except:
            print("Error in get_freq(): value returned: {0}".format(bla))
            output = -1.0
        return output

    def set_power(self, power_dBm):
        if (power_dBm >= -130) & (power_dBm <= 15):
            self.write(":SOURce:POWer {0}DBM".format(power_dBm))
        else:
            print("Error: power must be between -130 and 15 dBm")

    def get_power(self):
        bla = self.query(":SOURce:POWer?")
        try:
            output = float(bla)
        except:
            print("Error in get_power(): value returned: {0}".format(bla))
            output = -1.0
        return output

    # def set_frequency_sweep(self):

    def do_set_ext_trig_channel(self, ext_trig_channel):
        self.write(":LIST:TRIG:EXT:SOUR %s" % (ext_trig_channel))  # choose external trigger channel

    def do_get_ext_trig_channel(self):
        raise NotImplemented

    def set_freq_sweep(self):
        # LIST, CW OR FIXED
        # (CW and FIXED is the same and refers to the fixed frequency)
        self.write(":FREQuency:MODE LIST")

    def set_single_point(self):
        self.write(":FREQuency:MODE CW")

    def attn_hold_off(self):
        self.write(":SOUR:POW:ATT:AUTO ON")

    def set_sweep_type(self):
        # STEP - interval and number of pts | LIST - list ought to be loaded
        self.write(":LIST:TYPE STEP")

    def set_freq_limits(self, freq_limits):
        self.write(":FREQuency:STARt %f%s" % (freq_limits[0], "Hz")) # TODO: rename
        self.write(":FREQuency:STOP %f%s" % (freq_limits[-1], "Hz"))

    def do_set_nop(self, nop):
        self.write(":SWEep:POINts %i" % (nop))

    def do_get_nop(self):
        raise NotImplemented

    def do_set_InSweep_trg_src(self, InSweep_trg_src):
        # sweep event trigger source
        # (BUS is equivalent to GPIB source "*TRG" signal)
        self.write(":LIST:TRIG:SOUR %s" % (InSweep_trg_src))

    def do_get_InSweep_trg_src(self):
        raise NotImplemented

    def send_sweep_trigger(self):
        # starting trigger
        self.write("*TRG")

    def do_set_sweep_trg_src(self, sweep_trg_src):
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

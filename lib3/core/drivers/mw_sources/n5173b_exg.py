"""
SCPI command reference:
https://www.keysight.com/us/en/assets/9018-03690/programming-guides/9018-03690.pdf?success=true

Programming manual:
https://www.keysight.com/us/en/assets/9018-03690/programming-guides/9018-03690.pdf
main info on triggering can be found on p.83. Though it takes quite a
workaround to make it work.

datasheet:
https://www.keysight.com/us/en/assets/7018-04097/data-sheets/5991-3132.pdf
"""
# Standard library imports
from enum import Enum
from typing import Union, Dict, Any, List, Tuple

# Third party imports
import visa

# Local application imports
from lib3.core.drivers.mw_sources import MwSrcInterface
from lib3.core.drivers.mw_sources import MwSrcParameters
from lib3.core.drivers.mw_sources import MW_SRC_MODE, MW_SRC_TRG_SUBSYS


class N5173B(MwSrcInterface):
    def __init__(self, address):
        super().__init__()
        self._address = address
        rm = visa.ResourceManager()
        self._visainstrument = rm.open_resource(self._address)

    def read(self):
        return self._visainstrument.read()

    def write(self, msg):
        return self._visainstrument.write(msg)

    def query(self, msg):
        return self._visainstrument.query(msg)

    def set_parameters(self, params):
        """

        Parameters
        ----------
        params : MwSrcParameters
            structure containing all parameters

        Returns
        -------
        None
        """
        if params.mode == MW_SRC_MODE.SINGLE:
            self.set_freq_mode(freq_mode="FIXED")
            self.set_frequency(params.freq)
            self.set_power_mode(pow_mode='FIXED')
            self.set_power(params.power)
        elif params.mode == MW_SRC_MODE.SWEEP_FREQ_STEP_LINEAR:
            self.set_linear_frequency_sweep_step_triggered(
                freq_limits=params.freq_limits,
                nop=params.freq_nop,
                power=params.power,
                ext_trg_port=params.ext_trg_port,
                insweep_step_trg_subsys=params.insweep_step_trg_subsys,
                arm_sweep_trg_subsys=params.arm_sweep_trg_subsys
            )
        elif params.mode == MW_SRC_MODE.SWEEP_FREQ_LIST_LINEAR:
            """
            The maximum number of list sweep points is 3,201.
            STEP has no such restrictions.
            """
            raise NotImplemented
        elif params.mode == MW_SRC_MODE.SWEEP_POW_LIST_LINEAR:
            raise NotImplemented

    def set_freq_mode(self, mode="FIXED"):
        """
        Sets frequency mode
        Parameters
        ----------
        mode : str
            "FIXED", "CW" - fixed frequency
            "LIST" - sweep through list of frequencies supplied separately

        Returns
        -------
        None
        """
        # LIST, CW OR FIXED
        # (CW and FIXED is the same and refers to the fixed if_freq)
        self.write(":FREQ:MODE {}".format(mode))

    def get_parameters(self):
        """
        Returns a dictionary containing if_freq and power currently used
        by the device
        """
        return {"power": self.get_power(), "if_freq": self.get_frequency()}

    def set_output_state(self, output_state):
        """
        "ON" of "OFF"
        """
        self.write(":OUTput:STATe "+output_state)

    def get_output_state(self):
        return self.query(":OUTput:STATe?")

    def use_internal_clock(self, is_clock_internal):
        if is_clock_internal:
            self.write(":SOURce:ROSCillator:SOURce:AUTO OFF")
        else:
            self.write(":SOURce:ROSCillator:SOURce:AUTO ON")

    """ FREQUENCY settings """
    def set_freq_mode(self, freq_mode="FIXED"):
        """
        Sets power working mode
        Parameters
        ----------
        freq_mode : str
            "FIXED" - power is not changing (only manually through driver)
            "LIST" - sweep through list of powers supplied later
        Returns
        -------
        None
        """
        self.write(":FREQuency:MODE " + freq_mode)

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

    """ POWER settings """
    def set_power_mode(self, pow_mode="FIXED"):
        """
        Sets power working mode
        Parameters
        ----------
        pow_mode : str
            "FIXED" - power is not changing (only manually through driver)
            "LIST" - sweep through list of powers supplied later
        Returns
        -------
        None
        """
        self.write(":SOURce:POWer:MODE " + pow_mode)

    def set_power(self, power_dBm):
        """
        Sets output power for power SINGLE mode.
        Parameters
        ----------
        power_dBm : float
            For range 9 kHz - 13 GHz maximum output power is 18 dBm.
            See datasheet for more info.
        """
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

    def set_linear_frequency_sweep_step_triggered(
            self, freq_limits=None, nop=None, power=None,
            ext_trg_port="TRIG1",
            insweep_step_trg_subsys=MW_SRC_TRG_SUBSYS.EXT,
            arm_sweep_trg_subsys=MW_SRC_TRG_SUBSYS.BUS):
        # power mode - single power point
        self.set_power_mode(pow_mode="FIXED")
        self.set_power(power)

        """ FREQUENCY MODE SETTINGS """
        self.set_freq_mode(freq_mode="LIST")  # sweep mode
        # step mode requires only interval and number of points
        self.set_sweep_data_format(sweep_type="STEP")
        # linear spacing between points
        self.set_sweep_mesh_spacing(mesh="LIN")

        # set frequency interval
        self.set_freq_limits(freq_limits)
        # set number of points (both ends included)
        self.set_nop(nop)

        """ TRIGGER SETTINGS """
        # external subsystem reacts on positive edge of the trigger signal
        self.select_ext_trg_slope("POS")
        # select trigger subsystem that will cause to go to the next point
        # during the sweep
        self.select_insweep_trg_src(
            insweep_step_trg_subsys=insweep_step_trg_subsys.value
        )
        # select physical trigger source to be external source in sweep mode
        # that comes from EXT subsystem
        self.select_list_ext_trg_port(ext_trg_port=ext_trg_port)

        # select trigger subsystem that arms sweep
        self.select_sweep_arm_trg_src(
            arm_trg_subsys=arm_sweep_trg_subsys.value
        )

        # SWEEP will be armed repeatedly by its trigger source
        self.select_sweep_repetition("ON")

    """ SWEEP mode settings """
    def set_sweep_data_format(self, sweep_type="STEP"):
        """
        This command toggles between the two types of sweep

        Parameters
        ----------
        sweep_type : str
            "STEP" - This type of sweep has equally spaced frequencies and
                amplitudes.
            "LIST" - This type of sweep has arbitrary frequencies and
                amplitudes.
        """
        self.write(":LIST:TYPE {}".format(sweep_type))

    """ For SWEEP mode and STEP data format"""
    def set_sweep_mesh_spacing(self, mesh="LIN"):
        """
        This command enables the signal generator linear or logarithmic sweep
        modes.\n
        These commands require the signal generator to be in step mode.\n
        The instrument uses the specified start frequency, stop frequency,
        and number of points for both linear and log sweeps.

        References
        ----------
        p.94 programming manual

        Parameters
        ----------
        mesh : str
            "LIN" - linear division of interval
            "LOG" - logarithmic division of the interval
        """
        self.write(":SWEep:SPACing " + mesh)

    def set_freq_limits(self, freq_limits):
        """
        Set frequency sweep interval limits.
        Available only if
        `set_freq_mode("SWEEP");  set_sweep_data_format("STEP")`

        References
        -----------
        see p.78 of programming manual

        Parameters
        ----------
        freq_limits : Tuple[np.float64, np.float64]
            interval of frequencies to swept through
        """
        # This command sets the first frequency point in a step sweep.
        self.write(":FREQuency:STARt %f%s" % (freq_limits[0], "Hz"))
        # This command sets the last frequency point in a step sweep
        self.write(":FREQuency:STOP %f%s" % (freq_limits[1], "Hz"))

    def set_nop(self, nop):
        """
        This command defines the number of step sweep points.\n
        range from 2 to 65535.\n

        Parameters
        ----------
        nop : int
            number of points (including both ends) to be used during sweep.
        """
        self.write(":SWEep:POINts {:d}".format(nop))

    def select_sweep_repetition(self, mode="ON"):
        """
        This command selects either a continuous or single list or step sweep.
        Execution of this command does not affect a sweep in progress

        Parameters
        ----------
        mode : str
            "ON" - This choice selects continuous sweep where, after the
                completion of the previous sweep, the current sweep
                will restart automatically or wait until the appropriate
                trigger source is received.
            "OFF" - This choice selects a single sweep. Refer to
                :INITiate[:IMMediate][:ALL] for single sweep triggering
                information.
        """
        self.write(":INITiate:CONT {}".format(mode))

    """ TRIGGER options """
    def select_list_ext_trg_port(self, ext_trg_port="TRIG1"):
        """
        This command selects the external trigger source for usage in
        LIST/STEP sweep. Works if point sweep trigger and/or arm sweep trigger
        is chosen "EXT".
        With external triggering, the selected bi-directional BNC is
        configured as an input.

        Parameters
        ----------
        ext_trg_port : str
            "TRIGger1" - This choice selects the TRIG 1 BNC as the external
                trigger source for triggering sweep, point and function
                generator sweeps.
            "TRIGger2" - This choice selects the TRIG 2 BNC as the external
                trigger source for triggering sweep, point and function
                generator sweeps.
            "PULSE" - This choice selects the PULSE BNC as the external
                trigger source for triggering sweep, point and function
                generator sweeps.
        """
        self.write(":LIST:TRIG:EXT:SOUR {}".format(ext_trg_port))

    def select_ext_trg_slope(self, slope="POS"):
        """
        This command sets the polarity of an external signal at the TRIG 1,
        TRIG 2, or PULSE BNC (see :LIST:TRIGger:EXTernal:SOURce) or
        internal Pulse Video or Pulse Sync signal (see
        :LIST:TRIGger:INTernal:SOURce) that will trigger a list
        or step sweep.

        Parameters
        ----------
        slope : str
            "POS" - positive edge
            "NEG" - negative edge
        """
        self.write(":LIST:TRIGger:SLOPe {}".format(slope))

    def select_sweep_arm_trg_src(self, arm_trg_subsys="BUS"):
        """
        This command sets the sweep trigger source for a list or step sweep.
        Parameters
        ----------
        arm_trg_subsys : str
            "BUS" - This choice enables GPIB triggering using the *TRG or
                GET command. The *TRG SCPI command can be used
                with any combination of GPIB, LAN, or USB. The GET
                command requires USB, GPIB, or LAN–VXI–11.
            "IMM" - This choice enables immediate triggering of the sweep
                event.
            "EXT" - This choice enables the triggering of a sweep event by
                an externally applied signal at the TRIG 1, TRIG 2 or
                PULSE connector (see :TRIGger:EXTernal:SOURce).
        """
        self.write(":TRIGger:SEQuence:SOURce {}".format(arm_trg_subsys))

    def select_insweep_trg_src(self, insweep_step_trg_subsys="EXT"):
        """
        This command sets the point trigger source for a
        list or step sweep event

        Parameters
        ----------
        insweep_step_trg_subsys : str
            "BUS" - This choice enables GPIB triggering using the *TRG or
                GET command, or
                LAN and USB triggering using the *TRG command.
            "EXT" -  This choice enables the triggering of a sweep event by
                an externally applied signal at the TRIGGER IN (TRIG1,
                TRIG2, PULSE) connector.
            "IMM" - This choice enables immediate triggering of the sweep
            event.
        """
        self.write(":LIST:TRIG:SOUR {}".format(insweep_step_trg_subsys))

    def arm_sweep(self):
        """
        Send trigger from CPU (counts as "BUS" trigger)
        """
        self.write("*TRG")

    """ Additional options """
    def set_attn_hold_state(self, state="ON"):
        """
        This command sets the state of the attenuator auto mode function.

        Parameters
        ----------
        state : str
            "ON" - This selection allows the signal generator’s automatic
                level control (ALC) to adjust the attenuator so that a
                specified RF power level, at the Keysight MXG’s RF
                output connector, is maintained.
            "OFF" - This choice allows for a user–selected attenuator
                setting that is not affected by the signal generator’s
                ALC circuitry. Other settings become available, see
                Remarks.
                The OFF (0) selection can be used to eliminate power
                discontinuity normally associated with attenuator
                switching during power adjustments.
        """
        self.write(":SOUR:POW:ATT:AUTO ON")



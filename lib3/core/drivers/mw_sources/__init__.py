# Standard library imports
from typing import List
from enum import Enum
# Third party imports
import numpy as np
# Local application imports
# --------------------


class MW_SRC_MODE(str, Enum):
    """
    Operating mode for the microwave source.
    SINGLE - outputs at fixed frequency with fixed power
    SWEEP -  sweep through list of power/frequency list
        sweep behaviour depends on trigger settings introduced below
    """
    SINGLE = "SINGLE"
    SWEEP_FREQ_STEP_LINEAR = "SWEEP_FREQ_STEP_LINEAR"
    SWEEP_FREQ_LIST_LINEAR = "SWEEP_FREQ_LIST_LINEAR"
    SWEEP_POW_LIST_LINEAR = "SWEEP_POW_LIST_LINEAR"


class MW_SRC_TRG_SUBSYS(str, Enum):
    """
    Enumeration of possible triggering subsystems of the microwave source.

    BUS - trigger is armed manually via sending command from CPU.
    EXT - use external trigger.
    IMM - immediate trigger
    TIM - trigger from timer
    """
    BUS = "BUS"
    EXT = "EXT"
    IMM = "IMM"
    TIM = "TIM"


class MwSrcParameters:
    def __init__(self, mode=MW_SRC_MODE.SINGLE, power=None, freq=None,
                 ext_trg_port="TRIG1",
                 insweep_step_trg_subsys=MW_SRC_TRG_SUBSYS.EXT,
                 arm_sweep_trg_subsys=MW_SRC_TRG_SUBSYS.BUS,
                 frequencies_list=None, freq_limits=None, freq_nop=None,
                 powers_list=None):
        """

        Parameters
        ----------
        mode : MW_SRC_MODE
            operating mode
        power : float
            output power in dBm
        freq : float
            output single frequency Hz.
        ext_trg_port : str
            physical trigger port that will be used by EXTernal trigger
            subsystem.
            "TRIGN" - where N is external port number.
        insweep_step_trg_subsys : MW_SRC_TRG_SUBSYS
            trigger source for moving to the next step while performing
            sweeping through list of freqs/powers to output.
        arm_sweep_trg_subsys : MW_SRC_TRG_SUBSYS
            Source that triggers sweep output into active state. In active
            state it awaits next trigger, specified by `in_sweep_trg_src`.
        frequencies_list : List[float]
            list of frequencies in Hz. Only SWEEP+LIST output mode.
        freq_limits : Tuple[float, float]
            frequencies interval for SWEEP+STEP output mode
        freq_nop : int
            number of frequency points including both ends for `freq_limits`
            interval that will be swiped through during SWEEP+STEP output mode
        powers_list : List[float]
            list of powers in dBm. Used only in sweep output mode.
        """
        self.mode = mode
        self.power = power
        self.freq = freq

        self.ext_trg_port: str = ext_trg_port
        self.insweep_step_trg_subsys: MW_SRC_TRG_SUBSYS = \
            insweep_step_trg_subsys
        self.arm_sweep_trg_subsys: MW_SRC_TRG_SUBSYS = arm_sweep_trg_subsys

        self.frequencies_list = frequencies_list
        self.freq_limits = freq_limits
        self.freq_nop = freq_nop

        self.powers_list = powers_list

    def get_scan_freqs(self):
        if self.mode == MW_SRC_MODE.SINGLE:
            return [self.freq]
        elif self.mode == MW_SRC_MODE.SWEEP_FREQ_LIST_LINEAR:
            return self.frequencies_list
        elif self.mode == MW_SRC_MODE.SWEEP_FREQ_STEP_LINEAR:
            return np.linspace(*self.freq_limits, self.freq_nop)
        else:
            raise NotImplemented(
                "scan frequencies getter function is not "
                "implemented for mode requested: {}".format(self.mode)
            )

    def toJSON(self):
        return self.__dict__


class MwSrcInterface:
    """
    Base class that every microwave source should implement
    """
    def __init__(self, name=None):
        self.name = name
        self.parameters: MwSrcParameters = None

    def set_parameters(self, mw_src_params):
        """
        sets all parameters from MwSrcParameters class.
        Parameters
        ----------
        mw_src_params : MwSrcParameters

        Returns
        -------
        """
        raise NotImplementedError

    def get_parameters(self):
        """

        Returns
        -------
        res : MwSrcParameters
        """
        raise NotImplementedError

    def set_frequency(self, frequency):
        """

        Parameters
        ----------
        frequency : float
            Output frequency in Hz.

        Returns
        -------
        """
        raise NotImplementedError

    def set_power(self, power):
        """

        Parameters
        ----------
        power : float
            Output power in dBm

        Returns
        -------
        """
        raise NotImplementedError

    def set_output_state(self, state):
        """
        Turns RF output "ON" or "OFF"
        Parameters
        ----------
        state : str
            "ON" - output turns ON.
            "OFF" - output turns OFF.

        Returns
        -------
        None
        """
        raise NotImplementedError

    def set_linear_frequency_sweep_step_triggered(
            self, freq_limits=None, nop=None, power=None,
            ext_trg_port="TRIG1",
            insweep_step_trg_subsys=MW_SRC_TRG_SUBSYS.EXT,
            arm_sweep_trg_subsys=MW_SRC_TRG_SUBSYS.BUS):
        """
        Sets device ready to perform sweep upon arrival consecutive triggers to
        sweep step trigger source specified.
        Sets microwave source into mode where it outputs array of
        frequencies at given power.

        Equivalent to set

        def callback(i):
            self.set_frequency(freq[i])\n

        Such that `calllback()` call is triggered by signal coming from channel
        specified in
        `sweep_step_trg_src`.

        Parameters
        ----------
        freq_limits : Tuple[float,float]
            frequencies interval to sweep
        nop : int
            number of points in `freq_limits` interval (includeing both ends)
        power : float
            Sweep power value
        ext_trg_port : str
            physical port utilized by EXTernal trigger subsystem
            "TRIGN" - where N is number of input trigger port
        insweep_step_trg_subsys : MW_SRC_TRG_SUBSYS
            trigger subsystem for microwave source to step to the next sweep
            point.
        arm_sweep_trg_subsys : MW_SRC_TRG_SUBSYS
            trigger subsystem that turns microwave source ready for
            frequency sweep
        Returns
        -------

        """
        raise NotImplementedError

    def arm_sweep(self):
        """
        Send trigger from CPU (counts as "BUS" trigger)
        """
        raise NotImplementedError

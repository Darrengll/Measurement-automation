from typing import List
from enum import Enum


class MW_SRC_MODE(str, Enum):
    """
    Operating mode for the microwave source.
    SINGLE - outputs at fixed frequency with fixed power
    SWEEP -  sweep through list of power/frequency list
        sweep behaviour depends on trigger settings introduced below
    """
    SINGLE = "SINGLE"
    SWEEP = "SWEEP"


class MW_TRIG_SRC(str, Enum):
    """
    Trigger source for microwave source events.
    TRIG1 - use TRIG1 BNC input on the backplane of the device
    CONT - continuously trigger itself.
    """
    TRIG1 = "TRIG1"
    CONT = "CONT"


class MW_SWEEP_STEP_TRG_SRC(str, Enum):
    """
    Trigger source to initiate next step while working in SWEEPP mode.
    EXT - use external trigger.
    """
    EXT = "EXT"


class MW_SWEEP_ARMED_TRIG_SRC(str, Enum):
    """
    Source of trigger that will arm sweep. After sweep is armed, it awaits
    further trigger events from the source to be specified in
    `MW_SWEEP_STEP_TRG_SRC` to output next step from sweep list.
    the frequency/power sweep will make a step/full sweep.
    One step or full sweep will be performed depending on the
    MW_SWEEP_STEP_TRG_SRC value.

    BUS - trigger is armed manually via sending command from CPU.
    """
    BUS = "BUS"


class MwSrcParameters:
    def __init__(self, mode=MW_SRC_MODE.SINGLE, power=None, freq=None,
                 trig_src=MW_TRIG_SRC.TRIG1,
                 sweep_step_trg_src=MW_SWEEP_STEP_TRG_SRC.EXT,
                 sweep_arm_src=MW_SWEEP_ARMED_TRIG_SRC.BUS,
                 frequencies_list=None,
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
        trig_src : MW_TRIG_SRC
            physical trigger suource that will initiate device sweep stepping.
        sweep_arm_src : MW_SWEEP_ARMED_TRIG_SRC
            Source that triggers sweep output into active state. In active
            state it awaits next trigger, specified by `in_sweep_trg_src`.
        sweep_step_trg_src : MW_SWEEP_STEP_TRG_SRC
            trigger source for moving to the next step while performing
            sweeping through list of freqs/powers to output.
        frequencies_list : List[float]
            list of frequencies in Hz. Only sweep output mode.
        powers_list : List[float]
            list of powers in dBm. Used only in sweep output mode.
        """
        self.mode = mode
        self.power = power
        self.freq = freq

        self.trig_src = trig_src
        self.sweep_arm_src = sweep_arm_src
        self.in_sweep_trg_source = sweep_step_trg_src

        self.frequencies_list = frequencies_list
        self.powers_list = powers_list
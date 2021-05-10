# Standard library imports
# ------------------
# Third party imports
# -------------------
# Local application imports
from .mw_src_data_structures import MwSrcParameters, \
    MW_TRIG_SRC, MW_INSWEEP_TRG_SRC, MW_SWEEP_ARMED_TRIG_SRC


class MwSrcInterface:
    """
    Base class that every microwave source should implement
    """
    def __init__(self, name=None):
        self.name = name

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

    def set_frequency_sweep(self, frequencies=None, power=None,
                            insweep_step_trg_src=None, sweep_trg_src=None,
                            arm_trigger_src=None):
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
        frequencies : List[float]
            frequencies list to sweep
        power : float
            Sweep power value
        insweep_step_trg_src : MW_INSWEEP_TRG_SRC
            trigger source for microwave source to step to the next sweep
            point.
        sweep_trg_src : SWEEP_TRG_SRC
            trigger source that turns microwave source ready for frequency
            sweep
        arm_trigger_src : MW_SWEEP_ARMED_TRIG_SRC
            trigger source that
        Returns
        -------

        """
        raise NotImplementedError
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

    def arm_sweep(self, frequencies=None, powers=None,
                  sweep_step_trg_src=None, sweep_trg_src=None):
        """
        Sets device ready to perform sweep upon arrival consecutive triggers to
        sweep step trigger source specified.
        Outputs array of frequencies and powers
        Equivalent to set

        def callback(i):
            self.set_frequency(freq[i])\n
            self.set_power(power[i])

        Such that `calllback()` call is triggered by signal coming from channel
        specified in
        `sweep_step_trg_src`.

        Parameters
        ----------
        frequencies : List[float]
            frequencies list to sweep
        power : List[float]
            power list to sweep. Must have the same length as `frequencies
            argument`
        sweep_step_trg_src : MW_SWEEP_STEP_TRG_SRC
            trigger source for microwave source to step to the next sweep
            point.
        sweep_trg_src

        Returns
        -------

        """
        raise NotImplementedError
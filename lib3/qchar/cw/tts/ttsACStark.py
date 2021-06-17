# Standard library imports
from collections import OrderedDict
from typing import Iterable

# Third party imports
# ------------------

# Local application imports
from .ttsBase import TTSBase


class TTSAStark(TTSBase):
    READOUT_POWER_CAPTION = "Readout power, dBm"

    def sweep_readout_power(self, readout_powers_list_dbm):
        """
        Function sets sweep parameter to be the readout_power
        for dispersive measurement of a ACStark effect.

        Notes
        ----------
        Setter is not adaptive since resonator frequency depends only on
        flux and qubit state. Since flux is fixed and we wish to detect
        change from qubit state change we use first successful resonator state.

        Parameters
        ----------
        readout_powers_list_dbm : Iterable[float]
            list of VNA readout powers [dBm].
        """
        res_freq = self._last_resonator_result[0]
        self._vna[0].set_freq_limits(res_freq, res_freq)
        if not self._mw_triggered_by_vna:
            # if `mw_src` is not triggered by vna, we should change mw
            # frequency manually from CPU
            self.set_swept_parameters(
                # Note:since python 3.7 dict is ought to remember and preserve
                # insertion order inside dictionary
                **OrderedDict(
                    [
                        (
                            TTSAStark.READOUT_POWER_CAPTION,
                            (
                                self._set_readout_power(),
                                readout_powers_list_dbm
                            )
                         ),
                        (
                            "Scan Frequency, Hz",
                            (
                                self._mw_src[0].set_frequency,
                                self.scan_frequencies
                            )
                        )
                    ]
                )
            )
        else:
            self.set_swept_parameters(
                **{
                    TTSAStark.READOUT_POWER_CAPTION: (
                        self._set_readout_power, readout_powers_list_dbm
                    )
                }
            )
            pass

    def _set_readout_power(self, power_dbm):
        """
        Changes VNA readout power and keeps track in internal parameter
        storage.

        Parameters
        ----------
        power_dbm : float
            new output power value [dBm]
        """
        # TODO: add bandwidth and averaging change to keep SNR ration at
        #  every readout power
        self._vna[0].set_power(power_dbm)
        self._vna_pars["power"] = power_dbm
        self._mw_src[0].arm_sweep()


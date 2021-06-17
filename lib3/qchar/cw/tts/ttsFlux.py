# Standard library imports
from collections import OrderedDict

# Third party imports
# ------------------

# Local application imports
from .ttsBase import TTSBase


class TTSFlux(TTSBase):
    def sweep_flux(self, flux_values_list):
        if not self._mw_triggered_by_vna:
            # if `mw_src` is not triggered by vna, we should change mw
            # frequency manually from CPU
            self.set_swept_parameters(
                # Note:since python 3.7 dict is ought to remember and preserve
                # insertion order inside dictionary
                **OrderedDict(
                    [
                        (
                            self._flux_format_str,
                            (
                                self._adaptive_setter,
                                flux_values_list
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
                    self._flux_format_str: (
                        self._adaptive_setter, flux_values_list
                    )
                }
            )
            pass


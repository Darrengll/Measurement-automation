# Standard library imports
from collections import OrderedDict

# Third party imports
# ------------------

# Local application imports
from .ttsBase import TTSBase, TTSResultBase


class TTSFlux(TTSBase):
    def sweep_flux(self, flux_values):
        if not self._mw_triggered_by_vna:
            # if `mw_src` is not triggered by vna, we should change mw
            # frequency manually from CPU
            self.set_swept_parameters(
                OrderedDict(
                    [
                        (
                            self._flux_format_str,
                            (self._flux_parameter_setter, flux_values)
                         ),
                        (
                            "Scan Frequency, Hz",
                            (self._mw_src[0].set_frequency, self.scan_frequencies)
                        )
                    ]
                )
            )
        else:
            # if `mw_src` is not triggered by vna, we must set vna scan
            # points to be equal to the length of the
            # `self.scan_frequencies`.
            pass


from lib2.TwoToneSpectroscopy import *
from lib2.fulaut.SpectrumOracle import *
from lib2.ExperimentParameters import TTSRunnerParameters, GlobalParameters
from lib2.fulaut.qubit_spectra import *

from datetime import datetime

from loggingserver import LoggingServer


class TTSRunner:

    def __init__(self, sample_name, qubit_name, res_limits, fit_p0, vna=None,
                 exc_iqvg=None, cur_src=None):

        self._vna = vna
        self._cur_src = cur_src
        self._exc_iqvg = exc_iqvg
        self._sample_name = sample_name
        self._qubit_name = qubit_name
        self._res_limits = res_limits
        self._tts_name = "%s-two-tone" % qubit_name
        self._fit_p0 = fit_p0
        self._logger = LoggingServer.getInstance('fulaut')
        self._which_sweet_spot = GlobalParameters().which_sweet_spot[qubit_name]

        self._vna_parameters = {"freq_limits": self._res_limits,
                                "nop": 1,
                                "power": GlobalParameters().spectroscopy_readout_power,
                                "sweep_type": "LIN"}
        self._vna_parameters.update(TTSRunnerParameters().vna_parameters)

        self._mw_src_parameters = {"power": GlobalParameters().spectroscopy_excitation_power}

        res_freq, g, period, sweet_spot, max_q_freq, d = self._fit_p0

        if self._which_sweet_spot is "bottom":
            center = sweet_spot + period / 2
        else:
            center = sweet_spot

        span = 1/2 * period
        self._currents = linspace(center - span / 2,
                                  center + span / 2,
                                  201)

        min_q_freq = \
            transmon_spectrum(sweet_spot + period / 2,
                              period, sweet_spot, max_q_freq, d)
        min_at_the_scan_edge = \
            transmon_spectrum(sweet_spot + span / 2,
                              period, sweet_spot, max_q_freq, d)

        self._logger.debug(
            "Expected qubit frequency range (from AnticrossingOracle): %.3f to %.3f" % (min_q_freq, max_q_freq))
        mw_limits = (max(4e9, min_at_the_scan_edge), max_q_freq + .25e9)
        self._logger.debug("Two-tone frequency range: %.3f to %.3f" % mw_limits)

        self._mw_src_frequencies = linspace(*mw_limits, 401)

        self._tts_result = None
        self._launch_datetime = datetime.today()

    def run(self):

        # Check if today's spectrum is present

        known_results = \
            MeasurementResult.load(self._sample_name,
                                   self._tts_name,
                                   date=self._launch_datetime.strftime("%b %d %Y"),
                                   return_all=True)

        if known_results is not None:
            self._tts_result = known_results[-1]
        else:
            self._perform_TTS()

        if hasattr(self._tts_result, "_fit_params"):
            self._logger.debug("Using previous two-tone fit: %s" % str(self._tts_result._fit_params))
            return self._tts_result._fit_params

        try:
            so = SpectrumOracle("transmon",
                                self._tts_result,
                                self._fit_p0[2:], plot=True)
            params = period, sweet_spot, max_q_freq, d, alpha = so.launch()

        except:
            self._logger.warn("Two-tone fit failed")
            self._tts_result._name += "_fit-fail"
            self._tts_result.save()
            raise ValueError("Two-tone fit was unsuccessful")
        else:
            self._logger.debug("Two-tone fit: %s" % str(params))
            so.save()
            if known_results is None or not hasattr(self._tts_result, "_fit_params"):
                self._tts_result._fit_params = params
                self._tts_result.save()
            print("\n")

            return params

    def _perform_TTS(self):

        self._TTS = FluxTwoToneSpectroscopy("%s-two-tone" % self._qubit_name,
                                            self._sample_name,
                                            vna=self._vna,
                                            mw_src=self._exc_iqvg,
                                            current_src=self._cur_src)

        self._TTS.set_fixed_parameters(vna=[self._vna_parameters],
                                       mw_src=[self._mw_src_parameters],
                                       sweet_spot_current=float(mean(self._currents)),
                                       adaptive=True)

        self._TTS.set_swept_parameters(self._mw_src_frequencies,
                                       current_values=self._currents)
        self._TTS._measurement_result._unwrap_phase = False

        self._tts_result = self._TTS.launch()

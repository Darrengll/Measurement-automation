from lib2.TwoToneSpectroscopy import *
from lib2.fulaut.SpectrumOracle import *
from lib2.GlobalParameters import GlobalParameters
from lib2.fulaut.qubit_spectra import *

from datetime import datetime

from loggingserver import LoggingServer

class TTSRunner():

    def __init__(self, sample_name, qubit_name, res_limits, fit_p0, vna=None,
                 mw_src=None, cur_src=None, awgs=None):

        self._vna = vna
        self._cur_src = cur_src
        self._mw_src = mw_src
        self._sample_name = sample_name
        self._qubit_name = qubit_name
        self._res_limits = res_limits
        self._tts_name = "%s-two-tone" % qubit_name
        self._fit_p0 = fit_p0
        self._logger = LoggingServer.getInstance('fulaut')
        self._which_sweet_spot = GlobalParameters().which_sweet_spot[qubit_name]

        if awgs is not None:
            self._ro_awg = awgs["ro_awg"]
            self._q_awg = awgs["q_awg"]
            self._open_mixers()
            self._vna_power = GlobalParameters.spectroscopy_readout_power + 20
        else:
            self._vna_power = GlobalParameters.spectroscopy_readout_power

        self._vna_parameters = {"bandwidth": 25,
                                "freq_limits": self._res_limits,
                                "nop": 1,
                                "power": self._vna_power,
                                "averages": 1,
                                "sweep_type": "LIN",
                                "resonator_detection_nop": 501,
                                "resonator_detection_bandwidth":100}

        self._mw_src_parameters = {"power": GlobalParameters.spectroscopy_excitation_power}

        res_freq, g, period, sweet_spot, max_q_freq, d = self._fit_p0

        if self._which_sweet_spot is "bottom":
            center = sweet_spot + period / 2
        else:
            center = sweet_spot

        self._currents = linspace(center - period / 2,
                                  center + period / 2,
                                  201)

        min_q_freq = \
            transmon_spectrum(sweet_spot + period / 2, period, sweet_spot, max_q_freq, d)

        expected_q_freq = min_q_freq \
            if self._which_sweet_spot is "bottom" \
            else max_q_freq

        # if res_freq>expected_q_freq:
        #     mw_limits = (expected_q_freq-1.5e9, res_freq-1e9)
        # else:
        #     mw_limits = (res_freq-0.1e9, expected_q_freq+1e9)
        self._logger.debug("Two-tone frequency limits min: %.3f and max: %.3f"%(min_q_freq, max_q_freq))
        mw_limits = (min_q_freq-2e9, max_q_freq+.25e9)

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
            if known_results is None or not hasattr(self._tts_result, "_fit_params"):
                self._tts_result._fit_params = params
                self._tts_result.save()
            print("\n")

            return params

    def _perform_TTS(self):


        self._TTS = FluxTwoToneSpectroscopy("%s-two-tone" % self._qubit_name,
                                      self._sample_name,
                                      vna=self._vna,
                                      mw_src=self._mw_src,
                                      current_src=self._cur_src)

        self._TTS.set_fixed_parameters(vna = [self._vna_parameters],
                                 mw_src = [self._mw_src_parameters],
                                 sweet_spot_current=float(mean(self._currents)),
                                 adaptive=True)

        self._TTS.set_swept_parameters(self._mw_src_frequencies,
                                 current_values=self._currents)
        self._TTS._measurement_result._unwrap_phase = False

        self._tts_result = self._TTS.launch()

    def _open_mixers(self):
        self._ro_awg.output_continuous_IQ_waves(frequency=0,
                                                amplitudes=(0, 0),
                                                relative_phase=0,
                                                offsets=(1, 1),
                                                waveform_resolution=1)

        self._q_awg.output_continuous_IQ_waves(frequency=0,
                                               amplitudes=(0, 0),
                                               relative_phase=0,
                                               offsets=(1, 1),
                                               waveform_resolution=1)

from lib2.TwoToneSpectroscopy import *
from lib2.fulaut.SpectrumOracle import *
from lib2.fulaut.GlobalParameters import *
from lib2.fulaut.qubit_spectra import *

from datetime import datetime

from lib2.LoggingServer import *

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
        self._logger = LoggingServer.getInstance()
        self._which_sweet_spot = GlobalParameters.which_sweet_spot[qubit_name]

        if awgs is not None:
            self._ro_awg = awgs["ro_awg"]
            self._q_awg = awgs["q_awg"]
            self._open_mixers()
            self._vna_power = -20
        else:
            self._vna_power = -40

        self._vna_parameters = {"bandwidth": 100,
                                "freq_limits": self._res_limits,
                                "nop": 15,
                                "power": self._vna_power,
                                "averages": 1,
                                "sweep_type": "LIN"}

        self._mw_src_parameters = {"power": -15}

        res_freq, g, period, sweet_spot, max_q_freq, d = self._fit_p0

        if self._which_sweet_spot is "bottom":
            center = sweet_spot + period / 2
        else:
            center = sweet_spot

        self._currents = linspace(center - period / 4,
                                  center + period / 4,
                                  101)
        #self._currents = linspace(-2e-5, 7e-5, 201)

        min_q_freq = \
            transmon_spectrum(sweet_spot + period / 2, period, sweet_spot, max_q_freq, d)

        expected_q_freq = min_q_freq \
            if self._which_sweet_spot is "bottom" \
            else max_q_freq

        # if res_freq>expected_q_freq:
        #     mw_limits = (expected_q_freq-1.5e9, res_freq-1e9)
        # else:
        #     mw_limits = (res_freq-0.1e9, expected_q_freq+1e9)

        mw_limits = (5e9, 5.8e9)

        #mw_limits = (expected_q_freq - 1.5e9, expected_q_freq + 0.5e9)

        self._mw_src_frequencies = linspace(*mw_limits, 151)

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

        so = SpectrumOracle("transmon",
                            self._tts_result,
                            self._fit_p0[2:], plot=True)
        params = so.launch()
        self._logger.debug("Two-tone fit: %s" % str(params))
        if known_results is None:
            print("Saving...", end="")
            self._tts_result.save()
        print("\n")

        return params

    def _perform_TTS(self):

        f_res, g, period, sweet_spot, max_q_freq, d = \
            self._fit_p0

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
                                                offsets=(.5, .5),
                                                waveform_resolution=1)

        self._q_awg.output_continuous_IQ_waves(frequency=0,
                                               amplitudes=(0, 0),
                                               relative_phase=0,
                                               offsets=(1, 1),
                                               waveform_resolution=1)

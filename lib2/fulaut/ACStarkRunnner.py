from datetime import datetime
from time import sleep

from loggingserver import LoggingServer
from numpy import linspace, log10, around

from lib2.ExperimentParameters import GlobalParameters, ACSTTSRunnerParameters
from lib2.MeasurementResult import MeasurementResult
from lib2.TwoToneSpectroscopy import AcStarkTwoToneSpectroscopy
from lib2.fulaut import qubit_spectra
from lib2.fulaut.ACStarkOracle import ACStarkOracle


class ACStarkRunner:

    def __init__(self, sample_name, qubit_name, res_limits, spectrum_oracle_fit,
                 vna=None, exc_iqvg=None, cur_src=None, sa=None):

        self._vna = vna
        self._cur_src = cur_src
        self._exc_iqvg = exc_iqvg
        self._sa = sa
        self._sample_name = sample_name
        self._qubit_name = qubit_name
        self._res_limits = res_limits
        self._asts_name = "%s-ac-stark" % qubit_name
        self._spectrum_oracle_fit = spectrum_oracle_fit
        self._launch_datetime = datetime.today()
        self._vna_power = GlobalParameters().readout_power

        self._logger = LoggingServer.getInstance('fulaut')

    def launch(self, current):

        known_results = \
            MeasurementResult.load(self._sample_name,
                                   self._asts_name,
                                   date=self._launch_datetime.strftime("%b %d %Y"),
                                   return_all=True)

        if known_results is not None and not ACSTTSRunnerParameters().rerun:
            self._asts_result = known_results[-1]

        else:
            self._perform_asts(current)


        ACS = ACStarkOracle(self._asts_result, chi=.3e-3, plot=True)
        f_max, vna_power = ACS.launch()
        absolute_power = round(self._find_power(vna_power))

        self._asts_result.save()

        self._logger.debug(
            "Transmon bare if_freq: %.4f GHz, readout power (on SA): %d dBm (%d on VNA)" %
            (f_max / 1e9, absolute_power, vna_power))
        return f_max, absolute_power

    def _find_power(self, vna_power):

        res_freq = self._asts_result.get_context().get_equipment()["vna"][0]["freq_limits"][0]

        vna = self._vna[0]

        vna.set_power(vna_power)
        vna.set_freq_limits(res_freq, res_freq)
        vna.set_nop(1)
        vna.sweep_single()
        sleep(1)

        self._sa.setup_list_sweep([res_freq], [1000])

        self._sa.prepare_for_stb()
        self._sa.sweep_single()
        self._sa.wait_for_stb()
        power = self._sa.get_tracedata()[0]

        self._sa.setup_swept_sa(res_freq, 1e9, nop=1001, rbw=1e4)
        self._sa.set_continuous()
        vna.sweep_hold()

        return power

    def _perform_asts(self, current):
        ASTS = AcStarkTwoToneSpectroscopy("%s-AC-Stark" % self._qubit_name,
                                          self._sample_name,
                                          vna=self._vna,
                                          mw_src=self._exc_iqvg,
                                          current_src=self._cur_src)
        vna_parameters = {"freq_limits": self._res_limits,
                          "nop": 1}
        vna_parameters.update(ACSTTSRunnerParameters().vna_parameters)

        q_freq = qubit_spectra.transmon_spectrum(current, *self._spectrum_oracle_fit[:-1])

        mw_src_frequencies = linspace(q_freq - 150e6, q_freq + 50e6, 301)

        powers = linspace(self._vna_power - 15, self._vna_power + 5, 21)

        mw_src_parameters = {"power": GlobalParameters().excitation_power-10}

        ASTS.set_fixed_parameters(vna=[vna_parameters], mw_src=[mw_src_parameters], current=current)
        ASTS.set_swept_parameters(mw_src_frequencies, powers)

        ASTS._measurement_result._unwrap_phase = False

        self._asts_result = ASTS.launch()

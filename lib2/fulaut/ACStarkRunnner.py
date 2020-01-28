from datetime import datetime
from time import sleep

from loggingserver import LoggingServer
from numpy import linspace, log10, around

from lib2.GlobalParameters import GlobalParameters
from lib2.MeasurementResult import MeasurementResult
from lib2.TwoToneSpectroscopy import AcStarkTwoToneSpectroscopy
from lib2.fulaut import qubit_spectra
from lib2.fulaut.ACStarkOracle import ACStarkOracle


class ACStarkRunner:


    def __init__(self, sample_name, qubit_name, res_limits, spectrum_oracle_fit, vna=None,
                 mw_src=None, cur_src=None, awgs=None, sa = None):

        self._vna = vna
        self._cur_src = cur_src
        self._mw_src = mw_src
        self._sa = sa
        self._sample_name = sample_name
        self._qubit_name = qubit_name
        self._res_limits = res_limits
        self._asts_name = "%s-ac-stark" % qubit_name
        self._spectrum_oracle_fit = spectrum_oracle_fit
        self._launch_datetime = datetime.today()

        self._logger = LoggingServer.getInstance('fulaut')

        if awgs is not None:
            self._ro_awg = awgs["ro_awg"]
            self._q_awg = awgs["q_awg"]
            self._open_mixers()
            self._vna_power = GlobalParameters.spectroscopy_readout_power + 20
        else:
            self._vna_power = GlobalParameters.spectroscopy_readout_power

    def launch(self, current):

        self._open_mixers()

        known_results = \
            MeasurementResult.load(self._sample_name,
                                   self._asts_name,
                                   date=self._launch_datetime.strftime("%b %d %Y"),
                                   return_all=True)

        if known_results is not None:
            self._asts_result = known_results[-1]

        else:
            self._perform_asts(current)

        self._asts_result.save()

        ACS = ACStarkOracle(self._asts_result, chi=1e-3, plot=True)
        f_max, vna_power = ACS.launch()
        absolute_power = self._find_power(vna_power)

        self._logger.debug("Transmon bare frequency: %.4f GHz, readout power (on SA): %d dBm"%(f_max/1e9, absolute_power))
        return f_max, absolute_power


    def _find_power(self, vna_power):

        res_freq = self._asts_result.get_context().get_equipment()["vna"][0]["freq_limits"][0]

        vna = self._vna[0]

        vna.set_power(vna_power)
        vna.set_freq_limits(res_freq, res_freq)
        vna.sweep_continuous()
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
                                          mw_src=self._mw_src,
                                          current_src=self._cur_src)
        vna_parameters = {"bandwidth": 100,
                          "freq_limits": self._res_limits,
                          "nop": 1,
                          "averages": 50,
                          "resonator_detection_bandwidth": 500,
                          "resonator_detection_nop": 201}

        q_freq = qubit_spectra.transmon_spectrum(current, *self._spectrum_oracle_fit[:-1])

        mw_src_frequencies = linspace(q_freq - 150e6, q_freq + 50e6, 301)


        powers = linspace(self._vna_power-15, self._vna_power+5, 21)

        mw_src_parameters = {"power": GlobalParameters.spectroscopy_excitation_power}

        ASTS.set_fixed_parameters(vna=[vna_parameters], mw_src=[mw_src_parameters], current=current)
        ASTS.set_swept_parameters(mw_src_frequencies, powers)

        ASTS._measurement_result._unwrap_phase = True

        self._asts_result = ASTS.launch()


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
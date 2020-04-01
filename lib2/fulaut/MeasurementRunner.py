from lib.iq_mixer_calibration import *
from lib.data_management import *
from lib2.DispersiveHahnEcho import DispersiveHahnEcho
from lib2.fulaut.ACStarkRunnner import ACStarkRunner

from lib2.fulaut.AnticrossingOracle import *
from lib2.fulaut.SpectrumOracle import *
from lib2.fulaut.ResonatorOracle import *
from lib2.fulaut.STSRunner import *
from lib2.fulaut.TTSRunner import *
from lib2.DispersiveRabiOscillations import DispersiveRabiOscillations
from lib2.DispersiveRamsey import DispersiveRamsey
from lib2.DispersiveDecay import DispersiveDecay
from lib2.MeasurementResult import *
from lib2.SingleToneSpectroscopy import *
from lib2.TwoToneSpectroscopy import *
from lib2.fulaut.qubit_spectra import transmon_spectrum
from loggingserver import LoggingServer

from drivers.keysightAWG import *
from drivers.Tektronix_AWG5014 import *
from drivers.IQAWG import *
from drivers.Agilent_EXA import *

from scipy.constants import pi
import pickle

from time import sleep


class MeasurementRunner():

    def __init__(self, sample_name, s_parameter, devs_aliases_map):

        self._logger = LoggingServer.getInstance('fulaut')

        self._sample_name = sample_name
        self._s_parameter = s_parameter
        self._qubit_names = "I II III IV V VI VII VIII".split(" ")
        self._res_limits = {}
        self._sts_runners = {}
        self._sts_fit_params = {}
        self._tts_runners = {}
        self._tts_results = {}
        self._tts_fit_params = {}
        self._exact_qubit_freqs = {}
        self._dro_results = {}
        self._dr_results = {}
        self._dd_results = {}
        self._dhe_results = {}

        self._ramsey_offset = 5e3
        # self._vna = Znb("ZNB")
        self._sa = Agilent_EXA_N9010A("EXA")
        m = Measurement("", "", devs_aliases_map)
        self._vna = m._vna
        # self._sa = m._sa
        self._mw_src = m._mw_src
        self._cur_src = m._cur_src
        self._cur_src[0].set_status(1)

        self._ro_awg = m._ro_awg
        self._q_awg = m._q_awg
        
        self._launch_date = datetime.today()

    def run(self, qubits_to_measure=[0, 1, 2, 3, 4, 5], period_fraction=0):

        self._logger.debug("Started measurement for qubits ##:" + str(qubits_to_measure))

        self._open_only_readout_mixer()

        ro = ResonatorOracle(self._vna, self._s_parameter, 3e6)
        scan_areas = ro.launch()[:]

        for idx, res_limits in enumerate(scan_areas):
            if idx not in qubits_to_measure:
                continue

            qubit_name = self._qubit_names[idx]
            self._res_limits[qubit_name] = res_limits

            if qubit_name not in self._sts_fit_params.keys():
                STSR = STSRunner(self._sample_name,
                                 qubit_name,
                                 mean(res_limits),
                                 vna=self._vna,
                                 cur_src=self._cur_src,
                                 awgs={"q_awg": self._q_awg,
                                       "ro_awg": self._ro_awg})  # {"q_awg": self._q_awg,"ro_awg": self._ro_awg}

                self._sts_runners[qubit_name] = STSR
                self._sts_fit_params[qubit_name], loss = STSR.run()
                self._res_limits[qubit_name] = STSR.get_scan_area()
            # continue

            if qubit_name not in self._tts_fit_params.keys():
                TTSR = TTSRunner(self._sample_name,
                                 qubit_name,
                                 STSR.get_scan_area(),
                                 self._sts_fit_params[qubit_name],
                                 vna=self._vna,
                                 mw_src=self._mw_src,
                                 cur_src=self._cur_src,
                                 awgs={"q_awg": self._q_awg,
                                       "ro_awg": self._ro_awg})
                self._tts_runners[qubit_name] = TTSR
                self._tts_fit_params[qubit_name] = TTSR.run()


            sws_current = self._tts_fit_params[qubit_name][1]
            period = self._tts_fit_params[qubit_name][0]

            ASTSRunner = ACStarkRunner(self._sample_name,
                                       qubit_name,
                                       self._res_limits[qubit_name],
                                       self._tts_fit_params[qubit_name],
                                       vna=self._vna,
                                       mw_src=self._mw_src,
                                       cur_src=self._cur_src,
                                       awgs={"q_awg": self._q_awg,
                                             "ro_awg": self._ro_awg},
                                       sa=self._sa)
            q_freq_sws, power = ASTSRunner.launch(sws_current)


            # period_fraction = float(input("Enter current offset in period fraction: "))
            for period_fraction in linspace(0, 0.25, 10)[0:1]:
                current = sws_current + period_fraction * period
                # q_freq = transmon_spectrum(current, *self._tts_fit_params[qubit_name][:-1])
                q_freq = q_freq_sws

                GlobalParameters.ro_ssb_power[qubit_name] = power

                self._logger.debug("Pulsed measurements at %.3f GHz, %.2e A, "
                                   "%.2e periods away from sws" % (q_freq / 1e9, current, period_fraction))
                self._cur_src[0].set_current(current)

                self._exact_qubit_freqs[qubit_name] = q_freq

                self._ro_cal = self._calibrate_readout(qubit_name)
                self._exc_cal = self._calibrate_excitation(qubit_name)

                self._perform_Rabi_oscillations(qubit_name)

                self._max_ramsey_delay = .5e3
                self._ramsey_offset = 10e6
                self._ramsey_nop = 201

                self._perform_Ramsey_oscillations(qubit_name)
                detected_ramsey_freq, error = \
                    self._dr_results[qubit_name].get_ramsey_frequency()
                frequency_error = self._ramsey_offset - detected_ramsey_freq
                self._exact_qubit_freqs[qubit_name] -= frequency_error
                self._ramsey_offset = 1e6
                self._max_ramsey_delay = 10e3
                self._ramsey_nop = 201

                self._perform_Rabi_oscillations(qubit_name, True)
                self._perform_Ramsey_oscillations(qubit_name, True)

                self._max_echo_delay = 100e3
                self._echo_nop = 21
                self._perform_hahn_echo(qubit_name, True)

                self._decay_nop = 51
                self._decay_max_delay = 200e3
                self._decay_readout_duration = 20e3
                self._perform_decay(qubit_name, True)


    def _perform_decay(self, qubit_name, save=False):

        DD = DispersiveDecay("%s-decay" % qubit_name,
                             self._sample_name,
                             vna=self._vna,
                             ro_awg=[self._ro_awg],
                             q_awg=[self._q_awg],
                             q_lo=self._mw_src)

        vna_parameters = {"bandwidth": 10,
                          "freq_limits": self._res_limits[qubit_name],
                          "nop": 100,
                          "averages": 1,
                          "res_find_nop": 401}

        readout_delays = linspace(0, self._decay_max_delay, self._decay_nop)
        exc_frequency = self._exact_qubit_freqs[qubit_name]
        pi_pulse_duration = \
            self._dro_results[qubit_name].get_pi_pulse_duration() * 1e3

        pulse_sequence_parameters = {"awg_trigger_reaction_delay": 0,
                                     "readout_duration": self._decay_readout_duration,
                                     "repetition_period": self._decay_max_delay+self._decay_readout_duration + 10e3,
                                     "pi_pulse_duration": pi_pulse_duration}

        ro_awg_params = {"calibration": self._ro_cal}
        q_awg_params = {"calibration": self._exc_cal}

        q_lo_params = {'power': self._exc_cal.get_radiation_parameters()["lo_power"],
                       'frequency': exc_frequency + self._exc_cal.get_radiation_parameters()["if_frequency"]}

        DD.set_fixed_parameters(pulse_sequence_parameters,
                                vna=[vna_parameters],
                                ro_awg=[ro_awg_params],
                                q_awg=[q_awg_params],
                                q_lo=[q_lo_params])
        DD.set_swept_parameters(readout_delays)
        MeasurementResult.close_figure_by_window_name("Resonator fit")
        dd_result = DD.launch()
        self._dd_results[qubit_name] = dd_result
        if save:
            dd_result.save()

    def _perform_Ramsey_oscillations(self, qubit_name, save=False):

        DR = DispersiveRamsey("%s-ramsey" % qubit_name,
                              self._sample_name,
                              vna=self._vna,
                              ro_awg=[self._ro_awg],
                              q_awg=[self._q_awg],
                              q_lo=self._mw_src)

        vna_parameters = {"bandwidth": 10,
                          "freq_limits": self._res_limits[qubit_name],
                          "nop": 10,
                          "averages": 1,
                          "res_find_nop": 401}

        ramsey_delays = linspace(0, self._max_ramsey_delay, self._ramsey_nop)
        exc_frequency = self._exact_qubit_freqs[qubit_name] - self._ramsey_offset
        pi_pulse_duration = \
            self._dro_results[qubit_name].get_pi_pulse_duration() * 1e3

        pulse_sequence_parameters = \
            {"awg_trigger_reaction_delay": 0,
             "readout_duration": 15e3,
             "repetition_period": self._max_ramsey_delay + 100e3,
             "half_pi_pulse_duration": pi_pulse_duration / 2}

        ro_awg_params = {"calibration": self._ro_cal}
        q_awg_params = {"calibration": self._exc_cal}

        mw_src_params = {'power': self._exc_cal.get_radiation_parameters()["lo_power"],
                         'frequency': exc_frequency + self._exc_cal.get_radiation_parameters()["if_frequency"]}

        DR.set_fixed_parameters(pulse_sequence_parameters,
                                vna=[vna_parameters],
                                ro_awg=[ro_awg_params],
                                q_awg=[q_awg_params],
                                q_lo=[mw_src_params])
        DR.set_swept_parameters(ramsey_delays)
        MeasurementResult.close_figure_by_window_name("Resonator fit")

        dr_result = DR.launch()
        self._dr_results[qubit_name] = dr_result
        if save:
            dr_result.save()

    def _perform_hahn_echo(self, qubit_name, save=False):

        DHE = DispersiveHahnEcho("%s-echo" % qubit_name,
                              self._sample_name,
                              vna=self._vna,
                              ro_awg=[self._ro_awg],
                              q_awg=[self._q_awg],
                              q_lo=self._mw_src)

        vna_parameters = {"bandwidth": 10,
                          "freq_limits": self._res_limits[qubit_name],
                          "nop": 50,
                          "averages": 1,
                          "res_find_nop": 401}

        echo_delays = linspace(0, self._max_echo_delay, self._echo_nop)
        exc_frequency = self._exact_qubit_freqs[qubit_name]
        pi_pulse_duration = \
            self._dro_results[qubit_name].get_pi_pulse_duration() * 1e3

        pulse_sequence_parameters = \
            {"awg_trigger_reaction_delay": 0,
             "readout_duration": 15e3,
             "repetition_period": self._max_echo_delay + 100e3,
             "half_pi_pulse_duration": pi_pulse_duration / 2}

        ro_awg_params = {"calibration": self._ro_cal}
        q_awg_params = {"calibration": self._exc_cal}

        mw_src_params = {'power': self._exc_cal.get_radiation_parameters()["lo_power"],
                         'frequency': exc_frequency + self._exc_cal.get_radiation_parameters()["if_frequency"]}

        DHE.set_fixed_parameters(pulse_sequence_parameters,
                                vna=[vna_parameters],
                                ro_awg=[ro_awg_params],
                                q_awg=[q_awg_params],
                                q_lo=[mw_src_params])
        DHE.set_swept_parameters(echo_delays)
        MeasurementResult.close_figure_by_window_name("Resonator fit")

        dhe_result = DHE.launch()
        self._dhe_results[qubit_name] = dhe_result
        if save:
            dhe_result.save()

    def _perform_Rabi_oscillations(self, qubit_name, save=False):

        DRO = DispersiveRabiOscillations("%s-rabi" % qubit_name,
                                         self._sample_name,
                                         vna=self._vna,
                                         q_lo=self._mw_src,
                                         q_awg=[self._q_awg],
                                         ro_awg=[self._ro_awg],
                                         plot_update_interval=0.5)

        vna_parameters = {"bandwidth": 20,
                          "freq_limits": self._res_limits[qubit_name],
                          "nop": 10,
                          "averages": 1,
                          "res_find_nop": 401}

        exc_frequency = self._exact_qubit_freqs[qubit_name]
        excitation_durations = linspace(0, 500, 201)
        rabi_sequence_parameters = {"awg_trigger_reaction_delay": 0,
                                    "excitation_amplitude": 1,
                                    "readout_duration": 15000,
                                    "repetition_period": 100000}

        ro_awg_params = {"calibration": self._ro_cal}
        q_awg_params = {"calibration": self._exc_cal}

        mw_src_params = {'power': self._exc_cal.get_radiation_parameters()["lo_power"],
                         'frequency': exc_frequency + self._exc_cal.get_radiation_parameters()["if_frequency"]}

        DRO.set_fixed_parameters(rabi_sequence_parameters,
                                 vna=[vna_parameters],
                                 ro_awg=[ro_awg_params],
                                 q_awg=[q_awg_params],
                                 q_lo=[mw_src_params])
        DRO.set_swept_parameters(excitation_durations)
        DRO.set_ult_calib(False)
        MeasurementResult.close_figure_by_window_name("Resonator fit")

        dro_result = DRO.launch()
        self._dro_results[qubit_name] = dro_result
        if save:
            dro_result.save()

    def _set_vna_to_ro_lo(self):
        ro_lo = self._vna[0]
        ro_lo.set_frequency = lambda x: ro_lo.set_freq_limits(x, x)
        ro_lo.set_output_state = lambda x: x
        ro_lo.set_nop(1)
        ro_lo.sweep_hold()
        ro_lo.sweep_single()

    def _calibrate_readout(self, qubit_name):

        ro_resonator_frequency = self._sts_fit_params[qubit_name][0]
        ro_resonator_frequency = round(ro_resonator_frequency / 1e9, 2) * 1e9
        if_frequency = 0e6
        lo_power = -10
        ssb_power = GlobalParameters.ro_ssb_power[qubit_name]
        waveform_resolution = 1

        db = load_IQMX_calibration_database("CHGRO", 0)
        if db is not None:
            key = frozenset(dict(lo_power=lo_power,
                                      ssb_power=ssb_power,
                                      lo_frequency=ro_resonator_frequency,
                                      if_frequency=if_frequency,
                                      waveform_resolution=waveform_resolution) \
                                 .items())
            self._logger.debug("Requesting readout mixer cal: %s"%str(key))
            ro_cal = \
                db.get(key)
            if ro_cal is not None and not GlobalParameters.recalibrate_mixers[qubit_name]:
                return ro_cal

        self._logger.debug("Did not find calibration! Calibrating...")

        self._set_vna_to_ro_lo()

        ig = {"dc_offsets": (0.1, -0.1), "dc_offset_open": 0.3}
        cal = IQCalibrator(self._ro_awg,
                           self._sa,
                           self._vna[0],
                           "CHGRO",
                           0,
                           sidebands_to_suppress=1)

        ro_cal = cal.calibrate(lo_frequency=ro_resonator_frequency,
                               if_frequency=if_frequency,
                               lo_power=lo_power,
                               ssb_power=ssb_power,
                               waveform_resolution=1,
                               iterations=3,
                               minimize_iterlimit=20,
                               sa_res_bandwidth=100,
                               initial_guess=ig)
        save_IQMX_calibration(ro_cal)
        self._logger.debug("Calibration result: %s"%str(ro_cal))
        return ro_cal

    def _calibrate_excitation(self, qubit_name):
        qubit_frequency = self._tts_fit_params[qubit_name][2]
        qubit_frequency = round(qubit_frequency / 1e9 / 5, 2) * 5e9  # to 50 MHz
        if_frequency = 100e6
        lo_power = 14
        ssb_power = GlobalParameters.exc_ssb_power[qubit_name]
        waveform_resolution = 1

        db = load_IQMX_calibration_database("CHGQ", 0)
        exc_cal = \
            db.get(frozenset(dict(lo_power=lo_power,
                                  ssb_power=ssb_power,
                                  lo_frequency=qubit_frequency + if_frequency,
                                  if_frequency=if_frequency,
                                  waveform_resolution=waveform_resolution) \
                             .items()))
        if exc_cal is not None and not GlobalParameters.recalibrate_mixers[qubit_name]:
            return exc_cal

        ig = {"dc_offsets": (-0.017, -0.04),
              "if_amplitudes": (.1, .1),
              "if_phase": -pi * 0.54}
        cal = IQCalibrator(self._q_awg,
                           self._sa,
                           self._mw_src[0],
                           "CHGQ",
                           0,
                           sidebands_to_suppress=6)

        exc_cal = cal.calibrate(lo_frequency=qubit_frequency + if_frequency,
                                if_frequency=if_frequency,
                                lo_power=lo_power,
                                ssb_power=ssb_power,
                                waveform_resolution=waveform_resolution,
                                iterations=2,
                                minimize_iterlimit=20,
                                sa_res_bandwidth=500,
                                initial_guess=ig)
        save_IQMX_calibration(exc_cal)
        return exc_cal

    def _open_only_readout_mixer(self):
        self._ro_awg.output_continuous_IQ_waves(frequency=0, amplitudes=(0, 0),
                                                relative_phase=0, offsets=(1, 1),
                                                waveform_resolution=1)
        self._q_awg.output_continuous_IQ_waves(frequency=0, amplitudes=(0, 0),
                                               relative_phase=0, offsets=(0, 0),
                                               waveform_resolution=1)

    def _open_mixers(self):
        self._ro_awg.output_continuous_IQ_waves(frequency=0, amplitudes=(0, 0),
                                                relative_phase=0, offsets=(1, 1),
                                                waveform_resolution=1)
        self._q_awg.output_continuous_IQ_waves(frequency=0, amplitudes=(0, 0),
                                               relative_phase=0, offsets=(1, 1),
                                               waveform_resolution=1)

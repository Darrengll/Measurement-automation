from enum import Enum, auto

from lib2.DispersivePiPulseAmplitudeCalibration import DispersivePiPulseAmplitudeCalibrationResult
from lib2.Measurement import Measurement
from lib2.DispersiveRabiOscillations import DispersiveRabiOscillationsResult
import numpy as np
from importlib import reload
from drivers.keysightM3202A import KeysightM3202A
import inspect
import lib2.IQPulseSequence
from lib2.VNATimeResolvedDispersiveMeasurement1D import VNATimeResolvedDispersiveMeasurement1DResult
from drivers.Spectrum_m4x import SPCM

reload(lib2.IQPulseSequence)
from lib2.IQPulseSequence import IQPulseBuilder


def _default_args2dict():
    print(inspect.stack()[0])


class DigitizerTimeResolvedDirectMeasurement(Measurement):

    def __init__(self, name, sample_name, devs_aliases_map, plot_update_interval=1):

        # mandatory names for devices:
        self._q_iqawg = None
        self._q_lo = None
        self._dig: list[SPCM] = None
        super().__init__(name, sample_name, devs_aliases_map,
                         plot_update_interval=plot_update_interval)
        self._sequence_generator = None
        self._basis = None
        self._ult_calib = False
        self._adc_parameters = None
        self._n_samples_to_drop_by_dig_delay = 0
        self._n_samples_to_drop_in_end = 0
        self._pulse_sequence_parameters = \
            {"modulating_window": "rectangular", "excitation_amplitude": 1,
             "z_smoothing_coefficient": 0}

        # for debug purposes
        self.dataI = []
        self.dataQ = []

    def set_fixed_parameters(self, pulse_sequence_parameters, freq_limits = (0,50e6),
                             q_lo_params=[], q_iqawg_params=[], dig_params=[]):
        """
        :param dev_params:
            Minimum expected keys and elements expected in each:
                'vna': 0
                'q_iqawg': 0
                'ro_awg': 0

        Parameters
        ----------
        pulse_sequence_parameters
        """
        q_lo_params[0]["power"] = q_iqawg_params[0]["calibration"] \
            .get_radiation_parameters()["lo_power"]

        # TODO check carefully. All single device functions should be deleted?
        self._pulse_sequence_parameters.update(pulse_sequence_parameters)

        # Как это вообще работало, если в ContextBase нет такого метода get_pulse_sequence_parameters()?
        # self._measurement_result.get_context().get_pulse_sequence_parameters().update(pulse_sequence_parameters)
        self._measurement_result._iter = 0

        # convert dict with parameters into form that is demanded by 'super().set_fixed_parameters()'
        dev_params = {"q_lo": q_lo_params,
                      "q_iqawg": q_iqawg_params,
                      "dig": dig_params}
        super().set_fixed_parameters(**dev_params)
        self._measurement_result.get_context().update({
            "calibration_results": self._q_iqawg[0]._calibration.get_optimization_results(),
            "radiation_parameters": self._q_iqawg[0]._calibration.get_radiation_parameters(),
            "pulse_sequence_parameters": pulse_sequence_parameters
        })

    def set_basis(self, basis):
        d_real, d_imag = self._calculate_basis_complex_amplitudes(basis)
        relation = d_real / d_imag
        if relation > 5:
            # Imag quadrature is not oscillating, ignore it by making imag
            # distance equal to ten real distances so that new normalized values
            # obtained via that component would be small
            ground_state = np.real(basis[0]) + 1j * np.imag(basis[0])
            excited_state = np.real(basis[1]) + 1j * (np.imag(basis[0]) + 10 * d_real)
            basis = (ground_state, excited_state)
        elif relation < 0.2:
            # Real quadrature is not oscillating, ignore it
            ground_state = np.real(basis[0]) + 1j * np.imag(basis[0])
            excited_state = np.real(basis[0]) + 10 * d_imag + 1j * np.imag(basis[1])
            basis = (ground_state, excited_state)

        self._basis = basis

    @staticmethod
    def _calculate_basis_complex_amplitudes(self, basis):
        d_real = abs(np.real(basis[0] - basis[1]))
        d_imag = abs(np.imag(basis[0] - basis[1]))
        return d_real, d_imag

    def set_ult_calib(self, value=False):
        self._ult_calib = value

    def _single_measurement(self):
        dig = self._dig[0]
        dig_data = dig.measure(dig._bufsize)
        # convertion to mV is according to
        # https://spectrum-instrumentation.com/sites/default/files/download/m4i_m4x_22xx_manual_english.pdf
        # p.81
        dig_data = dig_data.astype(float) / dig.n_avg / 128 * dig.ch_amplitude

        data_i = dig_data[0::2]
        data_i = data_i.reshape(dig.n_seg, round(dig_data.shape[0] / 2 / dig.n_seg))
        data_i = data_i[:, self._n_samples_to_drop_by_dig_delay: -self._n_samples_to_drop_in_end]
        data_i = data_i.flatten()

        data_q = dig_data[1::2]
        data_q = data_q.reshape(dig.n_seg, round(dig_data.shape[0] / 2 / dig.n_seg))
        data_q = data_q[:, self._n_samples_to_drop_by_dig_delay: -self._n_samples_to_drop_in_end]
        data_q = data_q.flatten()

        freq = np.fft.fftfreq(len(data_i), d=1/self._dig[0].get_sample_rate())
        freq = np.fft.fftshift(freq)
        signal = np.fft.fftshift(np.fft.fft(data_i + 1j * data_q))
        idx = np.searchsorted(freq, -self._q_iqawg[0]._calibration._if_frequency)
        IQ = signal[idx] / len(data_i)

        # save full data in case of more detailed investigation
        self.dataI.append(data_i)
        self.dataQ.append(data_q)
        return IQ

    def _recording_iteration(self):
        if self._ult_calib:
            fg = self._single_measurement()
            self._output_zero_sequence()
            bg = self._single_measurement()
            mean_data = fg - bg
            # print(fg, bg, mean_data).
        else:
            mean_data = self._single_measurement()

        if self._basis is None:
            return mean_data
        else:
            basis = self._basis
            p_r = (np.real(mean_data) - np.real(basis[0])) / (np.real(basis[1]) - np.real(basis[0]))
            p_i = (np.imag(mean_data) - np.imag(basis[0])) / (np.imag(basis[1]) - np.imag(basis[0]))
            return p_r + 1j * p_i

    def _output_zero_sequence(self):
        prescaler = 0
        fs = KeysightM3202A.calc_sampling_rate(prescaler)
        pulses_period = self._pulse_sequence_parameters["repetition_period"]  # ns
        M = int(fs * pulses_period * 1e-9)
        wf = np.zeros(M)
        self._q_iqawg[0].output_modulated_IQ_waves(wf, prescaler)

    def _output_pulse_sequence(self, zero=False):
        # update a trigger delay of the digitizer
        dig = self._dig[0]
        timedelay = self._pulse_sequence_parameters["start_delay"] + \
                    self._pulse_sequence_parameters["excitation_duration"] + \
                    self._pulse_sequence_parameters["digitizer_delay"]
        dig.calc_and_set_trigger_delay(timedelay, include_pretrigger=True)
        self._n_samples_to_drop_by_dig_delay = dig.get_how_many_samples_to_drop_in_front()

        dig.calc_and_set_segment_size(extra=self._n_samples_to_drop_by_dig_delay)
        dig.setup_averaging_mode()
        self._n_samples_to_drop_in_end = dig.get_how_many_samples_to_drop_in_end()

        q_pbs = [q_iqawg.get_pulse_builder() for q_iqawg in self._q_iqawg]

        # TODO: 'and (self._q_z_awg[0] is not None)'  hotfix by Shamil (below)
        # I intend to declare all possible device attributes of the measurement class in it's child class definitions.
        # So hasattr(self, "_q_z_awg") is True
        # due to the fact that I had declared this parameter and initialized it with "[None]" in RabiFromFrequency.py
        if hasattr(self, '_q_z_awg') and (self._q_z_awg[0] is not None):
            q_z_pbs = [q_z_awg.get_pulse_builder() for q_z_awg in self._q_z_awg]
        else:
            q_z_pbs = [None]

        pbs = {'q_pbs': q_pbs,
               'q_z_pbs': q_z_pbs}

        if not zero:
            seqs = self._sequence_generator(self._pulse_sequence_parameters, **pbs)
        else:
            seqs = self._sequence_generator(self._pulse_sequence_parameters, **pbs)

        global global_seq
        global_seq = seqs["q_seqs"][0]

        for (seq, dev) in zip(seqs['q_seqs'], self._q_iqawg):
            dev.output_pulse_sequence(seq)
        if 'q_z_seqs' in seqs.keys():
            for (seq, dev) in zip(seqs['q_z_seqs'], self._q_z_awg):
                dev.output_pulse_sequence(seq, asynchronous=False)


class DigitizerDirectRabi(DigitizerTimeResolvedDirectMeasurement):
    def __init__(self, name, sample_name, plot_update_interval=1,
                 q_lo=[], q_iqawg=[], dig=[]):
        devs_aliases_map = {"q_lo": q_lo,
                            "q_iqawg": q_iqawg,
                            "dig": dig}
        super().__init__(name, sample_name, devs_aliases_map, plot_update_interval)
        self._measurement_result = DigitizerRabiResult(name, sample_name)
        self._sequence_generator = IQPulseBuilder.build_direct_rabi_sequences

    def sweep_excitation_durations(self, excitation_durations):
        super().set_swept_parameters(**{"excitation duration": (self._set_duration, excitation_durations)})

    def _set_duration(self, excitation):
        self._pulse_sequence_parameters["excitation_duration"] = excitation
        self._output_pulse_sequence()


class DigitizerRabiResult(VNATimeResolvedDispersiveMeasurement1DResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._x_axis_units = "ns"

    def _model(self, t, A_r, A_i, T_R, Omega_R, offset_r, offset_i, phase1, phase2):
        return -(A_r * np.cos(Omega_R * t + phase1) + 1j * A_i * np.cos(Omega_R * t + phase2)) * np.exp(-1 / T_R * t)\
               + offset_r + 1j * offset_i

    def _generate_fit_arguments(self, x, data):
        amp_r, amp_i = np.ptp(np.real(data))/2, np.ptp(np.imag(data))/2
        if np.abs(np.max(np.real(data)) - np.real(data[0])) < np.abs(np.real(data[0]) - np.min(np.real(data))):
            amp_r = -amp_r
        if np.abs(np.max(np.imag(data)) - np.imag(data[0])) < np.abs(np.imag(data[0]) - np.min(np.imag(data))):
            amp_i = -amp_i
        offset_r, offset_i = np.max(np.real(data)) - np.abs(amp_r), np.max(np.imag(data)) - np.abs(amp_i)

        time_step = x[1] - x[0]
        max_frequency = 1 / time_step / 10
        min_frequency = 1e-4
        frequency = np.random.random(1) * (max_frequency - min_frequency) + min_frequency
        p0 = [amp_r, amp_i, 1000, frequency * 2 * np.pi, offset_r, offset_i, 0, 0]

        bounds = ([-np.abs(amp_r) * 1.5, -np.abs(amp_i) * 1.5, 100, min_frequency * 2 * np.pi, -10, -10, -np.pi, -np.pi],
                  [np.abs(amp_r) * 1.5, np.abs(amp_i) * 1.5, 100000, max_frequency * 2 * np.pi, 10, 10, np.pi, np.pi])
        return p0, bounds

    def _generate_annotation_string(self, opt_params, err):
        return f"$T_R={opt_params[2]*1e-3:.2f}\pm {err[2]*1e-3:.2f}~\mu$s\n" \
               f"$\Omega_R/2\pi={opt_params[3] * 1e3 / 2 / np.pi:.2f}\pm {err[3] * 1e3 / 2 / np.pi:.2f}$ MHz\n" \
               f"$\Delta\phi={np.mod(opt_params[6] - opt_params[7], 2 * np.pi) - np.pi:.2f}$ rad"

    def _prepare_data_for_plot(self, data):
        return data[self._parameter_names[0]], data["data"]

    def get_pi_pulse_duration(self):
        return np.pi / self._fit_params[3] # ns

    def get_rabi_decay(self):
        return (self._fit_params[2] * 1e-3, self._fit_errors[2] * 1e-3)

    def get_rabi_frequency(self):
        return (self._fit_params[3] * 1e-3, self._fit_errors[3] * 1e-3)

    def get_basis(self):
        fit = self._fit_params
        A_r, A_i, offset_r, offset_i = fit[0], fit[1], fit[-4], fit[-3]
        ground_state = -A_r+offset_r+1j*(-A_i+offset_i)
        excited_state = A_r+offset_r+1j*(A_i+offset_i)
        return np.array((ground_state, excited_state))

    def get_betas(self):
        return [self._fit_params[0] + 1j*self._fit_params[1],  # beta_II
                self._fit_params[4] + 1j * self._fit_params[5]]  # beta_ZI or beta IZ depending on the qubit number in a qubit pair


class DigitizerDirectRabiAmplitudeCalibration(DigitizerTimeResolvedDirectMeasurement):
    def __init__(self, name, sample_name, plot_update_interval=1,
                 q_lo=[], q_iqawg=[], dig=[]):
        devs_aliases_map = {"q_lo": q_lo,
                            "q_iqawg": q_iqawg,
                            "dig": dig}
        super().__init__(name, sample_name, devs_aliases_map, plot_update_interval)
        self._measurement_result = DispersivePiPulseAmplitudeCalibrationResult(name, sample_name)
        self._sequence_generator = IQPulseBuilder.build_direct_rabi_sequences

    def sweep_amplitude(self, amplitudes):
        super().set_swept_parameters(**{"amplitude": (self._set_amplitude, amplitudes)})

    def _set_amplitude(self, amplitude):
        self._pulse_sequence_parameters["excitation_amplitude"] = amplitude
        self._output_pulse_sequence()


class DispersivePiPulseAmplitudeCalibrationResult(VNATimeResolvedDispersiveMeasurement1DResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._x_axis_units = "ratio"

    def _model(self, amplitude, A_r, A_i, pi_amplitude, offset_r, offset_i, phase_r, phase_i):
        return -(A_r * np.cos(np.pi * amplitude / pi_amplitude + phase_r)
                 + 1j * A_i * np.cos(np.pi * amplitude / pi_amplitude + phase_i)) + (offset_r + 1j * offset_i)

    def _generate_fit_arguments(self, x, data):
        amp_r, amp_i = np.ptp(np.real(data)) / 2, np.ptp(np.imag(data)) / 2
        if np.abs(np.max(np.real(data)) - np.real(data[0])) < np.abs(np.real(data[0]) - np.min(np.real(data))):
            amp_r = -amp_r
        if np.abs(np.max(np.imag(data)) - np.imag(data[0])) < np.abs(np.imag(data[0]) - np.min(np.imag(data))):
            amp_i = -amp_i
        offset_r, offset_i = np.max(np.real(data)) - np.abs(amp_r), np.max(np.imag(data)) - np.abs(amp_i)
        amp_step = x[1] - x[0]
        min_pi_pulse_amp = amp_step * 2 * 5
        max_pi_pulse_amp = (x[-1] - x[0]) * 2 * 10
        pi_pulse_amp = np.random.random(1) * (max_pi_pulse_amp - min_pi_pulse_amp) + min_pi_pulse_amp
        bounds = ([-np.abs(amp_r)*1.5,  -np.abs(amp_i)*1.5, min_pi_pulse_amp,   -10, -10, -np.pi, -np.pi],
                  [np.abs(amp_r)*1.5,   np.abs(amp_i)*1.5,  max_pi_pulse_amp,   10, 10, np.pi, np.pi])
        p0 = [amp_r, amp_i, pi_pulse_amp, offset_r, offset_i, 0, 0]
        return p0, bounds
    #
    # def _prepare_data_for_plot(self, data):
    #     return data["amplitude multiplier"], data["data"]

    def _generate_annotation_string(self, opt_params, err):
        return f"$(\pi) = {opt_params[2]:.2f} \pm {err[2]:.2f}$ a.u.\n" \
               f"$\Delta\phi={np.mod(opt_params[5] - opt_params[6], 2 * np.pi) - np.pi:.2f}$ rad"

    def get_pi_pulse_amplitude(self):
        return self._fit_params[2]

    def _prepare_data_for_plot(self, data):
        return data[self._parameter_names[0]], data["data"]

    def get_basis(self):
        fit = self._fit_params
        A_r, A_i, offset_r, offset_i = fit[0], fit[1], fit[-4], fit[-3]
        ground_state = -A_r + offset_r + 1j * (-A_i + offset_i)
        excited_state = A_r + offset_r + 1j * (A_i + offset_i)
        return np.array((ground_state, excited_state))

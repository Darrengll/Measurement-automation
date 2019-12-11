from lib2.Measurement import Measurement
import numpy as np
from importlib import reload
from drivers.keysightM3202A import KeysightM3202A
import inspect
import lib2.IQPulseSequence

reload(lib2.IQPulseSequence)
from lib2.IQPulseSequence import IQPulseBuilder

class MollowTripletMeasurement(Measurement):

    def __init__(self, name, sample_name, devs_aliases_map, plot_update_interval=1):

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
        self._fourier_method = FOURIER_METHOD.FFT

        self._full_data = []

    def set_fixed_parameters(self, pulse_sequence_parameters, **dev_params):
        """
        :param dev_params:
            Minimum expected keys and elements expected in each:
                'vna': 0
                'q_awg': 0
                'ro_awg': 0
        """
        # TODO check carefully. All single device functions should be deleted?
        self._pulse_sequence_parameters.update(pulse_sequence_parameters)

        # Как это вообще работало, если в ContextBase нет такого метода get_pulse_sequence_parameters()?
        # self._measurement_result.get_context().get_pulse_sequence_parameters().update(pulse_sequence_parameters)
        self._measurement_result._iter = 0

        super().set_fixed_parameters(**dev_params)
        self._measurement_result.get_context().update(
            {"calibration_results": self._q_awg[0]._calibration.get_optimization_results(),
             "radiation_parameters": self._q_awg[0]._calibration.get_radiation_parameters()}
        )
        # self._q_awg[0].setup_AM_and_carrier_from_calibration()

    def set_fourier_method(self, method: FOURIER_METHOD):
        self._fourier_method = method

    def get_fourier_method(self):
        return self._fourier_method

    def preset_Fourier_from_IQ(self, freq, segment_length, sample_rate):
        w = 2 * np.pi * freq
        t_total = segment_length / sample_rate
        t = np.linspace(0, t_total, segment_length)
        self._sin_t = np.sin(w * t)
        self._cos_t = np.cos(w * t)

    def get_Fourier_from_IQ(self, dataI, dataQ):
        """
        Computes a Fourier transform at a single freqiency from a complex signal z(t) = I(t) + 1j * Q(t)
        Took from 'Sank 2014' dissertation.

        Parameters
        ----------
        dataI: np.array, mV
        dataQ: np.array, mV
        freq: float, Hz

        Returns
        -------
            I + 1j * Q
        """
        I = np.sum(dataI * self._cos_t + dataQ * self._sin_t)
        Q = np.sum(dataQ * self._cos_t - dataI * self._sin_t)
        return (I + 1j * Q) / len(dataI)

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
        dig_data = self._dig[0].measure(self._dig[0]._bufsize)[2 * self._n_samples_to_drop_by_dig_delay:-2 * self._n_samples_to_drop_in_end]
        dig_data = dig_data * self._dig[0].ch_amplitude / 128 / self._dig[0].n_avg
        dataI = dig_data[0::2]
        dataQ = dig_data[1::2]

        self._full_data.append(dig_data)  # save full data in case of more detailed investigation

        IQ = 0 + 1j * 0
        if self._fourier_method is FOURIER_METHOD.SANK:
            self.preset_Fourier_from_IQ(self._q_awg[0]._calibration._if_frequency, len(dataI),
                                        self._dig[0].get_sample_rate())
            IQ = self.get_Fourier_from_IQ(dataI, dataQ)
        elif self._fourier_method is FOURIER_METHOD.FFT:
            freq = np.fft.fftfreq(len(dataI)) * self._dig[0].get_sample_rate()
            freq = np.fft.fftshift(freq)
            signal = np.fft.fftshift(np.fft.fft(dataI + 1j * dataQ))
            Ifft = np.real(signal)
            Qfft = np.imag(signal)
            idx = np.searchsorted(freq, -self._q_awg[0]._calibration._if_frequency)
            IQ = (Ifft + 1j * Qfft)[idx] / len(dataI)
        return IQ

    def _recording_iteration(self):

        if self._ult_calib:
            fg = self._single_measurement()
            self._output_zero_sequence()
            bg = self._single_measurement()
            mean_data = fg - bg
            # print(fg, bg, mean_data)
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
        self._q_awg[0].output_modulated_IQ_waves(wf, prescaler)

    def _output_pulse_sequence(self, zero=False):
        q_pbs = [q_awg.get_pulse_builder() for q_awg in self._q_awg]

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

        for (seq, dev) in zip(seqs['q_seqs'], self._q_awg):
            dev.output_pulse_sequence(seq)
        if 'q_z_seqs' in seqs.keys():
            for (seq, dev) in zip(seqs['q_z_seqs'], self._q_z_awg):
                dev.output_pulse_sequence(seq, asynchronous=False)
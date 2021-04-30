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
from drivers.Spectrum_m4x import SPCM, SPCM_MODE, SPCM_TRIGGER
from drivers.IQAWG import IQAWG
from drivers.E8257D import MXG
from drivers.Yokogawa_GS200 import Yokogawa_GS210

from typing import Dict, Union

reload(lib2.IQPulseSequence)
from lib2.IQPulseSequence import IQPulseBuilder


def _default_args2dict():
    print(inspect.stack()[0])


class DigitizerTimeResolvedDirectMeasurement(Measurement):

    def __init__(self, name, sample_name, devs_aliases_map,
                 plot_update_interval=1, save_traces=False):

        # mandatory names for devices:
        self._q_iqawg: list[IQAWG] = None
        self._q_lo: list[MXG] = None
        self._dig: list[SPCM] = None
        self._src: list[Yokogawa_GS210] = None
        super().__init__(name, sample_name, devs_aliases_map,
                         plot_update_interval=plot_update_interval)
        self._sequence_generator = None
        self._basis = None
        self._ult_calib = False
        self._adc_parameters = None
        self._n_samples_to_drop_by_delay = 0
        self._n_samples_to_drop_in_end = 0
        self._pulse_sequence_parameters: Dict[Union[str, int, float]] = \
            {"modulating_window": "rectangular", "excitation_amplitude": 1,
             "z_smoothing_coefficient": 0}
        self._down_conversion_calibration = None

        # ADC trace fourier component if_freq [Hz]
        self._downconv_freq = None

        ''' DEBUG '''
        # if 'True' all traces will be saved in 'dataI' and 'dataQ' for
        # further manual investigation
        self._save_traces = save_traces
        self.dataIQ = []  # `dataI + 1j*dataQ` traces

        # measurement result
        self._init_measurement_result()

    def set_fixed_parameters(
            self,
            pulse_sequence_parameters,
            freq_limits=(0,50e6), down_conversion_calibration=None,
            q_lo_params=[], q_iqawg_params=[], dig_params=[]
    ):
        """

        Parameters
        ----------
        pulse_sequence_parameters : Dict[str, Any]
        freq_limits : tuple[float]
        down_conversion_calibration : Any
        q_lo_params : list[dict[str, Any]]
        q_iqawg_params : list[dict[str, Any]]
        dig_params : list[dict[str, Any]]

        Returns
        -------
        None

        Notes
        ----------
        If digitizer 'mode' and 'trigger_source' are absent they
        are set to 'averaging' and 'EXT0' respectively
        """

        # LO source initialization
        calibration = q_iqawg_params[0]["calibration"]
        q_lo_params[0]["power"] = calibration._lo_power
        self._q_lo[0].set_output_state("ON")

        # store sequence parameters for further usage
        self._pulse_sequence_parameters.update(pulse_sequence_parameters)
        self._down_conversion_calibration = down_conversion_calibration
        self._downconv_freq = calibration._if_frequency

        # TODO: make check of the repetition period.
        #  in order to verify if it is dividable by both AWG and digitizer clocks.
        # repetition_period = self._pulse_sequence_parameters["repetition_period"]

        # convert dict with parameters into form that is demanded by 'super().set_fixed_parameters()'
        dev_params = {"q_lo": q_lo_params,
                      "q_iqawg": q_iqawg_params,
                      "dig": dig_params}
        # for all child experiments this parameters are default for
        # digitizer acquisition mode
        if "mode" not in dig_params[0]:
            dig_params[0]["mode"] = SPCM_MODE.AVERAGING
        if "trig_source" not in dig_params:
            dig_params[0]["trig_source"] = SPCM_TRIGGER.EXT0

        super().set_fixed_parameters(**dev_params)
        # initialize 'self._measurement_result' that is specific for particular child class
        self._measurement_result.get_context().update({
            "calibration_results": calibration.get_optimization_results(),
            "radiation_parameters": calibration.get_radiation_parameters(),
            "pulse_sequence_parameters": pulse_sequence_parameters
        })

    def _init_measurement_result(self):
        """
        Pure virtual function that allows child classes to initialize
        measurement_result attribute in a 'hook' fasion

        Returns
        -------
        None
        """
        raise NotImplementedError

    def set_swept_parameters(self, **swept_pars):
        super().set_swept_parameters(**swept_pars)
        self._pulse_sequence_parameters["longest_duration"] = \
            self._get_longest_pulse_sequence_duration(self._pulse_sequence_parameters, self._swept_pars)

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

    def _get_longest_pulse_sequence_duration(self, pulse_sequence_parameters, swept_pars):
        """
            Purely virtual function. Needs to be implemented in child classes.
        Function must calculate and return the 'longest pulse sequence duration' for the particular
        measurement child class based on its 'self._sequence_generator' implementation.
            'Longest pulse sequence duration' is time between start of the first pulse and end
        of the last pulse in sequnce.
            The value provided by function is needed when you need trace phase
        at the end of the pulse sequence to remain constant at the end of
        the last pulse through all iteration process. This value is used in
        'self._sequence_generator' in order to set constant phase at the end of the
        last pulse.

        Docstring for child classes:
            Function calculates and return the longest pulse sequence duration based
        on pulse sequence parameters provided and 'self._sequence_generator' implementation.
            'Longest pulse sequence duration' is time between start of the first pulse and end
        of the last pulse in sequnce.
            The value provided by function is needed when you need trace phase
        at the end of the pulse sequence to remain constant at the end of
        the last pulse through all iteration process. This value is used in
        'self._sequence_generator' in order to set constant phase at the end of the
        last pulse.

        Parameters
        ----------
        pulse_sequence_parameters : dict
            Dictionary that contain pulse sequence parameters for which
            you wish to calculate the longest duration. This parameters are fixed.

        swept_pars : dict
            Sweep parameters that are needed for calculation of the
            longest sequence.

        Returns
        -------
        float
            Longest sequence duration based on pulse sequence parameters in ns.

        Notes
        ------
            This function is introduced in the context of the solution to the phase jumps, caused
        by clock incompatibility between AWG and digitizer. The aim is to fix distance between
        digitizer measurement window and AWG trigger that obtains digitizer.
            The last pulse ending should stay at fixed distance from trigger event in contrary with previous
        implementation, where the start of the first control pulse was fixed relative to trigger event.
            The previous solution forced digitizer acquisition window (which is placed after the pulse sequence, usually)
        to shift further in timeline following the extension of the length of the pulse sequence.
        And due to the fact that extension length does not always coincide with acquisition
        window displacement (due to difference in AWG and digitizer clock period) the phase jumps
        arise as a problem.
            The solution is that end of the last pulse stays at the same distance from the trigger event and
        pulse sequence length extendends "back in timeline". Together with requirement that 'repetition_period"
        is dividable by both AWG and digitizer clocks this will ensure that phase jumps will be neglected completely.
        """
        raise NotImplementedError

    @staticmethod
    def _calculate_basis_complex_amplitudes(self, basis):
        d_real = abs(np.real(basis[0] - basis[1]))
        d_imag = abs(np.imag(basis[0] - basis[1]))
        return d_real, d_imag

    def set_ult_calib(self, value=False):
        self._ult_calib = value

    def _single_measurement(self):
        dig = self._dig[0]
        # digitizer measurement setup is already configured in
        # 'self.set_fixed_parameters'
        dig_data = dig.safe_measure()


        # I channel data exctraction
        data_i = dig_data[0::2]
        data_i = data_i.reshape(dig.n_seg, round(data_i.shape[0] / dig.n_seg))
        data_i = data_i[:, self._n_samples_to_drop_by_delay: -self._n_samples_to_drop_in_end]
        data_i = data_i.flatten()

        # Q channel data exctraction
        data_q = dig_data[1::2]
        data_q = data_q.reshape(dig.n_seg, round(data_q.shape[0] / dig.n_seg))
        data_q = data_q[:, self._n_samples_to_drop_by_delay: -self._n_samples_to_drop_in_end]
        data_q = data_q.flatten()

        data = data_i + 1j * data_q
        # save full data in case of more detailed investigation
        if self._save_traces:
            self.dataIQ.append(data)

        if self._down_conversion_calibration is not None:
            data = self._down_conversion_calibration.apply(data)

        # exctacting Fourier component that exactly matches
        # 'self._downconv_freq' (because if use FFT, if_freq mesh may not
        # exactly coincide with desired 'self._downconv_freq'
        dt = 1/self._dig[0].get_sample_rate()  # sec
        t = np.linspace(0, data.shape[-1]*dt, data.shape[-1], endpoint=False)
        IQ = np.dot(data, np.exp(-2 * np.pi * 1j * self._downconv_freq * t))

        # normalizing DFT
        IQ /= len(data)

        return IQ

    def _recording_iteration(self):
        if self._ult_calib:
            # pulse sequence already played buy AWG
            fg = self._single_measurement()
            # close input mixer to measure background
            self._output_zero_sequence()
            bg = self._single_measurement()
            mean_data = fg - bg
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
        """
        Closes input mixer and force AWG to continue generate trigger
        pulses for every period.

        Returns
        -------
        None

        Notes
        -------
        NOT TESTED
        Change is introduced on 16.03.2020 by Shamil
        """
        self._q_iqawg[0].output_zero(
            trigger_sync_every=self._pulse_sequence_parameters[
                "repetition_period"]
        )




    def _output_pulse_sequence(self, zero=False):
        # update a trigger delay of the digitizer
        dig = self._dig[0]
        # longest_duration
        timedelay = self._pulse_sequence_parameters["start_delay"] + \
                    self._pulse_sequence_parameters["longest_duration"] + \
                    self._pulse_sequence_parameters["digitizer_delay"]
        dig.calc_and_set_trigger_delay(timedelay, include_pretrigger=True)  # update how many samples drop in front
        self._n_samples_to_drop_by_delay = dig.get_how_many_samples_to_drop_in_front()
        dig.calc_segment_size()  # updates how many to drop in the end
        self._n_samples_to_drop_in_end = dig.get_how_many_samples_to_drop_in_end()
        dig.setup_averaging_mode()  # loads new segment size into device

        # DIAGNOSE PHASE JUMPS WITH THIS TIMINGS OUTPUT
        # ns_in_sample = 1e9 / dig.get_sample_rate()
        # print("")
        # print("segment duration: {:.3f} ns".format(dig._segment_size * ns_in_sample))
        # print("delay in fornt: {:.3f} ns".format((dig.delay_in_samples + dig._n_samples_to_drop_by_delay) * ns_in_sample))
        # print("drop in front: {:.3f} ns".format(dig._n_samples_to_drop_by_delay * ns_in_sample))
        # print("drop in end: {:.3f} ns".format(dig._n_samples_to_drop_in_end * ns_in_sample))

        q_pbs = [q_iqawg.get_pulse_builder() for q_iqawg in self._q_iqawg]

        # TODO: 'and (self._q_z_awg[0] is not None)'  hotfix by Shamil (below)
        # I intend to declare all possible device attributes of the measurement class in it's child class definitions.
        # So hasattr(self, "_q_z_awg") is True
        # due to the fact that I had declared this parameter and initialized it with "[None]" in RabiFromFrequencyTEST.py
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

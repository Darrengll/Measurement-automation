""" IN DEVELOPMENT. NOT USED ANYWHERE."""

from copy import deepcopy

from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

from .calibration import HetIQCalibration
from .pulse_sequence import PulseSequence

from itertools import cycle, islice

from typing import Dict, List, Iterable


class IQPulseSequence:
    """
    Class whose instances can be loaded directly to the AWG via AWG's
    ouptut_iq_pulse_sequence() method
    """

    def __init__(self, s, calib=None):
        """
        Parameters
        ----------
        s : Iterable
            list of complex numbers
        calib : HetIQCalibration
            AWG output sampling frequency in GHz
        """
        self._s = np.array(s, dtype=np.complex128, copy=True)
        self.calib = calib

    def get_sequence(self):
        return self._s.copy()

    def get_length(self):
        return len(self._s)

    def get_duration(self):
        """
        Returns
        -------
            Length of the sequence in nanoseconds
        """
        return len(self._s)/self.calib.awg_sample_rate

    def append(self, other):
        """
        Produces new `IQPulseSequence` instance with sequence of `other`
        appended to `self`

        Parameters
        ----------
        other : IQPulseSequence

        Returns : IQPulseSequence
        -------
        """
        return IQPulseSequence(np.hstack([self._s, other._s]))

    def __add__(self, other):
        """
        Produces new `IQPulseSequence` instance with sequence values equal
        added elementwise

        Parameters
        ----------
        other : IQPulseSequence

        Returns : IQPulseSequence
        -------
        """
        if self.get_length() == other.get_length():
            return IQPulseSequence(self._s + other._s, copy=True)
        else:
            raise ValueError("Sequences must have equal length to be added "
                             "element-wise")

    def get_sample_rate(self):
        return self.calib.awg_sample_rate

    def plot_signal(self, **kwargs):
        fig, (ax_i, ax_q) = plt.subplots(2,1)
        ax_i.plot(np.real(self._s), label="I", **kwargs)
        ax_q.plot(np.imag(self._s), label="I", **kwargs)
        fig.legend()

        return fig, (ax_i, ax_q)


class IQPulseBuilder:
    def __init__(self, iqmx_calibration):
        """
        Build a IQPulseBuilder instance for a previously calibrated IQ mixer.

        Parameters
        ----------
        iqmx_calibration : HetIQCalibration
            Calibration data for the IQ mixer that will be used to send out the pulse sequence.
            Make sure that the radiation parameters of this calibration are in match with your actual settings
        """
        self._iqmx_calibration = deepcopy(iqmx_calibration)
        self._waveform_resolution = iqmx_calibration.awg_sampling_period
        self._pulse_seq_I = PulseSequence(self._waveform_resolution)
        self._pulse_seq_Q = PulseSequence(self._waveform_resolution)

    def get_duration(self):
        """
        Returns : np.float64
        -------
            Returns duration of pulse sequence in nanoseconds
        """
        # always equal to `self._pulse_seq_Q.get_duration()
        return self._pulse_seq_I.get_duration()

    def get_calibration(self):
        return self._iqmx_calibration

    def add_dc_pulse(self, duration, dc_offsets_open=None):
        """
        Adds a pulse by putting a dc voltage at the I and Q inputs of the mixer
        This pulse will let trace through with calibrated power

        Parameters:
        -----------
        duration: float
            Duration of the pulse in nanoseconds
        dc_offsets_open: Tuple[float,float]
            Values of the dc voltage in Volts applied at the IQ mixer
            ports during the pulse.
            If not specified, calibration `dc_offsets_open` will be used to
            generate output with a given power.
        """
        if dc_offsets_open is None:
            vdc1, vdc2 = self._iqmx_calibration.dc_offsets_open
        else:
            vdc1, vdc2 = dc_offsets_open

        N_time_steps = int(round(duration / self._waveform_resolution))

        self._pulse_seq_I.append_pulse(np.zeros(N_time_steps) + vdc1)
        self._pulse_seq_Q.append_pulse(np.zeros(N_time_steps) + vdc2)
        return self

    def add_zero_pulse(self, duration, dc_offsets=None):
        """
        Adds a pulse with zero amplitude to the sequence
        This pulse will block LO->RF transmission effectively
         according to calibration results

        Parameters
        -----------
        duration: float
            Duration of the pulse in nanoseconds
        dc_offsets : Optional[Tuple[float,float]]
            Offsets to use instead of calibration values.
            If not provided, calibration values `dc_offsets_closed` are used.
            `dc_offsets_closed` are calibration result for minimizing mixer
            LO -> RF transmission.

        Returns
        --------
        self : IQPulseSequence
            returns `self` with insulating pulse added.
        """
        if dc_offsets is None:
            vdc1, vdc2 = self._iqmx_calibration.dc_offsets_close
        else:
            vdc1, vdc2 = dc_offsets

        N_time_steps = int(round(duration / self._waveform_resolution))
        self._pulse_seq_I.append_pulse(np.zeros(N_time_steps) + vdc1)
        self._pulse_seq_Q.append_pulse(np.zeros(N_time_steps) + vdc2)
        return self

    def add_sine_pulse(self, duration, if_offsets=None, phase=0,
                       amplitude_mult=1,
                       window="rectangular", hd_amplitude=0,
                       frequency=None, if_amplitudes=None,
                       window_parameter=0.5):
        """
        Adds a pulse with amplitude defined by the iqmx_calibration at frequency
        f_lo-f_if and some phase to the sequence. All sine pulses will be parts
        of the same continuous wave at frequency of f_if

        Parameters
        -----------
        duration: float
            Duration of the pulse in nanoseconds. For pulses other than rectangular
            will be interpreted as t_g (see F. Motzoi et al. PRL (2009))
        if_offsets : Tuple[float, float]
            Used instead of the corresponding parameter from
            `self._iqmx_calibration` if provided
        phase: float, rad
            Adds a relative phase to the outputted trace.
        amplitude: float
            Calibration if_amplitudes will be scaled by the
            amplitude_value.
        window: string
            List containing the name and the description of the modulating
            window of the pulse.
            Implemented modulations:
            "rectangular"
                Rectangular window.
            "gaussian"
                Gaussian window, see F. Motzoi et al. PRL (2009).
            "hahn"
                Hahn sin^2 window
        hd_amplitude: float
            correction for the Half Derivative method, theoretically should be 1
        if_freq : float, Hz
            Used instead of the corresponding parameter from self._iqmx_calibration if provided
        """
        if if_offsets is None:
            if_offs1, if_offs2 = self._iqmx_calibration.dc_offsets_open
        else:
            if_offs1, if_offs2 = if_offsets

        if if_amplitudes is None:
            if_amp1, if_amp2 = \
                self._iqmx_calibration.iqawg_i_amplitude,
                    self._iqmx_calibration.get_iqawg_q_amplitude()
        else:
            if_ampl1, if_amp2

        if_amp1, if_amp2 = \
            if_amp1 * amplitude_mult, if_amp2 * amplitude_mult

        if_phase = \
            self._iqmx_calibration.get_optimization_results()[0]["if_phase"]
        frequency = 2 * pi * self._iqmx_calibration.get_radiation_parameters()[
            "if_frequency"] / 1e9 if frequency is None else \
            2 * pi * frequency / 1e9

        N_time_steps = int(np.round(duration / self._waveform_resolution))

        duration = N_time_steps * self._waveform_resolution

        phase += self._pulse_seq_I.total_points() * self._waveform_resolution * frequency
        points = linspace(0, duration, N_time_steps,
                          endpoint=False)  # divide duration into N_time_steps intervals
        carrier_I = if_amp1 * exp(1j * (frequency * points + if_phase + phase))
        carrier_Q = if_amp2 * exp(1j * (frequency * points + phase))

        # print(real(carrier_Q)

        def rectangular():
            return ones_like(points), zeros_like(points)

        def gaussian():
            B = exp(-(duration / 2) ** 2 / 2 / (duration / 3) ** 2)
            window = (exp(-(points - duration / 2) ** 2 / 2 / (
                    duration / 3) ** 2) - B) / (1 - B)
            if (N_time_steps > 0):
                derivative = gradient(window, self._waveform_resolution)
                derivative[0] = derivative[-1] = 0
            else:
                derivative = 0
            return window, derivative

        def hahn():
            window = sin(pi * linspace(0, N_time_steps, N_time_steps,
                                       endpoint=False) / N_time_steps) ** 2
            if N_time_steps > 0:
                derivative = gradient(window, self._waveform_resolution)
                derivative[0] = derivative[-1] = 0
            else:
                derivative = 0
            return window, derivative

        def tukey():
            # https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.signal.tukey.html
            window = signal.tukey(N_time_steps, alpha=window_parameter)
            if N_time_steps > 1:
                derivative = gradient(window, self._waveform_resolution)
                derivative[0] = derivative[-1] = 0
            else:
                derivative = 0
            return window, derivative

        def kaiser():
            # https://docs.scipy.org/doc/scipy-1.0.0/reference/generated/scipy.signal.kaiser.html
            window = signal.kaiser(N_time_steps, beta=window_parameter)
            if N_time_steps > 0:
                derivative = gradient(window, self._waveform_resolution)
                derivative[0] = derivative[-1] = 0
            else:
                derivative = 0
            return window, derivative

        def decaying_exponent():
            window = exp(-window_parameter * linspace(0, N_time_steps,
                                                     N_time_steps,
                                                      endpoint=False) / N_time_steps)
            return window

        windows = {"rectangular": rectangular, "gaussian": gaussian,
                   "hahn": hahn, "tukey": tukey, "kaiser": kaiser,
                   "decaying_exponent": decaying_exponent}
        window, derivative = windows[window]()

        hd_correction = - derivative * hd_amplitude / 2 / (
                -2 * pi * 0.2)  # anharmonicity
        carrier_I = window * real(carrier_I) + hd_correction * imag(carrier_I)
        carrier_Q = window * real(carrier_Q) + hd_correction * imag(carrier_Q)

        self._pulse_seq_I.append_pulse(carrier_I + if_offs1)
        self._pulse_seq_Q.append_pulse(carrier_Q + if_offs2)
        return self

    def add_sine_pulse_from_string(self, pulse_string, pulse_duration,
                                   pulse_amplitude, window='gaussian'):
        """
        pulse_duration is pi_pulse_duraton for rectangular window
        and is arbitrary for gaussian window.

        pulse_amplitude is pi_pulse_amplitude for gaussian window and
        is arbitrary for rectangular window.
        """
        global_phase = 0
        pulse_ax = pulse_string[1]
        pulse_angle = eval(pulse_string.replace(pulse_ax, "1"))  # in pi's
        if window is "rectangular" or pulse_ax is "I":
            pulse_time = pulse_duration * abs(pulse_angle)
        else:
            pulse_time = pulse_duration
            pulse_amplitude = abs(pulse_angle) * pulse_amplitude
        pulse_phase = np.pi / 2 * (1 - np.sign(pulse_angle)) + global_phase
        if pulse_ax == "I":
            self.add_zero_pulse(pulse_time)
        elif pulse_ax == "X":
            self.add_sine_pulse(duration=pulse_time, phase=pulse_phase,
                                amplitude_mult=pulse_amplitude,
                                window=window)
        elif pulse_ax == "Y":
            self.add_sine_pulse(duration=pulse_time,
                                phase=pulse_phase + pi / 2,
                                amplitude_mult=pulse_amplitude,
                                window=window)
        elif pulse_ax == "Z":
            global_phase += np.pi * pulse_angle
            pulse_time = 0
        else:
            raise ValueError(
                "Axis of %s is not allowed. Check your sequence." % (
                    pulse_string))
        return self

    def add_zero_until(self, total_duration, if_offsets=None):
        """
        Adds a pulse with zero amplitude to the sequence of such length that the
        whole pulse sequence is of specified duration

        Should be used to end the sequence as the last call before build(...)

        Parameters
        -----------
        total_duration: float, ns
            Duration of the whole sequence
        if_offsets : tuple[float]
            Used instead of the corresponding parameter from self._iqmx_calibration if provided
        """
        total_time_steps = round(total_duration / self._waveform_resolution)
        current_time_steps = self._pulse_seq_I.total_points()
        residual_time_steps = total_time_steps - current_time_steps
        # from time import sleep
        # print(residual_time_steps,flush=True)
        self.add_zero_pulse(residual_time_steps * self._waveform_resolution,
                            if_offsets)
        return self

    def build(self):
        """
        Returns the IQ sequence containing I and Q pulse sequences and the total
        duration of the pulse sequence in ns
        """
        to_return = IQPulseSequence(self._pulse_seq_I, self._pulse_seq_Q)
        self._pulse_seq_I = PulseSequence(self._waveform_resolution)
        self._pulse_seq_Q = PulseSequence(self._waveform_resolution)
        return to_return

    @staticmethod
    def build_dispersive_rabi_sequences(pulse_sequence_parameters, **pbs):
        """
        Returns synchronized excitation and readout IQPulseSequences assuming that
        readout AWG is triggered by the excitation AWG
        """
        exc_pb = pbs['q_pbs'][0]
        ro_pb = pbs['ro_pbs'][0]
        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        excitation_duration = \
            pulse_sequence_parameters["excitation_duration"]
        window = \
            pulse_sequence_parameters["modulating_window"]
        amplitude = \
            pulse_sequence_parameters["excitation_amplitude"]

        exc_pb.add_zero_pulse(awg_trigger_reaction_delay) \
            .add_sine_pulse(excitation_duration, 0, amplitude_mult=amplitude,
                            window=window) \
            .add_zero_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        ro_pb.add_zero_pulse(excitation_duration + 10) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        return {'q_seqs': [exc_pb.build()],
                'ro_seqs': [ro_pb.build()]}

    @staticmethod
    def build_direct_rabi_sequences(pulse_sequence_parameters, **pbs):
        """

        Parameters
        ----------
        pulse_sequence_parameters : dict[str,float]
        pbs : Dict[str,List[IQPulseBuilder]]

        Returns
        -------
        Dict[str,List[IQPulseSequence]]
            Dictionary that contain pulse sequences for devices groups.
            Devices group lists id's are denoted by dictionary keys.

        Notes
        -------
            The previous solution forced digitizer acquisition window (which is placed after the pulse sequence, usually)
        to shift further in timeline following the extension of the length of the pulse sequence.
        And due to the fact that extension length does not always coincide with acquisition
        window displacement (due to difference in AWG and digitizer clock period) the phase jumps
        arise as a problem.
            The solution is that end of the last pulse stays at the same distance from the trigger event and
        pulse sequence length extendends "back in timeline". Together with requirement that 'repetition_period"
        is dividable by both AWG and digitizer clocks this will ensure that phase jumps will be neglected completely.
        """
        exc_pb = pbs['q_pbs'][0]
        start_delay = \
            pulse_sequence_parameters["start_delay"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        excitation_duration = \
            pulse_sequence_parameters["excitation_duration"]
        longest_pulse_duration = \
            pulse_sequence_parameters["longest_duration"]
        amplitude = \
            pulse_sequence_parameters["excitation_amplitude"]
        window = \
            pulse_sequence_parameters["modulating_window"]

        window_parameter = pulse_sequence_parameters["window_parameter"] \
            if "window_parameter" in pulse_sequence_parameters else 0.1

        # no need in starting_phase if end time of pulse ending is fixed
        # phase is accumulated and accounted for automatically
        # in pulse builders.
        starting_phase = 0

        exc_pb.add_zero_pulse(
            start_delay + longest_pulse_duration - excitation_duration) \
            .add_sine_pulse(excitation_duration,
                            window=window,
                            window_parameter=window_parameter,
                            phase=starting_phase,
                            amplitude_mult=amplitude) \
            .add_zero_until(repetition_period)
        return {'q_seqs': [exc_pb.build()]}\

    @staticmethod
    def build_decaying_exponent_sequence(pulse_sequence_parameters, **pbs):
        """

        Parameters
        ----------
        pulse_sequence_parameters : dict[str,float]
        pbs : Dict[str,List[IQPulseBuilder]]

        Returns
        -------
        Dict[str,List[IQPulseSequence]]
            Dictionary that contain pulse sequences for devices groups.
            Devices group lists id's are denoted by dictionary keys.

        Notes
        -------
            Actually, it is "build_direct_rabi_sequences" with the exponent
            window
        """
        exc_pb = pbs['q_pbs'][0]
        start_delay = \
            pulse_sequence_parameters["start_delay"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        excitation_duration = \
            pulse_sequence_parameters["excitation_duration"]
        longest_pulse_duration = \
            pulse_sequence_parameters["longest_duration"]
        amplitude = \
            pulse_sequence_parameters["excitation_amplitude"]
        window = \
            pulse_sequence_parameters["modulating_window"]

        if window == "decaying_exponent":
            window_parameter = pulse_sequence_parameters["gamma"] \
                if "gamma" in pulse_sequence_parameters else print(
                'WHERE IS GAMMA?????')
        else:
            print('not exponent window')

        # no need in starting_phase if end time of pulse ending is fixed
        # phase is accumulated and accounted for automatically
        # in pulse builders.
        starting_phase = 0

        exc_pb.add_zero_pulse(
            start_delay + longest_pulse_duration - excitation_duration) \
            .add_sine_pulse(excitation_duration,
                            window=window,
                            window_parameter=window_parameter,
                            phase=starting_phase,
                            amplitude_mult=amplitude) \
            .add_zero_until(repetition_period)
        return {'q_seqs': [exc_pb.build()]}

    @staticmethod
    def build_direct_ramsey_sequence(pulse_sequence_parameters, **pbs):
        """

        Parameters
        ----------
        pulse_sequence_parameters : dict[str,float]
        pbs : Dict[str,List[IQPulseBuilder]]

        Returns
        -------
        Dict[str,List[IQPulseSequence]]
            Dictionary that contain pulse sequences for devices groups.
            Devices group lists id's are denoted by dictionary keys.

        Notes
        -------
            The previous solution forced digitizer acquisition window (which is placed after the pulse sequence, usually)
        to shift further in timeline following the extension of the length of the pulse sequence.
        And due to the fact that extension length does not always coincide with acquisition
        window displacement (due to difference in AWG and digitizer clock period) the phase jumps
        arise as a problem.
            The solution is that end of the last pulse stays at the same distance from the trigger event and
        pulse sequence length extendends "back in timeline". Together with requirement that 'repetition_period"
        is dividable by both AWG and digitizer clocks this will ensure that phase jumps will be neglected completely.
        """
        exc_pb = pbs['q_pbs'][0]
        start_delay = \
            pulse_sequence_parameters["start_delay"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        pi_half_pulse_duration = \
            pulse_sequence_parameters["pi_half_pulse_duration"]
        longest_duration = \
            pulse_sequence_parameters["longest_duration"]
        amplitude = \
            pulse_sequence_parameters["pi_half_pulse_amplitude"]
        window = \
            pulse_sequence_parameters["modulating_window"]

        window_parameter = pulse_sequence_parameters["window_parameter"] \
            if "window_parameter" in pulse_sequence_parameters else 0.5

        # no need in starting_phase if end time of pulse ending is fixed
        # phase is accumulated and accounted for automatically
        # in pulse builders.
        starting_phase = 0

        exc_pb.add_zero_pulse(
            start_delay + longest_duration - pi_half_pulse_duration) \
            .add_sine_pulse(pi_half_pulse_duration,
                            window=window,
                            window_parameter=window_parameter,
                            phase=starting_phase,
                            amplitude_mult=amplitude) \
            .add_zero_until(repetition_period)
        return {'q_seqs': [exc_pb.build()]}

    @staticmethod
    def build_wave_mixing_pulses(pulse_sequence_parameters, **pbs):
        """

        Parameters
        ----------
        pulse_sequence_parameters : dict[str, Union[int,str]]
        pbs : Dict[str, List[IQPulseBuilder]]

        Returns
        -------
        seqs : dict[str,list[IQPulseSequence]]
        """
        exc_pb = pbs["q_pbs"][0]
        start_delay = pulse_sequence_parameters["start_delay"]
        repetition_period = pulse_sequence_parameters["repetition_period"]
        pulse_types = pulse_sequence_parameters["pulse_sequence"]
        pulses_n = len(pulse_types)

        # repeatedly extends 'excitation_durations' list to the length that equals the number of pulses
        excitation_durations = list(
            islice(cycle(pulse_sequence_parameters["excitation_durations"]),
                   pulses_n))

        # repeatedly extends 'excitation_durations' list to the length that equals the number of pulses
        # + [0.0] delay after last pulse, for convenience of iteration code
        # that constructs pulse sequences
        after_pulse_delays = list(
            islice(cycle(pulse_sequence_parameters["after_pulse_delays"]),
                   pulses_n - 1)) + [0.0]
        # check for non-negative pulse delays
        # negative pulse delays are implemented as 'pulse_shifts' parameters in
        # pulse_sequence_parameters
        for i, delay_after in enumerate(after_pulse_delays):
            if delay_after < 0:
                raise ValueError(f"'after_pulse_delays[{i}] is negative")

        # in you need change several pulses positions relative to the default setup
        pulse_shifts = pulse_sequence_parameters["pulse_shifts"]
        if len(pulse_shifts) != pulses_n:
            raise ValueError(f"len of 'pulse shifts': {pulse_shifts} \n"
                             f"is not equal to the len of the 'pulse_types': {pulse_types}")

        # periodically extending amplitudes list to match size of
        # `pulse_types`
        amplitudes = list(
            islice(cycle(pulse_sequence_parameters["excitation_amplitudes"]),
                   pulses_n))

        # periodically extending `phase_shifts` list to match size of
        # `pulse_types`
        if "phase_shifts" in pulse_sequence_parameters:
            phase_shifts = list(
                islice(cycle(pulse_sequence_parameters["phase_shifts"]),
                       pulses_n))
        else:
            phase_shifts = [0] * pulses_n

        window = pulse_sequence_parameters["modulating_window"]
        d_freq = pulse_sequence_parameters["d_freq"]
        freq = exc_pb.get_calibration().get_if_frequency()  # Hz

        # period when phases between two frequencies first time reach 2 pi
        envelope_duration = 1 / d_freq * 1e9  # ns

        params_zipped = list(
            zip(amplitudes,
                phase_shifts,
                excitation_durations,
                pulse_types)
        )

        '''
        Pulse construction procedure depends strongly on 
        whether pulses from different groups are overlapping or not.
        Output voltage always has to stay low enough in order to 
        guarantee linearity of overlapping pulse summation in mixer.
        
        Whether pulses are overlapping or not may change during the sweep. 
        This leads to changing trace power (ampltiude divided by 2 or not) 
        during the sweep, if pulses are overlapping.
        Hence, the behaviour of the pulse summation is forced to be the same 
        during the whole sweep by the assignment below, that does not allow
        overlapping of the pulses.
        '''
        pb0 = exc_pb
        # pulse sequence is separated into positive and negative sequences
        pb0._iqmx_calibration._if_offsets /= 2
        pb0._iqmx_calibration._dc_offsets /= 2
        pb_p = deepcopy(pb0)
        pb_n = deepcopy(pb0)
        pb0._iqmx_calibration._if_offsets *= 2
        pb0._iqmx_calibration._dc_offsets *= 2

        ''' 
        Calculating positions of the
        positive and negative frequency pulse groups respectively.
        No overlapping between pulses within the same frequency group is 
        allowed/supported yet.
        ==> all pulse delays in the same frequency group MUST BE >= 0 
        Also pulses shifts is not cumulative (previous pulses shifts do not 
        affect further pulses positions)
        '''

        '''
        Calculating positions for positive and negative 
        pulses respectively
        '''
        ''' positive frequency pulses block construction '''
        positive_n = pulse_types.count("P")
        p_positions = np.zeros(positive_n)  # start positions of the "P" pulses
        p_positions[0] += start_delay
        i = 0  # current pulse index
        for p_idx in range(positive_n):
            # flag indicates that it is time to break from inner cycle
            # and start calculating next "P" pulse position
            next_p_idx = False

            # cycling over all pulses (starting where ended last time)
            while (i < pulses_n) and (not next_p_idx):
                # adding delay time while "N" pulse
                if pulse_types[i] == "N":
                    p_positions[p_idx] += excitation_durations[i] + \
                                          after_pulse_delays[i]
                # if we meet 'p_idx' positive pulse
                elif pulse_types[i] == "P":
                    next_p_idx = True
                    # adding 'p_idx' pulse shift value
                    p_positions[p_idx] += pulse_shifts[i]

                    # if there is at least 1 more "P" pulse
                    # add previously accumulated result for next positive
                    # pulse start position:
                    # current pulse excitation duration and `delay_pulse_delay`
                    # as well
                    if (p_idx + 1) < len(p_positions):
                        p_positions[p_idx + 1] = p_positions[p_idx] + \
                                                 excitation_durations[i] + \
                                                 after_pulse_delays[i]
                i += 1

        ''' negative frequency pulses block construction '''
        # amount of negative pulses
        negative_n = pulse_types.count("N")
        n_positions = np.zeros(negative_n)  # start positions of the "N" pulses
        n_positions[0] += start_delay
        i = 0  # index through all pulses
        for n_idx in range(negative_n):
            # flag indicates that it is time to break from inner cycle
            # and start calculating next "N" pulse position
            next_n_idx = False

            # cycling over all pulses (starting where ended last time)
            while (i < pulses_n) and (not next_n_idx):
                # adding idle time for all positive pulses before the 'n_idx' negative pulse
                if pulse_types[i] == "P":
                    n_positions[n_idx] += excitation_durations[i] + \
                                          after_pulse_delays[i]
                # if we meet 'n_idx' positive pulse
                elif pulse_types[i] == "N":
                    next_n_idx = True
                    # adding 'n_idx' pulse shift value
                    n_positions[n_idx] += pulse_shifts[i]

                    # if there is at least 1 more "N" pulse
                    if (n_idx + 1) < len(n_positions):
                        # add previously accumulated result for next negative pulse start position
                        # add current pulse excitation duration as well
                        n_positions[n_idx + 1] = n_positions[n_idx] + \
                                                 excitation_durations[i] + \
                                                 after_pulse_delays[i]
                i += 1

        ''' Checking distances between pulses in the same group. They has to be non-negative '''
        # exctract positive and negative excitation durations
        p_excitation_durations = []
        n_excitation_durations = []
        for pulse_type, excitation_duration in zip(pulse_types,
                                                   excitation_durations):
            if pulse_type == "P":
                p_excitation_durations.append(excitation_duration)
            elif pulse_type == "N":
                n_excitation_durations.append(excitation_duration)

        # differences between start positions of successive "P" pulses
        p_diffs = np.diff(p_positions)
        n_diffs = np.diff(n_positions)

        # delay (zero time) between successive pulses inside their groups
        p_delays = p_diffs - p_excitation_durations[:-1]
        n_delays = n_diffs - n_excitation_durations[:-1]

        # check that pulses from the same group do not overlap
        if p_delays.any() < 0 or n_delays.any() < 0:
            raise ValueError(
                "pulses from the same frequency group found to be overlapping")

        envelopes_in_pulse_group = pulse_sequence_parameters[
            "envelopes_in_pulse_group"]
        n_pulse_groups = int(
            envelopes_in_pulse_group * envelope_duration / repetition_period)

        # restriction: envelope duration has to be multiple of the repetition period
        if abs(
                envelope_duration % repetition_period) > 1e-5:  # all values in ns
            raise ValueError(
                "pulse repetition frequency has to be a multiple of pulse frequency difference")

        ''' constructing pulses '''
        # appending zero values for convenience for code in cycle
        p_delays = np.concatenate((p_delays, [0.0]))
        n_delays = np.concatenate((n_delays, [0.0]))

        # print(p_positions, p_delays)
        # print(n_positions, n_delays)

        for i in range(n_pulse_groups):
            # indexes to iterate over each pulse frequency group
            p_idx = 0
            n_idx = 0
            pb_p = pb_p.add_zero_pulse(p_positions[0])
            pb_n = pb_n.add_zero_pulse(n_positions[0])
            for amplitude, phase_shift, excitation_duration, pulse_type in params_zipped:
                if pulse_type == "P":
                    pulse_freq = freq + d_freq
                    pb_p = pb_p.add_sine_pulse(excitation_duration,
                                               window=window,
                                               frequency=pulse_freq,
                                               phase=phase_shift,
                                               amplitude_mult=amplitude) \
                        .add_zero_pulse(p_delays[p_idx])
                    p_idx += 1
                elif pulse_type == "N":
                    pulse_freq = freq - d_freq
                    pb_n = pb_n.add_sine_pulse(excitation_duration,
                                               window=window,
                                               frequency=pulse_freq,
                                               phase=phase_shift,
                                               amplitude_mult=amplitude) \
                        .add_zero_pulse(n_delays[n_idx])
                    n_idx += 1

            pb_p = pb_p.add_zero_until((i + 1) * repetition_period)
            pb_n = pb_n.add_zero_until((i + 1) * repetition_period)

        # this parameters are needed by digitizer in order to properly perform
        # software filtering (the exact pulses end positions is needed)
        pulse_sequence_parameters.update(
            first_pulse_start=min(p_positions[0], n_positions[0]),
            last_pulse_end=max(p_positions[-1] + p_excitation_durations[-1],
                               n_positions[-1] + n_excitation_durations[-1])
        )

        # performing direct add on non-overlapping positive and negative waveforms
        return {'q_seqs': [pb_p.build().direct_add(pb_n.build())]}

    @staticmethod
    def build_stimulated_emission_sequence(pulse_sequence_parameters, **pbs):
        """
        Builds a pulse sequence with a repeated group of non-overlapping
        pulses.

        Use build_wave_mixing_pulses() to create sequences with overlapping
        pulses.

        Parameters
        ----------
        pulse_sequence_parameters : Dict[str, Union[int,str]]
        pbs : Dict[str, List[IQPulseBuilder]]

        Returns
        -------
        seqs : Dict[str,List[IQPulseSequence]]
        """
        exc_pb = pbs["q_pbs"][0]

        # initializing variables
        start_delay = pulse_sequence_parameters["start_delay"]
        repetition_period = pulse_sequence_parameters["repetition_period"]
        pulse_types = pulse_sequence_parameters["pulse_sequence"]
        excitation_durations = pulse_sequence_parameters[
            "excitation_durations"]
        amplitudes = pulse_sequence_parameters["excitation_amplitudes"]
        window = pulse_sequence_parameters["modulating_window"]
        window_parameter = 0.5
        if "window_parameter" in pulse_sequence_parameters:
            window_parameter = pulse_sequence_parameters["window_parameter"]
        after_pulse_delay = pulse_sequence_parameters["after_pulse_delay"]
        if after_pulse_delay < 0:
            raise ValueError(f"after_pulse_delay is negative")
        if "phase_shifts" in pulse_sequence_parameters:
            phase_shifts = pulse_sequence_parameters["phase_shifts"]
        else:
            phase_shifts = [0] * len(pulse_types)
        d_freq = pulse_sequence_parameters["d_freq"]
        ifreq = exc_pb.get_calibration().get_if_frequency()
        freqs = {"0": ifreq, "P": ifreq + d_freq, "N": ifreq - d_freq}

        # constructing a pulse sequence
        for i in range(pulse_sequence_parameters["periods_per_segment"]):
            exc_pb = exc_pb.add_zero_pulse(start_delay)
            for j in range(len(pulse_types)):
                exc_pb = exc_pb.add_sine_pulse(excitation_durations[j],
                                               window=window,
                                               phase=(phase_shifts[j]),
                                               frequency=freqs[pulse_types[j]],
                                               amplitude_mult=amplitudes[j],
                                               window_parameter=
                                               window_parameter)
            exc_pb = exc_pb.add_zero_until(repetition_period * (i + 1))

        # this parameters are set in order indicate timings for dynamically
        # created sophisticated pulses
        pulse_sequence_parameters.update(
            first_pulse_start=start_delay,
            last_pulse_end=start_delay + after_pulse_delay + sum(
                excitation_durations)
        )
        return {'q_seqs': [exc_pb.build()]}

    @staticmethod
    def build_stimulated_emission_sequence_old(pulse_sequence_parameters,
                                               **pbs):
        """
        Builds a pulse sequence with a repeated group of two non-overlapping pulse.

        Use build_wave_mixing_pulses() to create sequences with overlapping pulses.

        Parameters
        ----------
        pulse_sequence_parameters : dict[str, Union[int,str]]
        pbs : Dict[str, List[IQPulseBuilder]]

        Returns
        -------
        seqs : dict[str,list[IQPulseSequence]]
        """
        exc_pb = pbs["q_pbs"][0]
        start_delay = pulse_sequence_parameters["start_delay"]
        repetition_period = pulse_sequence_parameters["repetition_period"]
        pulse_types = pulse_sequence_parameters["pulse_sequence"]
        pulses_n = 2
        if len(pulse_types) != 2:
            raise ValueError("The number of pulses must be exactly two")

        # repeatedly extends 'excitation_durations' list to the length that equals the number of pulses
        excitation_durations = list(
            islice(cycle(pulse_sequence_parameters["excitation_durations"]),
                   pulses_n))

        # repeatedly extends 'excitation_durations' list to the length that equals the number of pulses
        # + [0.0] - for convenience of iteration
        after_pulse_delay = pulse_sequence_parameters["after_pulse_delay"]
        if after_pulse_delay < 0:
            raise ValueError(f"after_pulse_delay is negative")

        amplitudes = list(
            islice(cycle(pulse_sequence_parameters["excitation_amplitudes"]),
                   pulses_n))
        if "phase_shifts" in pulse_sequence_parameters:
            phase_shifts = list(
                islice(cycle(pulse_sequence_parameters["phase_shifts"]),
                       pulses_n))
        else:
            phase_shifts = [0] * pulses_n

        window = pulse_sequence_parameters["modulating_window"]

        d_freq = pulse_sequence_parameters["d_freq"]  # Hz
        freq = exc_pb.get_calibration().get_if_frequency()  # Hz
        freqs = {"P": freq + d_freq, "N": freq - d_freq, "0": freq}

        # period when phases between two frequencies first time reach 2 pi
        envelope_duration = pulse_sequence_parameters["repetition_period"] * 1
        # envelope_duration = 1 / d_freq * 1e9  # ns

        envelopes_in_pulse_group = pulse_sequence_parameters[
            "envelopes_in_pulse_group"]
        n_pulse_groups = int(
            envelopes_in_pulse_group * envelope_duration / repetition_period)

        # restriction: envelope duration has to be multiple of the repetition period
        # if envelope_duration % repetition_period > 0.:  # all values in ns
        #     raise ValueError("pulse repetition frequency has to be a multiple of pulse frequency difference")

        lo_freq = exc_pb._iqmx_calibration.get_radiation_parameters()[
            "lo_frequency"]

        ''' constructing pulses '''
        for i in range(n_pulse_groups):
            exc_pb = exc_pb.add_zero_pulse(start_delay) \
                .add_sine_pulse(excitation_durations[0],
                                window=window,
                                frequency=freqs[pulse_types[0]],
                                phase=(phase_shifts[0]),
                                amplitude_mult=amplitudes[0]) \
                .add_zero_pulse(after_pulse_delay) \
                .add_sine_pulse(excitation_durations[1],
                                window=window,
                                frequency=freqs[pulse_types[1]],
                                phase=(phase_shifts[1]),
                                amplitude_mult=amplitudes[1]) \
                .add_zero_until((i + 1) * repetition_period)
        # exc_pb = exc_pb.add_zero_until(envelope_duration * envelopes_in_pulse_group)
        # this parameters are needed by digitizer in order to properly perform
        # software filtering (the exact pulses end positions is needed)
        pulse_sequence_parameters.update(
            first_pulse_start=start_delay,
            last_pulse_end=start_delay + excitation_durations[
                0] + after_pulse_delay
        )
        # print("pulses are set")
        return {'q_seqs': [exc_pb.build()]}

    @staticmethod
    def build_direct_rabi_sequences_AM(pulse_sequence_parameters, **pbs):
        """
        building pulse sequence to measure Rabi oscillations
        Sequence is utilizing an amplitude modulation of the mixer

        Parameters
        ----------
        pulse_sequence_parameters
        pbs

        Returns
        -------

        """
        exc_pb = pbs['q_pbs'][0]
        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        excitation_duration = \
            pulse_sequence_parameters["excitation_duration"]
        amplitude = \
            pulse_sequence_parameters["excitation_amplitude"]

        exc_pb.add_zero_pulse(awg_trigger_reaction_delay, dc_offsets=(0, 0)) \
            .add_dc_pulse(excitation_duration, (1, 1)) \
            .add_zero_until(repetition_period, if_offsets=(0, 0))

        return {'q_seqs': [exc_pb.build()]}

    @staticmethod
    def build_dispersive_ramsey_sequences(pulse_sequence_parameters, **pbs):

        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        half_pi_pulse_duration = \
            pulse_sequence_parameters["half_pi_pulse_duration"]
        ramsey_delay = \
            pulse_sequence_parameters["ramsey_delay"]
        window = \
            pulse_sequence_parameters["modulating_window"]
        amplitude = \
            pulse_sequence_parameters["excitation_amplitude"]
        exc_pb = pbs['q_pbs'][0]
        ro_pb = pbs['ro_pbs'][0]

        exc_pb.add_zero_pulse(awg_trigger_reaction_delay) \
            .add_sine_pulse(half_pi_pulse_duration,
                            amplitude_mult=amplitude, window=window) \
            .add_zero_pulse(ramsey_delay) \
            .add_sine_pulse(half_pi_pulse_duration,
                            amplitude_mult=amplitude, window=window) \
            .add_zero_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        ro_pb.add_zero_pulse(2 * half_pi_pulse_duration + ramsey_delay + 10) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        return {'q_seqs': [exc_pb.build()],
                'ro_seqs': [ro_pb.build()]}

    @staticmethod
    def build_dispersive_decay_sequences(pulse_sequence_parameters, **pbs):
        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        pi_pulse_duration = \
            pulse_sequence_parameters["pi_pulse_duration"]
        readout_delay = \
            pulse_sequence_parameters["readout_delay"]
        exc_pb = pbs['q_pbs'][0]
        ro_pb = pbs['ro_pbs'][0]

        exc_pb.add_zero_pulse(awg_trigger_reaction_delay) \
            .add_sine_pulse(pi_pulse_duration, 0) \
            .add_zero_pulse(readout_delay + readout_duration) \
            .add_zero_until(repetition_period)

        ro_pb.add_zero_pulse(pi_pulse_duration + readout_delay) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        return {'q_seqs': [exc_pb.build()],
                'ro_seqs': [ro_pb.build()]}

    @staticmethod
    def build_dispersive_hahn_echo_sequences(pulse_sequence_parameters, **pbs):

        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        half_pi_pulse_duration = \
            pulse_sequence_parameters["half_pi_pulse_duration"]
        echo_delay = \
            pulse_sequence_parameters["echo_delay"]
        window = \
            pulse_sequence_parameters["modulating_window"]
        amplitude = \
            pulse_sequence_parameters["excitation_amplitude"]
        exc_pb = pbs['q_pbs'][0]
        ro_pb = pbs['ro_pbs'][0]
        exc_pb.add_zero_pulse(awg_trigger_reaction_delay) \
            .add_sine_pulse(half_pi_pulse_duration, amplitude_mult=amplitude,
                            window=window) \
            .add_zero_pulse(echo_delay / 2) \
            .add_sine_pulse(half_pi_pulse_duration,
                            amplitude_mult=2 * amplitude, window=window) \
            .add_zero_pulse(echo_delay / 2) \
            .add_sine_pulse(half_pi_pulse_duration, amplitude_mult=amplitude,
                            window=window) \
            .add_zero_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        ro_pb.add_zero_pulse(4 * half_pi_pulse_duration + echo_delay + 10) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        return {'q_seqs': [exc_pb.build()],
                'ro_seqs': [ro_pb.build()]}

    @staticmethod
    def build_radial_tomography_pulse_sequences(pulse_sequence_parameters,
                                                **pbs):
        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        tomo_phase = \
            pulse_sequence_parameters["tomo_phase"]
        prep_pulse = \
            pulse_sequence_parameters[
                "prep_pulse"]  # list with strings of pulses, i.e. '+X/2'
        prep_pulse_pi_amplitude = \
            pulse_sequence_parameters["prep_pulse_pi_amplitude"]
        tomo_delay = \
            pulse_sequence_parameters["tomo_delay"]
        padding = \
            pulse_sequence_parameters["padding"]
        pulse_length = \
            pulse_sequence_parameters["pulse_length"]
        z_pulse_offset_voltage = \
            pulse_sequence_parameters["z_pulse_offset_voltage"]
        z_pulse_duration = \
            pulse_sequence_parameters["z_pulse_duration"]
        z_smoothing_coefficient = \
            pulse_sequence_parameters["z_smoothing_coefficient"]
        tomo_pulse_amplitude = \
            pulse_sequence_parameters["tomo_pulse_amplitude"]
        window = \
            pulse_sequence_parameters["modulating_window"]

        exc_pb = pbs['q_pbs'][0]
        ro_pb = pbs['ro_pbs'][0]
        z_pb = pbs['q_z_pbs'][0]

        try:
            hd_amplitude = \
                pulse_sequence_parameters["hd_amplitude"]
        except KeyError:
            hd_amplitude = 0

        prep_total_duration = 0
        exc_pb.add_zero_pulse(awg_trigger_reaction_delay)
        for idx, pulse_str in enumerate(prep_pulse):
            if pulse_str[1] != "Z":
                exc_pb.add_sine_pulse_from_string(pulse_str,
                                                  pulse_length,
                                                  prep_pulse_pi_amplitude,
                                                  window=window)
                exc_pb.add_zero_pulse(padding)
                z_pb.add_zero_pulse(pulse_length + padding)
                prep_total_duration += pulse_length + padding
            elif pulse_str[1] == "Z":
                z_pb.add_rect_pulse(z_pulse_duration, z_pulse_offset_voltage,
                                    z_smoothing_coefficient)
                z_pb.add_zero_pulse(padding)
                exc_pb.add_zero_pulse(z_pulse_duration + padding)
                prep_total_duration += z_pulse_duration + padding

        exc_pb.add_zero_pulse(tomo_delay) \
            .add_sine_pulse(pulse_length, tomo_phase,
                            amplitude_mult=tomo_pulse_amplitude, window=window,
                            hd_amplitude=hd_amplitude) \
            .add_zero_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        ro_pb.add_zero_pulse(prep_total_duration + tomo_delay + pulse_length) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        z_pb.add_zero_until(repetition_period)

        return {'q_seqs': [exc_pb.build()],
                'q_z_seq': [z_pb.build()],
                'ro_seqs': [ro_pb.build()]}

    @staticmethod
    def build_dispersive_APE_sequences(pulse_sequence_parameters, **pbs):

        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        half_pi_pulse_duration = \
            pulse_sequence_parameters["half_pi_pulse_duration"]
        ramsey_angle = \
            pulse_sequence_parameters["ramsey_angle"]
        pseudo_I_pulses_count = \
            pulse_sequence_parameters["pseudo_I_pulses_count"]
        window = \
            pulse_sequence_parameters["modulating_window"]
        amplitude = \
            pulse_sequence_parameters["excitation_amplitude"]
        padding = \
            pulse_sequence_parameters["padding"]
        max_pseudo_I_pulses_count = \
            pulse_sequence_parameters["max_pseudo_I_pulses_count"]
        exc_pb = pbs['q_pbs'][0]
        ro_pb = pbs['ro_pbs'][0]
        try:
            hd_amplitude = \
                pulse_sequence_parameters["hd_amplitude"]
        except KeyError:
            hd_amplitude = 0

        exc_pb.add_zero_pulse(awg_trigger_reaction_delay) \
            .add_sine_pulse(half_pi_pulse_duration, 0,
                            amplitude_mult=amplitude, window=window,
                            hd_amplitude=hd_amplitude) \
            .add_zero_pulse(padding)

        for i in range(pseudo_I_pulses_count):
            exc_pb.add_sine_pulse(half_pi_pulse_duration, 0,
                                  amplitude_mult=amplitude, window=window,
                                  hd_amplitude=hd_amplitude) \
                .add_zero_pulse(padding) \
                .add_sine_pulse(half_pi_pulse_duration, pi,
                                amplitude_mult=amplitude, window=window,
                                hd_amplitude=hd_amplitude) \
                .add_zero_pulse(padding)

        for i in range(max_pseudo_I_pulses_count - pseudo_I_pulses_count):
            exc_pb.add_zero_pulse(2 * (half_pi_pulse_duration + padding))

        exc_pb.add_sine_pulse(half_pi_pulse_duration, ramsey_angle,
                              amplitude_mult=amplitude, window=window,
                              hd_amplitude=hd_amplitude) \
            .add_zero_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        ro_pb.add_zero_pulse(2 * half_pi_pulse_duration + padding + \
                             max_pseudo_I_pulses_count * 2 * (
                                     padding + half_pi_pulse_duration)) \
            .add_zero_pulse(padding).add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        return {'q_seqs': [exc_pb.build()],
                'ro_seqs': [ro_pb.build()]}

    @staticmethod
    def build_dispersive_pi_half_calibration_sequences(
            pulse_sequence_parameters, **pbs):
        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        half_pi_pulse_duration = \
            pulse_sequence_parameters["half_pi_pulse_duration"]
        twice_pi_half_pulses_count = \
            pulse_sequence_parameters["twice_pi_half_pulses_count"]
        window = \
            pulse_sequence_parameters["modulating_window"]
        amplitude = \
            pulse_sequence_parameters["excitation_amplitude"]
        padding = \
            pulse_sequence_parameters["padding"]
        exc_pb = pbs['q_pbs'][0]
        ro_pb = pbs['ro_pbs'][0]
        try:
            hd_amplitude = \
                pulse_sequence_parameters["hd_amplitude"]
        except KeyError:
            hd_amplitude = 0

        exc_pb.add_zero_pulse(awg_trigger_reaction_delay) \
            .add_sine_pulse(half_pi_pulse_duration, 0,
                            amplitude_mult=amplitude,
                            window=window,
                            hd_amplitude=hd_amplitude).add_zero_pulse(padding)

        for i in range(twice_pi_half_pulses_count):
            exc_pb.add_sine_pulse(half_pi_pulse_duration, 0,
                                  amplitude_mult=amplitude,
                                  window=window, hd_amplitude=hd_amplitude) \
                .add_zero_pulse(padding) \
                .add_sine_pulse(half_pi_pulse_duration, 0,
                                amplitude_mult=amplitude,
                                window=window, hd_amplitude=hd_amplitude) \
                .add_zero_pulse(padding)

        exc_pb.add_zero_until(repetition_period)

        ro_pb.add_zero_pulse((half_pi_pulse_duration + padding) * (
                1 + 2 * twice_pi_half_pulses_count)) \
            .add_dc_pulse(readout_duration).add_zero_until(repetition_period)

        return {'q_seqs': [exc_pb.build()],
                'ro_seqs': [ro_pb.build()]}

    @staticmethod
    def build_interleaved_benchmarking_sequence(pulse_sequence_parameters,
                                                **pbs):
        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        window = \
            pulse_sequence_parameters["modulating_window"]
        pulse_duration = \
            pulse_sequence_parameters["pulse_duration"]
        pi_pulse_amplitude = \
            pulse_sequence_parameters["pi_pulse_amplitude"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]

        padding = \
            pulse_sequence_parameters["padding"]
        benchmarking_sequence = \
            pulse_sequence_parameters["benchmarking_sequence"]
        exc_pb = pbs['q_pbs'][0]
        ro_pb = pbs['ro_pbs'][0]
        try:
            hd_amplitude = \
                pulse_sequence_parameters["hd_amplitude"]
        except KeyError:
            hd_amplitude = 0

        exc_pb.add_zero_pulse(awg_trigger_reaction_delay)
        global_phase = 0
        excitation_duration = 0
        for idx, pulse_str in enumerate(benchmarking_sequence):
            exc_pb.add_sine_pulse_from_string(pulse_str,
                                              pulse_duration,
                                              pi_pulse_amplitude, window)
            exc_pb.add_zero_pulse(padding)
            excitation_duration += pulse_duration + padding
        exc_pb.add_zero_until(repetition_period)
        ro_pb.add_zero_pulse(excitation_duration) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        return {'q_seqs': [exc_pb.build()],
                'ro_seqs': [ro_pb.build()]}

    @staticmethod
    def build_z_pulse_profile_scan_sequence(pulse_sequence_parameters, **pbs):
        """
        Returns synchronized excitation and readout IQPulseSequences assuming that
        readout AWG is triggered by the excitation AWG
        """
        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        pi_pulse_duration = \
            pulse_sequence_parameters["pi_pulse_duration"]
        window = \
            pulse_sequence_parameters["modulating_window"]
        z_pulse_offset_voltage = \
            pulse_sequence_parameters["z_pulse_offset_voltage"]
        z_pulse_duration = \
            pulse_sequence_parameters["z_pulse_duration"]
        pi_pulse_delay = \
            pulse_sequence_parameters["pi_pulse_delay"]
        amplitude = \
            pulse_sequence_parameters["excitation_amplitude"]
        z_smoothing_coefficient = \
            pulse_sequence_parameters["z_smoothing_coefficient"]
        exc_pb = pbs['q_pbs'][0]
        ro_pb = pbs['ro_pbs'][0]
        z_pb = pbs['q_z_pbs'][0]

        z_wait = abs(pi_pulse_delay) if pi_pulse_delay < 0 else 0
        exc_wait = abs(pi_pulse_delay) if pi_pulse_delay > 0 else 0

        exc_pb.add_zero_pulse(awg_trigger_reaction_delay + exc_wait) \
            .add_sine_pulse(pi_pulse_duration, 0,
                            amplitude_mult=amplitude, window=window) \
            .add_zero_pulse(readout_duration) \
            .add_zero_until(repetition_period)
        z_pb.add_zero_pulse(z_wait) \
            .add_rect_pulse(z_pulse_duration, z_pulse_offset_voltage,
                            z_smoothing_coefficient) \
            .add_zero_until(repetition_period)
        ro_pb.add_zero_pulse(max(pi_pulse_duration, z_pulse_duration)
                             + abs(pi_pulse_delay) + 10) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        return {'q_seqs': [exc_pb.build()],
                'q_z_seqs': [z_pb.build()],
                # <<<< " 'q_z_seq': [z_pb.build()], " haha
                'ro_seqs': [ro_pb.build()]}

    @staticmethod
    def build_z_pulse_ramsey_sequences(pulse_sequence_parameters, **pbs):
        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        pi_pulse_duration = \
            pulse_sequence_parameters["pi_pulse_duration"]
        window = \
            pulse_sequence_parameters["modulating_window"]
        z_pulse_offset_voltage = \
            pulse_sequence_parameters["z_pulse_offset_voltage"]
        z_pulse_duration = \
            pulse_sequence_parameters["z_pulse_duration"]
        padding = \
            pulse_sequence_parameters["padding"]
        amplitude = \
            pulse_sequence_parameters["excitation_amplitude"]
        z_smoothing_coefficient = \
            pulse_sequence_parameters["z_smoothing_coefficient"]
        exc_pb = pbs['q_pbs'][0]
        ro_pb = pbs['ro_pbs'][0]
        z_pb = pbs['q_z_pbs'][0]

        exc_pb.add_zero_pulse(awg_trigger_reaction_delay) \
            .add_sine_pulse(0.5 * pi_pulse_duration, 0,
                            amplitude_mult=amplitude, window=window) \
            .add_zero_pulse(2 * padding + z_pulse_duration) \
            .add_sine_pulse(0.5 * pi_pulse_duration, 0,
                            amplitude_mult=amplitude, window=window) \
            .add_zero_until(repetition_period)

        z_pb.add_zero_pulse(0.5 * pi_pulse_duration + padding) \
            .add_rect_pulse(z_pulse_duration, z_pulse_offset_voltage,
                            z_smoothing_coefficient) \
            .add_zero_until(repetition_period)

        ro_pb.add_zero_pulse(
            pi_pulse_duration + 2 * padding + z_pulse_duration) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        return {'q_seqs': [exc_pb.build()],
                'q_z_seqs': [z_pb.build()],  # <<<< 'q_z_seq': [z_pb.build()],
                'ro_seqs': [ro_pb.build()]}

    @staticmethod
    def build_ramsey_comparison_sequences0(pulse_sequence_parameters, **pbs):
        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        half_pi_pulse_duration = \
            pulse_sequence_parameters["half_pi_pulse_duration"]
        pi_pulse_control_duration = \
            pulse_sequence_parameters["pi_pulse_control_duration"]
        ramsey_delay = \
            pulse_sequence_parameters["ramsey_delay"]
        window = \
            pulse_sequence_parameters["modulating_window"]
        amplitude1 = \
            pulse_sequence_parameters["pi_pulse_amplitudes"][0]
        padding = \
            pulse_sequence_parameters["padding"]
        exc_pb1 = pbs['q_pbs'][0]
        exc_pb2 = pbs['q_pbs'][1]  # control qubit awg
        ro_pb = pbs['ro_pbs'][0]

        exc_pb1.add_zero_pulse(awg_trigger_reaction_delay) \
            .add_zero_pulse(pi_pulse_control_duration) \
            .add_sine_pulse(half_pi_pulse_duration,
                            amplitude_mult=amplitude1, window=window) \
            .add_zero_pulse(ramsey_delay) \
            .add_sine_pulse(half_pi_pulse_duration,
                            amplitude_mult=amplitude1, window=window) \
            .add_zero_pulse(padding) \
            .add_zero_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        exc_pb2.add_zero_until(repetition_period)  # nothing happens

        ro_pb.add_zero_pulse(
            pi_pulse_control_duration + half_pi_pulse_duration + ramsey_delay + half_pi_pulse_duration + padding) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        return {'q_seqs': [exc_pb1.build(), exc_pb2.build()],
                'ro_seqs': [ro_pb.build()]}

    @staticmethod
    def build_ramsey_comparison_sequences1(pulse_sequence_parameters, **pbs):
        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        half_pi_pulse_duration = \
            pulse_sequence_parameters["half_pi_pulse_duration"]
        pi_pulse_control_duration = \
            pulse_sequence_parameters["pi_pulse_control_duration"]
        ramsey_delay = \
            pulse_sequence_parameters["ramsey_delay"]
        window = \
            pulse_sequence_parameters["modulating_window"]
        amplitude1 = \
            pulse_sequence_parameters["pi_pulse_amplitudes"][0]
        amplitude2 = \
            pulse_sequence_parameters["pi_pulse_amplitudes"][1]
        padding = \
            pulse_sequence_parameters["padding"]
        exc_pb1 = pbs['q_pbs'][0]  # trigger master awg
        exc_pb2 = pbs['q_pbs'][1]  # control qubit awg
        ro_pb = pbs['ro_pbs'][0]

        exc_pb1.add_zero_pulse(awg_trigger_reaction_delay) \
            .add_zero_pulse(pi_pulse_control_duration) \
            .add_sine_pulse(half_pi_pulse_duration,
                            amplitude_mult=amplitude1, window=window) \
            .add_zero_pulse(ramsey_delay) \
            .add_sine_pulse(half_pi_pulse_duration,
                            amplitude_mult=amplitude1, window=window) \
            .add_zero_pulse(padding) \
            .add_zero_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        exc_pb2.add_sine_pulse(pi_pulse_control_duration,
                               amplitude_mult=amplitude2, window=window) \
            .add_zero_until(repetition_period)

        ro_pb.add_zero_pulse(
            pi_pulse_control_duration + half_pi_pulse_duration + ramsey_delay + half_pi_pulse_duration + padding) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        return {'q_seqs': [exc_pb1.build(), exc_pb2.build()],
                'ro_seqs': [ro_pb.build()]}

    @staticmethod
    def build_ramsey_comparison_sequences0_multiplexed(
            pulse_sequence_parameters, **pbs):

        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        half_pi_pulse_duration = \
            pulse_sequence_parameters["half_pi_pulse_duration"]
        ramsey_delay = \
            pulse_sequence_parameters["ramsey_delay"]
        window = \
            pulse_sequence_parameters["modulating_window"]
        amplitude = \
            pulse_sequence_parameters["excitation_amplitude"]
        padding = \
            pulse_sequence_parameters["padding"]
        exc_pb = pbs['q_pbs'][0][0]
        ro_pb = pbs['ro_pbs'][0]

        exc_pb.add_zero_pulse(awg_trigger_reaction_delay) \
            .add_sine_pulse(half_pi_pulse_duration,
                            amplitude_mult=amplitude, window=window) \
            .add_zero_pulse(ramsey_delay) \
            .add_sine_pulse(half_pi_pulse_duration,
                            amplitude_mult=amplitude, window=window) \
            .add_zero_pulse(padding) \
            .add_zero_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        ro_pb.add_zero_pulse(2 * half_pi_pulse_duration + ramsey_delay + 10) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        return {'q_seqs': [exc_pb.build()],
                'ro_seqs': [ro_pb.build()]}

    @staticmethod
    def build_ramsey_comparison_sequences1_multiplexed(
            pulse_sequence_parameters, **pbs):
        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        half_pi_pulse_duration = \
            pulse_sequence_parameters["half_pi_pulse_duration"]
        pi_pulse2_duration = \
            pulse_sequence_parameters["pi_pulse2_duration"]
        ramsey_delay = \
            pulse_sequence_parameters["ramsey_delay"]
        window = \
            pulse_sequence_parameters["modulating_window"]
        amplitude = \
            pulse_sequence_parameters["excitation_amplitude"]
        padding = \
            pulse_sequence_parameters["padding"]
        exc_pb1 = pbs['q_pbs'][0][0]
        exc_pb2 = pbs['q_pbs'][0][1]
        ro_pb = pbs['ro_pbs'][0]

        exc_ps1 = exc_pb2.add_zero_pulse(awg_trigger_reaction_delay) \
            .add_sine_pulse(pi_pulse2_duration,
                            amplitude_mult=amplitude, window=window) \
            .add_zero_pulse(padding).build()
        exc_ps2 = exc_pb1.add_sine_pulse(half_pi_pulse_duration,
                                         amplitude_mult=amplitude,
                                         window=window) \
            .add_zero_pulse(ramsey_delay) \
            .add_sine_pulse(half_pi_pulse_duration,
                            amplitude_mult=amplitude, window=window) \
            .add_zero_pulse(readout_duration) \
            .add_zero_until(repetition_period).build()
        exc_ps = exc_ps1 + exc_ps2

        ro_pb.add_zero_pulse(
            2 * half_pi_pulse_duration + ramsey_delay + pi_pulse2_duration + padding + 10) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        return {'q_seqs': [exc_ps],
                'ro_seqs': [ro_pb.build()]}

    @staticmethod
    def build_rabi_comparison_sequences0(pulse_sequence_parameters, **pbs):
        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        pi_pulse_control_duration = \
            pulse_sequence_parameters["pi_pulse_control_duration"]
        rabi_pulse_duration = \
            pulse_sequence_parameters["rabi_pulse_duration"]
        window = \
            pulse_sequence_parameters["modulating_window"]
        amplitude1 = \
            pulse_sequence_parameters["pi_pulse_amplitudes"][0]
        padding = \
            pulse_sequence_parameters["padding"]
        exc_pb1 = pbs['q_pbs'][0]
        exc_pb2 = pbs['q_pbs'][1]
        ro_pb = pbs['ro_pbs'][0]

        exc_pb1.add_zero_pulse(awg_trigger_reaction_delay) \
            .add_zero_pulse(pi_pulse_control_duration) \
            .add_sine_pulse(rabi_pulse_duration,
                            amplitude_mult=amplitude1, window=window) \
            .add_zero_pulse(padding) \
            .add_zero_until(repetition_period)

        exc_pb2.add_zero_until(repetition_period)

        ro_pb.add_zero_pulse(
            pi_pulse_control_duration + rabi_pulse_duration + padding) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        return {'q_seqs': [exc_pb1.build(), exc_pb2.build()],
                'ro_seqs': [ro_pb.build()]}

    @staticmethod
    def build_rabi_comparison_sequences1(pulse_sequence_parameters, **pbs):
        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        pi_pulse_control_duration = \
            pulse_sequence_parameters["pi_pulse_control_duration"]
        rabi_pulse_duration = \
            pulse_sequence_parameters["rabi_pulse_duration"]
        window = \
            pulse_sequence_parameters["modulating_window"]
        amplitude1 = \
            pulse_sequence_parameters["pi_pulse_amplitudes"][0]
        amplitude2 = \
            pulse_sequence_parameters["pi_pulse_amplitudes"][1]
        padding = \
            pulse_sequence_parameters["padding"]
        exc_pb1 = pbs['q_pbs'][0]
        exc_pb2 = pbs['q_pbs'][1]
        ro_pb = pbs['ro_pbs'][0]

        exc_pb1.add_zero_pulse(awg_trigger_reaction_delay) \
            .add_zero_pulse(pi_pulse_control_duration) \
            .add_sine_pulse(rabi_pulse_duration,
                            amplitude_mult=amplitude1, window=window) \
            .add_zero_pulse(padding) \
            .add_zero_until(repetition_period)

        exc_pb2.add_sine_pulse(pi_pulse_control_duration,
                               amplitude_mult=amplitude2, window=window) \
            .add_zero_until(repetition_period)

        ro_pb.add_zero_pulse(
            pi_pulse_control_duration + rabi_pulse_duration + padding) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        return {'q_seqs': [exc_pb1.build(), exc_pb2.build()],
                'ro_seqs': [ro_pb.build()]}

    @staticmethod
    def build_rabi_comparison_sequences0_multiplexed(pulse_sequence_parameters,
                                                     **pbs):
        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        pi_pulse2_duration = \
            pulse_sequence_parameters["pi_pulse2_duration"]
        rabi_pulse_duration = \
            pulse_sequence_parameters["rabi_pulse_duration"]
        window = \
            pulse_sequence_parameters["modulating_window"]
        amplitude = \
            pulse_sequence_parameters["excitation_amplitude"]
        padding = \
            pulse_sequence_parameters["padding"]
        exc_pb1 = pbs['q_pbs'][0][0]
        exc_pb2 = pbs['q_pbs'][0][1]
        ro_pb = pbs['ro_pbs'][0]

        exc_ps1 = exc_pb1.add_zero_pulse(awg_trigger_reaction_delay).build()
        exc_ps2 = exc_pb2.add_sine_pulse(rabi_pulse_duration) \
            .add_zero_pulse(padding) \
            .add_zero_pulse(readout_duration) \
            .add_zero_until(repetition_period).build()

        exc_ps = exc_ps1 + exc_ps2

        ro_pb.add_zero_pulse(rabi_pulse_duration + padding) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        return {'q_seqs': [exc_ps],
                'ro_seqs': [ro_pb.build()]}

    @staticmethod
    def build_rabi_comparison_sequences1_multiplexed(pulse_sequence_parameters,
                                                     **pbs):
        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        pi_pulse2_duration = \
            pulse_sequence_parameters["pi_pulse2_duration"]
        rabi_pulse_duration = \
            pulse_sequence_parameters["rabi_pulse_duration"]
        window = \
            pulse_sequence_parameters["modulating_window"]
        amplitude = \
            pulse_sequence_parameters["excitation_amplitude"]
        padding = \
            pulse_sequence_parameters["padding"]
        exc_pb1 = pbs['q_pbs'][0][0]
        exc_pb2 = pbs['q_pbs'][0][1]
        ro_pb = pbs['ro_pbs'][0]

        exc_ps1 = exc_pb1.add_zero_pulse(awg_trigger_reaction_delay) \
            .add_sine_pulse(pi_pulse2_duration,
                            amplitude_mult=amplitude, window=window) \
            .add_zero_pulse(padding).build()
        exc_ps2 = exc_pb2.add_sine_pulse(rabi_pulse_duration) \
            .add_zero_pulse(padding) \
            .add_zero_pulse(readout_duration) \
            .add_zero_until(repetition_period).build()

        exc_ps = exc_ps1 + exc_ps2

        ro_pb.add_zero_pulse(
            pi_pulse2_duration + padding + rabi_pulse_duration + padding) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        return {'q_seqs': [exc_ps],
                'ro_seqs': [ro_pb.build()]}

    @staticmethod
    def build_vacuum_ramsey_oscillations_sequences(pulse_sequence_parameters,
                                                   **pbs):
        awg_trigger_reaction_delay_z = \
            pulse_sequence_parameters["awg_trigger_reaction_delay_z"]
        awg_trigger_reaction_delay_ro = \
            pulse_sequence_parameters["awg_trigger_reaction_delay_ro"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        pi_pulse1_duration = \
            pulse_sequence_parameters["pi_pulse_duration_osc"]
        pi_pulse2_duration = \
            pulse_sequence_parameters["pi_pulse_duration_control"]
        window = \
            pulse_sequence_parameters["modulating_window"]
        z_pulse_offset_voltage = \
            pulse_sequence_parameters["z_pulse_offset_voltage"]
        z_pulse_duration = \
            pulse_sequence_parameters["interaction_duration"]
        padding = \
            pulse_sequence_parameters["padding"]
        amplitudes = \
            pulse_sequence_parameters["pi_pulse_amplitudes"]
        z_smoothing_coefficient = \
            pulse_sequence_parameters["z_smoothing_coefficient"]
        exc_pb_cal1 = pbs['q_pbs'][0][0]
        exc_pb_cal2 = pbs['q_pbs'][0][1]
        ro_pb = pbs['ro_pbs'][0]
        z_pb = pbs['q_z_pbs'][0]

        exc_ps1 = exc_pb_cal2.add_zero_pulse(awg_trigger_reaction_delay_z) \
            .add_sine_pulse(pi_pulse2_duration) \
            .add_zero_pulse(padding).build()
        exc_ps2 = exc_pb_cal1.add_sine_pulse(0.5 * pi_pulse1_duration, 0,
                                             amplitude_mult=amplitudes[0],
                                             window=window) \
            .add_zero_pulse(2 * padding + z_pulse_duration) \
            .add_sine_pulse(0.5 * pi_pulse1_duration, 0,
                            amplitude_mult=amplitudes[0], window=window) \
            .add_zero_until(repetition_period).build()

        exc_ps = exc_ps1 + exc_ps2

        z_pb.add_zero_pulse(
            pi_pulse2_duration + padding + 0.5 * pi_pulse1_duration + padding) \
            .add_rect_pulse(z_pulse_duration, z_pulse_offset_voltage,
                            z_smoothing_coefficient) \
            .add_zero_until(repetition_period)

        ro_trigger_compensation = awg_trigger_reaction_delay_z - awg_trigger_reaction_delay_ro
        ro_pb.add_zero_pulse(
            pi_pulse2_duration + padding + 0.5 * pi_pulse1_duration + \
            padding + z_pulse_duration + padding + 0.5 * pi_pulse1_duration + padding + \
            ro_trigger_compensation) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        return {'q_seqs': [exc_ps],
                'q_z_seqs': [z_pb.build()],  # <<<< 'q_z_seq': [z_pb.build()],
                'ro_seqs': [ro_pb.build()]}

    @staticmethod
    def build_vacuum_rabi_oscillations_sequences(pulse_sequence_parameters,
                                                 **pbs):
        pulse_sequence_parameters["z_pulse_duration"] = \
            pulse_sequence_parameters["interaction_duration"]

        return IQPulseBuilder.build_z_pulse_profile_scan_sequence(
            pulse_sequence_parameters, **pbs)

    @staticmethod
    def build_dispersive_rabi_2qubit_sequences(pulse_sequence_parameters,
                                               **pbs):  # TODO
        """
        TODO
        Synchronized pulse sequence for 2 qubits local excitations generators and one joint readout
        """

        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        excitation_duration = \
            pulse_sequence_parameters["excitation_duration"]
        window = \
            pulse_sequence_parameters["modulating_window"]
        amplitude_1, amplitude_2 = \
            (pulse_sequence_parameters["excitation_amplitude"],) * 2
        if 'excitation_amplitude_2' in pulse_sequence_parameters.keys():
            amplitude_2 = \
                pulse_sequence_parameters["excitation_amplitude_2"]

        exc_pbs = pbs['q_pbs']
        ro_pb = pbs['ro_pbs'][0]

        exc_pbs[0].add_zero_pulse(awg_trigger_reaction_delay) \
            .add_sine_pulse(excitation_duration, 0, amplitude_mult=amplitude_1,
                            window=window) \
            .add_zero_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        exc_pbs[1].add_zero_pulse(awg_trigger_reaction_delay) \
            .add_sine_pulse(excitation_duration, 0, amplitude_mult=amplitude_2,
                            window=window) \
            .add_zero_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        ro_pb.add_zero_pulse(excitation_duration + 10) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        return {'q_seqs': [exc_pbs[0].build(), exc_pbs[1].build()],
                'ro_seqs': [ro_pb.build()]}

    @staticmethod
    def build_dispersive_rabi_2qubit_sequences2(pulse_sequence_parameters,
                                                **pbs):  # TODO
        """
        TODO
        Synchronized pulse sequence for 2 qubits local excitations generators and one joint readout
        """
        # print("I just met you and this is crazy, but here's my number, so call me maybe")
        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        excitation_duration = \
            pulse_sequence_parameters["excitation_duration"]
        window = \
            pulse_sequence_parameters["modulating_window"]
        amplitude_1, amplitude_2 = \
            (pulse_sequence_parameters["excitation_amplitude"],) * 2
        if 'excitation_amplitude_2' in pulse_sequence_parameters.keys():
            amplitude_2 = \
                pulse_sequence_parameters["excitation_amplitude_2"]
        readout_offset_voltage = \
            pulse_sequence_parameters["readout_offset_voltage"]
        pulses_padding = \
            pulse_sequence_parameters["pulses_padding"]
        ro_padding = \
            pulse_sequence_parameters["ro_padding"]

        exc_pbs = pbs['q_pbs']
        ro_pb = pbs['ro_pbs'][0]
        z_pb = pbs['q_z_pbs'][0]

        exc_pbs[0][1]._iqmx_calibration._dc_offsets = array([0, 0])
        exc_pbs[0][1]._iqmx_calibration._if_offsets = array([0, 0])
        exc_pbs[0][0].add_zero_pulse(awg_trigger_reaction_delay) \
            .add_sine_pulse(excitation_duration, 0, amplitude_mult=amplitude_1,
                            window=window) \
            .add_zero_until(repetition_period)

        exc_pbs[0][1].add_zero_pulse(awg_trigger_reaction_delay) \
            .add_sine_pulse(excitation_duration, 0, amplitude_mult=amplitude_2,
                            window=window) \
            .add_zero_until(repetition_period)

        ro_pb.add_zero_pulse(excitation_duration + ro_padding) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        z_pb.add_zero_pulse(excitation_duration + ro_padding) \
            .add_rect_pulse(readout_duration, readout_offset_voltage, 1) \
            .add_zero_until(repetition_period)

        return {'q_seqs': [
            exc_pbs[0][0].build().direct_add(exc_pbs[0][1].build())],
            'ro_seqs': [ro_pb.build()],
            'q_z_seqs': [z_pb.build()]}

    @staticmethod
    def build_joint_tomography_pulse_sequences(pulse_sequence_parameters,
                                               **pbs):
        # TODO New, check required
        awg_trigger_reaction_delays = \
            pulse_sequence_parameters["awg_trigger_reaction_delays"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        padding = pulse_sequence_parameters["padding"]
        pi_pulse_lengths = \
            pulse_sequence_parameters["pi_pulse_lengths"]
        window = \
            pulse_sequence_parameters["modulating_window"]
        tomo_rotations = \
            pulse_sequence_parameters["tomo_local_rotations"]
        prep_pulses = \
            pulse_sequence_parameters[
                "prep_pulses"]  # Tuple of arrays with strings of pulses, i.e. '+X/2'
        # ( [qubit_1_preparation_list], [qubit_2_preparation_list], ...)
        pulse_pi_amplitudes = pulse_sequence_parameters["pulse_pi_amplitudes"]

        number_of_qubits = len(prep_pulses)
        exc_pbs = pbs['q_pbs']
        z_pbs = pbs['q_z_pbs']
        ro_pb = pbs['ro_pbs'][0]
        total_durations = [
                              0] * number_of_qubits  # total duration of pulses passed to each qubit

        # state pulse sequence build
        for qubit in range(number_of_qubits):
            exc_pbs[qubit].add_zero_pulse(awg_trigger_reaction_delays[qubit])
            for com in prep_pulses[qubit]:
                total_durations[qubit] += build_1q_pulse_from_command(com,
                                                                      exc_pbs[
                                                                          qubit],
                                                                      z_pbs[
                                                                          qubit],
                                                                      pulse_sequence_parameters,
                                                                      qubit=qubit)
            exc_pbs[qubit].add_zero_pulse(padding)
            total_durations[qubit] += padding

        total_durations_max = max(total_durations)
        # equalize all qubit pulse sequences lengths
        for qubit in range(number_of_qubits):
            exc_pbs[qubit].add_zero_until(
                total_durations_max + awg_trigger_reaction_delays[qubit])
        total_durations = [total_durations_max] * number_of_qubits

        # tomography rotations pulse sequence build
        for qubit in range(number_of_qubits):
            total_durations[qubit] += build_1q_pulse_from_command(
                tomo_rotations[qubit],
                exc_pbs[qubit], z_pbs[qubit],
                pulse_sequence_parameters,
                qubit=qubit)
            exc_pbs[qubit].add_zero_pulse(padding) \
                .add_zero_pulse(readout_duration) \
                .add_zero_until(repetition_period)
            z_pbs[qubit].add_zero_until(repetition_period)

        total_durations_max = max(total_durations)
        ro_pb.add_zero_pulse(total_durations_max + padding) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        return {'q_seqs': [exc_pbs[0].build(), exc_pbs[1].build()],
                'q_z_seqs': [z_pbs[0].build(), z_pbs[1].build()],
                'ro_seqs': [ro_pb.build()]}

    # version for single AWG and 2 qubits
    @staticmethod
    def build_joint_tomography_pulse_sequences_multiplex(
            pulse_sequence_parameters, **pbs):
        # TODO New, check required
        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        tomo_delay = \
            pulse_sequence_parameters["tomo_delay"]
        tomo_rotations = \
            pulse_sequence_parameters["tomo_local_rotations"]
        prep_pulses = \
            pulse_sequence_parameters[
                "prep_pulses"]  # Tuple of arrays with strings of pulses, i.e. '+X/2'
        # ( [qubit_1_preparation_list], [qubit_2_preparation_list], ...)
        padding = pulse_sequence_parameters["padding"]

        exc_pb1 = pbs['q_pbs'][0][0]
        exc_pb2 = pbs["q_pbs"][0][1]
        exc_pb2._iqmx_calibration._dc_offsets = array([0, 0])
        exc_pb2._iqmx_calibration._if_offsets = array([0, 0])

        z_pb = pbs['q_z_pbs'][0]
        ro_pb = pbs['ro_pbs'][0]
        prep_total_duration = 0
        tomo_total_duration = 0

        # prep for qubit 0
        exc_pb1.add_zero_pulse(awg_trigger_reaction_delay)
        exc_pb2.add_zero_pulse(awg_trigger_reaction_delay)

        for com in prep_pulses[0]:
            duration = build_1q_pulse_from_command(com, exc_pb1, z_pb,
                                                   pulse_sequence_parameters,
                                                   qubit=0)
            exc_pb2.add_zero_pulse(duration)
            prep_total_duration += duration

            # prep for qubit 1
        for com in prep_pulses[1]:
            duration = build_1q_pulse_from_command(com, exc_pb2, z_pb,
                                                   pulse_sequence_parameters,
                                                   qubit=1)
            exc_pb1.add_zero_pulse(duration)
            prep_total_duration += duration

        if pulse_sequence_parameters["gate_type"] is "SWAP":
            z_pulse_offset_voltage = \
                pulse_sequence_parameters["z_pulse_offset_voltage"]
            z_dur = pulse_sequence_parameters["z_pulse_duration"]
            exc_pb1.add_zero_pulse(z_dur + 2 * padding)
            exc_pb2.add_zero_pulse(z_dur + 2 * padding)
            z_smoothing_coefficient = pulse_sequence_parameters[
                "z_smoothing_coefficient"]
            z_pb.add_zero_pulse(padding) \
                .add_rect_pulse(z_dur, offset_voltage=z_pulse_offset_voltage,
                                tanh_sigma=z_smoothing_coefficient) \
                .add_zero_pulse(padding)
            prep_total_duration += z_dur + 2 * padding

        exc_pb1.add_zero_pulse(tomo_delay)
        exc_pb2.add_zero_pulse(tomo_delay)

        # tomo pulses for qubit 0

        duration = build_1q_pulse_from_command(tomo_rotations[0], exc_pb1,
                                               z_pb, pulse_sequence_parameters,
                                               qubit=0)
        exc_pb2.add_zero_pulse(duration)
        tomo_total_duration += duration

        # tomo pulses for qubit 1
        duration = build_1q_pulse_from_command(tomo_rotations[1], exc_pb2,
                                               z_pb, pulse_sequence_parameters,
                                               qubit=1)
        exc_pb1.add_zero_pulse(duration)
        tomo_total_duration += duration

        ro_pb.add_zero_pulse(
            prep_total_duration + tomo_total_duration + tomo_delay) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        exc_pb1.add_zero_until(repetition_period)
        exc_pb2.add_zero_until(repetition_period)

        pulse_seq = exc_pb1.build().direct_add(exc_pb2.build())

        return {'q_seqs': [pulse_seq],
                'q_z_seqs': [z_pb.build()],
                'ro_seqs': [ro_pb.build()]}

    @staticmethod
    def build_dispersive_shift_joint_sequences_multiplex(
            pulse_sequence_parameters, **pbs):

        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        prep_pulses = \
            pulse_sequence_parameters["prep_pulses"]

        exc_pb1 = pbs['q_pbs'][0][0]
        exc_pb2 = pbs["q_pbs"][0][1]
        exc_pb2._iqmx_calibration._dc_offsets = array([0, 0])
        exc_pb2._iqmx_calibration._if_offsets = array([0, 0])

        z_pb = pbs['q_z_pbs'][0]
        ro_pb = pbs['ro_pbs'][0]
        prep_total_duration = 0

        # prep for qubit 0
        exc_pb1.add_zero_pulse(awg_trigger_reaction_delay)
        exc_pb2.add_zero_pulse(awg_trigger_reaction_delay)

        for com in prep_pulses[0]:
            duration = build_1q_pulse_from_command(com, exc_pb1, z_pb,
                                                   pulse_sequence_parameters,
                                                   qubit=0)
            exc_pb2.add_zero_pulse(duration)
            prep_total_duration += duration

            # prep for qubit 1
        for com in prep_pulses[1]:
            duration = build_1q_pulse_from_command(com, exc_pb2, z_pb,
                                                   pulse_sequence_parameters,
                                                   qubit=1)
            exc_pb1.add_zero_pulse(duration)
            prep_total_duration += duration

        exc_pb1.add_zero_until(repetition_period)
        exc_pb2.add_zero_until(repetition_period)

        pulse_seq = exc_pb1.build().direct_add(exc_pb2.build())

        ro_pb.add_zero_pulse(prep_total_duration) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        return {'q_seqs': [pulse_seq],
                "q_z_seqs": [z_pb.build()],
                'ro_seqs': [ro_pb.build()]}

    @staticmethod
    def build_dispersive_shift_joint_sequences(pulse_sequence_parameters,
                                               **pbs):

        awg_trigger_reaction_delay_q1 = \
            pulse_sequence_parameters["awg_trigger_reaction_delay_q1"]
        awg_trigger_reaction_delay_q2 = \
            pulse_sequence_parameters["awg_trigger_reaction_delay_q2"]
        awg_trigger_reaction_delay_ro = \
            pulse_sequence_parameters["awg_trigger_reaction_delay_ro"]

        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        prep_pulses = \
            pulse_sequence_parameters["prep_pulses"]

        exc_pb1 = pbs['q_pbs'][0]
        exc_pb2 = pbs["q_pbs"][1]
        exc_pb2._iqmx_calibration._dc_offsets = array([0, 0])
        exc_pb2._iqmx_calibration._if_offsets = array([0, 0])

        z_pb = pbs['q_z_pbs'][0]
        ro_pb = pbs['ro_pbs'][0]
        prep_total_duration_q1 = 0
        prep_total_duration_q2 = 0

        # prep for qubit 0
        exc_pb1.add_zero_pulse(awg_trigger_reaction_delay_q1)
        exc_pb2.add_zero_pulse(awg_trigger_reaction_delay_q2)

        for com in prep_pulses[0]:
            duration = build_1q_pulse_from_command(com, exc_pb1, z_pb,
                                                   pulse_sequence_parameters,
                                                   qubit=0)
            exc_pb2.add_zero_pulse(duration)
            prep_total_duration_q1 += duration

            # prep for qubit 1
        for com in prep_pulses[1]:
            duration = build_1q_pulse_from_command(com, exc_pb2, z_pb,
                                                   pulse_sequence_parameters,
                                                   qubit=1)
            exc_pb1.add_zero_pulse(duration)
            prep_total_duration_q2 += duration

        exc_pb1.add_zero_until(repetition_period)
        exc_pb2.add_zero_until(repetition_period)

        ro_pb.add_zero_pulse(
            awg_trigger_reaction_delay_ro + max(prep_total_duration_q1,
                                                prep_total_duration_q2)) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        return {'q_seqs': [exc_pb1.build(), exc_pb2.build()],
                "q_z_seqs": [z_pb.build()],
                'ro_seqs': [ro_pb.build()]}

    @staticmethod
    def build_cz_calibration_sequence(pulse_sequence_parameters, **pbs):
        """
        Returns synchronized excitation and readout IQPulseSequences assuming that
        readout AWG is triggered by the excitation AWG
        """
        awg_trigger_reaction_delay = \
            pulse_sequence_parameters["awg_trigger_reaction_delay"]
        readout_duration = \
            pulse_sequence_parameters["readout_duration"]
        repetition_period = \
            pulse_sequence_parameters["repetition_period"]
        pi_pulse_duration = \
            pulse_sequence_parameters["pi_pulse_duration"]
        window = \
            pulse_sequence_parameters["modulating_window"]
        pi_pulse_delay = \
            pulse_sequence_parameters["pi_pulse_delay"]
        amplitude = \
            pulse_sequence_parameters["excitation_amplitude"]
        z_smoothing_coefficient = \
            pulse_sequence_parameters["z_smoothing_coefficient"]

        z_pulse_offset_voltage = \
            pulse_sequence_parameters["z_pulse_offset_voltage"]  # TODO name
        z_pulse_duration = \
            pulse_sequence_parameters["z_pulse_duration"]  # TODO name
        cz_shape_params = \
            pulse_sequence_parameters["cz_shape_params"] \
                if 'cz_shape_param' in pulse_sequence_parameters else [1]
        exc_pb = pbs['q_pbs'][0]
        ro_pb = pbs['ro_pbs'][0]
        z_pb = pbs['q_z_pbs'][0]

        z_wait = abs(pi_pulse_delay) if pi_pulse_delay < 0 else 0
        exc_wait = abs(pi_pulse_delay) if pi_pulse_delay > 0 else 0

        exc_pb.add_zero_pulse(awg_trigger_reaction_delay + exc_wait) \
            .add_sine_pulse(pi_pulse_duration, 0,
                            amplitude_mult=amplitude, window=window) \
            .add_zero_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        z_pb.add_zero_pulse(z_wait) \
            .add_cz_pulse(z_pulse_duration, z_pulse_offset_voltage,
                          *cz_shape_params) \
            .add_zero_until(repetition_period)

        ro_pb.add_zero_pulse(max(pi_pulse_duration, z_pulse_duration)
                             + abs(pi_pulse_delay) + 10) \
            .add_dc_pulse(readout_duration) \
            .add_zero_until(repetition_period)

        return {'q_seqs': [exc_pb.build()],
                'q_z_seqs': [z_pb.build()],
                'ro_seqs': [ro_pb.build()]}


def build_1q_pulse_from_command(command, exc_pb, z_pb,
                                pulse_sequence_parameters,
                                qubit=0):
    pi_pulse_length = pulse_sequence_parameters["pi_pulse_lengths"][qubit]
    pulse_pi_amplitude = pulse_sequence_parameters["pulse_pi_amplitudes"][
        qubit]
    window = pulse_sequence_parameters["modulating_window"]
    padding = pulse_sequence_parameters["padding"]

    # z_pulse_offset_voltage = pulse_sequence_parameters["z_pulse_offset_voltages"][qubit]
    # z_pulse_duration = pulse_sequence_parameters["z_pulse_duration"]
    # z_smoothing_coefficient = pulse_sequence_parameters["z_smoothing_coefficient"]

    total_duration = 0
    exc_pb.add_sine_pulse_from_string(command, pi_pulse_length,
                                      pulse_pi_amplitude, window=window)
    exc_pb.add_zero_pulse(padding)
    pulse_ax = command[1]
    pulse_angle = abs(eval(command.replace(pulse_ax, "1")))

    if window is "rectangular" or pulse_ax is "I":
        z_pb.add_zero_pulse(pi_pulse_length * pulse_angle + padding)
        total_duration += pi_pulse_length * pulse_angle + padding
    else:
        z_pb.add_zero_pulse(pi_pulse_length + padding)
        total_duration += pi_pulse_length + padding

    return total_duration

""" IN DEVELOPMENT. NOT USED ANYWHERE."""
import numpy as np
import copy as cpy
import matplotlib.pyplot as plt

from typing import Union, Tuple, List
from lib3.mixers.het_calibrator import CalibrationSingleUp


class PulseSequence:
    """
    Represents concept of sequence of voltages.
    Appending sequences and elementwise summation is implemented.
    Used by `PulseBuilder` to generate complex pulse sequences.
    """
    def __init__(self, waveform_resolution, waveform=None):
        """

        Parameters
        ----------
        waveform_resolution : np.float
            Time interval between adjacent points. Points assumed to be
            equidistant.
        """
        if waveform is None:
            self._waveform = np.empty(0)
        elif isinstance(waveform, (list, np.ndarray)):
            self._waveform = np.array(waveform, dtype=np.float64)
        # dt between pts in ns
        self._waveform_resolution = waveform_resolution

    def append_pulse(self, other, inplace=False):
        """
        Parameters
        ----------
        other : Union[PulseSequence, np.ndarray, List]
            Pulse to be appended representation.
            If not `PulseSequence` instance, `other` is interpreted as array
            of voltages.
        inplace : bool
            True - modifies and returns `self`
            False - creates new `PulseSequence` class instance and returns
            as a result

        Returns 
        -----------
        result : PulseSequence
            Result of appending `other` at the end of `self`.
        """
        if isinstance(other, (np.ndarray, list)):
            points_arr = np.array(other, np.float64)

        elif isinstance(other, PulseSequence):
            if self._waveform_resolution != other._waveform_resolution:
                raise ValueError("`_waveform_resolution` parameters are not "
                                 "equal for pulse sequences to append.")
            points_arr = other._waveform

        result_wf = None
        if len(points_arr) > 1:
            # np.concatenate always creates copy
            result_wf = np.concatenate(
                        (self._waveform, points_arr)
                    )
        else:
            result_wf = self._waveform

        if inplace:
            self._waveform = result_wf
            return self
        elif not inplace:
            return PulseSequence(
                self._waveform_resolution,
                waveform=result_wf
            )

    def __add__(self, other):
        """
        `self + other`.
        Adds another sequence with signal value added elementwise.
        Creates new `PulseSequence` instance.
        Parameters
        ----------
        other : Unionn[PulseSequence, np.ndarray, list]
            object to interpret as another `PulseSequence` with the same
            waveform resolution.

        Returns : PulseSequence
        -------
            Return new instance of `PulseSequence` class.
        """
        wf_res = None
        wf_other = None
        if isinstance(other, (np.ndarray, list)):
            wf_other = np.array(other, np.float64)
        elif isinstance(other, PulseSequence):
            if self._waveform_resolution != other._waveform_resolution:
                self.plot()
                other.plot()
                raise("`_waveform_resolution` parameters are not equal")
            else:
                wf_other = other._waveform

        try:
            wf_res = self._waveform + wf_other
        except Exception as e:
            print("Direct summation is not possible:", e)
            print(self._waveform.shape, other._waveform.shape)
            raise e

        return PulseSequence(
            self._waveform_resolution,
            waveform=wf_res
        )

    def __iadd__(self, other):
        """
        `self += other` - inplace addition. No new instances of
        `PulseSequence` are instantiated.
        Adds another sequence with signal values added elementwise.

        Parameters
        ----------
        other : Union[PulseSequence, np.ndarray, list]
            object to interpret as another `PulseSequence` with the same
            waveform resolution.

        Returns : PulseSequence
        -------
            Return modified self
        """
        wf_res = None
        wf_other = None
        if isinstance(other, (np.ndarray, list)):
            wf_other = np.array(other, np.float64)
        elif isinstance(other, PulseSequence):
            if self._waveform_resolution != other._waveform_resolution:
                self.plot()
                other.plot()
                raise ("`_waveform_resolution` parameters are not equal")
            else:
                wf_other = other._waveform

        self._waveform += wf_other
        return self

    def total_points(self):
        return len(self._waveform)

    def get_duration(self):
        return self._waveform_resolution * self.total_points()

    def get_waveform(self):
        return self._waveform

    def get_waveform_resolution(self):
        return self._waveform_resolution

    def plot(self, **kwargs):
        times = np.linspace(0, self.get_duration(), len(self._waveform))
        plt.plot(times, self._waveform, **kwargs)


class PulseBuilder:
    """
    Build a PulseBuilder instance for generating complex single-channel pulse
    sequences.
    """
    def __init__(self, calibration):
        """
        Parameters
        ----------
        calibration : CalibrationSingleUp
            calibration that shall be utilized while pulses are constructed
        """
        self._calibration = calibration
        self._waveform_resolution = \
            calibration.awg_sampling_period
        self._pulse_seq = PulseSequence(self._waveform_resolution)

    def add_zero_pulse(self, duration, dc_offset=None):
        """
        Adds a pulse with zero (calibrated) amplitude to the sequence

        Parameters:
        -----------
        duration: float
            Duration of the pulse in nanoseconds
        dc_offset : Tuple[(float,float)]

        """
        if dc_offset is None:
            dc_offset = self._calibration.dc_offsets_close[0]

        # Caution: if `duration%self._waveform_resolution != 0` then it will
        # add additional delay between pulses
        N_time_steps = int(round(duration / self._waveform_resolution))
        self._pulse_seq.append_pulse(np.zeros(N_time_steps + 1) + dc_offset)
        return self

    def add_rect_pulse(self, duration, offset_voltage, tanh_sigma=0):
        """
        Adds a pulse with offset_voltage with respect
        to the zero-calibrated voltage:
        absolute_voltage = zero_offset + offset_voltage

        Parameters:
        -----------
        duration: float
            Duration of the pulse in nanoseconds
        offset_voltage: float
            Offset voltage in Volts, that will be added to the zero_offset
            voltage.
        tanh_sigma: float
            Specifies the smoothing coefficient for tanh window,
            default=0 for no smoothing.
        """
        offset = self._calibration.dc_offsets_close[0] + offset_voltage
        N_time_steps = int(round(duration / self._waveform_resolution))

        if tanh_sigma == 0:
            waveform = np.zeros(N_time_steps + 1) + offset
        else:
            X = np.linspace(0, duration, N_time_steps + 1)
            start, end = (X - 2 * tanh_sigma) / tanh_sigma, \
                         (-X + duration - 2 * tanh_sigma) / tanh_sigma
            waveform = \
                (np.tanh(start) + 1) / 2 * (np.tanh(end) + 1) / 2 * \
                offset_voltage
            waveform -= min(abs(waveform)) * np.sign(offset_voltage)
            waveform += offset

        self._pulse_seq.append_pulse(waveform)
        return self

    def add_cz_pulse(self, duration, offset_voltage, *params):
        """
        Adds a pulse of the form Sum_n a_n * (1 - cos(2 pi n t / dur))

        Parameters:
        -----------
        duration: float
            Duration of the pulse in nanoseconds
        offset_voltage: float
            Offset voltage in Volts, that will be added to the zero_offset
            voltage
        params: float
            Array of a_n's in Volts

        TODO: add reference to the model and naming conventions used in this method

        Returns : self
        ---------------
        """
        offset = self._calibration.dc_offsets_close[0] + offset_voltage
        N_time_steps = int(round(duration / self._waveform_resolution))

        X = np.linspace(0, duration, N_time_steps + 1)
        components = [params[n] * (1 - np.cos(2 * np.pi * X * (n + 1) /
                                              duration))
                      for n in range(len(params))]
        waveform = sum(components, axis=0) * offset
        # What is the point of doing it?
        waveform -= min(abs(waveform)) * np.sign(offset_voltage)

        waveform += self._calibration.dc_offsets_close[0]

        self._pulse_seq.append_pulse(waveform)
        return self

    def add_zero_until(self, total_duration):
        """
        Adds a pulse with zero amplitude to the sequence of such length that the
        whole pulse sequence is of specified duration

        Should be used to end the sequence as the last call before build(...)

        Parameters:
        -----------
        total_duration: float
            Duration of the whole sequence

        Returns : self
        ---------
        """
        total_time_steps = round(total_duration / self._waveform_resolution)
        current_time_steps = self._pulse_seq.total_points() - 1
        residual_time_steps = total_time_steps - current_time_steps
        self.add_zero_pulse(residual_time_steps * self._waveform_resolution)
        return self

    def build(self):
        """
        Returns : PulseSequence
        -------
            Returns `PulseSequence` instance generated so far and resets
            internal constructed sequence state to new empty.
        """
        to_return = self._pulse_seq
        self._pulse_seq = PulseSequence(self._waveform_resolution)
        return to_return
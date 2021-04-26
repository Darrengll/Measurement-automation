from importlib import reload
import copy

from . import directRabi
reload(directRabi)
from .directRabi import DirectRabiBase

from .. import VNATimeResolvedDispersiveMeasurement2D
reload(VNATimeResolvedDispersiveMeasurement2D)
from ..VNATimeResolvedDispersiveMeasurement2D import VNATimeResolvedDispersiveMeasurement2DResult

from . import digitizerTimeResolvedDirectMeasurement
reload(digitizerTimeResolvedDirectMeasurement)
from .digitizerTimeResolvedDirectMeasurement import DigitizerTimeResolvedDirectMeasurement

from .. import IQPulseSequence
reload(IQPulseSequence)
from ..IQPulseSequence import IQPulseBuilder

import numpy as np


class DirectRabiChevrons(DirectRabiBase):
    """
    Class measures dependence of the averaged voltage
    response trace of the qubit radiation. Qubit is embedded in line and
    excited by applying pulse with carrier frequency close to qubit 0-1
    frequency and some duration.

    Qubit assumed directly coupled with CPW line.

    Only fourier component of trace with frequency corresponding to qubit
    frequency is extracted from trace and saved.

    Rabi Chevrons is dependence of aforementioned fourier component on pulse
    duration and carrier frequency.
    """

    def __init__(self, name, sample_name, plot_update_interval=1,
                 q_lo=[], q_iqawg=[], dig=[], save_traces=False):
        """

        Parameters
        ----------
        name : str
        sample_name : str
        plot_update_interval : float

        q_lo : List[EXG]
        q_iqawg : list[IQAWG]
        dig : list[SPCM]
        """
        self._measurement_result: DirectRabiChevronsResult = None
        super().__init__(
            name, sample_name,
            plot_update_interval=plot_update_interval,
            q_lo=q_lo, q_iqawg=q_iqawg, dig=dig, save_traces=save_traces
        )

        # exctracted as target sideband frequency of IQAWG's calibration
        self._qubit_frequency = None
        # IF frequency where frequency shift is counted from
        self._central_if_freq = None
        # downconvertion frequency should be changed according to IF
        # frequency change in order to exctract RF trace at qubit frequency
        # no matter the freq_shift value
        self._downconv_freq = None

    def _init_measurement_result(self):
        m_res = DirectRabiChevronsResult(self._name, self._sample_name)
        self.set_measurement_result(m_res)

    def set_fixed_parameters(self, pulse_sequence_parameters,
                             freq_limits = (0,50e6),
                             down_conversion_calibration=None,
                             q_lo_params=[], q_iqawg_params=[],
                             dig_params=[]):

        upconv_cal = q_iqawg_params[0]["calibration"]
        # if_frequency will be changed during frequency sweep, so we need
        # to force initial calibration structure to remain unchanged
        q_iqawg_params[0]["calibration"] = copy.deepcopy(upconv_cal)

        # make quick access to qubit frequency and
        # initial intermediate frequency
        self._qubit_frequency = upconv_cal._sideband_to_maintain_freq
        self._central_if_freq = upconv_cal._if_frequency
        self._upconv_sideband_calib = upconv_cal._sideband_to_maintain

        self._measurement_result._qubit_frequency = self._qubit_frequency
        self._measurement_result._central_if_freq = self._central_if_freq
        self._measurement_result._upconv_calib_sideband = self._upconv_sideband_calib

        super().set_fixed_parameters(
            pulse_sequence_parameters,
            freq_limits, down_conversion_calibration,
            q_lo_params, q_iqawg_params,
            dig_params
        )

    def sweep_rabi_chevrons(self, excitation_durations, frequency_shifts):
        """
        2D mesh for which response will be measured.

        Parameters
        ----------
        excitation_durations : np.ndarray
            excitation pulse durations in 'ns'
        frequency_shifts : np.ndarray
            excitation pulse carrier frequency shift

        Returns
        -------

        """
        # order of arguments is important
        # Order below sweeps frequency first, then changes duration
        # so we put 'self._output_pulse_sequence()' call into the
        # function that sets excitation duration
        # (See 'self._set_freq_shift' below)
        super().set_swept_parameters(
            **{
                "excitation_durations": (self._set_duration,
                                         excitation_durations),
                "frequency_shifts": (self._set_freq_shift,
                                     frequency_shifts)
            }
        )

    def _set_duration(self, duration):
        self._pulse_sequence_parameters["excitation_duration"] = duration

    def _set_freq_shift(self, freq_shift):
        calibration = self._q_iqawg[0].get_calibration()
        if calibration._sideband_to_maintain == "left":
            calibration._if_frequency = self._central_if_freq - freq_shift
            self._downconv_freq = self._central_if_freq - freq_shift
        else:
            calibration._if_frequency = self._central_if_freq + freq_shift
            self._downconv_freq = self._central_if_freq + freq_shift
        self._output_pulse_sequence()


class DirectRabiChevronsResult(VNATimeResolvedDispersiveMeasurement2DResult):
    def __init__(self, name, sample_name):
        self._qubit_frequency = None
        self._central_if_freq = None
        self._upconv_calib_sideband = None
        super().__init__(name, sample_name)

    def _prepare_data_for_plot(self, data):
        """
        Should be implemented in child classes
        """
        return ((data["frequency_shifts"] + self._qubit_frequency)/1e9,
                data["excitation_durations"],
                data["data"])

    def _annotate_axes(self, axes):
        axes[0].set_ylabel("Excitation duration [ns]")
        axes[-2].set_ylabel("Excitation duration [ns]")
        axes[-1].set_xlabel("Excitation frequency [GHz]")
        axes[-2].set_xlabel("Excitation frequency [GHz]")

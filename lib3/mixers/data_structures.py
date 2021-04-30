""" IN DEVELOPMENT. NOT USED ANYWHERE."""
import copy

import numpy as np
from typing import Hashable

from drivers.Spectrum_m4x import SPCM, ADCParameters
from drivers.E8257D import MXG


class HetIQCalibration:
    """
    Class for storing data of heterodyne scheme based on IQ mixers calibration.
    """
    def __init__(
            self, id, dc_offsets_close=(None, None),
            dc_offsets_open=(None, None),
            r_u=None, r_d=None, phi_up=None, phi_down=None, delay=None,
            phase_delay=None, iqawg_i_amplitude=None, lo_power=None,
            lo_freq=None, if_freq=None,
            awg_sample_period=None, adc_params=None
    ):
        """

        Parameters
        ----------
        id : Hashable
            identification that can be converted to string.
        dc_offsets_close : np.complex
            Pair of AWG I and Q channels offsets in volts for maximal
            isolation between LO and RF ports.
            Real part is for I channel. Imaginary part is for Q channel.
        dc_offsets_open : np.complex
            Pair of AWG I and Q channels offsets in volts for maximal
            image sideband cancellation.
            Real part is for I channel. Imaginary part is for I channel.
        r_u : float
            upconversion amplitude asymmetry coefficient
        r_d : float
            downconversion amplitude asymmetry coefficient
        phi_up : float
            upconversion phase difference in rads required for maximal
            image sideband cancellation.
        phi_down : float
            downconversion phase difference in rads.
            Calibrated so that I and Q downconverted channels consists of
            orthogonal quadratures of initial trace.
        delay : float
            time in nanoseconds that requires radiation to propagate from
            AWG to ADC.
        phase_delay : float
            phase scew from trace if_freq defines phase delay through the
            DUT.
        iqawg_i_amplitude : float
            Amplitude of sine trace outputted from I channel of IQAWG.
        lo_power : float
            Power of LO source in dBm.
        lo_freq : float
            Frequency of LO source.
        if_freq : float
            Intermediate if_freq. Carrier if_freq of signals generated
            by AWG.
        awg_sample_period : np.float
            AWG output sampling period in nanoseconds. Reverse to AWG's
            sample rate in GHz.
        adc_params : ADCParameters
            Digitizer parameters utilized in calibration
        """
        self.id = id
        self.dc_offsets_close = dc_offsets_close
        self.dc_offsets_open = dc_offsets_open
        self.r_u = r_u
        self.r_d = r_d
        self.phi_up = phi_up
        self.phi_down = phi_down
        self.delay = delay
        self.phase_delay = phase_delay
        self.iqawg_i_amplitude = iqawg_i_amplitude
        self.lo_power = lo_power
        self.lo_freq = lo_freq
        self.if_freq = if_freq
        self.awg_sampling_period = awg_sample_period
        self.adc_params = adc_params

    def get_iqawg_q_amplitude(self):
        return self.iqawg_i_amplitude*self.r_u

    def apply_to_IQ_trace(self, trace, time=None):
        """
        Parameters
        ----------
        trace : np.ndarray
            Complex valued signal with real part consisting of I quadrature
            and imaginary part consisting of Q quadrature.
        time : np.ndarray
            time points where signal was sampled.
        Returns
        -------
        s : np.ndarray
            Complex valued trace after downconvertion calibration is applied
        """
        pass


class CalibrationSingleUp(HetIQCalibration):
    """
    Class for storing data of single mixer upconversion calibration.
    All meaningful data is stored as 0-th element of any tuple attributes.

    E.g. to get `dc_offset_close` - optimized mixer's dc voltage offset to
    improve it's insulation, you should use:
        `calib.dc_offsets_close[0]`
    """
    pass
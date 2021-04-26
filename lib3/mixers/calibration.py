""" IN DEVELOPMENT. NOT USED ANYWHERE."""
import numpy as np
from typing import Tuple

from .iq_awg import IQAWG
from drivers.Spectrum_m4x import SPCM, ADCParameters
from drivers.E8257D import MXG


class HetIQCalibration:
    """
    Class for storing data of heterodyne scheme based on IQ mixers calibration.
    """
    def __init__(
            self, dc_offsets_close=(None, None), dc_offsets_open=(None, None),
            r_u=None, r_d=None, phi_up=None, phi_down=None, delay=None,
            phase_delay=None, iqawg_i_amplitude=None, lo_power=None,
            lo_freq=None, if_freq=None,
            awg_sample_period=None, adc_params=None
    ):
        """

        Parameters
        ----------
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
            phase scew from trace frequency defines phase delay through the
            DUT.
        iqawg_i_amplitude : float
            Amplitude of sine trace outputted from I channel of IQAWG.
        lo_power : float
            Power of LO source in dBm.
        lo_freq : float
            Frequency of LO source.
        if_freq : float
            Intermediate frequency. Carrier frequency of signals generated
            by AWG.
        awg_sample_period : np.float
            AWG output sampling period in nanoseconds. Reverse to AWG's
            sample rate in GHz.
        adc_params : ADCParameters
            Digitizer parameters utilized in calibration
        """
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
        Returns : np.ndarray
        -------
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


class HSCalibrator:
    # Heterodyne scheme calibrator class
    def __init__(self, iqawg, adc, lo_source):
        """

        Parameters
        ----------
        iqawg : IQAWG
        adc : SPCM
        lo_source : MXG
        """
        self.iqawg = iqawg
        self.adc = adc
        self.lo_src = lo_source
        self.adc_params = None

    def calibrate(self, iqawg_i_amplitude, lo_power, lo_freq, if_freq,
                  adc_params, initial_guess):
        """

        Parameters
        ----------
        iqawg_i_amplitude : float
            Amplitude of sine trace outputted from I channel of IQAWG.
        lo_power : float
            Power of LO source in dBm.
        lo_freq : float
            Frequency of LO source.
        if_freq : float
            Intermediate frequency. Carrier frequency of signals generated
            by AWG.
        adc_params : ADCParameters
            Digitizer parameters utilized in calibration
        initial_guess : HetIQCalibration
            Calibration result class instance that contains starting points
            for optimization routines. `r_up`, `r_down`, `phi_up`, `phi_down`
             and `delay`.

        Returns : HetIQCalibration
            Calibration of heterodyne scheme with cancelled image sideband
            and correct values for up and downconversion.
        -------

        """
        self.adc_params = adc_params
        if (initial_guess is not None) and isinstance(initial_guess, HetIQCalibration):
            calib = initial_guess
        else:
            calib = HetIQCalibration(
                dc_offsets_close=(0., 0.), dc_offsets_open=(0., 0.),
                r_u=1., r_d=1., phi_up=np.pi/2, phi_down=np.pi/4,
                delay=100,
                iqawg_i_amplitude=iqawg_i_amplitude, lo_power=lo_power,
                lo_freq=lo_freq, if_freq=if_freq
            )

        # setup ADC for measurements
        self.adc.reset_card()
        self.adc.set_parameters(self.adc_params)

        # setup AWG for measurements
        self.iqawg.reset_host_awgs()

        # TODO: OUTPUT several reference signals and perform each
        #  calibration stage.
        self.iqawg

        return calib

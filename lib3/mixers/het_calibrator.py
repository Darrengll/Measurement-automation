# Standard library imports
import copy
from typing import Hashable

# Third party imports
import numpy as np

# Local application imports
from lib3.core.compound_devices.iq_awg import IQAWG
from lib3.core.drivers.spectrum_m4x import ADCParameters, SPCM_MODE, \
    SPCM_TRIGGER
from lib3.mixers.data_structures import HetIQCalibration


class HetIQCalibrator:
    """
    Heterodyne scheme calibrator class
    """
    def __init__(self, iqawg, adc, lo_source):
        """

        Parameters
        ----------
        iqawg : IQAWG
        adc : SpcmAdcM4x
        lo_source : N5173B
        """
        self.iqawg = iqawg
        self.adc = adc  # adc channels must be set
        self.lo_src = lo_source

        # current calibration for internal usage
        # in `self.calibrate` function.
        self._current_calibration: HetIQCalibration = None
        # adc parameters calculated to comply with
        # `self.calibrate` function bandwidth and snr
        self._adc_params: ADCParameters = ADCParameters(
            SPCM_MODE.AVERAGING, oversampling_factor=1,

        )

    def calibrate(self, unique_id, iqawg_i_amplitude, lo_power, lo_freq, if_freq,
                  bandwidth, snr=10, initial_guess=None):
        """

        Parameters
        ----------
        unique_id : Hashable
            Any variable that can be converted to string in filename.
        iqawg_i_amplitude : float
            Amplitude of sine trace outputted from I channel of IQAWG.
        lo_power : float
            Power of LO source in dBm.
        lo_freq : float
            Frequency of LO source.
        if_freq : float
            Intermediate if_freq. Carrier if_freq of signals generated
            by AWG.
        bandwidth : float
            Bandwidth of the receiving device in Hz
        snr : float
            signal-to-noise ratio of the calibration
        initial_guess : HetIQCalibration
            Calibration result class instance that contains starting points
            for optimization routines. `r_up`, `r_down`, `phi_up`, `phi_down`
             and `delay`.

        Returns
        ---------------
        HetIQCalibration
            Calibration of heterodyne scheme with cancelled image sideband
            and correct data to perform up and downconversion of a signals.
        -------
        """

        # setting initial guess for calibration result
        if (initial_guess is not None) and isinstance(initial_guess, HetIQCalibration):
            calib = copy.deepcopy(initial_guess)
            calib.id = unique_id
        else:
            calib = HetIQCalibration(
                unique_id,
                dc_offsets_close=0, dc_offsets_open=0,
                r_u=1., r_d=1., phi_up=np.pi/2, phi_down=np.pi/4,
                delay=100,
                iqawg_i_amplitude=iqawg_i_amplitude, lo_power=lo_power,
                lo_freq=lo_freq, if_freq=if_freq
            )

        ## UPCONVERSION CALIBRATION START ##
        # setup AWG for measurements
        self.iqawg.reset_host_awgs()

        self._calibrate_dc_closed_offsets()
        ## UPCONVERSION CALIBRATION END ##
        # TODO: OUTPUT several reference signals and perform each
        #  calibration stage.

        ## DOWNCONVERSION CALIBRATION START ##
        # setup ADC for measurements
        self.adc.reset_card()
        self.adc.set_parameters(self.adc_params)
        self.adc_params = bandwidth  # parameters has to be altered due to
        # according to calibration needs (bandwidth, nops_density and maybe
        # other parameters).
        ## DOWNCONVERSION CALIBRATION END ##

        return calib

    def _calibrate_dc_closed_offsets(self):
        """
        Optimizes dc offsets to increasse isolation
        for LO->RF channel. Offsets used to close mixer "more tightly"
        in comparison when mixer is in idle state (i.e. zeros on both
        I-Q  inputs).

        Returns
        -------
        res : np.complex
            Re[res] - I channels dc offset
            Im[res] - Q channel dc offset
        """
        raise NotImplemented("Not implemented yet")
        def loss_function_if_offsets(if_offsets, args):
            if_amplitudes = args[0]
            phase = args[1]
            self._iqawg.output_continuous_IQ_waves(
                frequency=if_frequency,
                amplitudes=if_amplitudes, relative_phase=phase,
                offsets=if_offsets, waveform_resolution=waveform_resolution,
                optimized=self._optimized_awg_calls
            )
            self._sa.prepare_for_stb();
            self._sa.sweep_single();
            self._sa.wait_for_stb()
            data = self._sa.get_tracedata()

            print("\rIF offsets: ", format_number_list(if_offsets),
                  format_number_list(data),
                  end="            ", flush=True)
            clear_output(wait=True)

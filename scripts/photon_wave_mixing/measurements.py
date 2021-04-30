from importlib import reload
import lib2.stimulatedEmission
reload(lib2.stimulatedEmission)
from lib2.stimulatedEmission import StimulatedEmission
import copy
import scripts.photon_wave_mixing.devices as dev
from drivers.Spectrum_m4x import SPCM_MODE, SPCM_TRIGGER


def measure_reflection_from_sps(sps_devices, probe_devices, params):
    """
    Measurement of the average field reflected from the single-photon source.

    Parameters
    ----------
    sps_devices: dict
        Dictionary with devices. It contains an awg that we use to supply
        pulses to the single-photon source and a digitizer that measures the
        transmitted field
    probe_devices: dict
        Dictionary with devices. It contains a digitizer that measures the
        reflected field and an awg which we use to block LO with.
    Returns
    -------
        Measurement class
    """
    # Reset AWG
    dev.turn_off_awg(probe_devices)
    dev.turn_off_awg(sps_devices)

    # Check if all calibrations are in place
    status, message = check_calibrations(sps_devices)
    if not status:
        print(f"'sps_devices' does not contain {message}")
    status, message = check_calibrations(probe_devices)
    if not status:
        print(f"'probe_devices' does not contain {message}")

    # Mute the second IQ awg. Here it is not possible to turn off the
    # microwave source, because it is used by the probe downconversion IQ
    # mixer
    probe_devices['iqawg'].output_zero(trigger_sync_every=1000)

    # update digitizer for reflection
    devices = sps_devices.copy()
    devices['dig'] = probe_devices['dig']
    devices['downconv_cal'] = probe_devices['downconv_cal']
    devices['coil1'] = sps_devices['coil']
    devices['coil2'] = probe_devices['coil']

    return setup_stimulated_emission_measurement(devices, params)


def measure_transmission_through_sps(sps_devices, probe_devices,
                                  params):
    """
    Measurement of the average field transmitted through the single-photon
    source.

    Parameters
    ----------
    sps_devices: dict
        Dictionary with devices. It contains an awg that we use to supply
        pulses to the single-photon source and a digitizer that measures the
        transmitted field
    probe_devices: dict
        Dictionary with devices. It contains a digitizer that measures the
        reflected field and an awg which we use to block LO with.
    Returns
    -------
        Measurement class
    """
    # Reset AWG
    dev.turn_off_awg(probe_devices)
    dev.turn_off_awg(sps_devices)

    # Check if all calibrations are in place
    status, message = check_calibrations(sps_devices)
    if not status:
        print(f"'sps_devices' does not contain {message}")
    status, message = check_calibrations(probe_devices)
    if not status:
        print(f"'probe_devices' does not contain {message}")

    # Mute the second heterodyne completely
    probe_devices['mw'].set_output_state('OFF')

    devices = sps_devices.copy()
    devices['coil1'] = sps_devices['coil']
    devices['coil2'] = probe_devices['coil']

    return setup_stimulated_emission_measurement(devices, params)


def setup_stimulated_emission_measurement(devices_dict, params):
    """Sets parameters for the stimulated emission measurement class and 
    returns an object of that class
    :param devices_dict:
        dictionary with primary devices: microwave source 'mw', IQ AWG 
        driver 'iqawg', digitizer 'dig', current source 'coil' 
    :param params: 
        dictionary with all the required parameters
    :return: 
    """
    se = StimulatedEmission(params['name'], params['sample_name'],
                            params['comment'], q_lo=[devices_dict['mw']],
                            q_iqawg=[devices_dict['iqawg']],
                            dig=[devices_dict['dig']],
                            src=[devices_dict['coil1'], devices_dict['coil2']])

    repetition_period = params['period']  # ns
    upconv_cal = devices_dict['upconv_cal']

    se_sequence_parameters = {
        "start_delay": params['awg_delay'],
        "digitizer_delay": params['dig_delay'],
        "after_pulse_delay": 0,  # ns
        "readout_duration": params['readout_duration'], # ns
        "repetition_period": repetition_period, # ns
        "modulating_window": params['window'],
        "window_parameter": params['window_parameter'],
        "excitation_durations": params['pulse_length'], # ns
        "excitation_amplitudes": params['amplitude'],
        "pulse_sequence": ["0"],
        "periods_per_segment": 1,
        "phase_shifts": [0],
        "d_freq": 0, #Hz, not used if `pulse_sequence` includes only zeros
    }

    segment_time_length = se_sequence_parameters["periods_per_segment"] \
                          * repetition_period

    # digitizer driver must implement 'set_parameters' function to be compatible with Measurement class
    dig_params = {"channels": [0, 1],
                  "ch_amplitude": params['dig_ampl'],
                  "dur_seg": segment_time_length, # duration of a segment in ns
                  "n_avg": params['averages'], # number of averages
                  "n_seg": 1, # number of segments
                  "oversampling_factor": 1,
                  "pretrigger": 32,
                  "mode": SPCM_MODE.AVERAGING,
                  "trig_source": SPCM_TRIGGER.EXT0
                  }

    dev_params = {'q_lo_params':  [
                            {"if_freq": upconv_cal.get_lo_frequency()}
                        ],
                  'q_iqawg_params': [
                            {"calibration": upconv_cal}
                        ],
                  'dig_params':   [dig_params]}

    freq_limits = (-params['cutoff'], params['cutoff']) # Hz

    se.set_fixed_parameters(
        se_sequence_parameters, freq_limits=freq_limits, delay_correction=0,
        down_conversion_calibration=devices_dict['downconv_cal'],
        subtract_pi=False, filter=params['digital_filter'], pulse_edge_mult=20,
        **dev_params)
    freq = upconv_cal.get_if_frequency()

    # Setup sweep parameters
    # pulse_amplitudes = np.linspace(0, 0.5, 51)
    # se.sweep_pulse_amplitude(pulse_amplitudes)
    # repetition_periods = np.array([995, 1000, 1005]) # np.linspace(1000, 2000, 3)
    # se.sweep_repetition_period(repetition_periods)
    # pulse_phases = np.linspace(0, 2 * np.pi, 21, endpoint=True)
    # se.sweep_pulse_phase(pulse_phases)

    se.option_abs = False  # True
    se._save_traces = params['save_traces']
    se._subtract_shifted = params['subtract_shifted_traces']
    se._main_current = params['main_current']
    se._shifted_current = params['shifted_current']
    se._cut_everything_outside_the_pulse = False

    return se


def stimulated_emission_sweep_lo_frequency(se, shifts):
    upconv_cal = se._q_iqawg[0].get_calibration()
    downconv_cal = se._down_conversion_calibration
    lo_freq = upconv_cal.get_lo_frequency()
    upconv_cals = [copy.deepcopy(upconv_cal) for i in range(len(shifts))]
    downconv_cals = [copy.deepcopy(downconv_cal) for i in range(len(shifts))]
    for i in range(len(shifts)):
        upconv_cals[i]._lo_frequency = lo_freq + shifts[i]
        downconv_cals[i].set_shift(shifts[i])
    se.sweep_lo_shift(shifts, upconv_cals, downconv_cals)


def check_calibrations(devices_dict):
    """Checks if calibrations are in the dict. Returns True if everything is
    fine. Otherwise, returns False and a message string in a tuple"""
    if not ("upconv_cal" in devices_dict.keys()) \
            or devices_dict['upconv_cal'] is None:
        return (False, "upconversion calibration")

    if not ("downconv_cal" in devices_dict.keys()) \
        or devices_dict['downconv_cal'] is None:
        return (False, "downconversion calibration")

    return (True, None)
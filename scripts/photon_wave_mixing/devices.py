from importlib import reload, import_module
from dataclasses import dataclass
import drivers  # all measurement device drivers package
import numpy as np
import sys
from time import sleep
import matplotlib.pyplot as plt
import lib.plotting as plt2
from tqdm.notebook import tqdm
from copy import deepcopy
from lib import data_management as dm
import scipy.fft as fp
from drivers.Spectrum_m4x import SPCM_TRIGGER, SPCM_MODE
import pickle

def my_import(modulename, classnames):
    """
    Reduces imports and reloads to one line
    Usage: exec(my_import("drivers.Spectrum_m4x",
                          "SPCM, SPCM_MODE, SPCM_TRIGGER"))
    """
    return(f"import {modulename};"
           f"reload({modulename});"
           f"from {modulename} import {classnames}")

## DEVICES IMPORT ##
exec(my_import("lib.iq_mixer_calibration_v2", "IQCalibrator"))

print("REPORT ON LOADED DEVICES")

# ## RF generators ##
exec(my_import("drivers.sc5502a", "SC5502A"))
mw_sps = SC5502A(idx=1, master=False)
mw_probe = SC5502A(idx=0, master=False)
print("Microwave source 'mw_sps' is loaded")
print("Microwave source 'mw_probe' is loaded")

## AWGs ##
exec(my_import("drivers.keysightM3202A", "KeysightM3202A"))

channelI = 1
channelQ = 2

awg_sps_slot = 13
awg_sps = KeysightM3202A("M3202A", awg_sps_slot, chassis=1)
awg_sps.synchronize_channels(channelI, channelQ)
awg_sps.trigger_output_config(channel=channelI, trig_length=100)
awg_sps.stop_AWG(channel=channelI)

awg_probe_slot = 14
awg_probe = KeysightM3202A("M3202A", awg_probe_slot, chassis=1)
awg_probe.synchronize_channels(channelI, channelQ)
awg_probe.trigger_output_config(channel=channelI, trig_length=100)
awg_probe.stop_AWG(channel=channelI)
print(f"AWG 'awg_sps' is loaded. Slot #{awg_sps_slot}")
print(f"AWG 'awg_probe' is loaded. Slot #{awg_probe_slot}")


# ## IQAWGs ##
exec(my_import("drivers.IQAWG", "IQAWG, AWGChannel"))
iqawg_sps = IQAWG(AWGChannel(awg_sps, channelI),
                  AWGChannel(awg_sps, channelQ))
iqawg_probe = IQAWG(AWGChannel(awg_probe, channelI),
                    AWGChannel(awg_probe, channelQ))
print("IQAWG 'iqawg_sps' is loaded")
print("IQAWG 'iqawg_probe' is loaded")

# Digitizers
exec(my_import("drivers.Spectrum_m4x", "SPCM, SPCM_MODE, SPCM_TRIGGER"))
if "dig_probe" not in globals():
    global dig_sps
    global dig_probe
    dig_sps = SPCM(b"/dev/spcm0")
    dig_probe = SPCM(b"/dev/spcm1")
print(f"Digitizer 'dig_sps' is loaded. Slot #{dig_sps.get_slot_number()}")
print(f"Digitizer 'dig_probe' is loaded. Slot #{dig_probe.get_slot_number()}")

# stabilized current sources ##
exec(my_import("drivers.Yokogawa_GS200", "Yokogawa_GS210"))
sps_coil = Yokogawa_GS210("yok1")
sps_coil.set_src_mode_curr()  # set current source mode
sps_coil.set_range(0.01)  # set 10 mA range regime
probe_coil = Yokogawa_GS210("yok2")
probe_coil.set_src_mode_curr()  # set current source mode
probe_coil.set_range(0.01)  # set 10 mA range regimeregime
# sps_loop = Yokogawa_GS210("yok3")
# sps_loop.set_src_mode_curr()  # set current source mode
# sps_loop.set_range(0.01)  # set 10 mA range regime
print(f"Current source 'sps_coil' is loaded. Address {sps_coil._address}")
print(f"Current source 'probe_coil' is loaded. Address {probe_coil._address}")
# print(f"Current source 'sps_loop' is loaded. Address {sps_loop._address}")

# Spectral analyzers
from drivers.Agilent_EXA import Agilent_EXA_N9010A
exa = Agilent_EXA_N9010A("EXA")
print("Spectral analyzer 'exa' is loaded")

# Devices dictionaries
sps_devices = {
    'awg': awg_sps,
    'dig': dig_sps,
    'iqawg': iqawg_sps,
    'mw': mw_sps,
    'coil': sps_coil,
    'downconv_cal': None,
    'upconv_cal': None,
}
probe_devices = {
    'awg': awg_probe,
    'dig': dig_probe,
    'iqawg': iqawg_probe,
    'mw': mw_probe,
    'coil': probe_coil,
    'downconv_cal': None,
    'upconv_cal': None
}


# IQ MIXERS CALIBRATION
exec(my_import("lib.iq_mixer_calibration_v2",
               "IQCalibrator, IQCalibrationData"))
exec(my_import("lib.iq_downconversion_calibration",
               "IQDownconversionCalibrator"))


def load_last_upconv_calibration(params):
    """Loads last upconversion calibration with provided parameters
    Parametrs example:
    sps_upconv_mixer_calibration_parameters = {
        'id': 'sps_mixer_upconv_cal',dBm
        'lo_power': 15, # dBm
        'ssb_power': -30, #
        'if_freq': 50e6, #
        'q_freq': 4.9e9, # Hz
        'shift': 0, # Hz
        'sideband_to_maintain': 'right',
    }
    """
    lo_freq = params['q_freq'] - params['if_freq'] - params['shift']
    upcals = dm.load_IQMX_calibration_database(params['id'], 0)
    key = frozenset(dict(lo_power=params['lo_power'],
                         sideband_to_maintain=params['sideband_to_maintain'],
                         ssb_power=params['ssb_power'],
                         lo_frequency=lo_freq,
                         if_frequency=params['if_freq'],
                         waveform_resolution=1,
                         ).items())
    return upcals.get(key)


def load_last_downconv_calibration(params):
    return dm.load_downconversion_calibration(params['id'], params['if_freq'])


def launch_upconv_calibration(devices_dict, params, ig=None):
    cal = IQCalibrator(devices_dict['iqawg'], exa, devices_dict['mw'],
                       params['id'], 0,
                       sideband_to_maintain=params['sideband_to_maintain'])

    lo_freq = params['q_freq'] - params['if_freq'] - params['shift']
    ro_cal = cal.calibrate(
        lo_freq, params['if_freq'], params['lo_power'], params['ssb_power'],
        waveform_resolution=1, iterations=3, minimize_iterlimit=100,
        sa_res_bandwidth=5e3, sa_vid_bandwidth=1000, initial_guess=ig)
    dm.save_IQMX_calibration(ro_cal)
    devices_dict['iqawg'].set_parameters({"calibration": ro_cal})
    devices_dict['upconv_cal'] = ro_cal
    return ro_cal


def launch_downconv_calibration(devices_dict, params, plot=False):
    if devices_dict['iqawg'].get_calibration() is None:
        print("Set calibration parameter in IQ AWG")
        return None

    amps = (params['amplitude'], params['amplitude'])
    signal_period = params['signal_period']
    up_cal = devices_dict['iqawg'].get_calibration()

    dig_params = {"channels": [0, 1],  # a list of channels to measure
                  "ch_amplitude": params['dig_ampl'],
                  # mV, amplitude for every channel (allowed values are 200, 500, 1000, 2500 mV)
                  "dur_seg": signal_period - 160,
                  # duration of a segment in ns
                  "n_avg": params['averages'],  # number of averages
                  "n_seg": 1,  # number of segments
                  "oversampling_factor": 1,
                  # sample_rate = max_sample_rate / oversampling_factor
                  "pretrigger": 32,  # samples
                  "mode": SPCM_MODE.AVERAGING,
                  "trig_source": SPCM_TRIGGER.EXT0,
                  "digitizer_delay": 0
                  }

    cal = IQDownconversionCalibrator(devices_dict['iqawg'],
                                     devices_dict['dig'],
                                     params['id'])
    downconv_cal = cal.calibrate(up_cal, dig_params=dig_params,
                                 trigger_period=signal_period, amps=amps)
    dm.save_downconversion_calibration(downconv_cal)
    devices_dict['downconv_cal'] = downconv_cal

    if plot:
        downconv_cal.show_signal_spectra_before_and_after()
        downconv_cal.show_before_and_after()

    return downconv_cal


## HELPER FUNCTIONS ##
def turn_off_current_sources():
    for i in range(1, 3):
        cur_src = Yokogawa_GS210("yok" + str(i))
    cur_src.set_src_mode_curr()  # set current source mode
    cur_src.set_range(1e-3)  # set 1 mA range regime
    cur_src.set_current(0)


def turn_off_mw_sources():
    mw_sps.set_output_state("OFF")
    mw_probe.set_output_state("OFF")


def turn_on_mw_sources():
    mw_sps.set_output_state("ON")
    mw_probe.set_output_state("ON")


def plot_trace(trace, dt):
    fig, (ax_re, ax_im) = plt.subplots(2, 1, sharex=True)
    time = dt*np.linspace(0, len(trace), len(trace), endpoint=False)
    ax_re.plot(time, np.real(trace))
    ax_re.set_ylabel("Real")
    ax_im.plot(time, np.imag(trace))
    ax_im.set_ylabel("Imag")
    ax_im.set_xlabel("t")


def plot_trace_fft(trace, sampling_freq):
    fig, ax = plt.subplots(1,1)
    ax.magnitude_spectrum(trace, Fs=sampling_freq, scale='dB')
    ax.margins(x=0)


def plot_trace_IQ(trace):
    fig, ax = plt.subplots(1,1)
    ax.plot(np.real(trace), np.imag(trace))


def turn_off_awg(devices_dict):
    awg = devices_dict['awg']
    awg.clear()
    awg.synchronize_channels(channelI, channelQ)
    awg.trigger_output_config(channel=channelI, trig_length=100)
    awg.stop_AWG(channel=channelI)


def downconvert(trace, if_freq = 50):
    # if_freq is in MHz
    N = len(trace)
    time = np.linspace(0, N * 0.8, N, endpoint=False) # ns
    return trace * np.exp(-2j * np.pi * if_freq / 1000 * time)

import lib2.directMeasurements.directRabi
reload(lib2.directMeasurements.directRabi)
from lib2.directMeasurements.directRabi import DirectRabiFromPulseDuration,\
    DirectRabiFromAmplitude

def setup_rabi_from_delay(devices_dict, params):
    """Returns the Measurement class"""
    name = params['name']
    dr = DirectRabiFromPulseDuration(params['name'], params['sample_name'],
                                     q_lo=[devices_dict['mw']],
                                     q_iqawg=[devices_dict['iqawg']],
                                     dig=[devices_dict['dig']],
                                     save_traces=params['save_traces'])

    excitation_durations = params['excitation_durations']
    rabi_sequence_parameters = {"start_delay": 0,
                                "readout_duration": params['readout_duration'],
                                "repetition_period": params['period'],
                                "modulating_window": params['window'],
                                "window_parameter": params['window_parameter'],
                                "digitizer_delay": params['dig_delay'],
                                "excitation_amplitude": params['amplitude'],
                                "excitation_duration": 20
                                }
    ro_cal = devices_dict['iqawg'].get_calibration()
    q_lo_params = {"frequency": ro_cal.get_lo_frequency()}
    iqawg_in_params = {"calibration": ro_cal}

    dig_params = {"channels": [0, 1],  # a list of channels to measure
                  "ch_amplitude": params['dig_ampl'],  # amplitude for every
                  # channel
                  "dur_seg": rabi_sequence_parameters["readout_duration"],
                  # duration of a segment in ns
                  "n_avg": params['averages'],  # number of averages
                  "n_seg": 1,  # number of segments
                  "oversampling_factor": 1,
                  # sample_rate = max_sample_rate / oversampling_factor
                  "pretrigger": 32,  # samples
                  }

    dev_params = {'q_lo_params': [q_lo_params],
                  'q_iqawg_params': [iqawg_in_params],
                  'dig_params': [dig_params]}
    dr.set_fixed_parameters(rabi_sequence_parameters,
                            down_conversion_calibration=devices_dict[
                                'downconv_cal'],
                            **dev_params)
    dr.sweep_excitation_durations(excitation_durations)
    dr.set_ult_calib(params['ult_calib'])
    return dr


exec(my_import("lib2.directMeasurements.directRamsey", "DirectRamseyFromDelay"))


def setup_ramsey_measurement(devices_dict, params):
    """Return an object of the Measurement class"""
    name = params['name']
    sample_name = params['sample_name']
    pi_pulse_duration = params['pi_pulse']
    DRam = DirectRamseyFromDelay(name, sample_name,
                                 q_lo=[devices_dict['mw']],
                                 q_iqawg=[devices_dict['iqawg']],
                                 dig=[devices_dict['dig']])

    ramsey_sequence_parameters = {
        "start_delay": params[''],
        "readout_duration": 500,
        "repetition_period": 1000,
        "modulating_window": "tukey",
        "window_parameter": 0.1,
        "digitizer_delay": 0,
        "pi_half_pulse_amplitude": 1,
        "pi_half_pulse_duration": pi_pulse_duration / 2,
    }

    lo_shift = -15e6  # Hz
    q_lo_params = {"frequency": lo_freq + lo_shift}  # shifts[k]}
    iqawg_in_params = {"calibration": ro_cal}  # ro_cals[k]}

    # digitizer driver must implement 'set_parameters' function to be compatible with Measurement class
    dig_params = {"channels": [0, 1],  # a list of channels to measure
                  "ch_amplitude": 200,  # amplitude for every channel
                  "dur_seg": ramsey_sequence_parameters["readout_duration"],
                  # duration of a segment in ns
                  "n_avg": 1 << 24,  # number of averages
                  "n_seg": 1,  # number of segments
                  "oversampling_factor": 1,
                  # sample_rate = max_sample_rate / oversampling_factor
                  "pretrigger": 32,  # samples
                  }

    dev_params = {'q_lo_params': [q_lo_params],
                  'q_iqawg_params': [iqawg_in_params],
                  'dig_params': [dig_params]}

    DRam.set_fixed_parameters(ramsey_sequence_parameters,
                              down_conversion_calibration=downconv_cal,
                              **dev_params)
    DRam.sweep_ramsey_delay(digital_filtering=True)
    DRam.setup_pi_subtraction(False)
    DRam.set_ult_calib(False)
    DRam._fir_cutoff = 50e6


def check_amplitude_linearity(devices_dict, params):
    """Measures the amplitude response of the heterodyne scheme vs amplitude

    Usage example:
        al_params = {
            'awg_amps': np.linspace(0, 1, 101),
            'period': 1000,  # ns
            'dig_ampl': 200,  # mV
            'averages': 1 << 20,
        }
        check_amplitude_linearity({
                'iqawg': sps_devices['iqawg'],
                'dig': probe_devices['dig']
            },
            al_params)
    """
    dig_params = {
        "channels": [0, 1],  # a list of channels to measure
        "ch_amplitude": params['dig_ampl'],
        "dur_seg": params['period'] - 100,
        "n_avg": params['averages'],  # number of averages
        "n_seg": 1,  # number of segments
        "oversampling_factor": 1,
        "pretrigger": 32,  # samples
        "mode": SPCM_MODE.AVERAGING,
        "trig_source": SPCM_TRIGGER.EXT0,
        "digitizer_delay": 0
    }
    devices_dict['dig'].set_parameters(dig_params)
    awg_amps = params['awg_amps']
    dig_probe.set_parameters(dig_params)
    dig_amps = np.zeros_like(awg_amps)
    freq = devices_dict['iqawg'].get_calibration().get_if_frequency()
    for i in tqdm(range(len(awg_amps))):
        devices_dict['iqawg'].output_IQ_waves_from_calibration(
                                        trigger_sync_every=params['period'],
                                        amp_coeffs=(awg_amps[i], awg_amps[i]))
        data = devices_dict['dig'].safe_measure()
        trace = data[0::2] + 1j * data[1::2]
        # trace_cal = devices_dict['downconv_cal'].apply(trace)
        nfft = fp.next_fast_len(len(trace))
        xf = fp.fftshift(
            fp.fftfreq(nfft, d=(1/devices_dict['dig'].get_sample_rate())))
        idx = np.argmin(np.abs(xf - freq))
        yf = fp.fftshift(fp.fft(trace)) / nfft
        dig_amps[i] = np.abs(yf[idx])

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(awg_amps, dig_amps)
    plt.xlabel("awg amp, ratio")
    plt.ylabel("dig amplitude, mV")
    plt.grid(True)
    plt.xlim(0, 1)

    plt.subplot(1, 2, 2)
    plt.plot(awg_amps, np.gradient(dig_amps))
    plt.xlabel("awg amp, ratio")
    plt.ylabel("dig amplitude gradient")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.tight_layout(True)
    plt.show()

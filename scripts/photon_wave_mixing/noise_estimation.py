import numpy as np
from drivers.Spectrum_m4x import SPCM_MODE, SPCM_TRIGGER
import scipy.signal as sg
import scipy.fft as fp
from lib2.IQPulseSequence import IQPulseBuilder

main_current = 0
shifted_current = 0
devices = {"coil": None, "mw": None, "iqawg": None, "dig": None}
dig_params = {
    "dig_ampl": 200,
    "averages": 1 << 24,
}
cutoff = 5e7  # Hz
iffreq = 5e7  # Hz


def measure_absolute_snr():
    sigma = measure_sigma_amplitude()
    noise = measure_noise_std()
    return np.log(sigma**2 / noise)


def measure_sigma_amplitude():
    output_pulses()
    tune_qubit_on_resonance()
    s0 = apply_filter(downconvert(measure_averaged_signal()))
    tune_qubit_off_resonance()
    s1 = apply_filter(downconvert(measure_averaged_signal()))
    s = s0 - s1
    tune_qubit_on_resonance()
    return find_amplitude(s)


def measure_noise_std():
    turn_off_drive()
    tune_qubit_on_resonance()
    h = apply_filter(downconvert(measure_signal()))
    power = calculate_noise_power(h)
    return power


def measure_signal():
    devices["dig"].set_parameters({
        "channels": [0, 1],
        "ch_amplitude": dig_params['dig_ampl'],
        "dur_seg": int(1e7),  # duration of a segment in ns
        "n_avg": 1,
        "n_seg": 1,
        "oversampling_factor": 1,
        "pretrigger": 32,
        "mode": SPCM_MODE.STANDARD,
        "trig_source": SPCM_TRIGGER.AUTOTRIG
    })
    data = devices["dig"].safe_measure()
    return data[0::2] + 1j * data[1::2]


def measure_averaged_signal():
    devices["dig"].set_parameters({
        "channels": [0, 1],
        "ch_amplitude": dig_params['dig_ampl'],
        "dur_seg": 800,  # duration of a segment in ns
        "n_avg": dig_params['averages'],  # number of averages
        "n_seg": 1,  # number of segments
        "oversampling_factor": 1,
        "pretrigger": 32,
        "mode": SPCM_MODE.AVERAGING,
        "trig_source": SPCM_TRIGGER.EXT0
    })
    data = devices["dig"].safe_measure()
    return data[0::2] + 1j * data[1::2]


def apply_filter(signal):
    b = sg.firwin(len(signal), cutoff, fs=1.25e9)
    signal = sg.convolve(signal, b, "same")
    return signal


def downconvert(signal):
    time = np.linspace(0, 0.8e-10 * len(signal), len(signal), endpoint=False)
    c = np.exp(-2j * np.pi * iffreq * time)
    return signal * c


def tune_qubit_on_resonance():
    devices['coil'].set_current(main_current)


def tune_qubit_off_resonance():
    devices['coil'].set_current(shifted_current)


def output_pulses():
    seq_params = {
        "start_delay": 300,
        "digitizer_delay": 0,
        "after_pulse_delay": 0,  # ns
        "readout_duration": 1000,  # ns
        "repetition_period": 1000,  # ns
        "modulating_window": "rectangular",
        "window_parameter": 1,
        "excitation_durations": [100],  # ns
        "excitation_amplitudes": [1],
        "pulse_sequence": ["0"],
        "periods_per_segment": 1,
        "phase_shifts": [0],
        "d_freq": 0,  # Hz, not used if `pulse_sequence` includes only zeros
    }
    pb = devices['iqawg'].get_pulse_builder()
    seq = IQPulseBuilder.build_stimulated_emission_sequence(seq_params,
                                                            **{"q_pbs": [pb]})
    devices['iqawg'].output_pulse_sequence(seq['q_seqs'][0])


def turn_off_drive():
    devices["iqawg"].output_zero(trigger_sync_every=1000)


def find_amplitude(signal):
    # freqs = fp.fftfreq(len(signal), d=8e-10)
    # idx = np.abs(freqs - 5e7).argmin()
    spectrum = fp.fft(signal)
    return np.max(np.abs(spectrum)) / len(signal)


def calculate_noise_power(signal):
    return np.std(signal)**2

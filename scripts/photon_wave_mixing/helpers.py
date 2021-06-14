from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
import scipy.fft as fp
import numpy as np
import pickle
import matplotlib.pyplot as plt
import lib.plotting as plt2


def parse_probe_qubit_sts(freqs, S21):
    amps = np.abs(S21)
    frequencies = freqs[gaussian_filter1d(amps, sigma=1).argmin(axis=-1)]
    return frequencies


def parse_sps_sts(freqs, S21):
    amps = np.abs(S21)
    frequencies = freqs[gaussian_filter1d(amps, sigma=10).argmax(axis=-1)]
    return frequencies


def qubit_fit_func(x, a, b, c):
    return a * (x - b)**2 + c


def fit_probe_qubit_sts(filename, plot=True):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        currents = data['current, [A]']
        freqs = data['Frequency [Hz]']
        S21 = data['data']
    frequencies = parse_probe_qubit_sts(freqs, S21)
    popt, conv = curve_fit(qubit_fit_func, currents, frequencies,
                           p0=(-1e16, -2.5e-3, 5.15e9))
    if plot:
        xx, yy = np.meshgrid(currents, freqs)
        plt2.plot_2D(xx, yy,
                     np.transpose(gaussian_filter1d(np.abs(S21), sigma=20)))
        plt.figure()
        plt.plot(currents, frequencies, 'o')
        plt.plot(currents, qubit_fit_func(currents, *popt))
        plt.margins(x=0)
        plt.xlabel("Current, A")
        plt.ylabel("Qubit frequency, Hz")
        plt.show()
    return popt


def fit_sps_sts(filename, plot=True):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        currents = data['current, [A]']
        freqs = data['Frequency [Hz]']
        S21 = data['data']
    frequencies = parse_sps_sts(freqs, S21)
    popt, conv = curve_fit(qubit_fit_func, currents, frequencies,
                           p0=(-1e15, -5e-4, 5.15e9))
    if plot:
        xx, yy = np.meshgrid(currents, freqs)
        plt2.plot_2D(xx, yy,
                     np.transpose(gaussian_filter1d(np.abs(S21), sigma=10)))
        plt.figure()
        plt.plot(currents, frequencies, 'o')
        plt.plot(currents, qubit_fit_func(currents, *popt))
        plt.margins(x=0)
        plt.xlabel("Current, A")
        plt.ylabel("Qubit frequency, Hz")
        plt.show()
    return popt


def get_current(frequency, a, b, c):
    current = b + np.sqrt((frequency - c) / a)
    return current


def remove_outliers():
    pass


def get_signal_amplitude(downconverted_trace):
    N = len(downconverted_trace)
    return np.abs(fp.fft(downconverted_trace)[0] / N)


def get_noise(downconverted_trace):
    return np.std(downconverted_trace)

def measure_snr(devices_dict):
    # turn off microwave
    devices_dict['mw'].set_output_state("ON")

    # turn off AWG
    devices_dict['awg'].reset()
    devices_dict['awg'].synchronize_channels(channelI, channelQ)
    devices_dict['awg'].trigger_output_config(channel=channelI,
                                              trig_length=100)
    devices_dict['awg'].stop_AWG(channel=channelI)
    devices_dict['iqawg'].set_parameters({"calibration": devices_dict['upconv_cal']})
    devices_dict['iqawg'].output_IQ_waves_from_calibration(
        amp_coeffs=(0.5, 0.5))
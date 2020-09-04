from matplotlib import pyplot as plt
from numpy import meshgrid, unwrap, append
import numpy as np
from scipy import fftpack as fp

"""
Frequently used plotting functions
"""


def generate_mesh(X, Y):
    step_X = X[1] - X[0]
    step_Y = Y[1] - Y[0]
    return meshgrid(append(X - step_X / 2, X[-1] + step_X / 2),
                    append(Y - step_Y / 2, Y[-1] + step_Y / 2))


def plot_measurement(measurement, kwargs_amp={}, kwargs_phas={},
                     unwrap_phase=False, cmap="RdBu_r", figsize=(15, 10)):
    if measurement.get_type_str() == "pna-p1D-2D":
        measurement = measurement.get_data()
        if len(measurement) < 4:
            print("Obsolete data structure, plot manually, please")
            return -1
        fig_amps = plt.figure(figsize=figsize)
        fig_phas = plt.figure(figsize=figsize)
        ax_amps = fig_amps.add_axes([0.1, 0.1, 0.75, 0.75])
        ax_phas = fig_phas.add_axes([0.1, 0.1, 0.75, 0.75])

        ax_amps.plot(measurement[0], measurement[2], **kwargs_amp)
        ax_phas.plot(measurement[0], measurement[3], **kwargs_phas)
        fig_phas.canvas.set_window_title("Phase")
        fig_amps.canvas.set_window_title("Amplitude")
        ax_phas.set_title("Phase")
        ax_amps.set_title("Amplitude")
        return fig_amps, fig_phas

    if measurement.get_type_str() == "pna-p2D-2D":
        measurement = measurement.get_data()
        if len(measurement) < 5:
            print("Obsolete data structure, plot manually, please")
            return -1
        fig_amps = plt.figure(figsize=figsize)
        fig_phas = plt.figure(figsize=figsize)
        ax_amps = fig_amps.add_axes([0.1, 0.1, 0.75, 0.75])
        ax_phas = fig_phas.add_axes([0.1, 0.1, 0.75, 0.75])
        X = measurement[1] if len(measurement[0]) == 1 else measurement[0]
        ax_amps.plot(X, measurement[3][0], **kwargs_amp)
        ax_phas.plot(X, measurement[4][0], **kwargs_phas)
        fig_phas.canvas.set_window_title("Phase")
        fig_amps.canvas.set_window_title("Amplitude")
        ax_phas.set_title("Phase")
        ax_amps.set_title("Amplitude")
        return fig_amps, fig_phas

    if measurement.get_type_str() == "pna-p1D-3D":
        measurement = measurement.get_data()
        if len(measurement) < 4:
            print("Obsolete data structure, plot manually, please")
            return -1
        fig_amps = plt.figure(figsize=figsize)
        fig_phas = plt.figure(figsize=figsize)
        ax_amps = fig_amps.add_axes([0.1, 0.1, 0.75, 0.75])
        ax_phas = fig_phas.add_axes([0.1, 0.1, 0.75, 0.75])
        ax_amps.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_phas.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        XX, YY = meshgrid(measurement[0], measurement[1])

        amps_map = ax_amps.pcolormesh(XX, YY, measurement[2].T, cmap=cmap,
                                      **kwargs_amp)
        phas_map = ax_phas.pcolormesh(XX, YY,
                                      measurement[3].T if not unwrap_phase
                                      else unwrap(unwrap(measurement[3]).T),
                                      cmap=cmap, **kwargs_phas)
        fig_phas.canvas.set_window_title("Phase")
        fig_amps.canvas.set_window_title("Amplitude")
        ax_phas.set_title("Phase")
        ax_amps.set_title("Amplitude")
        plt.colorbar(amps_map, ax=ax_amps)
        plt.colorbar(phas_map, ax=ax_phas)
        ax_amps.grid()
        ax_phas.grid()
        ax_amps.axis("tight")
        ax_phas.axis("tight")
        return fig_amps, fig_phas

    if measurement.get_type_str() == "pna-p2D-3D":
        data = measurement.get_data()

        fig_amps = plt.figure(figsize=figsize)
        fig_phas = plt.figure(figsize=figsize)
        ax_amps = fig_amps.add_axes([0.1, 0.1, 0.75, 0.75])
        ax_amps.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_phas = fig_phas.add_axes([0.1, 0.1, 0.75, 0.75])
        ax_phas.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))

        XX, YY = generate_mesh(data[0], data[1])
        amps_map = ax_amps.pcolormesh(XX, YY, data[3].T, cmap=cmap,
                                      **kwargs_amp)
        phas_map = ax_phas.pcolormesh(XX, YY, data[4].T if not unwrap_phase
                                      else unwrap(unwrap(data[4]).T),
                                      cmap=cmap,
                                      **kwargs_phas)

        fig_phas.canvas.set_window_title("Phase")
        fig_amps.canvas.set_window_title("Amplitude")
        ax_phas.set_title("Phase")
        ax_amps.set_title("Amplitude")
        plt.colorbar(amps_map, ax=ax_amps)
        plt.colorbar(phas_map, ax=ax_phas)
        ax_amps.grid()
        ax_phas.grid()
        ax_amps.axis("tight")
        ax_phas.axis("tight")
        return fig_amps, fig_phas


def plot_one_trace(trace: np.ndarray, sampling_resolution: float = 0.8,
                   time_limits: tuple = None, comment: str = ""):
    """
    Quickly plot the data trace measured by a digitizer

    Parameters
    ----------
    trace: np.ndarray
        complex trace:Frequently used plotting functions trace_i + 1j * trace_q
    sampling_resolution: float
        sampling resolution in nanoseconds
    time_limits: tuple of size 2
        xlims for plotting in nanoseconds
    comment: str
        some comment to add in title
    """
    time = np.r_[0:len(trace)] * sampling_resolution
    if time_limits is None:
        time_limits = (time[0], time[-1])
    fig, axs = plt.subplots(2, sharex=True)
    if len(comment) > 0:
        fig.canvas.set_window_title(comment)
        fig.suptitle(comment)
    else:
        fig.suptitle("One trace")
    axs[0].plot(time, np.real(trace))
    axs[0].set_xlim(time_limits)
    axs[0].set_title("channel I")
    axs[0].set_ylabel("Voltage, mV")
    axs[0].grid(True)
    axs[1].plot(time, np.imag(trace))
    axs[1].set_xlim(time_limits)
    axs[1].set_title("channel Q")
    axs[1].set_xlabel("time, ns")
    axs[1].set_ylabel("Voltage, mV")
    axs[1].grid(True)

    plt.show()


def plot_fourier_of_trace(trace: np.ndarray, sampling_resolution: float = 0.8,
                          freq_limits: tuple = None, comment: str = ""):
    nfft = fp.next_fast_len(len(trace))
    xf = fp.fftshift(fp.fftfreq(nfft, d=sampling_resolution/1e9))
    yf = np.abs(fp.fftshift(fp.fft(trace, n=nfft)) / nfft)

    fig, ax = plt.subplots()
    if len(comment) > 0:
        fig.canvas.set_window_title(comment)
        ax.set_title(comment)
    else:
        ax.set_title("FFT")
    ax.plot(xf, 20 * np.log10(yf))
    ax.set_xlim(freq_limits if freq_limits is not None else (xf[0], xf[-1]))
    ax.set_xlabel("Frequency, Hz")
    ax.set_ylabel("Voltage, dBm")
    ax.grid(True)
    plt.show()

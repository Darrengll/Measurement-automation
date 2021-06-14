from dataclasses import dataclass
from tqdm.notebook import tqdm
import os
import pickle
import lib.plotting as plt2
import numpy as np


@dataclass
class ParametersSA:
    center_frequency: float
    span: float
    r_bandwidth: float
    v_bandwidth: float
    number_of_points: int = 1001
    averages: int = 1

    def setup_sa(self, sa_driver):
        sa_driver.set_centerfreq(self.center_frequency)
        sa_driver.set_span(self.span)
        sa_driver.set_averages(self.averages)
        sa_driver.set_video_bandwidth(self.v_bandwidth)
        sa_driver.set_bandwidth(self.r_bandwidth)  # BW/NOP < 100 Hz
        sa_driver.set_nop(self.number_of_points)  # BW/NOP < 100 Hz


def power_sweep_sa(devices, power_list, sa_parameters):
    data_list = []
    sa_parameters.setup_sa(devices["sa"])
    for power in tqdm(power_list):
        devices["mw"].set_power(power)  # dBm
        data = measure_with_sa(devices['sa'], sa_parameters.v_bandwidth)
        data_list.append(data)
    return data_list


# Launch mixing measurement on the probe qubit when the sps is off resonance
def run_mixing_on_probe(devices, power_constant_probe, power_list_sps,
                        delta, sa_parameters):
    data_list = []
    omega_plus = sa_parameters.center_frequency + delta
    omega_minus = sa_parameters.center_frequency - delta

    # initial setup of MW sources
    devices['mw_probe'].set_output_state("ON")
    devices['mw_sps'].set_output_state("ON")
    devices['mw_probe'].set_frequency(omega_plus)
    devices['mw_sps'].set_frequency(omega_minus)
    devices['mw_probe'].set_power(power_constant_probe)

    sa_parameters.setup_sa(devices['sa'])

    for power in tqdm(power_list_sps):
        devices['mw_sps'].set_power(power)
        data = measure_with_sa(devices['sa'], sa_parameters.v_bandwidth)
        data_list.append(data)
    return data_list


# Launch mixing measurement on the SPS when the probe qubit is off resonance
def run_mixing_on_sps(devices, power_constant_probe, power_list_sps,
                        delta, sa_parameters):
    data_list = []
    omega_plus = sa_parameters.center_frequency + delta
    omega_minus = sa_parameters.center_frequency - delta

    # initial setup of MW sources
    devices['mw1'].set_output_state("ON")
    devices['mw2'].set_output_state("ON")
    devices['mw1'].set_frequency(omega_plus)
    devices['mw2'].set_cw_time(omega_minus, -1)
    devices['mw2'].set_power(power_constant_probe)

    sa_parameters.setup_sa(devices['sa'])

    for power in tqdm(power_list_sps):
        devices['mw1'].set_power(power)
        data = measure_with_sa(devices['sa'], sa_parameters.v_bandwidth)
        data_list.append(data)
    return data_list


def run_mixing_on_sps_through_circulator(devices, power_constant_probe,
                                         power_list_sps, delta, sa_parameters):
    data_list = []
    omega_plus = sa_parameters.center_frequency + delta
    omega_minus = sa_parameters.center_frequency - delta

    # initial setup of MW sources
    devices['mw1'].set_output_state("ON")
    devices['mw2'].set_output_state("ON")
    devices['mw1'].set_frequency(omega_plus)
    devices['mw2'].set_frequency(omega_minus)
    devices['mw2'].set_power(power_constant_probe)

    sa_parameters.setup_sa(devices['sa'])

    for power in tqdm(power_list_sps):
        devices['mw1'].set_power(power)
        data = measure_with_sa(devices['sa'], sa_parameters.v_bandwidth)
        data_list.append(data)
    return data_list


def measure_with_sa(sa, vidbw):
    sa.prepare_for_stb()
    sa.set_video_bandwidth(vidbw)
    sa.sweep_single()
    sa.wait_for_stb()
    data = sa.get_tracedata()
    return data


def save_data(timestamp, name, data, context):
    the_folder = get_folder_path(timestamp, name)
    filename = f"{the_folder}/{name}.pkl"
    context_file = f"{the_folder}/context.txt"

    # create folders
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # save data
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

    # save context
    with open(context_file, 'w') as f:
        f.write(str(context))


def get_folder_path(timestamp, name):
    date_str = timestamp.strftime("%b %d %Y")
    time_str = timestamp.strftime("%H-%M-%S")
    the_path = f"data/Photon_wave_mixing/{date_str}/{time_str} - {name}"
    return the_path


def plot_sa_traces_2d(freqs, swept_vars, data, xlabel, ylabel, name,
                      timestamp):
    XX, YY = np.meshgrid(freqs, swept_vars)
    ZZ = np.array(data)
    the_path = f"{get_folder_path(timestamp, name)}/{name}.png"
    # the_path = f"data/Photon_wave_mixing/SPS mixing pictures/{name}.png"
    plt2.plot_2D(XX, YY, ZZ, xlabel=xlabel, ylabel=ylabel, title=name,
                 vmin=-142, vmax=-90,
                 savepath=the_path, cmap="magma")

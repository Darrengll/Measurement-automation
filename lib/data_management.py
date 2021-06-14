import time
import pickle as pkl
import os
from lib import plotting as pl
from lib.iq_downconversion_calibration import IQDownconversionCalibrationResult
from lib.measurement import Measurement
import numpy as np


#directory = 'Data\\IQMXCalibration'
directory = r'D:\GitHub\Measurement-automation\data\IQMXCalibrationData\IQMXCalibration'

def save_downconversion_calibration(downconv_calibration):
    if not os.path.exists(directory):
        os.makedirs(directory)

    calibration = downconv_calibration.get_dict()

    filename = calibration["_mixer_id"]
    iffreq = downconv_calibration.get_if_frequency()
    try:
        with open(directory + "\\" + filename + '.pkl', 'rb') as f:
            known_cal_data = pkl.load(f)

            known_cal_data[iffreq] = calibration

        with open(directory + "\\" + filename + '.pkl', 'wb') as f:
            pkl.dump(known_cal_data, f)

    except FileNotFoundError:
        new_cal_data = {iffreq: calibration}

        with open(directory + "\\" + filename + '.pkl', 'w+b') as f:
            pkl.dump(new_cal_data, f)


def load_downconversion_calibration(mixer_id, iffreq):
    # directory = 'data\\IQMXCalibration'
    filename = mixer_id

    try:
        with open(directory + "\\" + filename + '.pkl', 'rb') as f:
            known_cal_data = pkl.load(f)

    except FileNotFoundError:
        return None
    cal_dict = known_cal_data[iffreq]
    return IQDownconversionCalibrationResult.load_dict(cal_dict)


def save_IQMX_calibration(iqmx_calibration):
    # directory = 'data\\IQMXCalibration'
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = iqmx_calibration.get_mixer_parameters()["mixer_id"]
    try:
        with open(directory + "\\" + filename + '.pkl', 'rb') as f:
            known_cal_data = pkl.load(f)

        try:
            cal_for_known_attenuation = known_cal_data[iqmx_calibration.get_mixer_parameters()["iq_attenuation"]]
            cal_for_known_attenuation[frozenset(iqmx_calibration.get_radiation_parameters().items())] = iqmx_calibration
        except KeyError:
            known_cal_data[iqmx_calibration.get_mixer_parameters()["iq_attenuation"]] = \
                {frozenset(iqmx_calibration.get_radiation_parameters().items()): iqmx_calibration}

        with open(directory + "\\" + filename + '.pkl', 'wb') as f:
            pkl.dump(known_cal_data, f)

    except FileNotFoundError:
        new_cal_data = {iqmx_calibration.get_mixer_parameters()["iq_attenuation"]: {
            frozenset(iqmx_calibration.get_radiation_parameters().items()): iqmx_calibration}}

        with open(directory + "\\" + filename + '.pkl', 'w+b') as f:
            pkl.dump(new_cal_data, f)


def load_IQMX_calibration_database(mixer_id, iq_attenuation):
    # directory = 'data\\IQMXCalibration'
    filename = mixer_id

    try:
        with open(directory + "\\" + filename + '.pkl', 'rb') as f:
            known_cal_data = pkl.load(f)

    except FileNotFoundError:
        return None
    return known_cal_data[iq_attenuation]


def save_measurement(measurement, filename, plot_amps_kwargs={}, plot_phas_kwargs={}, plot_kwargs={}):
    directory = 'Data\\' + measurement.get_datetime().strftime("%b %d %Y")
    if not os.path.exists(directory):
        os.makedirs(directory)

    subdirectory = "\\" + measurement.get_datetime().strftime("%H-%M-%S") + " - " + filename
    if not os.path.exists(directory + subdirectory):
        os.makedirs(directory + subdirectory)

    with open(directory + subdirectory + "\\" + filename + '.pkl', 'w+b') as f:
        pkl.dump((measurement.get_type(), measurement.get_data(), measurement.get_datetime(), \
                  measurement.get_recording_time(), measurement.get_context()), f)

    with open(directory + subdirectory + "\\" + filename + '_context.txt', 'w') as f:
        f.write(str(measurement.get_context()))

    for fig in pl.plot_measurement(measurement, plot_amps_kwargs, plot_phas_kwargs, **plot_kwargs):
        fig.savefig(directory + subdirectory + "\\" + filename + " " + fig.get_axes()[0].get_title() + ".png", dpi=600,
                    bbox_inches="tight")
    return directory + subdirectory + "\\"


def load_measurement(filename):
    with open(filename + '.pkl', 'r+b') as f:
        meas = pkl.load(f)
    measurement = Measurement()
    try:
        measurement.set_type(meas[0])
        measurement.set_data(meas[1])
        measurement.set_datetime(meas[2])
        measurement.set_recording_time(meas[3])
        measurement.set_context(meas[4])
    except IndexError:
        print("Loaded a measurement with an outdated format. Some fields will be undefined.")
    return measurement


def save_data(timestamp, name, data, context):
    the_folder = get_folder_path(timestamp, name)
    filename = f"{the_folder}/{name}.pkl"
    context_file = f"{the_folder}/context.txt"

    # create folders
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # save data
    with open(filename, 'wb') as f:
        pkl.dump(data, f)

    # save context
    with open(context_file, 'w') as f:
        f.write(str(context))


def get_folder_path(timestamp, name):
    date_str = timestamp.strftime("%b %d %Y")
    time_str = timestamp.strftime("%H-%M-%S")
    the_path = f"data/Photon_wave_mixing/{date_str}/{time_str} - {name}"
    return the_path

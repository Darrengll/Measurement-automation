
from matplotlib._pylab_helpers import Gcf
from lib2.SingleToneSpectroscopy import *
import datetime
import numpy as np


def test_save_load_no_excess_plot():

    result = SingleToneSpectroscopyResult("test_no_excess_plot", "test")
    frequencies = np.linspace(0, 20000, 201)
    currents = np.linspace(0, 0.5, 20)

    XX, YY = np.meshgrid(frequencies, currents)

    S21s = np.cos(XX)+ 1j*np.sin(YY)


    data = {"Frequency [Hz]":frequencies, "Current [A]":currents, "data":S21s}
    result.set_data(data)
    result._anim = None
    result.set_start_datetime(datetime.datetime(2005, 7, 14, 12, 30))
    result._parameter_names = ["Frequency [Hz]", "Current [A]"]
    result.save()


    result1 = MeasurementResult.load("test", "test_no_excess_plot")

    # assert  not hasattr(result1,  "_lines")
    # assert not hasattr(result1, "_fit_lines")
    # assert np.all(result1.get_data()["data"] == data["data"])
    # assert np.all(result1.get_data()["echo_delay"] == data["echo_delay"])
    assert len(Gcf.get_all_fig_managers()) == 0
import pytest
from matplotlib._pylab_helpers import Gcf
from lib2.MeasurementResult import *
from lib2.DispersiveHahnEcho import *
import datetime
import numpy as np


@pytest.fixture(scope="module")
def result():
    result = DispersiveHahnEchoResult("test_no_excess_plot", "test")
    echo_delays = np.linspace(0, 20000, 201)
    S21s = np.cos(echo_delays) + 1j * np.sin(echo_delays)
    data = {"echo_delay": echo_delays, "data": S21s}
    result.set_data(data)
    result.set_start_datetime(datetime.datetime(2005, 7, 14, 12, 30))
    result.set_parameter_names(['echo_delay'])
    result._anim = None
    result._iter_idx_ready = [len(S21s)]
    return result

def test_get_state(result):
    d = result.__getstate__()
    assert d["_lines"] == [None]*2
    assert d["_fit_lines"] == [None]*2
    assert d["_anno"] == [None]*2

def test_save_load_no_excess_plot(result):

    result.save()
    result1 = MeasurementResult.load("test", "test_no_excess_plot")

    # assert np.all(result1.get_data()["data"] == result1.get_data()["data"])
    # assert np.all(result1.get_data()["echo_delay"] == result1.get_data()["echo_delay"])
    assert len(Gcf.get_all_fig_managers()) == 0

    MeasurementResult.delete("test", "test_no_excess_plot", delete_all=True)
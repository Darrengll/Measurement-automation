from datetime import datetime

import pytest
from matplotlib._pylab_helpers import Gcf
from numpy.random import random

from lib2.DispersiveRabiChevrons import DispersiveRabiChevronsResult
from lib2.MeasurementResult import MeasurementResult, find
from PIL import ImageChops
from PIL import Image
from numpy import array, zeros_like, std, linspace


@pytest.fixture(scope="module")
def result():
    result = DispersiveRabiChevronsResult("test_no_excess_plot", "test")
    durations = linspace(0, 20000, 201)
    S21s = random(size=(100, 100)) + 1j * random(size=(100, 100))
    data = {"excitation_duration": durations,
            "excitation_frequency":durations,
            "data": S21s}
    result.set_data(data)
    result.set_start_datetime(datetime(2005, 7, 14, 12, 30))
    result.set_parameter_names(['excitation_duration', 'excitation_frequency'])
    result._anim = None
    return result

def test_visualize():

    baseline_result = MeasurementResult.load("test", "rabi-chevrons-plotting-baseline")
    baseline_result._maps = [None]*4
    baseline_result._cbs = [None] * 4
    baseline_result._name = 'rabi-chevrons-plotting-baseline-test'
    baseline_result.save(plot_maximized=False)
    image1 = Image.open(find('rabi-chevrons-plotting-baseline.png', 'data')[0])
    image2 = Image.open(find('rabi-chevrons-plotting-baseline-test.png', 'data')[0])
    im = ImageChops.difference(image1.convert('RGB'), image2.convert('RGB'))
    differ = (array(list(image2.getdata())) - array(list(image1.getdata()))).ravel()

    assert std(differ) < 15

def test_save_load_no_excess_plot(result):

    result.save()
    baseline_result = MeasurementResult.load("test",
                                             "test_no_excess_plot")

    # assert np.all(result1.get_data()["data"] == result1.get_data()["data"])
    # assert np.all(result1.get_data()["echo_delay"] == result1.get_data()["echo_delay"])
    assert len(Gcf.get_all_fig_managers()) == 0

    MeasurementResult.delete("test", "test_no_excess_plot", delete_all=True)
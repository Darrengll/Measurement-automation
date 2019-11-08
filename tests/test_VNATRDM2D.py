from lib2.MeasurementResult import MeasurementResult, find
from PIL import ImageChops
from PIL import Image
from numpy import array, zeros_like, std

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
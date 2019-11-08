import datetime

from lib2.MeasurementResult import MeasurementResult, find
from matplotlib import pyplot as plt



def test_save_load():
    result = MeasurementResult("test_delete", "test")
    result._datetime = datetime.datetime(2005, 11, 11)
    result.save()

    result1 = MeasurementResult.load("test", "test_delete")
def test_save_delete():

    result = MeasurementResult("test_delete", "test")
    result._datetime = datetime.datetime(2005, 11, 11)

    result.save()
    assert len(find("*test_delete*", "data")) == 5

    MeasurementResult.delete("test", "test_delete", delete_all=True)
    assert len(find("*test_delete*", "data")) == 0

    plt.close("all")
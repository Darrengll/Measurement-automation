import pytest
from drivers.keysight11713C import *


@pytest.mark.skip
def test_set_get():
    attenuator = Keysight11713C("swc1", "Y")
    for i in range(82):
        attenuator.set_attenuation(i)
        assert i == attenuator.get_attenuation()



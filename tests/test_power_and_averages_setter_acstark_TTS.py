from lib2.FastTwoToneSpectroscopy import *
from unittest.mock import MagicMock, Mock, call
from time import sleep
from numpy import linspace

from lib2.FastTwoToneSpectroscopyBase import FluxControlType


class DataGen:

    def __init__(self):
        self._counter = 0
        self._nop = 1
        self._freq = 0.1

    def fun(self, x):
        return [cos(self._freq * x) + 1j * cos(self._freq * x)] * self._nop

    def get_sdata(self):
        retval = self.fun(self._counter)
        self._counter += 1
        sleep(0.001)
        return retval

    def get_frequencies(self):
        return linspace(self._freq_limits[0], self._freq_limits[1], self._nop)

    def set_parameters(self, params):
        self._nop = params["nop"]
        self._freq_limits = params["freq_limits"]

    def set_freq_limits(self, *limits):
        self._freq_limits = limits

    def set_nop(self, nop):
        self._nop = nop

    def detect(self):
        return 5.9e9, 0, 0

dg = DataGen()
vna = MagicMock()
cur_src = MagicMock()
vol_src = MagicMock()
q_lo = MagicMock()
q_awg = MagicMock()
ro_awg = MagicMock()
q_z_awg = Mock()
mw_src = MagicMock()
RD = Mock()
RD.detect = dg.detect
vna.get_sdata = dg.get_sdata
vna.get_frequencies = dg.get_frequencies
vna.set_freq_limits = dg.set_freq_limits
vna.set_nop = dg.set_nop

def test_power_and_averages_setter():

    equipment = {"vna": [vna], "mw_src": [mw_src], 'voltage_src': [vol_src]}
    FASTS = FastAcStarkTwoToneSpectroscopy("test_delete", 'test', FluxControlType.VOLTAGE,
                                           vna=[vna], mw_src=[mw_src], voltage_src=[vol_src])
    FASTS._mw_src[0].write(":SWEep:ATTen:PROTection ON")
    mw_src_frequencies = linspace(5.5e9, 5.7e9, 101)
    res_limits = (4e9, 4e9)
    vol = 0.32

    mw_src_parameters = {"power": 5, "freq_limits": (mw_src_frequencies[0], mw_src_frequencies[-1]),
                         "nop": len(mw_src_frequencies), "ext_trig_channel": "TRIG1"}

    vna_parameters = {"bandwidth": 100, "freq_limits": res_limits, "nop": len(mw_src_frequencies),
                      "sweep_type": "LIN",
                      "power": -12, "averages": 10, "aux_num": 1, "trig_dur": 3e-3, "res_find_power": -20,
                      "res_find_nop": 401}

    FASTS._resonator_detector = RD
    vna_powers = linspace(-10, 6, 51)
    dev_params = {'vna': [vna_parameters],
                  'mw_src': [mw_src_parameters]}
    power = 100
    FASTS.set_fixed_parameters(flux_control_parameter=vol, bandwidth_factor=10, **dev_params)
    FASTS.set_swept_parameters(vna_powers)

    start_averages = vna_parameters["averages"]
    avg_factor = exp((power - vna_powers[0]) / vna_powers[0] * log(10))

    vna_parameters1 = {"bandwidth": 100, "freq_limits": res_limits, "nop": len(mw_src_frequencies),
                       "sweep_type": "LIN",
                       "power": 100, "averages": 0, "aux_num": 1, "trig_dur": 3e-3, "res_find_power": -20,
                       "res_find_nop": 401, 'trig_per_point': True, 'pos': True, 'bef': False}

    acs_result = FASTS._power_and_averages_setter(power)
    vna.set_parameters.assert_called_with(vna_parameters1)
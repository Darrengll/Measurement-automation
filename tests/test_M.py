##проверить, что в приборы загружаются правильные данные
from lib2.Measurement import Measurement
from lib2.MeasurementResult import MeasurementResult, ContextBase
from unittest.mock import MagicMock, Mock
from time import sleep
from numpy import linspace

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
        sleep(0.05)
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
vna = Mock()
cur_src = Mock()
q_lo = MagicMock()
q_awg = MagicMock()
ro_awg = MagicMock()
q_z_awg = Mock()
mw_src = Mock()
RD = Mock()
RD.detect = dg.detect
vna.get_sdata = dg.get_sdata
vna.get_frequencies = dg.get_frequencies
vna.set_freq_limits = dg.set_freq_limits
vna.set_nop = dg.set_nop

def test_loading_pars_to_devices():

    equipment = {"vna": [vna], "q_lo":[q_lo, q_lo], "q_awg": [q_awg], \
     "ro_awg": [ro_awg], "q_z_awg": [q_z_awg], "mw_src": [mw_src], "bias_src": [cur_src]}
    meas = Measurement("test_delete", "test", {"vna":[vna], "q_lo":[q_lo], "q_awg":[q_awg],\
                         "ro_awg":[ro_awg], "q_z_awg":[q_z_awg], "mw_src":[mw_src], "bias_src": [cur_src]})
    vna_parameters = {"bandwidth": 10, "freq_limits": [6e9] * 2, "nop": 10, "averages": 1}
    sequence_parameters = {"awg_trigger_reaction_delay": 0, "readout_duration": 3000,
                           "repetition_period": 30000, "half_pi_pulse_duration": 100 / 2}
    ro_awg_params = {"calibration": MagicMock()}
    q_awg_params = {"calibration": MagicMock()}
    mw_src_frequencies = linspace(8.5e9, 9.2e9, 51)
    mw_src_parameters = {"power": 0, 'freq_limits': mw_src_frequencies}
    q_z_awg_params = {"calibration": MagicMock()}
    q_freq = 8.7e9
    currents = linspace(0,0.5, 20)
    q_lo_params = {'power': -10, 'if_freq': q_freq + 100e6}
    dev_params = {'vna': [vna_parameters],
                  'ro_awg': [ro_awg_params],
                  'q_awg': [q_awg_params],
                  'q_lo': [q_lo_params, q_lo_params],
                  'mw_src':[mw_src_parameters],
                  'q_z_awg': [q_z_awg_params]}

    meas.set_measurement_result(MagicMock())
    meas.set_fixed_parameters(**dev_params)
    meas.set_swept_parameters(current_values = currents)

    for key in dev_params.keys():
        for dev, param in list(zip(equipment[key], dev_params[key])):
            dev.set_parameters.assert_called_with(param)




    # equip_param_pair = []
    # for key in equipment.keys():
    #     equip_param_one_pair = [equipment[key], dev_params[key]]
    #     equip_param_pair.append(equip_param_one_pair)
    # for one_pair in equip_param_pair:
    #     for i in one_pair[0]:
    #         b = one_pair[0].index(i)
    #         i.set_parameters.assert_called_with(one_pair[1][b])



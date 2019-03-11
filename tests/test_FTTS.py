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



def test_set_fixed_pars_in_FFTTS_cur():

    equipment = {"vna": [vna], "mw_src": [mw_src], 'current_src': [cur_src]}
    TTS = FastFluxTwoToneSpectroscopy("test_delete", "test", FluxControlType.CURRENT,
                                      vna=[vna], mw_src=[mw_src], current_src=[cur_src])
    res_limits = [4e9, 6e9]
    vna_parameterz = {"bandwidth": 250, "freq_limits": res_limits, "nop": 10, "power": -15, "averages": 1}
    TTS._resonator_detector = RD
    mw_src_frequencies = linspace(8.5e9, 9.2e9, 51)
    mw_src_parameters = {"power": 0, 'freq_limits': mw_src_frequencies}

    center = 0
    sweet_spot = 0
    period = 1
    currents = linspace(sweet_spot - period / 8, sweet_spot + period / 8, 51)
    dev_params = {'vna': [vna_parameterz],
                  'mw_src': [mw_src_parameters]}
    TTS.set_fixed_parameters(adaptive=True, **dev_params)
    TTS.set_swept_parameters(flux_parameter_values=currents)
    TTS._measurement_result._unwrap_phase = True

    for key in dev_params.keys():
        for dev, param in list(zip(equipment[key], dev_params[key])):
            dev.set_parameters.assert_called_with(param)


    tts_result = TTS.launch()
    [cur_src][0].set_current.assert_has_calls([call(current) for current in currents])


def test_set_fixed_pars_in_FFTTS_vol():

    equipment = {"vna": [vna], "mw_src": [mw_src], 'voltage_src': [vol_src]}
    TTS = FastFluxTwoToneSpectroscopy("test_delete", "test", FluxControlType.VOLTAGE,
                                      vna=[vna], mw_src=[mw_src], voltage_src=[vol_src])
    res_limits = [4e9, 6e9]
    vna_parameterz = {"bandwidth": 250, "freq_limits": res_limits, "nop": 10, "power": -15, "averages": 1}
    TTS._resonator_detector = RD
    mw_src_frequencies = linspace(8.5e9, 9.2e9, 51)
    mw_src_parameters = {"power": 0, 'freq_limits': mw_src_frequencies}

    center = 0
    sweet_spot = 0
    period = 1
    voltages = linspace(sweet_spot - period / 8, sweet_spot + period / 8, 51)
    dev_params = {'vna': [vna_parameterz],
                  'mw_src': [mw_src_parameters]}
    TTS.set_fixed_parameters(adaptive=True, **dev_params)
    TTS.set_swept_parameters(voltages)
    TTS._measurement_result._unwrap_phase = True

    for key in dev_params.keys():
        for dev, param in list(zip(equipment[key], dev_params[key])):
            dev.set_parameters.assert_called_with(param)


    tts_result = TTS.launch()
    [vol_src][0].set_voltage.assert_has_calls([call(voltage) for voltage in voltages])


def test_set_fixed_pars_in_FPTTS_cur():

    equipment = {"vna": [vna], "mw_src": [mw_src], 'current_src': [cur_src]}
    PTS = FastPowerTwoToneSpectroscopy("test_delete", "test", FluxControlType.CURRENT,
                                      vna=[vna], mw_src=[mw_src], current_src=[cur_src])

    current = 0
    mw_freq_limits = (5.5e9, 5.9e9)
    mw_freq_nop = 201
    res_limits = [4e9, 6e9]
    mw_src_frequencies = linspace(mw_freq_limits[0], mw_freq_limits[-1], mw_freq_nop)
    PTS._resonator_detector = RD
    vna_parameters = {"bandwidth": 50, "freq_limits": res_limits, "nop": mw_freq_nop,
                      "res_find_power": -0, "res_find_nop": 801,
                      "power": 0, "sweep_type": "LIN", "averages": 1,
                      "aux_num": 1, "trig_dur": 2.5e-3}
    mw_src_parameters = {"power": 7, "freq_limits": mw_freq_limits, "nop": mw_freq_nop,
                         "ext_trig_channel": "TRIG1"}
    dev_params = {'vna': [vna_parameters],
                  'mw_src': [mw_src_parameters]}
    mw_src_powers = linspace(-20, -5, 21)
    PTS.set_fixed_parameters(flux_control_parameter=current, adaptive=True, **dev_params)
    PTS.set_swept_parameters(mw_src_powers)
    PTS._measurement_result.set_phase_units("deg")
    PTS._measurement_result.set_unwrap_phase(False)


    for key in dev_params.keys():
        for dev, param in list(zip(equipment[key], dev_params[key])):
            dev.set_parameters.assert_called_with(param)


    pts_result = PTS.launch()
    [mw_src][0].set_power.assert_has_calls([call(power) for power in mw_src_powers])
    [cur_src][0].set_current.assert_called_with(current)


def test_set_fixed_pars_in_FPTTS_vol():

    equipment = {"vna": [vna], "mw_src": [mw_src], 'voltage_src': [vol_src]}
    PTS = FastPowerTwoToneSpectroscopy("test_delete", "test", FluxControlType.VOLTAGE,
                                      vna=[vna], mw_src=[mw_src], voltage_src=[vol_src])

    voltage = 3.14
    mw_freq_limits = (5.5e9, 5.9e9)
    mw_freq_nop = 201
    res_limits = [4e9, 6e9]
    mw_src_frequencies = linspace(mw_freq_limits[0], mw_freq_limits[-1], mw_freq_nop)
    PTS._resonator_detector = RD
    vna_parameters = {"bandwidth": 50, "freq_limits": res_limits, "nop": mw_freq_nop,
                      "res_find_power": -0, "res_find_nop": 801,
                      "power": 0, "sweep_type": "LIN", "averages": 1,
                      "aux_num": 1, "trig_dur": 2.5e-3}
    mw_src_parameters = {"power": 7, "freq_limits": mw_freq_limits, "nop": mw_freq_nop,
                         "ext_trig_channel": "TRIG1"}
    dev_params = {'vna': [vna_parameters],
                  'mw_src': [mw_src_parameters]}
    mw_src_powers = linspace(-20, -5, 21)
    PTS.set_fixed_parameters(flux_control_parameter=voltage, adaptive=True, **dev_params)
    PTS.set_swept_parameters(mw_src_powers)
    PTS._measurement_result.set_phase_units("deg")
    PTS._measurement_result.set_unwrap_phase(False)


    for key in dev_params.keys():
        for dev, param in list(zip(equipment[key], dev_params[key])):
            dev.set_parameters.assert_called_with(param)


    pts_result = PTS.launch()
    [mw_src][0].set_power.assert_has_calls([call(power) for power in mw_src_powers])
    [vol_src][0].set_voltage.assert_called_with(voltage)


def test_set_fixed_pars_in_FACSTTS_cur():


    equipment = {"vna": [vna], "mw_src": [mw_src], 'current_src': [cur_src]}
    FASTS = FastAcStarkTwoToneSpectroscopy("test_delete", 'test', FluxControlType.CURRENT,
                                           vna=[vna], mw_src=[mw_src], current_src=[cur_src])
    FASTS._mw_src[0].write(":SWEep:ATTen:PROTection ON")
    mw_src_frequencies = linspace(5.5e9, 5.7e9, 101)
    res_limits = (4e9, 4e9)
    cur = 1.8

    mw_src_parameters = {"power": 5, "freq_limits": (mw_src_frequencies[0], mw_src_frequencies[-1]),
                         "nop": len(mw_src_frequencies), "ext_trig_channel": "TRIG1"}

    vna_parameters = {"bandwidth": 100, "freq_limits": res_limits, "nop": len(mw_src_frequencies),
                      "sweep_type": "LIN",
                      "power": -10, "averages": 10, "aux_num": 1, "trig_dur": 3e-3, "res_find_power": -20,
                      "res_find_nop": 401}

    FASTS._resonator_detector = RD


    vna_powers = linspace(-10, 6, 51)
    dev_params = {'vna': [vna_parameters],
                  'mw_src': [mw_src_parameters]}
    FASTS.set_fixed_parameters(flux_control_parameter=cur, bandwidth_factor=10,  **dev_params)
    FASTS.set_swept_parameters(vna_powers)

    start_averages = vna_parameters["averages"]
    avg_factors = exp((vna_powers - vna_powers[0]) / vna_powers[0] * log(start_averages))
    avg_factors = around(start_averages * avg_factors)


    for key in dev_params.keys():
        for dev, param in zip(equipment[key], dev_params[key]):
            dev.set_parameters.assert_called_with(param)

    cur_src.set_current.assert_called_with(cur)

    test_args = []
    for pow, avg in zip(vna_powers,avg_factors):
        vna_pars = vna_parameters.copy()
        vna_pars["averages"]=avg
        vna_pars['power']=pow
        test_args.append(vna_pars)



    acs_result = FASTS.launch()
    [vna][0].set_parameters.assert_has_calls([call(test_arg) for test_arg in test_args])

def test_set_fixed_pars_in_FACSTTS_vol():


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
                      "res_find_nop": 401, 'trig_per_point': True, 'pos': True, 'bef': False}

    FASTS._resonator_detector = RD

    vna_powers = linspace(-10, 6, 3)
    dev_params = {'vna': [vna_parameters],
                  'mw_src': [mw_src_parameters]}
    FASTS.set_fixed_parameters(flux_control_parameter=vol, bandwidth_factor=10, **dev_params)
    FASTS.set_swept_parameters(vna_powers)

    start_averages = vna_parameters["averages"]
    avg_factors = exp((vna_powers - vna_powers[0]) / vna_powers[0] * log(start_averages))
    avg_factors = around(start_averages * avg_factors)


    for key in dev_params.keys():
        for dev, param in zip(equipment[key], dev_params[key]):
            dev.set_parameters.assert_called_with(param)

    [vol_src][0].set_voltage.assert_called_with(vol)

    test_args = []
    for pow, avg in zip(vna_powers,avg_factors):
        vna_pars = vna_parameters.copy()
        vna_pars["averages"]=avg
        vna_pars['power']=pow
        test_args.append(vna_pars)

    acs_result = FASTS.launch()
    vna.set_parameters.assert_has_calls([call(test_arg) for test_arg in test_args])
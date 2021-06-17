import pytest

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
mw_src = MagicMock()
RD = Mock()
RD.detect = dg.detect
vna.get_sdata = dg.get_sdata
vna.get_frequencies = dg.get_frequencies
vna.set_freq_limits = dg.set_freq_limits
vna.set_nop = dg.set_nop
vna.set_parameters = MagicMock(side_effect = dg.set_parameters)

equipment = {"vna": [vna], "mw_src": [mw_src],
             'current_src': [cur_src], 'voltage_src': [vol_src]}

vna_parameters = {"bandwidth": 25,
                  "freq_limits": [4e9, 6e9],
                  "nop": 10,
                  "power": -15,
                  "averages": 100,
                  "resonator_detection_bandwidth": 100,
                  "resonator_detection_nop":101,
                  "sweep_type": "LIN",
                  "aux_num": 1,
                  "trig_dur": 2.5e-3}

mw_src_parameters = {"power": 0,
                     'frequencies': linspace(3e9, 6e9, 101),
                     "ext_trig_channel": "TRIG1"}

mw_src.write(":SWEep:ATTen:PROTection ON")


@pytest.fixture(params = [FluxControlType.CURRENT, FluxControlType.VOLTAGE])
def FFTTS(request):
    tts = FastFluxTwoToneSpectroscopy("test_delete",
                                      "test",
                                      request.param,
                                      **equipment)
    tts._resonator_detector = RD
    return tts


@pytest.fixture(params = [FluxControlType.CURRENT, FluxControlType.VOLTAGE])
def FPTTS(request):
    tts = FastPowerTwoToneSpectroscopy("test_delete",
                                       "test",
                                       request.param,
                                       **equipment)
    tts._mw_src[0].write(":SWEep:ATTen:PROTection ON")
    tts._resonator_detector = RD
    return tts


@pytest.fixture(params = [FluxControlType.CURRENT, FluxControlType.VOLTAGE])
def FACSTTS(request):
    tts = FastAcStarkTwoToneSpectroscopy("test_delete",
                                       "test",
                                       request.param,
                                       **equipment)
    tts._mw_src[0].write(":SWEep:ATTen:PROTection ON")
    tts._resonator_detector = RD
    return tts

#@pytest.mark.skip
def test_FFTTS(FFTTS):

    flux_param_values = linspace(-1, 1, 51)
    dev_params = {'vna': [vna_parameters.copy()],
                  'mw_src': [mw_src_parameters.copy()]}

    FFTTS.set_fixed_parameters(adaptive=True, **dev_params)
    FFTTS.set_swept_parameters(flux_parameter_values=flux_param_values)

    for key in dev_params.keys():
        for dev, param in list(zip(equipment[key], dev_params[key])):
            dev.set_parameters.assert_called_with(param)


    FFTTS.launch()
    if FFTTS.get_flux_control_type() is FluxControlType.CURRENT:
        cur_src.set_current.assert_has_calls([call(current) for current in flux_param_values])
    elif FFTTS.get_flux_control_type() is FluxControlType.VOLTAGE:
        vol_src.set_voltage.assert_has_calls([call(voltage) for voltage in flux_param_values])


# @pytest.mark.skip
def test_FPTTS(FPTTS):

    dev_params = {'vna': [vna_parameters.copy()],
                  'mw_src': [mw_src_parameters.copy()]}
    mw_src_powers = linspace(-20, -5, 21)
    flux_parameter_value = 0
    FPTTS.set_fixed_parameters(flux_control_parameter=flux_parameter_value, adaptive=True, **dev_params)
    FPTTS.set_swept_parameters(mw_src_powers)


    for key in dev_params.keys():
        for dev, param in list(zip(equipment[key], dev_params[key])):
            dev.set_parameters.assert_called_with(param)


    FPTTS.launch()
    mw_src.set_power.assert_has_calls([call(power) for power in mw_src_powers])

    if FPTTS.get_flux_control_type() is FluxControlType.CURRENT:
        cur_src.set_current.assert_called_with(flux_parameter_value)
    elif FPTTS.get_flux_control_type() is FluxControlType.VOLTAGE:
        vol_src.set_voltage.assert_called_with(flux_parameter_value)


def test_FACSTTS(FACSTTS):

    dev_params = {'vna': [vna_parameters.copy()],
                  'mw_src': [mw_src_parameters.copy()]}

    flux_param = 1.8
    vna_powers = linspace(-10, 6, 3)
    FACSTTS.set_fixed_parameters(flux_control_parameter=flux_param,
                                 **dev_params)
    FACSTTS.set_swept_parameters(vna_powers)

    start_averages = vna_parameters["averages"]
    avg_factors = exp((vna_powers - vna_powers[0]) / vna_powers[0] * log(start_averages))
    avg_factors = around(start_averages * avg_factors)


    for key in dev_params.keys():
        for dev, param in zip(equipment[key], dev_params[key]):
            dev.set_parameters.assert_called_with(param)

    if FACSTTS.get_flux_control_type() is FluxControlType.CURRENT:
        cur_src.set_current.assert_called_with(flux_param)
    elif FACSTTS.get_flux_control_type() is FluxControlType.VOLTAGE:
        vol_src.set_voltage.assert_called_with(flux_param)

    test_args = []
    vna_parameters.update({'trig_per_point': True, 'pos': True, 'bef': False})
    test_args.append(vna_parameters.copy())
    for power, avg in zip(vna_powers, avg_factors):
        test_args.append({"nop": vna_parameters["resonator_detection_nop"],
         "freq_limits": vna_parameters["freq_limits"],
         "power": power,
         "bandwidth": vna_parameters["resonator_detection_bandwidth"],
         "averages":avg})

        vna_pars = vna_parameters.copy()
        vna_pars['freq_limits'] = (5.9e9, 5.9e9)
        vna_pars["averages"] = avg
        vna_pars["power"] = power
        test_args.append(vna_pars)



    FACSTTS.launch()
    vna.set_parameters.assert_has_calls([call(test_arg) for test_arg in test_args])
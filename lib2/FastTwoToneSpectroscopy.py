from numpy import *
from lib2.FastTwoToneSpectroscopyBase import FastTwoToneSpectroscopyBase
from time import sleep

class FastFluxTwoToneSpectroscopy(FastTwoToneSpectroscopyBase):

    def __init__(self, name, sample_name, flux_control_type, **devs_aliases_map):
        super().__init__(name, sample_name,
                         flux_control_type, devs_aliases_map)

    def set_fixed_parameters(self, flux_control_parameter = None,
                             bandwidth_factor=10, **dev_params):

        vna_parameters = dev_params['vna'][0]
        mw_src_parameters = dev_params['mw_src'][0]
        self._resonator_area = vna_parameters["freq_limits"]
        self._adaptive = True if flux_control_parameter is None else False

        # trigger layout is detected via mw_src_parameters in TTSBase class

        super().set_fixed_parameters(vna=dev_params['vna'], mw_src=dev_params['mw_src'],
                                     flux_control_parameter=flux_control_parameter,
                                     detect_resonator=True if flux_control_parameter is not None else False,
                                     bandwidth_factor=bandwidth_factor)

    def set_swept_parameters(self, flux_parameter_values):
        setter = self._adaptive_setter if self._adaptive else self._triggering_setter
        swept_pars = {self._parameter_name: (setter, flux_parameter_values)}
        super().set_swept_parameters(**swept_pars)


class FastPowerTwoToneSpectroscopy(FastTwoToneSpectroscopyBase):

    def __init__(self, name, sample_name, flux_control_type, **devs_aliases_map):
        super().__init__(name, sample_name, flux_control_type, devs_aliases_map)

    def set_fixed_parameters(self, flux_control_parameter,
                             bandwidth_factor=10, **dev_params):

        vna_parameters = dev_params['vna'][0]
        mw_src_parameters = dev_params['mw_src'][0]
        self._resonator_area = vna_parameters["freq_limits"]
        # trigger layout is detected via mw_src_parameters in TTSBase class

        super().set_fixed_parameters(flux_control_parameter, vna=dev_params['vna'], mw_src=dev_params['mw_src'],
                                     detect_resonator=True,
                                     bandwidth_factor=bandwidth_factor)

    def set_swept_parameters(self, power_values):
        self._base_parameter_setter = self._mw_src[0].set_power
        swept_pars = {"Power [dBm]": (self._triggering_setter, power_values)}
        super().set_swept_parameters(**swept_pars)


class FastAcStarkTwoToneSpectroscopy(FastTwoToneSpectroscopyBase):

    def __init__(self, name, sample_name, flux_control_type, **devs_aliases_map):

        super().__init__(name, sample_name, flux_control_type, devs_aliases_map)

    def set_fixed_parameters(self, flux_control_parameter, bandwidth_factor=10, **dev_params):

        vna_parameters = dev_params['vna'][0]
        mw_src_parameters = dev_params['mw_src'][0]
        vna_parameters = dev_params['vna'][0]
        self._resonator_area = vna_parameters["freq_limits"]
        super().set_fixed_parameters(flux_control_parameter, detect_resonator=False,
                                     bandwidth_factor=bandwidth_factor,
                                     vna=dev_params['vna'],
                                     mw_src=dev_params['mw_src'])

    def set_swept_parameters(self, power_values):

        swept_pars = \
            {"Readout power [dBm]": (self._power_and_averages_setter, power_values)}
        super().set_swept_parameters(**swept_pars)

    def _power_and_averages_setter(self, power):
        powers = self._swept_pars["Readout power [dBm]"][1]
        vna_parameters = self._fixed_pars["vna"][0].copy()
        start_averages = vna_parameters["averages"]
        avg_factor = exp((power - powers[0]) / powers[0] * log(start_averages))
        vna_parameters["averages"] = round(start_averages * avg_factor)
        vna_parameters["power"] = power
        vna_parameters["freq_limits"] = self._resonator_area

        self._mw_src[0].set_output_state("OFF")
        if vna_parameters["freq_limits"][0] != vna_parameters["freq_limits"][-1]:
            # print("\rDetecting a resonator within provided frequency range of the VNA %s\
            #        "%(str(vna_parameters["freq_limits"])), flush=True, end="")

            res_result = self._detect_resonator(vna_parameters, plot=False)

            if (res_result is None):
                print("Failed to fit resonator, trying to use last successful fit, power = ", power,
                      " A")
                if (self._last_resonator_result is None):
                    print("no successful fit is present, terminating")
                    return None
                else:
                    res_result = self._last_resonator_result
            else:
                self._last_resonator_result = res_result

            res_freq, res_amp, res_phase = self._last_resonator_result
            # print("\rDetected frequency is %.5f GHz, at %.2f mU and %.2f \
            #        degrees"%(res_freq/1e9, res_amp*1e3, res_phase/pi*180), end="")
        else:
            res_freq = vna_parameters["freq_limits"][0]

        self._mw_src[0].set_output_state("ON")
        vna_parameters["freq_limits"] = (res_freq, res_freq)

        self._vna[0].set_parameters(vna_parameters)
        self._mw_src[0].send_sweep_trigger()  # telling mw_src to start

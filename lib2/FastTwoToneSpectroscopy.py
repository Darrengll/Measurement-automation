from numpy import *
from lib2.FastTwoToneSpectroscopyBase import FastTwoToneSpectroscopyBase
from time import sleep


class FastFluxTwoToneSpectroscopy(FastTwoToneSpectroscopyBase):

    def __init__(self, name, sample_name, line_attenuation_db=60, **devs_aliases_map):
        super().__init__(name, sample_name,
                         line_attenuation_db, devs_aliases_map)

    def set_swept_parameters(self, current_values=None,
                             voltage_values=None):
        
        base_parameter_values = current_values if voltage_values is None else voltage_values
        self._base_parameter_setter = self._current_src[0].set_current\
                                        if voltage_values is None else self._voltage_src[0].set_voltage

        self._base_parameter_name = "Voltage [V]" if voltage_values is not None else "Current [A]"

        swept_pars = {self._base_parameter_name: [self._base_parameter_setter, base_parameter_values]}
        super().set_swept_parameters(**swept_pars)


class FastPowerTwoToneSpectroscopy(FastTwoToneSpectroscopyBase):

    def __init__(self, name, sample_name, line_attenuation_db=60, **devs_aliases_map):
        super().__init__(name, sample_name, line_attenuation_db, devs_aliases_map)

    def set_swept_parameters(self, power_values):
        self._base_parameter_setter = self._mw_src[0].set_power
        swept_pars = {"Power [dBm]": [self._base_parameter_setter, power_values]}
        super().set_swept_parameters(**swept_pars)


class FastAcStarkTwoToneSpectroscopy(FastTwoToneSpectroscopyBase):

    def __init__(self, name, sample_name, line_attenuation_db=60, **devs_aliases_map):
        super().__init__(name, sample_name, line_attenuation_db, devs_aliases_map)

    def set_swept_parameters(self, power_values):
        self._base_parameter_setter = self._power_and_averages_setter
        swept_pars = \
            {"Readout power [dBm]": [self._base_parameter_setter, power_values]}
        super().set_swept_parameters(**swept_pars)

    def _power_and_averages_setter(self, power):
        powers = self._swept_pars["Readout power [dBm]"][1]
        vna_parameters = self._fixed_pars["vna"]
        start_averages = vna_parameters[0]["averages"]
        avg_factor = exp((power - powers[0]) / powers[0] * log(start_averages))
        vna_parameters[0]["averages"] = round(start_averages * avg_factor)
        vna_parameters[0]["power"] = power

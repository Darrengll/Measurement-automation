from enum import Enum, auto
from json import load

class ResonatorType(Enum):
    REFLECTION = auto()
    NOTCH = auto()
    TRANSMISSION = auto()

resonator_type_map = {"reflection": ResonatorType.REFLECTION,
                      "notch": ResonatorType.NOTCH,
                      "transmission": ResonatorType.TRANSMISSION}

class TTSRunnerParameters:
    parameters = None
    path = "lib2/fulaut/fulaut_parameters.json"

    def __init__(self):
        with open(self.path) as f:
            self.parameters = load(f)["tts_runner"]

    @property
    def vna_parameters(self):
        return self.parameters["vna_parameters"]

class STSRunnerParameters:
    parameters = None
    path = "lib2/fulaut/fulaut_parameters.json"

    def __init__(self):
        with open(self.path) as f:
            self.parameters = load(f)["sts_runner"]

    @property
    def flux_nop(self):
        return self.parameters["flux_nop"]

    @property
    def vna_parameters(self):
        return self.parameters["vna_parameters"]

    @property
    def anticrossing_oracle_hints(self):
        return self.parameters["anticrossing_oracle_hints"]

class ResonatorOracleParameters:
    parameters = None
    path = "lib2/fulaut/fulaut_parameters.json"

    def __init__(self):
        with open(self.path) as f:
            self.parameters = load(f)["resonator_oracle"]

    @property
    def peak_number(self):
        return self.parameters["peak_number"]

    @property
    def vna_parameters(self):
        return self.parameters["vna_parameters"]


class GlobalParameters:

    parameters = None
    path = "lib2/global_parameters.json"

    def __init__(self):
        with open(self.path) as f:
            self.parameters = load(f)
    
    @property
    def resonator_type(self):
        return resonator_type_map[self.parameters["resonator_type"]]

    @property
    def which_sweet_spot(self):
        return self.parameters["which_sweet_spot"]

    @property
    def ro_ssb_power(self):
        return self.parameters["ro_ssb_power"]

    @property
    def exc_ssb_power(self):
        return self.parameters["exc_ssb_power"]

    @property
    def spectroscopy_readout_power(self):
        return self.parameters["spectroscopy_readout_power"]

    @property
    def spectroscopy_excitation_power(self):
        return self.parameters["spectroscopy_excitation_power"]



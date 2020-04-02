from enum import Enum, auto
from json import load

class ResonatorType(Enum):
    REFLECTION = auto()
    NOTCH = auto()
    TRANSMISSION = auto()

resonator_type_map = {"reflection": ResonatorType.REFLECTION,
                      "notch": ResonatorType.NOTCH,
                      "transmission": ResonatorType.TRANSMISSION}

class FulautParameters:
    parameters = None
    path = "lib2/fulaut/fulaut_parameters.json"

    def __init__(self):
        with open(self.path) as f:
            self.parameters = load(f)

    @property
    def resonator_oracle(self):
        return self.parameters["resonator_oracle"]

    @property
    def sts_runner(self):
        return self.parameters["sts_runner"]

class GlobalParameters:

    parameters = None
    path = "lib2/global_parameters.json"

    def __init__(self):
        with open(self.path) as f:
            self.parameters = load(f)
    
    @property
    def resonator_type(self):
        return resonator_type_map[self.parameters["resonator_type"]]

    def which_sweet_spot(self):
        return self.parameters["which_sweet_spot"]

    def ro_ssb_power(self):
        return self.parameters["ro_ssb_power"]

    def exc_ssb_power(self):
        return self.parameters["exc_ssb_power"]

    def spectroscopy_readout_power(self):
        return self.parameters["spectroscopy_readout_power"]

    def spectroscopy_excitation_power(self):
        return self.parameters["spectroscopy_excitation_power"]

    def anticrossing_oracle_hits(self):
        return self.parameters["anticrossing_oracle_hits"]




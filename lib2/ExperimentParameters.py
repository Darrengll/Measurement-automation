from enum import Enum, auto
from json import load


class ResonatorType(Enum):
    REFLECTION = auto()
    NOTCH = auto()
    TRANSMISSION = auto()


resonator_type_map = {"reflection": ResonatorType.REFLECTION,
                      "notch": ResonatorType.NOTCH,
                      "transmission": ResonatorType.TRANSMISSION}


class ExperimentParameters:
    _group_name = None
    _subgroup_name = None
    _path = "lib2/experiment_parameters.json"
    _parameters = {}

    def __init__(self):
        with open(self._path) as f:
            self._parameters = load(f)[self._group_name]

        if self._subgroup_name is not None:
            self._parameters = self._parameters[self._subgroup_name]

        self._init_fields()

    def set_group_name(self, name):
        self._group_name = name

    def set_subgroup_name(self, name):
        self._subgroup_name = name

    def _init_fields(self):
        for property_name in vars(self).keys():
            if property_name[0] is not "_":
                self.__setattr__(property_name, self._parameters[property_name])


class GlobalParameters(ExperimentParameters):
    def __init__(self):
        self.set_group_name("global")
        self.which_sweet_spot = None
        self.readout_power = None
        self.excitation_power = None
        super().__init__()

    @property
    def resonator_type(self):
        return resonator_type_map[self._parameters["resonator_type"]]


class FulautParameters(ExperimentParameters):

    def __init__(self):
        self.set_group_name("fulaut")
        self.rerun = None
        super().__init__()


class ResonatorOracleParameters(FulautParameters):
    def __init__(self):
        self.set_subgroup_name("resonator_oracle")
        self.peak_number = None
        self.vna_parameters = None
        self.default_scan_area = None
        self.window = None
        super().__init__()


class STSRunnerParameters(FulautParameters):
    def __init__(self):
        self.set_subgroup_name("sts_runner")
        self.flux_nop = None
        self.vna_parameters = None
        self.anticrossing_oracle_hints = None
        super().__init__()


class TTSRunnerParameters(FulautParameters):
    def __init__(self):
        self.set_subgroup_name("tts_runner")
        self.vna_parameters = None
        self.frequency_span = None
        self.periods = None
        self.flux_nop = None
        self.frequency_nop = None
        super().__init__()


class ACSTTSRunnerParameters(FulautParameters):
    def __init__(self):
        self.set_subgroup_name("acstts_runner")
        self.vna_parameters = None
        super().__init__()


class TimeResolvedParameters(FulautParameters):

    def __init__(self):
        self.readout_duration = None
        self.repetition_period = None
        self.nop = None
        self.averages = None
        super().__init__()


class RabiParameters(TimeResolvedParameters):
    def __init__(self):
        self.set_subgroup_name("rabi")
        self.max_excitation_duration = None
        super().__init__()


class RamseyParameters(TimeResolvedParameters):
    def __init__(self):
        self.set_subgroup_name("ramsey")
        self.max_ramsey_delay = None
        self.detuning = None
        super().__init__()


class DecayParameters(TimeResolvedParameters):
    def __init__(self):
        self.set_subgroup_name("decay")
        self.max_readout_delay = None
        super().__init__()


class HahnEchoParameters(TimeResolvedParameters):
    def __init__(self):
        self.set_subgroup_name("hahn_echo")
        self.max_echo_delay = None
        super().__init__()

from drivers.BiasType import BiasType


class AWGVoltageSource:

    def __init__(self, awg, channel_number):
        self._awg = awg
        self._channel_number = channel_number
        self._voltage = None
        self._bias_type = BiasType.VOLTAGE

    def get_range(self):
        return self._awg.get_voltage_range()

    def set_voltage(self, voltage):
        self._voltage = voltage
        self._awg.output_arbitrary_waveform([voltage] * 100, 1e6,
                                            channel=self._channel_number)

    def get_voltage(self):
        return self._voltage

    def set(self, parameter):
        self.set_voltage(parameter)

    def set_appropriate_range(self, voltage_value):
        pass  # this driver does not support ranges

    def get_bias_type(self):
        return self._bias_type
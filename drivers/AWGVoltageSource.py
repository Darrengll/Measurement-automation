
class AWGVoltageSource():

    def __init__(self, awg, channel_number):
        self._awg = awg
        self._channel_number = channel_number
        self._voltage = None

    def set_voltage(self, voltage):
        self._voltage = voltage
        self._awg.output_arbitrary_waveform([voltage]*3, 1e6,
            channel=self._channel_number, asynchronous=False)

    def get_voltage(self):
        return self._voltage

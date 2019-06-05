import os
import sys

sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')
import keysightSD1
import unittest

sys.path.append(os.path.abspath('..\\..\\src\\Python\\'))
from ktqet_qubit import *

AWG_MODEL = 'M3202A'
DIG_MODEL = 'M3102A'


class KtQetQubitTestCase(unittest.TestCase):

    def setUp(self):
        # Create the baseband equipment driver instances (required by the qubit
        # objects)
        # The SD1 driver does not really support simulation mode.  However,
        # you can still create an instance of the driver and bang away on it
        # but the calls will always return error codes.
        self.__control_awg = keysightSD1.SD_AOU()
        self.__readout_awg = keysightSD1.SD_AOU()
        self.__readout_dig = keysightSD1.SD_AIN()

        self.__control_awg.openWithSlot(AWG_MODEL, 1, 2)
        self.__readout_awg.openWithSlot(AWG_MODEL, 1, 3)
        self.__readout_dig.openWithSlot(DIG_MODEL, 1, 3)

        # Create a new qubit object
        self.qubit = KtQetQubit(self.__control_awg, self.__readout_awg, self.__readout_dig, 1, 1, 1)

    def tearDown(self):
        self.qubit.dispose()
        self.qubit = None

        if self.__control_awg is not None:
            self.__control_awg.close()
        if self.__readout_awg is not None:
            self.__readout_awg.close()
        if self.__readout_dig is not None:
            self.__readout_dig.close()

        self.__control_awg = None
        self.__readout_awg = None
        self.__readout_dig = None

    def test_constructor(self):
        self.assertNotEqual(self.qubit, None)


def suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(KtQetQubitTestCase)
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())

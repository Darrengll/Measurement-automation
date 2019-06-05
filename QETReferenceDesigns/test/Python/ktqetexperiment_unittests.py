import os
import sys

sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')
import keysightSD1
import unittest

sys.path.append(os.path.abspath('..\\..\\src\\Python\\'))
from ktqet_experiment import *

AWG_MODEL = 'M3202A'
DIG_MODEL = 'M3102A'


class KtQetExperimentTestCase(unittest.TestCase):
    """
    KtQetExperimentTestCase is intended to be a base class for all of the
    reference design classes unit tests.  All of the reference designs
    aggregate a single qubit object, and so this class helps with the common
    object creation required.
    """

    def setUp(self):
        # Create the baseband equipment driver instances (required by the qubit
        # objects)
        # The SD1 driver does not really support simulation mode.  However,
        # you can still create an instance of the driver and bang away on it
        # but the calls will always return error codes.
        self.__control_awg = keysightSD1.SD_AOU()
        self.__readout_awg = keysightSD1.SD_AOU()
        self.__readout_dig = keysightSD1.SD_AIN()

        self.__control_awg.openWithSlot(AWG_MODEL, 1, 4)
        self.__readout_awg.openWithSlot(AWG_MODEL, 1, 4)
        self.__readout_dig.openWithSlot(DIG_MODEL, 1, 5)

        # Create a new qubit object
        self.qubit = KtQetQubit(self.__control_awg, self.__readout_awg, self.__readout_dig, 1, 3, 1)
        self.experiment = KtQetExperiment(self.qubit)

    def tearDown(self):
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

    def test_configure_acquisition(self):
        # The configure_acquisition function requires two parameters both of which must be
        # multiples of 10 ns
        acquisition_delay = 4
        acquisition_length = 100

        try:
            self.experiment.configure_acquisition(acquisition_delay, acquisition_length)
            self.fail()
        except InvalidParameterException as e:
            self.assertEqual(str(e), 'Invalid acquisition_delay. Value must be a multiple of 10 ns.')

        acquisition_delay = 100
        acquisition_length = 7

        try:
            self.experiment.configure_acquisition(acquisition_delay, acquisition_length)
            self.fail()
        except InvalidParameterException as e:
            self.assertEqual(str(e), 'Invalid acquisition_length. Value must be an even integer.')

        acquisition_length = 0
        try:
            self.experiment.configure_acquisition(acquisition_delay, acquisition_length)
            self.fail()
        except InvalidParameterException as e:
            self.assertEqual(str(e), 'Invalid acquisition_length. Value must be greater than 20.')        

        acquisition_delay = 100
        acquisition_length = 100
        self.experiment.configure_acquisition(acquisition_delay, acquisition_length)
        self.assertEqual(self.experiment._acquisition_delay, acquisition_delay)
        self.assertEqual(self.experiment._acquisition_length, acquisition_length)

    def test_configure_time(self):
        # The configure_time_parameters function requires two parameters, both of which need to be multiples of 10 ns.
        initial_tau = 4
        tau_step_size = 20

        try:
            self.experiment.configure_time_parameters(initial_tau, tau_step_size)
            self.fail()
        except InvalidParameterException as e:
            self.assertEqual(str(e), 'Invalid initial_tau.  Value must be a multiple of 10 ns.')

        initial_tau = 100
        tau_step_size = 7

        try:
            self.experiment.configure_time_parameters(initial_tau, tau_step_size)
            self.fail()
        except InvalidParameterException as e:
            self.assertEqual(str(e), 'Invalid tau_step_size.  Value must be a multiple of 10 ns.')

        initial_tau = 100
        tau_step_size = 100
        self.experiment.configure_time_parameters(initial_tau, tau_step_size)
        self.assertEqual(self.experiment._initial_tau, initial_tau)
        self.assertEqual(self.experiment._tau_step_size, tau_step_size)

    def test_configure_flow_control(self):
        # The configure_flow_control_parameters function requires three parameters, one of which need to be multiples of 10 ns.
        step_delay = 4
        num_steps = 100
        num_loops = 4

        try:
            self.experiment.configure_flow_control_parameters(step_delay, num_steps, num_loops)
            self.fail()
        except InvalidParameterException as e:
            self.assertEqual(str(e), 'Invalid step_delay.  Value must be a multiple of 10 ns.')

        step_delay = 100
        self.experiment.configure_flow_control_parameters(step_delay, num_steps, num_loops)
        self.assertEqual(self.experiment._step_delay, step_delay)
        self.assertEqual(self.experiment._num_steps, num_steps)
        self.assertEqual(self.experiment._num_loops, num_loops)

    def test_configure_readout(self):
        # The configure_readout function requires three parameters, one of which must be a multiple of 10
        # and two that must be instances of SD_Wave 
        test_wave_i = None
        test_wave_q = SD1.SD_Wave()
        test_wave_q.newFromFile(os.path.abspath('..\\..\\include\\waveforms\\') + '\\' + 'Pi2wave.csv')
        readout_delay = 30

        try:
            self.experiment.configure_readout(readout_delay, test_wave_i, test_wave_q)
            self.fail()
        except InvalidParameterException as e:
            self.assertEqual(str(e), 'Invalid readout_wave_i. Variable must be an instance of SD_Wave.')

        test_wave_i = SD1.SD_Wave()
        test_wave_i.newFromFile(os.path.abspath('..\\..\\include\\waveforms\\') + '\\' + 'Pi2wave.csv')
        test_wave_q = None
        readout_delay = 30


        try:
            self.experiment.configure_readout(readout_delay, test_wave_i, test_wave_q)
            self.fail()
        except InvalidParameterException as e:
            self.assertEqual(str(e), 'Invalid readout_wave_q. Variable must be an instance of SD_Wave.')


        test_wave_i = SD1.SD_Wave()
        test_wave_i.newFromFile(os.path.abspath('..\\..\\include\\waveforms\\') + '\\' + 'Pi2wave.csv')
        test_wave_q = SD1.SD_Wave()
        test_wave_q.newFromFile(os.path.abspath('..\\..\\include\\waveforms\\') + '\\' + 'Pi2wave.csv')
        readout_delay = 7

        try:
            self.experiment.configure_readout(readout_delay, test_wave_i, test_wave_q)
            self.fail()
        except InvalidParameterException as e:
            self.assertEqual(str(e), 'Invalid readout_delay.  Value must be a multiple of 10 ns.')

        readout_delay = 100
        self.experiment.configure_readout(readout_delay, test_wave_i, test_wave_q)
        self.assertEqual(self.experiment._readout_delay, readout_delay)
        self.assertEqual(self.experiment._readout_wave_i, test_wave_i)
        self.assertEqual(self.experiment._readout_wave_q, test_wave_q)

    def test_configure_pi_waveforms(self):
        # The configure_pi_waveforms function requires two parameters, both of which must be
        # be instances of SD_Wave 
        test_wave_i = None
        test_wave_q = SD1.SD_Wave()
        test_wave_q.newFromFile(os.path.abspath('..\\..\\include\\waveforms\\') + '\\' + 'Pi2wave.csv')

        try:
            self.experiment.configure_pi_waveforms(test_wave_i, test_wave_q)
            self.fail()
        except InvalidParameterException as e:
            self.assertEqual(str(e), 'Invalid pi_wave_i. Variable must be an instance of SD_Wave.')


        test_wave_i = SD1.SD_Wave()
        test_wave_i.newFromFile(os.path.abspath('..\\..\\include\\waveforms\\') + '\\' + 'Pi2wave.csv')
        test_wave_q = None

        try:
            self.experiment.configure_pi_waveforms(test_wave_i, test_wave_q)
            self.fail()
        except InvalidParameterException as e:
            self.assertEqual(str(e), 'Invalid pi_wave_q. Variable must be an instance of SD_Wave.')


        test_wave_i = SD1.SD_Wave()
        test_wave_i.newFromFile(os.path.abspath('..\\..\\include\\waveforms\\') + '\\' + 'Pi2wave.csv')
        test_wave_q = SD1.SD_Wave()
        test_wave_q.newFromFile(os.path.abspath('..\\..\\include\\waveforms\\') + '\\' + 'Pi2wave.csv')

        self.experiment.configure_pi_waveforms(test_wave_i, test_wave_q)
        self.assertEqual(self.experiment._pi_wave_i, test_wave_i)
        self.assertEqual(self.experiment._pi_wave_q, test_wave_q)

    def test_configure_pi2_waveforms(self):
        # The configure_pi2_waveforms function requires two parameters, both of which must be
        # be instances of SD_Wave 
        test_wave_i = None
        test_wave_q = SD1.SD_Wave()
        test_wave_q.newFromFile(os.path.abspath('..\\..\\include\\waveforms\\') + '\\' + 'Pi2wave.csv')

        try:
            self.experiment.configure_pi2_waveforms(test_wave_i, test_wave_q)
            self.fail()
        except InvalidParameterException as e:
            self.assertEqual(str(e), 'Invalid pi2_wave_i. Variable must be an instance of SD_Wave.')


        test_wave_i = SD1.SD_Wave()
        test_wave_i.newFromFile(os.path.abspath('..\\..\\include\\waveforms\\') + '\\' + 'Pi2wave.csv')
        test_wave_q = None

        try:
            self.experiment.configure_pi2_waveforms(test_wave_i, test_wave_q)
            self.fail()
        except InvalidParameterException as e:
            self.assertEqual(str(e), 'Invalid pi2_wave_q. Variable must be an instance of SD_Wave.')


        test_wave_i = SD1.SD_Wave()
        test_wave_i.newFromFile(os.path.abspath('..\\..\\include\\waveforms\\') + '\\' + 'Pi2wave.csv')
        test_wave_q = SD1.SD_Wave()
        test_wave_q.newFromFile(os.path.abspath('..\\..\\include\\waveforms\\') + '\\' + 'Pi2wave.csv')

        self.experiment.configure_pi2_waveforms(test_wave_i, test_wave_q)
        self.assertEqual(self.experiment._pi2_wave_i, test_wave_i)
        self.assertEqual(self.experiment._pi2_wave_q, test_wave_q)


if __name__ == '__main__':
    pass

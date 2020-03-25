import os
import sys
sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')
import keysightSD1 as SD1
from time import sleep

from ktqet_exceptions import *
from ktqet_qubit import *


class KtQetExperiment(object):
    """
    KtQetExperiment is a base class for quantum engineering experiments.  The class contains common functionality
    used by various quantum  experiments and is not intended to be used as a standalone module, but should be
    extended by a concrete implementation of an experiment.
    """

    HVI_DONE = 1
    HVI_BUSY = 0
    INFINITE_CYCLES = 0
    QUEUE_MODE_CYCLIC = 1
    QUEUE_MODE_ONE_SHOT = 0
    NANOSECONDS_PER_CYCLE = 10

    def __init__(self, qubit: KtQetQubit):
        """
        Create a new instance of KtQetExperiment.
        :param qubit: The KtQetQubit object that encapsulates the hardware to be used.
        """
        self.qubit = qubit
        self.hvi = None

        # Acquisition parameters
        self._acquisition_delay = None
        self._acquisition_length = None
        # Time measurement parameters
        self._initial_tau = None
        self._tau_step_size = None        
        # Readout parameters
        self._readout_delay = None
        self._readout_wave_i = None
        self._readout_wave_q = None
        # Experiment flow control parameters
        self._step_delay = None
        self._num_steps = None
        self._num_loops = None
        # Pi waveform objects
        self._pi2_wave_i = None
        self._pi2_wave_q = None
        self._pi_wave_i = None
        self._pi_wave_q = None

    def dispose(self):
        """
        Disposes of all components.
        """
        if self.hvi is not None:
            self.hvi.close()
            self.hvi = None

        if self.qubit is not None:
            self.qubit.dispose()
            self.qubit = None

    def configure_acquisition(self, acquisition_delay, acquisition_length):
        """
        Configure the acquisition parameters.  This function MUST be called before calling run.
        :param acquisition_delay:  The time between readout acquisitions, in ns.  Must be in multiples of 10 ns.
        :param acquisition_length:  The length of the acquisition, in samples. For 
        M3102A (500MSa/s) ->  2 nanoseconds / sample
        M3100A (100MSa/s) -> 10 nanoseconds / sample
        """

        # Validate that we received values that are multiples of 10 ns.
        if acquisition_delay % 10 != 0:
            raise InvalidParameterException('Invalid acquisition_delay. Value must be a multiple of 10 ns.')
        if acquisition_length % 2 != 0:
            raise InvalidParameterException('Invalid acquisition_length. Value must be an even integer.')
        if acquisition_length < 20:
            raise InvalidParameterException('Invalid acquisition_length. Value must be greater than 20.')

        self._acquisition_delay = acquisition_delay
        self._acquisition_length = acquisition_length


    def configure_time_parameters(self, initial_tau, tau_step_size):
        """
        Configure the stimulus parameters.  This function MUST be called before calling run.

        :param initial_tau:  The initial time delay between the control and readout pulse, in ns.  
        Must be in multiples of 10 ns.
        :param tau_step_size:  The amount of time to add to the delay between the control and readout 
        pulse between iterations, in ns.  Must be in multiples of 10 ns.
        """

        # Validate that we received values that are multiples of 10 ns.
        if initial_tau % 10 != 0:
            raise InvalidParameterException('Invalid initial_tau.  Value must be a multiple of 10 ns.')
        if tau_step_size % 10 != 0:
            raise InvalidParameterException('Invalid tau_step_size.  Value must be a multiple of 10 ns.')

        self._initial_tau = initial_tau
        self._tau_step_size = tau_step_size


    def configure_flow_control_parameters(self, step_delay, num_steps, num_loops):
        """
        Configure the experiment's flow control parameters.  This function MUST be called before calling run.

        :param step_delay: The amount of time to wait after a single pulse sequence and acquisition are performed, in
        ns.  Must be in multiples of 10 ns.
        :param num_steps:  The number of times to increase the delay between the control and readout pulses (inner loop)
        :param num_loops:  The number of times to repeat the entire experiment (outer loop).
        """

        # Validate that we received values that are multiples of 10 ns.

        if step_delay % 10 != 0:
            raise InvalidParameterException('Invalid step_delay.  Value must be a multiple of 10 ns.')        
        if not isinstance(num_steps, int) or num_steps < 1:
            raise InvalidParameterException('Invalid num_steps. Value must be a positive integer.')
        if not isinstance(num_loops, int) or num_loops < 1:
            raise InvalidParameterException('Invalid num_loops. Value must be a positive integer.')

        self._step_delay = step_delay
        self._num_steps = num_steps
        self._num_loops = num_loops


    def configure_readout(self, readout_delay, readout_wave_i: SD1.SD_Wave, readout_wave_q: SD1.SD_Wave):
        """
        Configure the waveforms to use on the readout AWG's.
        :param readout_delay:  An additional delay that is applied after the last control pulse, but before the readout
        pulse in nanoseconds.  Must be a multiple of 10 ns.
        :param readout_wave_i:  The readout waveform for the I-channel of the AWG.
        :param readout_wave_q:  The readout waveform for the Q-channel of the AWG.

        """

        # Validate that we received values that are multiples of 10 ns.
        if readout_delay % 10 != 0:
            raise InvalidParameterException('Invalid readout_delay.  Value must be a multiple of 10 ns.')
        #Validate we received SD_Wave objects
        if not isinstance(readout_wave_i, SD1.SD_Wave):
            raise InvalidParameterException('Invalid readout_wave_i. Variable must be an instance of SD_Wave.')
        if not isinstance(readout_wave_q, SD1.SD_Wave):
            raise InvalidParameterException('Invalid readout_wave_q. Variable must be an instance of SD_Wave.')

        self._readout_delay = readout_delay
        self._readout_wave_i = readout_wave_i
        self._readout_wave_q = readout_wave_q


    def configure_pi_waveforms(self, pi_wave_i: SD1.SD_Wave, pi_wave_q : SD1.SD_Wave):
        """
        Configure the waveforms to use on the control AWG's.

        :param pi_wave_i:  The pi control waveform for the I-channel of the AWG.
        :param pi_wave_q:  The pi control waveform for the Q-channel of the AWG.
        """

        #Validate we received SD_Wave objects
        if not isinstance(pi_wave_i, SD1.SD_Wave):
            raise InvalidParameterException('Invalid pi_wave_i. Variable must be an instance of SD_Wave.')

        if not isinstance(pi_wave_q, SD1.SD_Wave):
            raise InvalidParameterException('Invalid pi_wave_q. Variable must be an instance of SD_Wave.')

        self._pi_wave_i = pi_wave_i
        self._pi_wave_q = pi_wave_q


    def configure_pi2_waveforms(self, pi2_wave_i: SD1.SD_Wave, pi2_wave_q : SD1.SD_Wave ):
        """
        Configure the waveforms to use on the control AWG's.

        :param pi2_wave_i:  The pi/2 control waveform for the I-channel of the AWG.
        :param pi2_wave_q:  The pi/2 control waveform for the Q-channel of the AWG.

        """

        #Validate we received SD_Wave objects
        if not isinstance(pi2_wave_i, SD1.SD_Wave):
            raise InvalidParameterException('Invalid pi2_wave_i. Variable must be an instance of SD_Wave.')

        if not isinstance(pi2_wave_q, SD1.SD_Wave):
            raise InvalidParameterException('Invalid pi2_wave_q. Variable must be an instance of SD_Wave.')

        self._pi2_wave_i = pi2_wave_i
        self._pi2_wave_q = pi2_wave_q


    def open_hvi(self, file_path):
        """
        Creates an instance of HVI and opens the requested HVI file.
        :param file_path: the path to the compiled HVI sequence (.HVI extention)
        """
        self.hvi = SD1.SD_HVI()

        # Check to see that the file actually exists
        if os.path.isfile(file_path):
            error = self.hvi.open(file_path)            
            if error < 0:
                if error != -8031 and error != -8038:
                    self.hvi.close()
                    self.hvi = None
                    raise HviException('Error opening HVI file: {}'.format(self.format_SD1_error(error)))
        else:
            self.hvi.close()
            self.hvi = None
            raise HviException('Failed to open HVI file: {}.  File does not exist.'.format(file_path))

    def assign_hardware_to_hvi(self, name: str, hardware):
        """
        Assign hardware to an open HVI project.
        :param name: The name of the hardware
        :param hardware: The hardware driver.
        :return:
        """
        if self.hvi is None:
            raise HviException('No HVI project open.  Open a valid HVI project using the open_hvi function.')

        error = self.hvi.assignHardwareWithUserNameAndModuleID(name, hardware)
        #
        if error < 0 and error != -8069:
            raise HviException('Error occurred assigning hardware. {}'.format(self.format_SD1_error(error)))

    def assign_parameter_to_hvi(self, hardware_name: str, parameter_name: str, value: int):
        """
        Assign an integer constant to an open HVI project.
        :param hardware_name: The name of the HVI hardware module.
        :param parameter_name: The name of the HVI project parameter to assign.
        :param value: The value to assign to the parameter.
        :return:
        """
        if self.hvi is None:
            raise HviException('No HVI project open.  Open a valid HVI project using the open_hvi function.')

        error = self.hvi.writeIntegerConstantWithUserName(hardware_name, parameter_name, value)
        if error < 0:
            raise HviException('Error occurred writing integer constant. {}'.format(self.format_SD1_error(error)))

    def compile_and_load_hvi(self):
        """
        Compiles an HVI project after all of the parameters have been applied.
        :return:
        """
        if self.hvi is None:
            raise HviException('No HVI project open.  Open a valid HVI project using the open_hvi function.')

        errors = self.hvi.compile()
        
        if errors > 0:
            print('Compile errors: {}'.format(errors))
            for error in range(errors):
                print('Error ' + error + ' : ' + self.hvi.compilationErrorMessage(error))

        error = self.hvi.load()
        if error < 0:
            print('load: {}'.format(self.format_SD1_error(error)))

    def start_and_wait_for_hvi(self, hvi_done_register_module, 
                                hvi_done_register_name='hvi_done', timeout=10):
        """
        Begins the currently loaded HVI sequence and waits for the 
        signaling 'HVI done register' to be set by the HVI sequence
        :param hvi_done_register_module: The SD_Module object that 
        contains the register used as a flag to signal that the
        HVI sequence has finished.
        :param hvi_done_register_name: The name of the register 
        (defined in HVI project) that signals the completion of
        the HVI sequence.
        return:
        """

        # How often we query the HVI seqeunce if it is done (in seconds)
        sleep_time = 0.1

        # Check we have a valid SD_HVI
        if self.hvi is None:
            raise HviException('No HVI project open.  Open a valid HVI project using the open_hvi function.')

        # Set the flag as BUSY
        error = hvi_done_register_module.writeRegisterByName(hvi_done_register_name, self.HVI_BUSY)
        if error < 0:
            raise HviException('Error occurred writing register. {}'.format(self.format_SD1_error(error)))

        # Start HVI sequence
        error = self.hvi.start()    
        if error < 0:
            raise HviException('Error occurred starting HVI sequence. {}'.format(self.format_SD1_error(error)))

        # We calculate how many times we have to query HVI if done 
        # for the given sleep_time
        attempts = int(timeout / sleep_time)
        
        # Read flag
        temp = hvi_done_register_module.readRegisterByName(hvi_done_register_name)
        is_hvi_done = temp[1]
        error = temp[0]

        if error < 0:
            raise HviException('Error occurred writing register. {}'.format(self.format_SD1_error(error)))

        # Keep checking flag. Stop if done or timeout reached
        while is_hvi_done != self.HVI_DONE and attempts > 0:
            
            # Read flag
            temp = hvi_done_register_module.readRegisterByName(hvi_done_register_name)
            is_hvi_done = temp[1]
            error = temp[0]
            
            if error < 0:
                raise HviException('Error occurred writing register. {}'.format(self.format_SD1_error(error)))
            
            sleep(sleep_time)
            attempts -= 1

        # When we run out of attempts we have reached a timeout. 
        if attempts == 0:
            print('HVI wait timeout. Timeout too short? Was HVI sequence loaded correctly?')


    @staticmethod
    def format_SD1_error(error):
        return 'SD1 Error: {} - {}'.format(error, SD1.SD_Error.getErrorMessage(error))

    #Check that variables have been configured
    def _validate_acquisition_parameters(self):
        if self._acquisition_length is None:
            return False
        if self._acquisition_delay is None:
            return False
        return True

    def _validate_time_parameters(self):
        if self._initial_tau is None:
            return False
        if self._tau_step_size is None:
            return False
        return True

    def _validate_flow_control_parameters(self):
        if self._step_delay is None:
            return False
        if self._num_steps is None:
            return False
        if self._num_loops is None:
            return False
        return True

    def _validate_pi_waveform_parameters(self):
        if self._pi_wave_i is None:
            return False
        if self._pi_wave_q is None:
            return False
        return True

    def _validate_pi2_waveform_parameters(self):
        if self._pi2_wave_i is None:
            return False
        if self._pi2_wave_q is None:
            return False
        return True

    def _validate_readout_parameters(self):
        if self._readout_wave_i is None:
            return False
        if self._readout_wave_q is None:
            return False
        if self._readout_delay is None:
            return False            
        return True

    def _validate_parameters(self):
        if not self._validate_acquisition_parameters():
            raise ExperimentNotConfiguredException('Acquisition parameters not configured.')

        if not self._validate_readout_parameters():
            raise ExperimentNotConfiguredException('Readout parameters not configured.')


if __name__ == "__main__":
    pass

import sys

sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')
import keysightSD1 as SD1

from ktqet_experiment import *


class Ramsey(KtQetExperiment):
    """
    EnergyRelaxation is an extension of KtQetExperiment and is designed to demonstrate 
    T2 measurements on qubits. Two Pi pulses are played separated by a variable
    amount of time and followed by a readout pulse.
    
    """

    def __init__(self, qubit: KtQetQubit):
        super(self.__class__, self).__init__(qubit)

    def run(self, hvi_file_path):
        """
        Run the experiment.  WARNING - This is a blocking call!
        :param hvi_file_path: The path to the HVI file to use for the experiment.
        :return:
        """
        # Validate that the client has called configure first
        self._validate_parameters()

        print('Starting Ramsey experiment...')

        # Open the HVI file
        super(self.__class__, self).open_hvi(hvi_file_path)

        # Assign the hardware modules to the HVI project
        awg_name = 'RamseyAWG'
        digitizer_name = 'RamseyDIG'
        super(self.__class__, self).assign_hardware_to_hvi(awg_name, self.qubit.awg_control)
        super(self.__class__, self).assign_hardware_to_hvi(digitizer_name, self.qubit.dig_readout)

        # Assign all of the parameters to the HVI project.
        self.__assign_parameters_to_hvi(awg_name, digitizer_name)

        # Compile and load the HVI sequence
        super(self.__class__, self).compile_and_load_hvi()

        control_waveform_id_i = 0
        control_waveform_id_q = 1
        readout_waveform_id_i = 2
        readout_waveform_id_q = 3

        # Flush the waveform memory
        self.qubit.flush_awgs()

        # Upload the waveforms to memory
        self.qubit.awg_control.waveformLoad(self._pi2_wave_i, control_waveform_id_i)
        self.qubit.awg_control.waveformLoad(self._pi2_wave_q, control_waveform_id_q)
        self.qubit.awg_control.waveformLoad(self._readout_wave_i, readout_waveform_id_i)
        self.qubit.awg_control.waveformLoad(self._readout_wave_q, readout_waveform_id_q)

        # Queue the waveforms
        delay = 0
        prescalar = 0

        self.qubit.awg_control.AWGqueueWaveform(self.qubit.awg_control_channel_i, 
                control_waveform_id_i, SD1.SD_TriggerModes.SWHVITRIG_CYCLE, delay, 
                self.INFINITE_CYCLES, prescalar)
        self.qubit.awg_control.AWGqueueWaveform(self.qubit.awg_control_channel_q, 
                control_waveform_id_q, SD1.SD_TriggerModes.SWHVITRIG_CYCLE, delay, 
                self.INFINITE_CYCLES, prescalar)
        self.qubit.awg_control.AWGqueueWaveform(self.qubit.awg_readout_channel_i, 
                readout_waveform_id_i, SD1.SD_TriggerModes.SWHVITRIG_CYCLE, delay, 
                self.INFINITE_CYCLES, prescalar)
        self.qubit.awg_control.AWGqueueWaveform(self.qubit.awg_readout_channel_q, 
                readout_waveform_id_q, SD1.SD_TriggerModes.SWHVITRIG_CYCLE, delay, 
                self.INFINITE_CYCLES, prescalar)

        # Digitizer setup
        readout_points_per_cycle = int(self._acquisition_length)
        readout_cycles = int(self._num_loops * self._num_steps)
        readout_points = int(readout_points_per_cycle * readout_cycles)
        readout_points_per_read = readout_points
        while readout_points_per_read > (readout_points_per_cycle * 500):
            readout_points_per_read /= 2

        self.qubit.setup_readout(readout_points, readout_points_per_cycle, 
                                    readout_cycles, readout_points_per_read)

        print('Running...')

        self.qubit.start_awgs_dig()

        # Begin the experiment
        super(self.__class__, self).start_and_wait_for_hvi(self.qubit.awg_control, 
                            timeout = self._num_loops * self._num_steps)

        # Release resources
        self.hvi.stop()
        self.hvi.close()
        self.qubit.read_raw_data()
        self.qubit.stop_awgs_dig()        

        print('Done')


    def __assign_parameters_to_hvi(self, awg_name: str, digitizer_name: str):
        awg_parameters = {
            'nLoops': self._num_loops,
            'nSteps': self._num_steps,
            'initTau': int(self._initial_tau / self.NANOSECONDS_PER_CYCLE),
            'tauStep': int(self._tau_step_size / self.NANOSECONDS_PER_CYCLE),
            'ROdelay': int(self._readout_delay / self.NANOSECONDS_PER_CYCLE),
            'stepDelay': int(self._step_delay / self.NANOSECONDS_PER_CYCLE),
        }

        digitizer_parameters = {
            'tauStep': int(self._tau_step_size / self.NANOSECONDS_PER_CYCLE),
            'initAcqDelay': int(self._acquisition_delay / self.NANOSECONDS_PER_CYCLE),
        }

        for key, value in awg_parameters.items():
            super(self.__class__, self).assign_parameter_to_hvi(awg_name, key, value)

        for key, value in digitizer_parameters.items():
            super(self.__class__, self).assign_parameter_to_hvi(digitizer_name, key, value)


    def _validate_parameters(self):
        super(self.__class__, self)._validate_parameters()

        if not super(self.__class__, self)._validate_pi2_waveform_parameters():
            raise ExperimentNotConfiguredException('Pi/2 waveforms not configured.') 

        if not super(self.__class__, self)._validate_flow_control_parameters():
            raise ExperimentNotConfiguredException('Flow control parameters not configured.')

        if not super(self.__class__, self)._validate_time_parameters():
            raise ExperimentNotConfiguredException('Time parameters not configured.')


if __name__ == "__main__":
    pass

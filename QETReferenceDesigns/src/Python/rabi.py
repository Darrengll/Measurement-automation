import sys

sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')
import keysightSD1 as SD1

from ktqet_experiment import *


class Rabi(KtQetExperiment):
    """
	Rabi is an extension of KtQetExperiment and is designed to demonstrate 
    Rabi oscillations on qubits. Increasingly long pulses are played by
    the AWG followed by a readout pulse and digitizer aquisition.
	"""

    def __init__(self, qubit: KtQetQubit):
        super(self.__class__, self).__init__(qubit)
        self._rampup_wave_i = None
        self._rampup_wave_q = None
        self._rampdown_wave_i = None
        self._rampdown_wave_q = None

    def run(self, hvi_file_path):
        """
		Run the experiment.  WARNING - This is a blocking call!
		:param hvi_file_path: The path to the HVI file to use for the experiment.
		:return:
		"""
        # Validate that the client has called configure first
        self._validate_parameters()

        print('Starting Rabi experiment...')

        # Open the HVI file

        super(self.__class__, self).open_hvi(hvi_file_path)

        # Assign the hardware modules to the HVI project
        awg_name = 'RabiAWG'
        digitizer_name = 'RabiDIG'
        super(self.__class__, self).assign_hardware_to_hvi(awg_name, self.qubit.awg_control)
        super(self.__class__, self).assign_hardware_to_hvi(digitizer_name, self.qubit.dig_readout)

        # Assign all of the parameters to the HVI project.
        self.__assign_parameters_to_hvi(awg_name, digitizer_name)

        # Compile and load the HVI sequence
        super(self.__class__, self).compile_and_load_hvi()

        rampup_waveform_id_i = 0
        rampup_waveform_id_q = 1
        rampdown_waveform_id_i = 2
        rampdown_waveform_id_q = 3
        readout_waveform_id_i = 4
        readout_waveform_id_q = 5

        # Flush the waveform memory
        self.qubit.flush_awgs()

        # Upload the waveforms to memory
        self.qubit.awg_control.waveformLoad(self._rampup_wave_i, rampup_waveform_id_i)
        self.qubit.awg_control.waveformLoad(self._rampup_wave_q, rampup_waveform_id_q)
        self.qubit.awg_control.waveformLoad(self._rampdown_wave_i, rampdown_waveform_id_i)
        self.qubit.awg_control.waveformLoad(self._rampdown_wave_q, rampdown_waveform_id_q)
        self.qubit.awg_control.waveformLoad(self._readout_wave_i, readout_waveform_id_i)
        self.qubit.awg_control.waveformLoad(self._readout_wave_q, readout_waveform_id_q)

        # Queue the waveforms
        self.qubit.awg_control.AWGqueueWaveform(self.qubit.awg_control_channel_i, rampup_waveform_id_i,
                                                SD1.SD_TriggerModes.SWHVITRIG, 0, 1, 0)
        self.qubit.awg_control.AWGqueueWaveform(self.qubit.awg_control_channel_q, rampup_waveform_id_q,
                                                SD1.SD_TriggerModes.SWHVITRIG, 0, 1, 0)
        self.qubit.awg_control.AWGqueueWaveform(self.qubit.awg_control_channel_i, rampdown_waveform_id_i,
                                                SD1.SD_TriggerModes.SWHVITRIG, 0, 1, 0)
        self.qubit.awg_control.AWGqueueWaveform(self.qubit.awg_control_channel_q, rampdown_waveform_id_q,
                                                SD1.SD_TriggerModes.SWHVITRIG, 0, 1, 0)
        # Configure the queue for cyclical behavior
        self.qubit.awg_control.AWGqueueConfig(self.qubit.awg_control_channel_i, 1)
        self.qubit.awg_control.AWGqueueConfig(self.qubit.awg_control_channel_q, 1)

        self.qubit.awg_control.AWGqueueWaveform(self.qubit.awg_readout_channel_i, readout_waveform_id_i,
                                                SD1.SD_TriggerModes.SWHVITRIG_CYCLE, 0, 0, 0)
        self.qubit.awg_control.AWGqueueWaveform(self.qubit.awg_readout_channel_q, readout_waveform_id_q,
                                                SD1.SD_TriggerModes.SWHVITRIG_CYCLE, 0, 0, 0)

        # Digitizer setup
        readout_points_per_cycle = int(self._acquisition_length)
        readout_cycles = int(self._num_loops * self._num_steps)
        readout_points = int(readout_points_per_cycle * readout_cycles)
        readout_points_per_read = readout_points
        while readout_points_per_read > (readout_points_per_cycle * 500):
            readout_points_per_read /= 2  # look carefully

        self.qubit.setup_readout(readout_points, readout_points_per_cycle, readout_cycles, readout_points_per_read)

        # Running experiment
        print('Running...')

        self.qubit.start_awgs_dig()

        super(self.__class__, self).start_and_wait_for_hvi(self.qubit.awg_control,
                                                           timeout=self._num_loops * self._num_steps)

        # Release resources
        self.qubit.read_raw_data()
        self.qubit.stop_awgs()
        self.hvi.stop()
        self.hvi.close()

        print('Done')

    def __assign_parameters_to_hvi(self, awg_name: str, digitizer_name: str):
        awg_parameters = {
            'nLoops': int(self._num_loops),
            'nSteps': int(self._num_steps),
            'initialTau': int(self._initial_tau / self.NANOSECONDS_PER_CYCLE),
            'tauStep': int(self._tau_step_size / self.NANOSECONDS_PER_CYCLE),
            'ROdelay': int(self._readout_delay / self.NANOSECONDS_PER_CYCLE),
            'stepDelay': int(self._step_delay / self.NANOSECONDS_PER_CYCLE),
            'tauRepetition': int(1e6)
        }

        digitizer_parameters = {
            'tauStep': int(self._tau_step_size / self.NANOSECONDS_PER_CYCLE),
            'initAcqDelay': int(self._acquisition_delay / self.NANOSECONDS_PER_CYCLE),
            'tauRepetition': int(1e6)
        }

        for key, value in awg_parameters.items():
            super(self.__class__, self).assign_parameter_to_hvi(awg_name, key, value)

        for key, value in digitizer_parameters.items():
            super(self.__class__, self).assign_parameter_to_hvi(digitizer_name, key, value)

    def _validate_parameters(self):
        super(self.__class__, self)._validate_parameters()

        if not super(self.__class__, self)._validate_flow_control_parameters():
            raise ExperimentNotConfiguredException('Flow control parameters not configured.')

        if not self._validate_ramp_waveform_parameters():
            raise ExperimentNotConfiguredException('Ramp waveforms not configured.')

        if not super(self.__class__, self)._validate_time_parameters():
            raise ExperimentNotConfiguredException('Time parameters not configured.')

    def _validate_ramp_waveform_parameters(self):
        if self._rampup_wave_i is None:
            return False
        if self._rampup_wave_q is None:
            return False
        if self._rampdown_wave_i is None:
            return False
        if self._rampdown_wave_q is None:
            return False
        return True

    def _configure_rabi_waveforms(self, rampup_wave_i: SD1.SD_Wave, rampup_wave_q: SD1.SD_Wave,
                                  rampdown_wave_i: SD1.SD_Wave, rampdown_wave_q: SD1.SD_Wave):
        if not isinstance(rampup_wave_q, SD1.SD_Wave):
            raise InvalidParameterException('Invalid pi_wave_i. Variable must be an instance of SD_Wave.')

        if not isinstance(rampup_wave_i, SD1.SD_Wave):
            raise InvalidParameterException('Invalid pi_wave_q. Variable must be an instance of SD_Wave.')

        if not isinstance(rampdown_wave_q, SD1.SD_Wave):
            raise InvalidParameterException('Invalid pi_wave_i. Variable must be an instance of SD_Wave.')

        if not isinstance(rampdown_wave_i, SD1.SD_Wave):
            raise InvalidParameterException('Invalid pi_wave_q. Variable must be an instance of SD_Wave.')

        self._rampup_wave_i = rampup_wave_i
        self._rampup_wave_q = rampup_wave_q
        self._rampdown_wave_i = rampdown_wave_i
        self._rampdown_wave_q = rampdown_wave_q


if __name__ == "__main__":
    pass

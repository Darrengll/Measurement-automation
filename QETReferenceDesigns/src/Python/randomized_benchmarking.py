import sys

sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')
import keysightSD1 as SD1

from ktqet_experiment import *


class RandomizedBenchmarking(KtQetExperiment):
    """
    RandomizedBenchmarking is designed to demonstrate how to efficiently load 
    and play thousands of waveforms using the M320xA AWGs. A user-defined 
    sequence of waveform is played followed by a readout pulse and 
    digitizer aquisition.
    """

    CYCLES_PER_IQ_SAMPLE = 2    # When using IQ, it takes two clock cycles to output an IQ sample

    def __init__(self, qubit: KtQetQubit):
        super(self.__class__, self).__init__(qubit)
        self._gate_sequence_II = []
        self._gate_sequence_IQ = []
        self._gate_sequence_QI = []
        self._gate_sequence_QQ = []
        self._experiment_delay = None
        self._ng = None
        self._nl = None
        self._np = None
        self._ne = None        


    def run(self, hvi_file_path):
        """
        Run the experiment.  WARNING - This is a blocking call!
        :param hvi_file_path: The path to the HVI file to use for the experiment.
        :return:
        """

        # Validate that the client has called configure first
        self._validate_parameters()

        print('Starting Randomized Benchmarking experiment...')


        # Open the HVI file
        super(self.__class__, self).open_hvi(hvi_file_path)

        # Assign the hardware modules to the HVI project
        awg_name = 'RBAWG'
        digitizer_name = 'RBDIG'
        super(self.__class__, self).assign_hardware_to_hvi(awg_name, self.qubit.awg_control)
        super(self.__class__, self).assign_hardware_to_hvi(digitizer_name, self.qubit.dig_readout)

        # Assign all of the parameters to the HVI project.
        self.__assign_parameters_to_hvi(awg_name, digitizer_name)

        # Compile and load the HVI sequence
        super(self.__class__, self).compile_and_load_hvi()


        # Digitizer setup        
        readout_points_per_cycle = int(self._acquisition_length)
        readout_cycles = int(self._sequences)
        readout_points = int(readout_points_per_cycle * readout_cycles)
        readout_points_per_read = readout_points

        while readout_points_per_read > (readout_points_per_cycle * 500):
            readout_points_per_read /= 2

        self.qubit.setup_readout(readout_points, readout_points_per_cycle, readout_cycles, 
                readout_points_per_read)

        self.qubit.start_dig()
        # Create all SD_Wave objects in the PC
        gate_sequence_i = []
        gate_sequence_q = []
        gate_sequence_length = []
        
        for sequence in range(self._sequences):
            gate_sequence_i.append(SD1.SD_Wave())
            gate_sequence_i[sequence].newFromArrayDouble(SD1.SD_WaveformTypes.WAVE_IQ, 
                    self._gate_sequence_II[sequence], self._gate_sequence_IQ[sequence])
            
            gate_sequence_q.append(SD1.SD_Wave())
            gate_sequence_q[sequence].newFromArrayDouble(SD1.SD_WaveformTypes.WAVE_IQ, 
                    self._gate_sequence_QI[sequence], self._gate_sequence_QQ[sequence])

            gate_sequence_length.append( int( 
                len(self._gate_sequence_II[sequence]) 
                * self.CYCLES_PER_IQ_SAMPLE / self.NANOSECONDS_PER_CYCLE))


        #DDR waveforms IDs created for convenience (first half is I, second half is Q)
        gate_sequence_id_i = []
        gate_sequence_id_q = []
        max_number_of_sequences = int((N_MAX_WAVEFORMS_DDR - 2) / 2)
        for sequence in range(max_number_of_sequences):
            gate_sequence_id_i.extend([sequence])
            gate_sequence_id_q.extend([int(sequence + N_MAX_WAVEFORMS_DDR / 2)])
        
        readout_waveform_id_i = int((N_MAX_WAVEFORMS_DDR - 2) / 2)
        readout_waveform_id_q = int(N_MAX_WAVEFORMS_DDR - 1)


        # Load readout waveforms. _readout_wave_x must be SD_Wave objects
        self.qubit.awg_control.waveformLoad(self._readout_wave_i, readout_waveform_id_i)
        self.qubit.awg_control.waveformLoad(self._readout_wave_q, readout_waveform_id_q)


        print('Running...')

        error = self.hvi.start()
        if error < 0:
            self.hvi.close()
            return error

        sequence = 0
        total_sequences_loaded = 0
        consumed_loaded_sequences = 0
        
        while sequence < self._sequences:

            # wait for HVI
            check_experiment_running = parse("(0, {the.value})", 
                    str(self.qubit.awg_control.readRegisterByName('expRunningFlag')))

            # Loop while HVI is running
            while (int(check_experiment_running['the.value'])==1):
                check_experiment_running = parse("(0, {the.value})",
                        str(self.qubit.awg_control.readRegisterByName('expRunningFlag')))

            # HVI sequence is now waiting for PC

            self.qubit.stop_awgs()


            if (consumed_loaded_sequences >= total_sequences_loaded): #Check if new waveforms need to get loaded
                # Load and queue new sequences
                self.qubit.flush_awgs()
                consumed_loaded_sequences = 0

                total_sequences_loaded = self._load_new_sequences(consumed_loaded_sequences, max_number_of_sequences, 
                        gate_sequence_i, gate_sequence_q, gate_sequence_id_i, gate_sequence_id_q, 
                        readout_waveform_id_i, readout_waveform_id_q)

                self._queue_new_sequence_after_flush(gate_sequence_id_i[consumed_loaded_sequences],
                        gate_sequence_id_q[consumed_loaded_sequences], readout_waveform_id_i, readout_waveform_id_q)                

            else:
                # Only queue new sequence
                self._queue_new_sequence(gate_sequence_id_i[consumed_loaded_sequences], 
                        gate_sequence_id_q[consumed_loaded_sequences])

                consumed_loaded_sequences += 1            
            

            self.qubit.start_awgs()

            #Update HVI delay values
            self.qubit.awg_control.writeRegisterByName('ROdelayReg', 
                    gate_sequence_length[sequence] + int(self._readout_delay / self.NANOSECONDS_PER_CYCLE))
            
            self.qubit.awg_control.writeRegisterByName('expDelayReg',
                    gate_sequence_length[sequence] + int(self._experiment_delay / self.NANOSECONDS_PER_CYCLE))

            self.qubit.dig_readout.writeRegisterByName('acqDelayReg',
                    int(gate_sequence_length[sequence]) + int(self._acquisition_delay  / self.NANOSECONDS_PER_CYCLE))
            
            #Resume HVI
            self.qubit.awg_control.writeRegisterByName('PCreadyFlag', 1)

             #*************** HVI is running *********************

            sequence += 1


        # Release resources
        self.hvi.stop()
        self.hvi.close()
        self.qubit.read_raw_data()
        self.qubit.stop_awgs_dig()        

        print('Done')



    def add_gates_to_sequence(self, sequence_number, gate_II, gate_IQ, gate_QI, gate_QQ, number_of_gates):
        """
        Adds new gates to the specified sequence number. The first 'number_of_gates' elements will be 
        added from the provided gate arrays.
        :param: sequence: The sequence number or index that the gates to add the gate to. as an integer.
        :param: gateII: The array of gates (waveforms/double array) to play as the I component of the I channel. 
        :param: gateIQ: The array of gates (waveforms/double array) to play as the Q component of the I channel.
        :param: gateQI: The array of gates (waveforms/double array) to play as the I component of the Q channel.
        :param: gateQQ: The array of gates (waveforms/double array) to play as the Q component of the Q channel.
        :param: number_of_gates: The number of gates to add to the sequence.
        :return:
        """
        # TODO use arrays of SD_Waves instead of doubles. We don't want to restrict users to files or arrays.

        for gate_number in range(number_of_gates):
            self._gate_sequence_II[sequence_number].extend(gate_II[gate_number])
            self._gate_sequence_IQ[sequence_number].extend(gate_IQ[gate_number])
            self._gate_sequence_QI[sequence_number].extend(gate_QI[gate_number])
            self._gate_sequence_QQ[sequence_number].extend(gate_QQ[gate_number])


    def _queue_new_sequence(self, gate_sequence_id_i, gate_sequence_id_q):

        delay = 0        
        prescalar = 0

        #queue gate sequence
        self.qubit.awg_control.AWGflush(self.qubit.awg_control_channel_i)
        self.qubit.awg_control.AWGflush(self.qubit.awg_control_channel_q)
        self.qubit.awg_control.AWGqueueWaveform(self.qubit.awg_control_channel_i, 
                gate_sequence_id_i, SD1.SD_TriggerModes.SWHVITRIG_CYCLE, delay, 
                self.INFINITE_CYCLES, prescalar)
        self.qubit.awg_control.AWGqueueWaveform(self.qubit.awg_control_channel_q, 
                gate_sequence_id_q, SD1.SD_TriggerModes.SWHVITRIG_CYCLE, delay, 
                self.INFINITE_CYCLES, prescalar)


    def _queue_new_sequence_after_flush(self, gate_sequence_id_i, gate_sequence_id_q, 
                readout_pulse_id_i, readout_pulse_id_q):

        delay = 0        
        prescalar = 0

        #queue gate sequence
        self.qubit.awg_control.AWGqueueWaveform(self.qubit.awg_control_channel_i, 
                gate_sequence_id_i, SD1.SD_TriggerModes.SWHVITRIG_CYCLE, delay, 
                self.INFINITE_CYCLES, prescalar)
        self.qubit.awg_control.AWGqueueWaveform(self.qubit.awg_control_channel_q, 
                gate_sequence_id_q, SD1.SD_TriggerModes.SWHVITRIG_CYCLE, delay, 
                self.INFINITE_CYCLES, prescalar)
        
        #queue RO pulses
        self.qubit.awg_control.AWGqueueWaveform(self.qubit.awg_readout_channel_i, 
                int(readout_pulse_id_i), SD1.SD_TriggerModes.SWHVITRIG_CYCLE, delay, 
                self.INFINITE_CYCLES, prescalar)
        self.qubit.awg_control.AWGqueueWaveform(self.qubit.awg_readout_channel_q, 
                int(readout_pulse_id_q), SD1.SD_TriggerModes.SWHVITRIG_CYCLE, delay, 
                self.INFINITE_CYCLES, prescalar)


    def _load_new_sequences(self, firstSequence, max_number_of_sequences, gate_sequence_i, 
            gate_sequence_q, gate_sequence_id_i, gate_sequence_id_q, readout_pulse_id_i, readout_pulse_id_q):

        total_sequences_loaded = firstSequence
        newSequencesLoadedDDR = 0

        while total_sequences_loaded < self._sequences and \
                newSequencesLoadedDDR < max_number_of_sequences:

            self.qubit.awg_control.waveformLoad(gate_sequence_i[total_sequences_loaded],
                    gate_sequence_id_i[total_sequences_loaded])

            self.qubit.awg_control.waveformLoad(gate_sequence_q[total_sequences_loaded],
                    gate_sequence_id_q[total_sequences_loaded])

            newSequencesLoadedDDR += 1 
            total_sequences_loaded += 1
        
        self.qubit.awg_readout.waveformLoad(self._readout_wave_i, int(readout_pulse_id_i))
        self.qubit.awg_readout.waveformLoad(self._readout_wave_q, int(readout_pulse_id_q))
        return total_sequences_loaded        


    def configure_rb_parameters(self, Ng, Nl, Np, Ne, experiment_delay):
        """
        Configure randomized benchmarking parameters. This function MUST be called before
        calling run.
        :param Ng: The number of random computational sequences.
        :param Nl: The length of the subset of random lengths of Clifford operations.
        :param Np: The number of Pauli randomizations.
        :param Ne: The number of repetitions for each specific sequence.
        :param experiment_delay: the amount of time to wait between experiments in nanoseconds.
        Must be a multiple of 10 ns.
        """

        if not isinstance(Ng, int) or Ng < 1:
            raise InvalidParameterException('Invalid Ng. Value must be a positive integer')
        if not isinstance(Nl, int) or Nl < 1:
            raise InvalidParameterException('Invalid Nl. Value must be a positive integer')
        if not isinstance(Np, int) or Np < 1:
            raise InvalidParameterException('Invalid Np. Value must be a positive integer')
        if not isinstance(Ne, int) or Ne < 1:
            raise InvalidParameterException('Invalid Ne. Value must be a positive integer')
        if experiment_delay % 10 != 0:
            raise InvalidParameterException('Invalid experiment_delay. Value must be a multiple of 10 ns.')
        if experiment_delay < 0:
            raise InvalidParameterException('Invalid experiment_delay. Value must be a positive integer')

        self._ng = Ng
        self._nl = Nl
        self._np = Np
        self._ne = Ne
        self._experiment_delay = experiment_delay
        self.sequences =  self._sequences = Ng*Nl*Np

        # Create an array with Ng*Nl*Np elements
        for i in range(self._sequences):
            self._gate_sequence_II.append([])
            self._gate_sequence_IQ.append([])
            self._gate_sequence_QI.append([])
            self._gate_sequence_QQ.append([])
 

    def _validate_rb_parameters(self):
        if self._ng is None:
            return False
        if self._nl is None:
            return False
        if self._np is None:
            return False
        if self._ne is None:
            return False
        return True


    def _validate_parameters(self):
        super(self.__class__, self)._validate_parameters()

        if not self._validate_rb_parameters():
            raise ExperimentNotConfiguredException('Randomized Benchmarking parameters ' +\
                    'not configured.')   


    def __assign_parameters_to_hvi(self, awg_name: str, digitizer_name: str):

            awg_parameters = {
                    'Ne': self._ne,
                    'nSequences': self._sequences,
                    'PCisRunning': 0,
                    'HVIisRunning': 0,
                    'End': 0,
                    'NeISrunning': 0,
                    'NewSequence': 0
            }            

            digitizer_parameters = {
                    'acqDelay': int(self._acquisition_delay / self.NANOSECONDS_PER_CYCLE)
            }

            for key, value in awg_parameters.items():
                    super(self.__class__, self).assign_parameter_to_hvi(awg_name, key, value)

            for key, value in digitizer_parameters.items():
                    super(self.__class__, self).assign_parameter_to_hvi(digitizer_name, key, value)


if __name__ == "__main__":    
    pass
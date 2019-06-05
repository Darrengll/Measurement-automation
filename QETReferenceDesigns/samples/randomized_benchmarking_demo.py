import os
import sys


sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')
import keysightSD1 as SD1

sys.path.append(os.path.abspath('..\\src\\Python\\'))
import ktqet_qubit
import randomized_benchmarking

def main():

    awg_chassis = 1
    dig_chassis = 1
    awg_slot = 4
    dig_slot = 5
    awg_control_channel_i = 1
    awg_readout_channel_i = 3
    dig_readout_channel_i = 1


    awg_control = SD1.SD_AOU()
    awg_readout = awg_control
    dig_readout = SD1.SD_AIN()

    error = awg_control.openWithSlot('M3202A', awg_chassis, awg_slot)
    if error < 0:
        print('open awg error: {} {}'.format(error, SD1.SD_Error.getErrorMessage(error)))
        awg_control.close()
        exit()

    error = dig_readout.openWithSlot('M3102A', dig_chassis, dig_slot)
    if error < 0:
        print('open dig error: {} {}'.format(error, SD1.SD_Error.getErrorMessage(error)))
        awg_control.close()
        dig_readout.close()
        exit()

    qet_experiment = randomized_benchmarking.RandomizedBenchmarking(
        ktqet_qubit.KtQetQubit(awg_control, awg_readout, dig_readout, awg_control_channel_i, awg_readout_channel_i,
                               dig_readout_channel_i))


    awg_control_amp_i = 1.5
    awg_control_amp_q = 1.5
    awg_control_if_frequency = 0*200e6
    awg_control_offset_i = 0
    awg_control_offset_q = 0
    awg_readout_amp_i = 1.5
    awg_readout_amp_q = 1.5
    awg_readout_if_frequency = 100e6
    awg_readout_offset_i = 0
    awg_readout_offset_q = 0
    dig_readout_fullscale_i = 1.5
    dig_readout_fullscale_q = 1.5

    qet_experiment.qubit.setup_awg_control_channel(awg_control_amp_i, awg_control_amp_q, awg_control_offset_i,
                                                   awg_control_offset_q, awg_control_if_frequency)
    qet_experiment.qubit.setup_awg_readout_control(awg_readout_amp_i, awg_readout_amp_q, awg_readout_offset_i,
                                                   awg_readout_offset_q, awg_readout_if_frequency)
    qet_experiment.qubit.setup_dig_readout_channel(dig_readout_fullscale_i, dig_readout_fullscale_q)


    #Set up readout waveforms
    readout_wave_i = SD1.SD_Wave()
    readout_wave_q = SD1.SD_Wave()


    readout_wave_i.newFromFile(os.path.abspath('..\\include\\waveforms\\') + '\\' + 'ROwave.csv')
    readout_wave_q.newFromFile(os.path.abspath('..\\include\\waveforms\\') + '\\' + 'ROwave.csv')


    # RANDOMIZED BENCHMARKING DEMO START

    Ng = 2               #number of random computational sequences
    Nl = 2               #number of lenghts
    Np = 2               #number of Pauli randomizations
    Ne = 5               #number of experiments for each specific sequence
    
    readout_delay = 0                #in nanoseconds - must be a multiple of 10
    acquisition_delay = 500          #in nanoseconds - must be a multiple of 10
    acquisition_length = 80          #in samples - 2ns/sample for M3102A
    experiment_delay = 0             #in nanoseconds - must be a multiple of 10

    qet_experiment.configure_readout(readout_delay, readout_wave_i, readout_wave_q)
    qet_experiment.configure_rb_parameters(Ng, Nl, Np, Ne, experiment_delay)
    qet_experiment.configure_acquisition(acquisition_delay, acquisition_length)


    nGates = 10
    gatesII = []
    gatesIQ = []
    gatesQI = []
    gatesQQ = []
    for i in range (nGates):
        gatesII.append([])
        gatesIQ.append([])
        gatesQI.append([])
        gatesQQ.append([])


    for i in range(nGates):
        x = (i + 1) / nGates
        gatesII[i] = [0,0,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,0,0]
        gatesIQ[i] = [0,0,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,0,0]
        gatesQI[i] = [0,0,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,0,0]
        gatesQQ[i] = [0,0,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,0,0]

    # # gate 1
    # gatesII[0] = [0,0,0,1,1,1,1,0,0,0]
    # gatesIQ[0] = [0,0,0,1,1,1,1,0,0,0]
    # gatesQI[0] = [0,0,0,1,1,1,1,0,0,0]
    # gatesQQ[0] = [0,0,0,1,1,1,1,0,0,0]
    # # gate 2
    # gatesII[1] = [0,0,0,1,1,1,1,0,0,0]
    # gatesIQ[1] = [0,0,0,1,1,1,1,0,0,0]
    # gatesQI[1] = [0,0,0,1,1,1,1,0,0,0]
    # gatesQQ[1] = [0,0,0,1,1,1,1,0,0,0]
    # # gate 3
    # gatesII[2] = [0,0,0,1,1,1,1,0,0,0]
    # gatesIQ[2] = [0,0,0,1,1,1,1,0,0,0]
    # gatesQI[2] = [0,0,0,1,1,1,1,0,0,0]
    # gatesQQ[2] = [0,0,0,1,1,1,1,0,0,0]


    # rbNsequences was created in rbSetup and it equals Ng*Nl*Np
    for sequence in range(qet_experiment.sequences):
        # each element in the sequence will have 
        ###################
        ##
        ## User can add randomization in this section
        ##
        ###################
        qet_experiment.add_gates_to_sequence(sequence, gatesII, gatesIQ, gatesQI, gatesQQ, nGates)

    
    # Run the experiment
    file_name = 'RB.HVI'
    hvi_folder_path = os.path.abspath('..\\include\\hvi\\')
    file_path = hvi_folder_path + '\\' + file_name
    qet_experiment.run(file_path)


    # Plot acquisition data
    if (not qet_experiment.qubit.acquisition_successful):
        print('Full acquisition not successful. Check experiment parameters.')
    qet_experiment.qubit.plot_raw_readout_data()    

    # Save aquisition data to disk
    # rb_i_path = os.path.abspath('..\\include\\readout_data\\Randomized_Benchmarking_I.csv')
    # rb_q_path = os.path.abspath('..\\include\\readout_data\\Randomized_Benchmarking_Q.csv')
    # qet_experiment.qubit.save_raw_readout_data(rb_i_path, rb_q_path)

    awg_control.close()
    dig_readout.close()





if __name__ == "__main__":    
    main()

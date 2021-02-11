import os
import sys

sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')
import keysightSD1 as key

sys.path.append(os.path.abspath('..\\src\\Python\\'))
import ktqet_qubit
import rabi


def main():
    awg_chassis = 1
    dig_chassis = 1
    awg_slot = 4
    dig_slot = 5
    awg_control_channel_i = 1
    awg_readout_channel_i = 3
    dig_readout_channel_i = 1

    awg_control = key.SD_AOU()
    awg_readout = awg_control
    dig_readout = key.SD_AIN()

    error = awg_control.openWithSlot('M3202A', awg_chassis, awg_slot)
    if error < 0:
        print('open awg error: {} {}'.format(error, key.SD_Error.getErrorMessage(error)))
        awg_control.close()
        exit()

    error = dig_readout.openWithSlot('M3102A', dig_chassis, dig_slot)
    if error < 0:
        print('open dig error: {} {}'.format(error, key.SD_Error.getErrorMessage(error)))
        dig_readout.close()
        awg_control.close()
        exit()

    qet_experiment = rabi.Rabi(
        ktqet_qubit.KtQetQubit(awg_control, awg_readout, dig_readout, awg_control_channel_i, awg_readout_channel_i,
                               dig_readout_channel_i))
    # Configure the AWG and Digitizer
    awg_control_amp_i = 1.5
    awg_control_amp_q = 1.5
    awg_control_if_frequency = 100e6
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

    # Create waveform objects from .csv files
    rampup_wave_i = key.SD_Wave()
    rampup_wave_q = key.SD_Wave()
    rampdown_wave_i = key.SD_Wave()
    rampdown_wave_q = key.SD_Wave()
    readout_wave_i = key.SD_Wave()
    readout_wave_q = key.SD_Wave()
    
    rampup_wave_i.newFromFile(os.path.abspath('..\\include\\waveforms\\') + '\\' + 'RampUp.csv')
    rampup_wave_q.newFromFile(os.path.abspath('..\\include\\waveforms\\') + '\\' + 'RampUp.csv')
    rampdown_wave_i.newFromFile(os.path.abspath('..\\include\\waveforms\\') + '\\' + 'RampDown.csv')
    rampdown_wave_q.newFromFile(os.path.abspath('..\\include\\waveforms\\') + '\\' + 'RampDown.csv')
    readout_wave_i.newFromFile(os.path.abspath('..\\include\\waveforms\\') + '\\' + 'ROwave.csv')
    readout_wave_q.newFromFile(os.path.abspath('..\\include\\waveforms\\') + '\\' + 'ROwave.csv')

    # Configure the experiment
    acquisition_delay = 500
    acquisition_length = 100
    step_delay = 1500
    initial_tau = 10
    tau_step_size = 1000
    num_steps = 5
    num_loops = 2
    readout_delay = 0

    qet_experiment.configure_acquisition(acquisition_delay, acquisition_length)
    qet_experiment._configure_rabi_waveforms(rampup_wave_i, rampup_wave_q, rampdown_wave_i, rampdown_wave_q)
    qet_experiment.configure_readout(readout_delay, readout_wave_i, readout_wave_q)
    qet_experiment.configure_time_parameters(initial_tau, tau_step_size)
    qet_experiment.configure_flow_control_parameters(step_delay, num_steps, num_loops)

    # Run the experiment
    file_name = 'Rabi.HVI'
    hvi_folder_path = os.path.abspath('..\\include\\hvi\\')
    file_path = hvi_folder_path + '\\' + file_name
    qet_experiment.run(file_path)

    # Plot acquisition data
    if (not qet_experiment.qubit.acquisition_successful):
        print('Full acquisition not successful. Check experiment parameters.')
    qet_experiment.qubit.plot_raw_readout_data()

    # rabi_i_path = os.path.abspath('..\\include\\readout_data\\Rabi_I.csv')
    # rabi_q_path = os.path.abspath('..\\include\\readout_data\\Rabi_Q.csv')
    # qet_experiment.qubit.save_raw_readout_data(rabi_i_path, rabi_q_path)

    # input('Done. Press enter to exit.')
    awg_control.close()
    dig_readout.close()


if __name__ == "__main__":
    main()
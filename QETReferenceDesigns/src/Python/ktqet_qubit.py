import msvcrt
import sys
import matplotlib.pyplot as plt
import numpy as np
import time

sys.path.append('C:\Program Files (x86)\Keysight\SD1\Libraries\Python')
import keysightSD1 as key
from parse import *

QUEUE_CONFIG_ONE_SHOT = 0
QUEUE_CONFIG_CYCLIC = 1
N_MAX_WAVEFORMS_DDR = 100
N_MAX_WAVEFORMS_QUEUE = 1024
N_MAX_WAVEFORM_CYCLES_QUEUE = 65535
N_MAX_WAVEFORM_START_DELAY_QUEUE = 65535
MODULATION_ON = 1
MODULATION_OFF = 0
ALL_CHANNELS_MASK = 0xF


class KtQetQubit:
    """
    KtQetQubit is a wrapper around the instrumentation required to control a single qubit.  This is currently
    targeting superconducting qubits.  The intent of this class is to provide a simple API for controlling a qubit.
    """

    def __init__(self, awg_control, awg_readout, dig_readout, awg_control_channel_i, awg_readout_channel_i,
                 dig_readout_channel_i):
        self.awg_control = awg_control
        self.awg_readout = awg_readout
        self.dig_readout = dig_readout
        # TODO - IQ pairs are hard-coded to be adjacent. HVI API is needed to assign.
        self.awg_control_channel_i = awg_control_channel_i
        self.awg_control_channel_q = awg_control_channel_i + 1
        self.awg_readout_channel_i = awg_readout_channel_i
        self.awg_readout_channel_q = awg_readout_channel_i + 1
        self.dig_readout_channel_i = dig_readout_channel_i
        self.dig_readout_channel_q = dig_readout_channel_i + 1

        self.acquisition_successful = False

        self.readoutPoints = 10
        self.readoutPointsPerRead = 10

        self.tempReadoutBufferI = []
        self.tempReadoutBufferIlength = 0
        self.tempReadoutBufferQ = []
        self.tempReadoutBufferQlength = 0

        self.rawReadoutBufferI = []
        self.rawReadoutBufferIlength = 0
        self.rawReadoutBufferQ = []
        self.rawReadoutBufferQlength = 0

        self.dig_readout.channelPrescalerConfig(self.dig_readout_channel_i, 0)
        self.dig_readout.channelPrescalerConfig(self.dig_readout_channel_q, 0)

    def dispose(self):
        """
        Dispose of all managed components.
        """
        if self.awg_control is not None:
            self.awg_control.close()
            self.awg_control = None

        if self.awg_readout is not None:
            self.awg_readout.close()
            self.awg_readout = None

        if self.dig_readout is not None:
            self.dig_readout.close()
            self.dig_readout = None

    def setup_awg_control_channel(self, amplitude_i, amplitude_q, offset_i, offset_q, if_frequency,
                                  modulation_type="AM"):
        # TODO - Merge the two SetupAwg functions and make a new parameter to indicate control vs readout.
        self.awg_control.channelOffset(self.awg_control_channel_i, offset_i)
        self.awg_control.channelOffset(self.awg_control_channel_q, offset_q)
        if if_frequency == 0:
            self.awg_control.channelAmplitude(self.awg_control_channel_i, amplitude_i)
            self.awg_control.channelAmplitude(self.awg_control_channel_q, amplitude_q)
            self.awg_control.channelWaveShape(self.awg_control_channel_i, key.SD_Waveshapes.AOU_AWG)
            self.awg_control.channelWaveShape(self.awg_control_channel_q, key.SD_Waveshapes.AOU_AWG)

        else:
            self.awg_control.channelAmplitude(self.awg_control_channel_i, 0)
            self.awg_control.channelAmplitude(self.awg_control_channel_q, 0)
            self.awg_control.channelWaveShape(self.awg_control_channel_i, key.SD_Waveshapes.AOU_SINUSOIDAL)
            self.awg_control.channelWaveShape(self.awg_control_channel_q, key.SD_Waveshapes.AOU_SINUSOIDAL)
            self.awg_control.channelFrequency(self.awg_control_channel_i, if_frequency)
            self.awg_control.channelFrequency(self.awg_control_channel_q, if_frequency)
            if modulation_type == 'AM':
                self.awg_control.modulationAmplitudeConfig(self.awg_control_channel_i,
                                                           key.SD_ModulationTypes.AOU_MOD_AM,
                                                           amplitude_i)
                self.awg_control.modulationAmplitudeConfig(self.awg_control_channel_q,
                                                           key.SD_ModulationTypes.AOU_MOD_AM,
                                                           amplitude_q)
                self.awg_control.channelPhase(self.awg_control_channel_i, 0)
                self.awg_control.channelPhase(self.awg_control_channel_q, 90)
            if modulation_type == 'IQ':
                self.awg_control.modulationIQConfig(self.awg_control_channel_i, MODULATION_ON)  # value 1 to enable modulation
                self.awg_control.modulationIQConfig(self.awg_control_channel_q, MODULATION_ON)
            self.awg_control.channelPhaseResetMultiple(ALL_CHANNELS_MASK)

    def setup_awg_readout_control(self, amplitude_i, amplitude_q, offset_i, offset_q, if_frequency):
        self.awg_readout.channelOffset(self.awg_readout_channel_i, offset_i)
        self.awg_readout.channelOffset(self.awg_readout_channel_q, offset_q)
        if if_frequency == 0:
            self.awg_readout.channelAmplitude(self.awg_readout_channel_i, amplitude_i)
            self.awg_readout.channelAmplitude(self.awg_readout_channel_q, amplitude_q)
            self.awg_readout.channelWaveShape(self.awg_readout_channel_i, key.SD_Waveshapes.AOU_AWG)
            self.awg_readout.channelWaveShape(self.awg_readout_channel_q, key.SD_Waveshapes.AOU_AWG)
        else:
            self.awg_readout.channelAmplitude(self.awg_readout_channel_i, 0)
            self.awg_readout.channelAmplitude(self.awg_readout_channel_q, 0)
            self.awg_readout.channelWaveShape(self.awg_readout_channel_i, key.SD_Waveshapes.AOU_SINUSOIDAL)
            self.awg_readout.channelWaveShape(self.awg_readout_channel_q, key.SD_Waveshapes.AOU_SINUSOIDAL)
            self.awg_readout.channelFrequency(self.awg_readout_channel_i, if_frequency)
            self.awg_readout.channelFrequency(self.awg_readout_channel_q, if_frequency)
            self.awg_readout.modulationAmplitudeConfig(self.awg_readout_channel_i, key.SD_ModulationTypes.AOU_MOD_AM,
                                                       amplitude_i)
            self.awg_readout.modulationAmplitudeConfig(self.awg_readout_channel_q, key.SD_ModulationTypes.AOU_MOD_AM,
                                                       amplitude_q)
            self.awg_readout.channelPhase(self.awg_readout_channel_i, 0)
            self.awg_readout.channelPhase(self.awg_readout_channel_q, 90)
            self.awg_readout.channelPhaseResetMultiple(ALL_CHANNELS_MASK)

    def setup_dig_readout_channel(self, fullscale_i, fullscale_q):
        self.dig_readout.channelInputConfig(self.dig_readout_channel_i, fullscale_i, key.AIN_Impedance.AIN_IMPEDANCE_50,
                                            key.AIN_Coupling.AIN_COUPLING_DC)
        self.dig_readout.channelInputConfig(self.dig_readout_channel_q, fullscale_q, key.AIN_Impedance.AIN_IMPEDANCE_50,
                                            key.AIN_Coupling.AIN_COUPLING_DC)

    def __get_time_stamp(self):
        time_stamp = str(datetime.now())
        time_stamp = time_stamp[0:time_stamp.find('.')].replace(' ', '_').replace(':', '-')
        return time_stamp

    def save_raw_readout_data(self, filename_i, filename_q):
        print('Saving raw data to disk...')

        np.savetxt(filename_i, np.transpose(self.rawReadoutBufferI), delimiter=',')
        np.savetxt(filename_q, np.transpose(self.rawReadoutBufferQ), delimiter=',')

        print('Done')

    def plot_raw_readout_data(self):
        plt.clf()

        plt.plot(np.arange(len(self.rawReadoutBufferI)), self.rawReadoutBufferI)
        plt.plot(np.arange(len(self.rawReadoutBufferQ)), self.rawReadoutBufferQ)

        plt.show()

    def read_raw_data(self, timeout=100, max_read_attempts=10):
        """
        This method reads data from dig_readout if available. 
        :param timeout: The timeout in milliseconds for each read attempt.
        :param max_read_attempts: The max number of attempts at reading all expected data.
        """

        totalAcquiredPointsI = 0
        totalAcquiredPointsQ = 0

        current_read_attempt = 0

        while current_read_attempt < max_read_attempts:
            
            self.tempReadoutBufferI = self.dig_readout.DAQread(self.dig_readout_channel_i,
                    self.tempReadoutBufferIlength, timeout)
            
            acquiredPointsI = len(self.tempReadoutBufferI)
            
            self.tempReadoutBufferQ = self.dig_readout.DAQread(self.dig_readout_channel_q,
                    self.tempReadoutBufferQlength, timeout)

            acquiredPointsQ = len(self.tempReadoutBufferQ)

            for i in range(acquiredPointsI):
                self.rawReadoutBufferI.append(self.tempReadoutBufferI[i])

            for i in range(acquiredPointsQ):
                self.rawReadoutBufferQ.append(self.tempReadoutBufferQ[i])

            totalAcquiredPointsI += acquiredPointsI
            totalAcquiredPointsQ += acquiredPointsQ

            current_read_attempt += 1

            print(totalAcquiredPointsI, current_read_attempt, self.dig_readout.DAQcounterRead(1))
            if totalAcquiredPointsI == totalAcquiredPointsQ and totalAcquiredPointsI >= self.readoutPoints:
                self.acquisition_successful = True
                return

        self.acquisition_successful = False


    def setup_readout(self, n_readout_points, n_readout_points_per_cycle, n_readout_cycles, n_readout_points_per_read):
        triggerDelay = 0
        self.readoutPoints = n_readout_points
        self.readoutPointsPerRead = n_readout_points_per_read

        # raw data buffers
        self.readoutBufferIlength = n_readout_points
        self.readoutBufferQlength = n_readout_points

        self.tempReadoutBufferIlength = int(n_readout_points_per_read)
        self.tempReadoutBufferQlength = int(n_readout_points_per_read)

        self.dig_readout.DAQconfig(self.dig_readout_channel_i, n_readout_points_per_cycle, n_readout_cycles, triggerDelay,
                                   key.SD_TriggerModes.SWHVITRIG)
        self.dig_readout.DAQconfig(self.dig_readout_channel_q, n_readout_points_per_cycle, n_readout_cycles, triggerDelay,
                                   key.SD_TriggerModes.SWHVITRIG)

    def flush_awgs(self):
        self.awg_control.AWGqueueConfig(self.awg_control_channel_i, QUEUE_CONFIG_ONE_SHOT)
        self.awg_control.AWGqueueConfig(self.awg_control_channel_q, QUEUE_CONFIG_ONE_SHOT)
        self.awg_readout.AWGqueueConfig(self.awg_readout_channel_i, QUEUE_CONFIG_ONE_SHOT)
        self.awg_readout.AWGqueueConfig(self.awg_readout_channel_q, QUEUE_CONFIG_ONE_SHOT)

        self.awg_control.waveformFlush()
        self.awg_readout.waveformFlush()

        self.awg_control.AWGflush(self.awg_control_channel_i)
        self.awg_control.AWGflush(self.awg_control_channel_q)
        self.awg_readout.AWGflush(self.awg_readout_channel_i)
        self.awg_readout.AWGflush(self.awg_readout_channel_q)

    def start_awgs(self):
        self.awg_control.AWGstart(self.awg_control_channel_i)
        self.awg_control.AWGstart(self.awg_control_channel_q)
        self.awg_readout.AWGstart(self.awg_readout_channel_i)
        self.awg_readout.AWGstart(self.awg_readout_channel_q)

    def stop_awgs(self):
        self.awg_control.AWGstop(self.awg_control_channel_i)
        self.awg_control.AWGstop(self.awg_control_channel_q)
        self.awg_readout.AWGstop(self.awg_readout_channel_i)
        self.awg_readout.AWGstop(self.awg_readout_channel_q)

    def start_dig(self):
        self.dig_readout.DAQflush(self.dig_readout_channel_i)
        self.dig_readout.DAQflush(self.dig_readout_channel_q)
        self.dig_readout.DAQstart(self.dig_readout_channel_i)
        self.dig_readout.DAQstart(self.dig_readout_channel_q)

    def stop_dig(self):
        self.dig_readout.DAQstop(self.dig_readout_channel_i)
        self.dig_readout.DAQstop(self.dig_readout_channel_q)

    def start_awgs_dig(self):
        self.dig_readout.DAQflush(self.dig_readout_channel_i)
        self.dig_readout.DAQflush(self.dig_readout_channel_q)
        self.dig_readout.DAQstart(self.dig_readout_channel_i)
        self.dig_readout.DAQstart(self.dig_readout_channel_q)

        self.awg_control.AWGstart(self.awg_control_channel_i)
        self.awg_control.AWGstart(self.awg_control_channel_q)
        self.awg_readout.AWGstart(self.awg_readout_channel_i)
        self.awg_readout.AWGstart(self.awg_readout_channel_q)

    def stop_awgs_dig(self):
        self.dig_readout.DAQstop(self.dig_readout_channel_i)
        self.dig_readout.DAQstop(self.dig_readout_channel_q)
        self.awg_control.AWGstop(self.awg_control_channel_i)
        self.awg_control.AWGstop(self.awg_control_channel_q)
        self.awg_readout.AWGstop(self.awg_readout_channel_i)
        self.awg_readout.AWGstop(self.awg_readout_channel_q)


if __name__ == "__main__":
    pass

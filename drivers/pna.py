import instr
import numpy as np
from time import sleep

class Pna(instr.Instr):

    def __init__(self, ip_address):
        super(Pna, self).__init__(ip_address)
        self.cls()
        # self.set_current_channel_and_trace(1,1)

        # bla = self.ask("SYSTem:ACTive:CHANnel?")
        # self.current_channel = int(bla)
        # bla = self.ask("SYSTem:MEAS:CATalog? {0}".format(self.current_channel))
        # self.current_measurement_number = int(bla[1:-1])

        self.visa_instr.chunk_size = 32001*2*8
        self.visa_instr.read_termination = '\n'
        self.visa_instr.write_termination = '\n'
        self.visa_instr.send_end = True
        self.visa_instr.query_delay = 0

        self.current_channel = self.get_current_channel()
        self.current_measurement_number = None
        self.current_measurement_name = None

    def get_nb_errors(self):
        bla = self.query("SYSTem:ERRor:COUNt?")
        try:
            output = int(bla)
        except:
            print("Error in get_nb_errors: value returned: {0}".format(bla))
            output = -1
        return output

    def get_last_error(self):
        nb_errors = self.get_nb_errors()
        if nb_errors > 0:
            output = self.query("SYSTem:ERRor?")
        elif nb_errors == -1:
            output = "Could not figure out if an error message is available."
        else:
            output = "No error in queue."
        return output

    def get_all_errors(self):
        nb_errors = self.get_nb_errors()
        if nb_errors > 0:
            output = ""
            for n in range(nb_errors-1):
                output += self.query("SYSTem:ERRor?") + "\n"
            output += self.query("SYSTem:ERRor?")
        elif nb_errors == -1:
            output = "Could not figure out if an error message is available."
        else:
            output = "No error in queue."
        return output


    def set_data_format(self, data_format):
        self.write("FORMat {1}".format(self.current_channel, data_format))

    def get_current_channel(self):
        chlist = self.list_channels()
        if chlist != []:
            bla = self.ask("SYSTem:ACTive:CHANnel?")
            try:
                output = int(bla)
            except:
                print("Error in get_current_channel(): value returned: {0}".format(bla))
                output = -1
        else:
            output = None
        return output

    def get_current_TrX(self):
        # /!\ the "Trace" is actually the "Measurement number" as defined in
        # the manual of the PNA. It is the "X" in the "TrX" displayed on the
        # screen
        bla = self.ask("CALCulate{0}:PARameter:MNUMber?".format(self.current_channel))
        try:
            output = int(bla)
        except:
            print("Error in get_current_trace(): value returned: {0}".format(bla))
            output = -1
        # else:
        #     self.current_measurement_number = output
        return output

    def get_current_measurement_name(self):
        measurement_name = self.ask("CALCulate{0}:PARameter:SELect?".format(self.current_channel))
        return measurement_name[1:-1]   # removing the quotes surrounding the name

    def set_current_channel_and_trace(self, channel_number, measurement_name):
        self.write("CALCulate{0}:PARameter:SELect {1}".format(channel_number, measurement_name))
        self.current_channel = self.get_current_channel();
        # print self.current_channel
        self.current_measurement_name = self.get_current_measurement_name();
        # print self.current_measurement_name
        self.current_measurement_number = self.get_current_TrX();
        # print self.current_measurement_number
        if self.current_channel != channel_number:
            print("Problem with setting bias channel")
        if self.current_measurement_name != measurement_name.upper():
            print("Problem with setting bias measurement_name")
        return self.current_measurement_number

    # def set_current_trace(self, measurement_number):
    #     # /!\ the "Trace" is actually the "Measurement number" as defined in
    #     # the manual of the PNA. It is the "X" in the "TrX" displayed on the
    #     # screen
    #     self.write("CALCulate{0}:PARameter:MNUMber:SELect {1}".format(self.current_channel, measurement_number))
    #     self.current_measurement_number = self.get_current_trace();

    def set_current_trace_by_name(self, measurement_name):
        self.write("CALCulate{0}:PARameter:SELect {1}".format(self.current_channel, measurement_name))
        self.current_measurement_name = self.get_current_measurement_name();
        if self.current_measurement_name != measurement_name.upper():
            print("Problem with setting bias measurement_name")
        self.current_measurement_number = self.get_current_TrX();
        return self.current_measurement_number        

    def get_nb_points(self):
        return self.ask("SENSe{0}:SWEep:POINts?".format(self.current_channel))

    def get_frequencies(self):
        freqtext = self.ask("CALCulate{0}:X?".format(self.current_channel))
        return np.array([float(txt) for txt in freqtext.split(',')])

    def get_fdata(self):
        text = self.ask("CALCulate{0}:DATA? FDATA".format(self.current_channel))
        return np.array([float(txt) for txt in text.split(',')])

    def get_fmem(self):
        text = self.ask("CALCulate{0}:DATA? FMEM".format(self.current_channel))
        return np.array([float(txt) for txt in text.split(',')])

    def get_sdata(self):
        text = self.ask("CALCulate{0}:DATA? SDATA".format(self.current_channel))
        values_interlaced = np.array([float(txt) for txt in text.split(',')])
        return values_interlaced[0::2] + 1j*values_interlaced[1::2]

    def get_smem(self):
        text = self.ask("CALCulate{0}:DATA? SMEM".format(self.current_channel))
        values_interlaced = np.array([float(txt) for txt in text.split(',')])
        return values_interlaced[0::2] + 1j*values_interlaced[1::2]

    def smoothing(self, is_on):
        if is_on:
            self.write("CALCulate{0}:SMOothing ON".format(self.current_channel))
        else:
            self.write("CALCulate{0}:SMOothing OFF".format(self.current_channel))

    def set_smoothing_points(self, nb_points):
        self.write("CALCulate{0}:SMOothing:POINts {1}".format(self.current_channel, nb_points))

    def sweep_hold(self):
        self.write("SENSe{0}:SWEep:MODE HOLD".format(self.current_channel))

    def sweep_single(self):
        self.write("SENSe{0}:SWEep:MODE SINGle".format(self.current_channel))

    def average_off(self):
        self.write("SENSe{0}:AVERage:STATe OFF".format(self.current_channel))

    def average_restart(self):
        self.write("SENSe{0}:AVERage:STATe OFF".format(self.current_channel))
        self.write("SENSe{0}:AVERage:STATe ON".format(self.current_channel))

    def set_average(self, nb_averages):
        self.write("SENSe{0}:AVERage:COUNt {1}".format(self.current_channel, nb_averages))
        self.write("SENSe{0}:AVERage:STATe ON".format(self.current_channel))

    def delete_trace(self, channel_number, measurement_name):
        trace_list = self.list_traces(channel_number)
        if measurement_name.upper() in trace_list:
            self.write("CALCulate{0}:PARameter:DELete {1}".format(channel_number, measurement_name))
        else:
            print("Cannot delete trace: non-existent trace")

    def list_traces(self, channel_number):
        bla = self.ask("CALCulate{0}:PARameter:CATalog:EXTended?".format(channel_number))
        output = bla[1:-1].split(",")
        if output == "NO CATALOG":
            return []
        else:
            return output[0::2]

    def list_channels(self):
        bla = self.ask("SYSTem:CHANnels:CATalog?")
        blabla = [int(X) for X in bla[1:-1].split(",") if X != ""]
        # if blabla == []:
        #     return []
        # else:
        return blabla

    def create_channel_and_trace(self, channel_number, measurement_name, S_parameter, trace_number, window_number):
        used_channels = self.list_channels()
        if not(S_parameter in ["S21", "S11"]):
            print("ERROR: S_parameter for new channel should be S11 or S21 (or udpate the program !)")
            return None
        elif channel_number in used_channels:
            print("Channel already exists. Create another one or delete it first.")
            return None
        else:
            for chan in used_channels:
                used_measurement_names = self.list_traces(chan)
                if measurement_name.upper() in used_measurement_names:
                    print("Measurement name already exists. Choose another name or delete the measurement first.")
                    return None
            self.write("CALCulate{0}:PARameter:EXTended {1},{2}".format(channel_number, measurement_name, S_parameter))
            self.write("DISPlay:WINDow{0}:TRACe{1}:FEED {2}".format(window_number, trace_number, measurement_name))
            self.set_current_channel_and_trace(channel_number, measurement_name)
            self.sweep_hold()
            self.set_power(-60)
            self.set_power_off()
            self.write("SENSe{0}:SWEep:TIME:AUTO ON".format(channel_number))
            self.set_average(1)
            self.average_off()
            self.set_if_bw(1000)
            self.smoothing(False)
            self.set_freq_start_stop(2e9, 12e9)
            self.set_format("MLOGarithmic")
            return self.current_measurement_number

    def set_power(self, power):
        self.write("SOURce{0}:POWer1:LEVel {1}".format(self.current_channel, power))
        self.write("SOURce{0}:POWer1:MODE ON".format(self.current_channel))

    def set_power_off(self):
        self.write("SOURce{0}:POWer1:MODE OFF".format(self.current_channel))

    def set_nb_points(self, nb_points):
        self.write("SENSe{0}:SWEep:POINts {1}".format(self.current_channel, nb_points))

    def set_freq_start_stop(self, fstart, fstop):
        self.write("SENSe{0}:SWEep:TYPE LINear".format(self.current_channel))
        self.write("SENSe{0}:FREQuency:STARt {1}".format(self.current_channel, int(fstart)))
        self.write("SENSe{0}:FREQuency:STOP {1}".format(self.current_channel, int(fstop)))

    def set_freq_center_span(self, fcenter, fspan):
        self.write("SENSe{0}:SWEep:TYPE LINear".format(self.current_channel))
        self.write("SENSe{0}:FREQuency:CENTer {1}".format(self.current_channel, int(fcenter)))
        self.write("SENSe{0}:FREQuency:SPAN {1}".format(self.current_channel, int(fspan)))

    def set_if_bw(self, if_bw):
        self.write("SENSe{0}:BANDwidth {1}".format(self.current_channel, if_bw))

    def set_format(self, format):
        accepted = ["MLINear", "MLIN", "MLOGarithmic", "MLOG", "PHASe", "PHAS", "UPHase", "UPH", "IMAGinary", "IMAG", "REAL", "POLar", "POL", "SMITh", "SMIT", "SADMittance", "SADM", "SWR", "GDELay", "GDEL", "KELVin", "KELV", "FAHRenheit", "FAHR", "CELSius", "CELS"]
        accepted_lower_case = [x.lower() for x in accepted]
        if format.lower() in accepted_lower_case:
            self.write("CALCulate{0}:FORMat {1}".format(self.current_channel, format))
        else:
            print("Error: format must be MLINear, MLOGarithmic, PHASe, UPHase, IMAGinary, REAL, POLar, SMITh, SADMittance, SWR, GDELay, KELVin, FAHRenheit or CELSius.")

    def send_trigger(self):
        self.write("TRIGger:SCOPe CURRent")
        self.write("TRIGger:SOURce MANual")
        self.write("INITiate{0}:IMMediate".format(self.current_channel))
        # self.write("TRIGger:SOURce IMMediate")
        # self.write("SENSe{0}:SWEep:MODE SINGle".format(channel))

    def scale_auto(self, window_number, trace_number):
        self.write("DISPlay:WINDow{0}:TRACe{1}:Y:SCALe:AUTO".format(window_number, trace_number))

    def scale_auto_all(self, window_number):
        self.scale_couple(True)
        self.write("DISPlay:WINDow{0}:Y:AUTO".format(window_number))
        # self.write("DISPlay:WINDow{0}:TRACe:Y:SCALe:AUTO".format(window_number))

    def scale_couple(self, is_coupled):
        if is_coupled:
            self.write("DISPlay:WINDow:TRACe:Y:SCALe:COUPle:METHod WINDow")
        else:
            self.write("DISPlay:WINDow:TRACe:Y:SCALe:COUPle:METHod OFF")

    def get_ai(self, ai_channel):
        return float(self.ask("CONT:AUX:INPut{0}:VOLT?".format(ai_channel)))

    def get_ao(self, ao_channel):
        return float(self.ask("CONT:AUX:OUTput{0}:VOLT?".format(ao_channel)))

    # smoothly sets the output voltage on the Power I/O connector
    # sweep_time in seconds
    def set_ao(self, ao_channel, voltage, sweep_time):
        if (voltage <= 10) & (voltage >= -10):
            nb_steps = 50
            voltage_init = self.get_ao(ao_channel)
            for v in np.linspace(voltage_init, voltage, nb_steps):
                self.write("CONT:AUX:OUTput{0}:VOLT {1}".format(ao_channel, v))
                sleep(float(sweep_time) /nb_steps)
            vo = self.get_ao(ao_channel)
            vi = self.get_ai(ao_channel)
            if v != 0:
                print("V = {0:.3f}\t Vo = {1:.3f}\tVi = {2:.3f}\tdVi = {3:.2f}%\tdVo = {4:.2f}%\tdVio = {5:.2f}%".format(v, vo, vi, 100*(vi-v)/v, 100*(vo-v)/v, 100*(vi-vo)/vo))
            else:
                print("V = {0:.3f}\t Vo = {1:.3f}\tVi = {2:.3f}".format(v, vo, vi))
        else:
            print("Error: Analog output voltage must be in the range -10,+10")

    def gpib_bridge_open(self, device_gpib_address):
        self.write("SYST:COMM:GPIB:RDEV:OPEN 0,{0},100".format(device_gpib_address))
        bla = self.ask("SYST:COMM:GPIB:RDEV:OPEN?")
        try:
            output = int(bla)
        except:
            print("Error in initialize_gpib_pass_through(): value returned: {0}".format(bla))
            output = -1
        return output

    def gpib_bridge_write(self, handle, command):
        self.write("SYST:COMM:GPIB:RDEV:WRITE {0},'{1}'".format(handle, command))

    def gpib_bridge_read(self, handle):
        return self.ask("SYST:COMM:GPIB:RDEV:READ? {0}".format(handle))

    def gpib_bridge_ask(self, handle, command):
        self.gpib_bridge_write(handle, command)
        return self.gpib_bridge_read(handle)

    def gpib_bridge_close(self, handle):
        self.write("SYST:COMM:GPIB:RDEV:CLOSE {0}".format(handle))
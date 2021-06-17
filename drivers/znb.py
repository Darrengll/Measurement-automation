import drivers.instr as instr
import numpy as np


class Znb(instr.Instr):

    def __init__(self, visa_name):
        super(Znb, self).__init__(visa_name)
        self.cls()
        self._visainstrument.read_termination = '\n'
        self._visainstrument.timeout = 5000
        self.write("ROSCillator EXT")
        channel = self.list_channels()
        if channel:
            self.current_channel = channel[0]
            trace = self.list_traces(self.current_channel)
            if trace:
                self.current_measurement_name = trace[0]
            else:
                return False
        else:
            return False
        self.set_current_channel_and_trace(self.current_channel, self.current_measurement_name)
        self.set_data_format("ASCII")

    def get_parameters(self):
        """
        Returns a dictionary containing bandwidth, nop, power, averages and
        freq_limits currently used by the VNA
        """
        return {"bandwidth":self.get_bandwidth(),
                  "nop":self.get_nop(),
                  "power":self.get_power(),
                  "averages":self.get_averages(),
                  "freq_limits":self.get_freq_limits()}

    def set_parameters(self, parameters_dict):
        """
        Method allowing to set all or some of the VNA parameters at once
        (bandwidth, nop, power, averages, freq_limits and sweep type)
        """
        if "bandwidth" in parameters_dict.keys():
            self.set_bandwidth(parameters_dict["bandwidth"])
        if "averages" in parameters_dict.keys():
            self.set_averages(parameters_dict["averages"])
        if "power" in parameters_dict.keys():
            self.set_power(parameters_dict["power"])
        if "nop" in parameters_dict.keys():
            self.set_nop(parameters_dict["nop"])
        if "freq_limits" in parameters_dict.keys():
            self.set_freq_limits(*parameters_dict["freq_limits"])
        if "trigger_type" in parameters_dict.keys():
            self.set_trigger_type(parameters_dict["trigger_type"])

    def select_S_param(self, S_param, channel=1):
        """
        Set all active traces to measure S parameter specified in the argument.

        Parameters:
        -----------
        S_param: string
            name of the coefficient of the S matrix, i.e. "S21" or "S14"
        channel=1: int
            channel number, usually 1
        """
        for tracename in self.list_traces(channel):
            self.write("CALCulate{0}:PARameter:MEASure '{1}', '{2}'".format(1, tracename, S_param))

    def clear_error_queue(self):
        self.write("*CLS")

    def set_data_format(self, data_format):
        if not(data_format.upper() in ["ASCII", "REAL,32", "REAL,64", "REAL, 32", "REAL, 64"]):
            print("ERROR: data_format must be 'ASCII', 'REAL, 32' or 'REAL, 64'")
            return None
        else:
            self.write("FORMat {1}".format(self.current_channel, data_format))

    def set_power(self, power_dBm):
        self.write("SOURce{0}:POWer {1}".format(self.current_channel, power_dBm))
        # self.write("SOURce{0}:POWer:MODE ON".format(self.current_channel))

    def get_power(self):
        return float(self.query("SOURce%d:POWer?"%self.current_channel))

    def set_power_off(self):
        self.write("SOURce{0}:POWer:STATe OFF".format(self.current_channel))

    def set_power_on(self):
        self.write("SOURce{0}:POWer:STATe ON".format(self.current_channel))

    def get_nop(self):
        return int(self.query("SENSe{0}:SWEep:POINts?".format(self.current_channel)))

    def set_nop(self, nb_points):
        self.write("SENSe{0}:SWEep:POINts {1}".format(self.current_channel, nb_points))

    def set_average_mode(self, mode):
        """
        Sets the averaging mode for ZNB. See manual for available modes.
        """
        self.write("SENS:AVER:MODE "+str(mode))

    def set_averages(self, nb_averages):
        if nb_averages >= 1:
            # self.write("SENS:AVER:MODE FLATTEN")
            self.write("SENSe{0}:AVERage:COUNt {1}".format(self.current_channel, nb_averages))
            self.write("SENSe{0}:SWEep:COUNt {1}".format(self.current_channel, nb_averages))
            self.write("SENSe{0}:AVERage:STATe ON".format(self.current_channel))
        elif nb_averages == 1:
            self.average_off()
        else:
            print("ERROR in set_average: nb_averages should be >1 or =1 to turn off averaging.")

    def get_averages(self):
        return int(self.query("SENSe%d:SWEep:COUNt?"%self.current_channel))

    def avg_clear(self):
        self.write("SENSe{0}:AVERage:CLEar".format(self.current_channel))

    def average_off(self):
        self.write("SENSe{0}:AVERage:STATe OFF".format(self.current_channel))
        self.write("SENSe{0}:AVERage:COUNt 1".format(self.current_channel))
        self.write("SENSe{0}:SWEep:COUNt 1".format(self.current_channel))


    def smoothing(self, aperture):
        if (aperture >= 0.05) and (aperture <=100.00):
            self.write("CALCulate{0}:SMOothing:APERture {1}".format(self.current_channel, aperture))
            self.write("CALCulate{0}:SMOothing ON".format(self.current_channel))
        elif (aperture == 0) | (aperture == False):
            self.write("CALCulate{0}:SMOothing OFF".format(self.current_channel))
        # elif aperture == None:
        #     return self.query("CALCulate{0}:SMOothing?".format(self.current_channel))
        else:
            print("ERROR: error in function smoothing(). Accepted parameters: 0.05<=aperture<=100 or 0 or False to turn off")


    def set_bandwidth(self, if_bw):
        self.write("SENSe{0}:BANDwidth {1}".format(self.current_channel, if_bw))
        bla = self.query("SENSe{0}:BANDwidth?".format(self.current_channel))
        try:
            actual_bw = int(bla)
        except:
            print("ERROR in set_if_bw(): value returned by ZNB is not an integer.")
            actual_bw = -1
        return actual_bw

    def get_bandwidth(self):
        return int(self.query("SENSe{0}:BANDwidth?".format(self.current_channel)))

    def get_frequencies(self):
        freqtext = self.query("CALCulate{0}:DATA:STIMulus?".format(self.current_channel))
        return np.array([float(txt) for txt in freqtext.split(',')])

    def get_sdata(self):
        text = self.query("CALCulate{0}:DATA? SDATA".format(self.current_channel))
        values_interlaced = np.array([float(txt) for txt in text.split(',')])
        return values_interlaced[0::2] + 1j*values_interlaced[1::2]

    def get_fdata(self):
        text = self.query("CALCulate{0}:DATA? FDATA".format(self.current_channel))
        return np.array([float(txt) for txt in text.split(',')])


    def set_format(self, format):
        accepted = ["MLINear", "MLIN", "MLOGarithmic", "MLOG", "PHASe", "PHAS", "UPHase", "UPH", "IMAGinary", "IMAG", "REAL", "POLar", "POL", "SMITh", "SMIT", "ISMith", "ISM", "SWR", "GDELay", "GDEL", "COMPlex", "COMP", "MAGNitude", "MAGN"]
        accepted_lower_case = [x.lower() for x in accepted]
        if format.lower() in accepted_lower_case:
            self.write("CALCulate{0}:FORMat {1}".format(self.current_channel, format))
        else:
            print("Error: format must be 'MLINear', 'MLOGarithmic', 'PHASe', 'UPHase', 'IMAGinary', 'REAL', 'POLar', 'SMITh', 'ISMith', 'SWR', 'GDELay', 'COMPlex' or'MAGNitude'.")

    # def memorize(self):
    #     self.write("CALCulate{0}:MATH:MEMorize".format(self.current_channel))


    def delete_trace(self, channel_number, measurement_name):
        trace_list = self.list_traces(channel_number)
        if measurement_name in trace_list:
            self.write("CALCulate{0}:PARameter:DELete '{1}'".format(channel_number, measurement_name))
        else:
            print("Cannot delete trace: non-existent trace")

    def delete_all_traces(self, channel_number):
        if channel_number in self.list_channels():
            self.write("CALCulate{0}:PARameter:DELete:CALL".format(channel_number))
        else:
            print("Cannot delete traces: non-existent channel")

    def delete_all_memory(self, channel_number):
        if channel_number in self.list_channels():
            self.write("CALCulate{0}:PARameter:DELete:CMEMory".format(channel_number))
        else:
            print("Cannot delete traces: non-existent channel")

    def delete_really_all_traces(self):
        self.write("CALCulate:PARameter:DELete:ALL")

    def delete_really_all_memory(self):
        self.write("CALCulate:PARameter:DELete:MEMory")

    def set_freq_limits(self, start, stop):
        self.set_xlim(start, stop)

    def get_freq_limits(self):
        start = float(self.query("SENSe{0}:FREQuency:STARt?".format(self.current_channel)))
        stop = float(self.query("SENSe{0}:FREQuency:STOP?".format(self.current_channel)))
        return start, stop

    def set_xlim(self, fstart, fstop):
        self.write("SENSe{0}:SWEep:TYPE LINear".format(self.current_channel))
        self.write("SENSe{0}:FREQuency:STARt {1}".format(self.current_channel, int(fstart)))
        self.write("SENSe{0}:FREQuency:STOP {1}".format(self.current_channel, int(fstop)))

    def set_freq_center_span(self, fcenter, fspan):
        self.write("SENSe{0}:SWEep:TYPE LINear".format(self.current_channel))
        self.write("SENSe{0}:FREQuency:CENTer {1}".format(self.current_channel, int(fcenter)))
        self.write("SENSe{0}:FREQuency:SPAN {1}".format(self.current_channel, int(fspan)))

    def set_span(self, span):
        self.write("SENSe{0}:FREQuency:SPAN {1}".format(self.current_channel, int(span)))

    def set_center(self, center):
        self.write("SENSe{0}:FREQuency:CENTer {1}".format(self.current_channel, int(center)))

    def get_center(self):
        return self.query("SENSe{0}:FREQuency:CENTer?".format(self.current_channel))

    # /!\ DOES NOT KEEP LAST AVERAGING !!! HOLDS WITH ONLY THE LAST SINGLE SWEEP
    def sweep_hold(self):
        # self.write("SENSe{0}:SWEep:MODE HOLD".format(self.current_channel))
        self.write("INITiate{0}:CONTinuous OFF".format(self.current_channel))

    # TO CHECK
    def sweep_single(self):
        # self.write("SENSe{0}:SWEep:MODE SINGle".format(self.current_channel))
        self.write("INITiate{0}:IMMediate".format(self.current_channel))

    def sweep_continuous(self):
        # self.write("SENSe{0}:SWEep:MODE HOLD".format(self.current_channel))
        self.write("INITiate{0}:CONTinuous ON".format(self.current_channel))

    def set_trigger_type(self, trigger_type):
        if trigger_type == "single":
            self.sweep_single()
        elif trigger_type == "continuous":
            self.sweep_continuous()
        else:
            raise ValueError("Sweep type %s not supported"%sweep_type)

    def set_trigger_manual(self):
        self.write("TRIGger{0}:SOURce MANual".format(self.current_channel))
        self.write("INITiate{0}:CONTinuous OFF".format(self.current_channel))

    def send_trigger(self):
        # self.write("TRIGger:SCOPe CURRent")
        self.write("TRIGger:SOURce IMMediate")
        self.write("INITiate{0}:IMMediate".format(self.current_channel))
        # self.write("SENSe{0}:SWEep:MODE SINGle".format(channel))

    def free_run(self):
        self.write("INITiate{0}:CONTinuous ON".format(self.current_channel))
        # self.write("TRIGger{0}:SOURce IMMediate".format(self.current_channel))


    def create_channel_and_trace(self, channel_number, measurement_name, S_parameter, window_number):
        used_channels = self.list_channels()
        if not(S_parameter in ["S12", "S11"]):
            print("ERROR: S_parameter for new channel should be S11 or S12 (or udpate the program !)")
            return None
        elif channel_number in used_channels:
            print("Channel already exists. Create another one or delete it first.")
            return None
        else:
            for chan in used_channels:
                used_measurement_names = self.list_traces(chan)
                if measurement_name in used_measurement_names:
                    print("Measurement name already exists. Choose another name or delete the measurement first.")
                    return None
                else:
                    self.write("CALCulate{0}:PARameter:SDEFine '{1}', '{2}'".format(channel_number, measurement_name, S_parameter))
                    self.set_current_channel_and_trace(channel_number, measurement_name)
                    self.sweep_hold()
                    self.set_power(-60)
                    self.set_power_off()
                    self.write("SENSe{0}:SWEep:TIME:AUTO ON".format(self.current_channel))
                    self.set_average(1)
                    self.average_off()
                    self.set_if_bw(1000)
                    self.smoothing(False)
                    self.set_freq_start_stop(2e9, 12e9)
                    self.set_format("MLOGarithmic")
                    self.write("DISPlay:WINDow{0}:STATe ON".format(window_number))
                    self.write("DISPlay:WINDow{0}:TRACe{1}:FEED '{2}'".format(window_number, self.current_trace_number, self.current_measurement_name))
                    return True

    def set_current_channel_and_trace(self, channel_number, measurement_name):
        used_channels = self.list_channels()
        if channel_number not in used_channels:
            print("Channel doesn't exist. Create it first.")
            return None
        else:
            used_measurement_names = self.list_traces(channel_number)
            if measurement_name not in used_measurement_names:
                print("Trace doesn't exists. Create it first.")
                return None
            else:
                self.write("CALCulate{0}:PARameter:SELect '{1}'".format(channel_number, measurement_name))
                self.current_channel = channel_number
                self.current_measurement_name = measurement_name
                self.current_trace_number = self.get_trace_number_from_trace_name(self.current_measurement_name)
                return True

    def delete_channel(self, channel_number):
        self.write("CONFigure:CHANnel{0} OFF".format(channel_number))

    def delete_trace_by_window(self, window_number, measurement_name):
        trace_number = self.get_trace_number_from_trace_name(measurement_name)
        self.write("DISPlay:WINDow{0}:TRACE{1}:DELete".format(window_number, trace_number))

    def list_channels(self):
        bla = self.query("CONFigure:CHANnel:CATalog?")
        blabla = [int(X) for X in bla[1:-1].split(",")[0::2] if X != ""]
        # if blabla == []:
        #     return []
        # else:
        return blabla

    def list_traces(self, channel_number):
        bla = self.query("CONFigure:CHANnel{0}:TRACe:CATalog?".format(channel_number))
        blabla = bla[1:-1]
        if (blabla == "NO CATALOG") or (blabla == ""):
            return []
        else:
            # trace_numbers = [int(X) for X in blabla.split(",")[0::2] if X != ""]
            trace_names   = [X for X in blabla.split(",")[1::2] if X != ""]
            # return trace_numbers, trace_names
            return trace_names

    def get_trace_number_from_trace_name(self, trace_name):
        bla = self.query("CONFigure:TRACe:NAME:ID? '{0}'".format(trace_name))
        try:
            n = int(bla)
        except:
            print("ERROR in get_trace_number_from_trace_name(): not an integer.")
            n = None
        return n

    # TO CHECK
    def scale_auto(self, window_number, trace_number):
        self.write("DISPlay:WINDow{0}:TRACe{1}:Y:SCALe:AUTO ONCE".format(window_number, trace_number))

    def autoscale_all(self):
        for i in range(1, 4):
            self.write("DISPlay:WINDow:TRACe:Y:SCALe:AUTO ONCE, '{0}'".format("Trc"+str(i)))

    def scale_auto_by_trace_name(self, trace_name):
        self.write("DISPlay:WINDow:TRACe:Y:SCALe:AUTO ONCE, '{0}'".format(trace_name))

    # DOESNT WORK
    def screenshot(self, base_filename, filetype):
        if filetype == None:
            filetype = "PNG"
        elif filetype.upper() not in ["BMP", "JPG", "PNG", "PDF", "SVG"]:
            print('ERROR: filetype should be "BMP", "JPG", PNG", "PDF" or "SVG".')
            return None
        self.write("MMEMory:NAME {0}.{1}".format(base_filename,filetype.lower()))
        self.write("HCOPy:DEVice:LANGuage {0}".format(filetype))
        self.write("HCOPy:ITEM:MLISt ON")
        self.write("HCOPy:ITEM:LOGO OFF")
        self.write("HCOPy:ITEM:TIME ON")
        self.write("HCOPy:PAGE:COLor ON")
        self.write("HCOPy:PAGE:WINDow ACTive")
        self.write("HCOPy:DESTination 'MMEMory'")
        self.write("HCOPy")
        print("Screenshot saved to {0}.{1}".format(base_filename,filetype.lower()))

    def get_errors(self):
        output = self.query("SYSTem:ERRor:ALL?")
        return output

    def set_electrical_delay(self, delay):
        self.write("SENSe{0}:CORRection:EDELay2:TIME {1}".format(self.current_channel, delay))

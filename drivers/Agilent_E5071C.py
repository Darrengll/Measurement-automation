
from numpy import array
from drivers.Agilent_PNA_L import *
from numpy import sqrt

class Agilent_E5071C(Agilent_PNA_L):

    def __init__(self, address, channel_index = 1):
        """
        Initializes

        Input:
            name (string)    : name of the instrument
            address (string) : GPIB address
        """

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.WARNING)

        Instrument.__init__(self, "", tags=['physical'])

        self._address = address
        rm = visa.ResourceManager()
        self._visainstrument = rm.open_resource(self._address)# no term_chars for GPIB!!!!!
        self._zerospan = False
        self._freqpoints = 0
        self._ci = channel_index
        self._start = 0
        self._stop = 0
        self._nop = 0

        # Implement parameters

        self.add_parameter('nop', type=int,
            flags=Instrument.FLAG_GETSET,
            minval=1, maxval=100000,
            tags=['sweep'])

        self.add_parameter('bandwidth', type=float,
            flags=Instrument.FLAG_GETSET,
            minval=0, maxval=1e9,
            units='Hz', tags=['sweep'])

        self.add_parameter('averages', type=int,
            flags=Instrument.FLAG_GETSET,
            minval=1, maxval=1024, tags=['sweep'])

        self.add_parameter('average', type=bool,
            flags=Instrument.FLAG_GETSET)

        self.add_parameter('centerfreq', type=float,
            flags=Instrument.FLAG_GETSET,
            minval=0, maxval=20e9,
            units='Hz', tags=['sweep'])

        self.add_parameter('center', type=float,
            flags=Instrument.FLAG_GETSET,
            minval=0, maxval=20e9,
            units='Hz', tags=['sweep'])

        self.add_parameter('startfreq', type=float,
            flags=Instrument.FLAG_GETSET,
            minval=0, maxval=20e9,
            units='Hz', tags=['sweep'])

        self.add_parameter('stopfreq', type=float,
            flags=Instrument.FLAG_GETSET,
            minval=0, maxval=20e9,
            units='Hz', tags=['sweep'])

        self.add_parameter('CWfreq', type=float,
            flags=Instrument.FLAG_GETSET,
            minval=300e3, maxval=20e9,
            units='Hz', tags=['sweep'])

        self.add_parameter('span', type=float,
            flags=Instrument.FLAG_GETSET,
            minval=0, maxval=20e9,
            units='Hz', tags=['sweep'])

        self.add_parameter('power', type=float,
            flags=Instrument.FLAG_GETSET,
            minval=-90, maxval=12,
            units='dBm', tags=['sweep'])

        self.add_parameter('zerospan', type=bool,
            flags=Instrument.FLAG_GETSET)

        self.add_parameter('channel_index', type=int,
            flags=Instrument.FLAG_GETSET)

        #Triggering Stuff
        self.add_parameter('trigger_source', type=bytes,
            flags=Instrument.FLAG_GETSET)

        # output trigger stuff by Elena
        self.add_parameter('aux_num', type=int,
                           flags=Instrument.FLAG_GETSET)

        self.add_parameter('trig_per_point', type=bool,
                           flags=Instrument.FLAG_GETSET)

        self.add_parameter('pos', type=bool,
                           flags=Instrument.FLAG_GETSET)

        self.add_parameter('bef', type=bool,
                           flags=Instrument.FLAG_GETSET)

        self.add_parameter('trig_dur', type=float,
                           flags=Instrument.FLAG_GETSET,
                           minval=2e-3, units='s')


        # sets the S21 setting in the PNA X
        # self.define_S21() # this two lines is uncommented by Shamil 06/26/2017 due to the fact that
        # self.set_S21()  # by using high level measurement child classes it is not possible to continue proper operation
                        # of PNA-L after self._visaintrument.write( "SYST:FPReset" ) command, it seem like without this
                        # lines of code there is no trace selected after self.select_default_trace()
                        # and self.get_all seem do interrupt the program with timeout exception thrown by low-level visa
                        # GPIB drivers. The reason is that PNA-L doesn't have any number of points in sweep (get_all start
                        # by quering number of points in current sweep), because there is no traces defined, hence there
                        # is no number of points available to read
        # self.select_default_trace()


        # Implement functions
        self.add_function('get_frequencies')
        self.add_function("get_freqpoints")
        self.add_function('get_tracedata')
        self.add_function('get_sdata')
        self.add_function('init')
        self.add_function('set_S21')
        self.add_function('set_xlim')
        self.add_function('get_xlim')
        self.add_function('get_sweep_time')
        self.add_function('sweep_single')
        self.add_function("prepare_for_stb")
        self.add_function('wait_for_stb')
        self.add_function('set_electrical_delay')
        self.add_function('get_electrical_delay')
        self.add_function('sweep_hold')
        self.add_function('sweep_continuous')
        self.add_function('autoscale_all')

        #self.add_function('avg_clear')
        #self.add_function('avg_status')

        #self._oldspan = self.get_span()
        #self._oldnop = self.get_nop()
        #if self._oldspan==0.002:
        #  self.set_zerospan(True)

        self.get_all()

    def get_sdata(self):
        self._visainstrument.write("FORM:DATA REAL32;")

        self._visainstrument.write("FORM:BORD SWAP")

        values = self._visainstrument.query_binary_values("CALC1:DATA:SDAT?")
        return array(values)[0:-1:2] + 1j*array(values)[1::2]

    def set_parameters(self, parameters_dict):
        # """
        # Method allowing to set all or some of the VNA parameters at once
        # (bandwidth, nop, power, averages and freq_limits)
        # """
        # if "bandwidth" in parameters_dict.keys():
        #     self.set_bandwidth(parameters_dict["bandwidth"])
        # if "averages" in parameters_dict.keys():
        #     self.set_averages(parameters_dict["averages"])
        # if "power" in parameters_dict.keys():
        #     self.set_power(parameters_dict["power"])
        # if "nop" in parameters_dict.keys():
        #     self.set_nop(parameters_dict["nop"])
        # if "freq_limits" in parameters_dict.keys():
        #     self.set_freq_limits(*parameters_dict["freq_limits"])
        # if "span" in parameters_dict.keys():
        #     self.set_span(parameters_dict["span"])
        # if "centerfreq" in parameters_dict.keys():
        #     self.set_centerfreq(parameters_dict["centerfreq"])
        # if "sweep_type" in parameters_dict.keys():
        #     self.set_sweep_type(parameters_dict["sweep_type"])
        # if "trig_source" in parameters_dict.keys():
        #     self.set_trigger_source(parameters_dict["trig_source"])
        #
        # if "aux_num" in parameters_dict.keys():
        #     self.set_aux_num(parameters_dict["aux_num"])
        # if "stepped_triggered_sweep" in parameters_dict.keys():
        #     self.set_stepped_triggered_sweep(parameters_dict["stepped_triggered_sweep"])
        # if "pos" in parameters_dict.keys():
        #     self.set_pos(parameters_dict["pos"])
        # if "bef" in parameters_dict.keys():
        #     self.set_bef(parameters_dict["bef"])
        # if "trig_dur" in parameters_dict.keys():
        #     self.set_trig_dur(parameters_dict["trig_dur"])
        if "stepped_triggered_sweep" in parameters_dict.keys():
            self.set_stepped_triggered_sweep(parameters_dict["stepped_triggered_sweep"])
            parameters_dict.pop("stepped_triggered_sweep")
        if "freq_limits" in parameters_dict.keys():
            self.set_freq_limits(*parameters_dict["freq_limits"])
            parameters_dict.pop("freq_limits")

        super().set_parameters(parameters_dict)


    def sweep_single(self):
        self.write(":INIT1:IMM")

    def sweep_continuous(self):
        self._visainstrument.write(":INIT1:CONT ON")

    def sweep_hold(self):
        self._visainstrument.write(":INIT1:CONT OFF")

    def reset_sweep(self):
        self._visainstrument.write(":ABOR")

    def autoscale_all(self):
        pass

    def set_stepped_triggered_sweep(self, stepped_triggered_sweep):
        if stepped_triggered_sweep is True:
            self._visainstrument.write(":TRIG:SEQ:POINT ON")
        elif stepped_triggered_sweep is False:
            self._visainstrument.write(":TRIG:SEQ:POINT OFF")
        else:
            raise ValueError("stepped_triggered_sweep can be either True or False")

    def prepare_for_stb(self):
        self._visainstrument.write(":STAT:OPER:PTR 0")
        self._visainstrument.write(":STAT:OPER:NTR 16")
        self._visainstrument.write(":STAT:OPER:ENAB 16")
        self._visainstrument.write("*SRE 128")
        self._visainstrument.write("*CLS")


    def do_set_trigger_source(self, source):
        if source.upper()[:3] in ["INT", "MAN", "EXT", "BUS"]:
            self._visainstrument.write('TRIG:SEQ:SOUR %s' % source.upper())
        else:
            raise ValueError('set_trigger_source(): must be INTernal | MANual | EXTernal | BUS')

    def wait_for_stb(self):
        while True:
            bla = self._visainstrument.query("*STB?")
            try:
                stb_value = int(bla)
                if stb_value == 192:
                    break
            except:
                print("Error in wait(): value returned: {0}".format(bla))
            else:
                sleep(0.001)



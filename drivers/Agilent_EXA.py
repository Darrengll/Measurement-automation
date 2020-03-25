from drivers.instrument import Instrument
from numpy import *
import numpy
import visa
import types
import logging
from time import sleep

class Agilent_EXA_N9010A(Instrument):
    """
    This is the python driver for the Agilent EXA N9010A Signal Analyzer

    Usage:
    Initialise with
    <name> = instruments.create('<name>', address='<GPIB address>', reset=<bool>)

    """

    def __init__(self, address, channel_index = 1):
        """
        Initializes

        Input:
            address (string) : VISA address
        """

        Instrument.__init__(self, "EXA", tags=['physical'])

        self._address = address
        rm = visa.ResourceManager()
        self._visainstrument = rm.open_resource(self._address, timeout=10000)# no term_chars for GPIB!!!!!
        self._freqpoints = 0
        self._zerospan=False
        self._list_sweep = False
        self._ci = channel_index
        self._start = 0
        self._stop = 0
        self._nop = 0

        self._visainstrument.timeout = None

        # Implement parameters

        self.add_parameter('nop', type=int,
            flags=Instrument.FLAG_GETSET,
            minval=1, maxval=100000,
            tags=['sweep'])

        self.add_parameter('bandwidth', type=float,
            flags=Instrument.FLAG_GETSET,
            minval=1, maxval=1e9,
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

        self.add_parameter('startfreq', type=float,
            flags=Instrument.FLAG_GETSET,
            minval=0, maxval=20e9,
            units='Hz', tags=['sweep'])

        self.add_parameter('stopfreq', type=float,
            flags=Instrument.FLAG_GETSET,
            minval=0, maxval=20e9,
            units='Hz', tags=['sweep'])

        self.add_parameter('span', type=float,
            flags=Instrument.FLAG_GETSET,
            minval=0, maxval=20e9,
            units='Hz', tags=['sweep'])


        self.add_parameter('zerospan', type=bool,
            flags=Instrument.FLAG_GETSET)

        self.add_parameter('channel_index', type=int,
            flags=Instrument.FLAG_GETSET)

        #Triggering Stuff
        self.add_parameter('trigger_source', type=bytes,
            flags=Instrument.FLAG_GETSET)

        # sets the S21 setting in the PNA X

        # Implement functions
        self.add_function('get_freqpoints')
        self.add_function('get_tracedata')
        self.add_function('init')
        self.add_function('set_xlim')
        self.add_function('get_xlim')
        self.add_function('get_sweep_time')
        self.add_function('sweep_single')
        self.add_function("prepare_for_stb")
        self.add_function('wait_for_stb')
        self.add_function('set_electrical_delay')
        self.add_function("setup_list_sweep")
        self.add_function("set_parameters")
        #self.add_function('avg_clear')
        #self.add_function('avg_status')

        #self._oldspan = self.get_span()
        #self._oldnop = self.get_nop()
        #if self._oldspan==0.002:
        #  self.set_zerospan(True)

        self.get_all()
        #self.setup_swept_sa()


    def get_all(self):
        self.get_nop()
        self.get_centerfreq()
        self.get_startfreq()
        self.get_stopfreq()
        self.get_span()
        self.get_bandwidth()
        self.get_trigger_source()
        self.get_average()
        self.get_averages()
        self.get_freqpoints()
        self.get_channel_index()
        #self.get_zerospan()


    ###
    #Communication with device
    ###

    def init(self):
        if self._zerospan:
          self._visainstrument.write('INIT1')
        else:
          if self.get_average():
            for i in range(self.get_averages()):
              self._visainstrument.write('INIT1')
          else:
              self._visainstrument.write('INIT1')

    def get_parameters(self):
        """
        Returns a dictionary containing bandwidth, nop, power, averages and
        freq_limits currently used by the EXA
        """
        return {"bandwidth":self.get_bandwidth(),
                  "nop":self.get_nop(),
                  "centerfreq":self.get_centerfreq(),
                  "span":self.get_span(),
                  "avs":self.get_averages(),
                  "av_status":self.get_average()}

    def set_parameters(self, pars_dict):
        """
        Method allowing to set all of the EXA parameters at once (bandwidth, nop,
        centerfreq, spanfreq, averages and averaging status)
        """
        keys = pars_dict.keys()
        if "bandwidth" in keys: self.set_bandwidth(pars_dict["bandwidth"])
        if "nop" in keys: self.set_nop(pars_dict["nop"])
        if "centerfreq" in keys: self.set_centerfreq(pars_dict["centerfreq"])
        if "span" in keys: self.set_span(pars_dict["span"])
        if "averages" in keys: self.set_averages(pars_dict["averages"])
        if "avg_status" in keys: self.set_average(pars_dict["avg_status"])


    def reset_windows(self):
        self._visainstrument.write('DISP:WIND Off')
        self._visainstrument.write('DISP:WIND On')

    def set_autoscale(self):
        self._visainstrument.write("DISP:WIND:TRAC:Y:AUTO")

    def set_continuous(self,ON=True):
        if ON:
            self._visainstrument.write( "INITiate:CONTinuous ON")
        else:
            self._visainstrument.write( "INITiate:CONTinuous Off")

    def get_sweep(self):
        self._visainstrument.write( "ABORT; INITiate:IMMediate;*wai")

    def avg_clear(self):
        self._visainstrument.write(':SENS%i:AVER:CLE' %(self._ci))

    def avg_status(self):
        # this does not work the same way than the VNA:
        #return int(self._visainstrument.query(':SENS%i:AVER:COUN?' %(self._ci))
        pass

    def get_avg_status(self):
        return self._visainstrument.query('STAT:OPER:AVER1:COND?')

    def still_avg(self):
        if int(self.get_avg_status()) == 1: return True
        else: return False

    def get_tracedata(self):
        """
        Get the data of the current trace

        Output: (Frequency_List, Signal_List)

        """
        self._visainstrument.write(":FORMat:DATA REAL,32")
        self._visainstrument.write(":FORMat:BORDer SWAP")
        data = self._visainstrument.query_binary_values(":CALC:DATA1?")

        if self._list_sweep:
            return data

        data_size = numpy.size(data)
        freq = numpy.array(data[0:data_size:2])
        signal = numpy.array(data[1:data_size:2])

        return signal


    def get_freqpoints(self, query = False):
      self._freqpoints = numpy.linspace(self._start,self._stop,self._nop)
      return self._freqpoints

    def set_electrical_delay(self, delay):
        self._visainstrument.write("CALC{0}:CORRection:EDELay:TIME {1}".format(self._ci, delay))

    def set_xlim(self, start, stop):
        self._visainstrument.write('SENS%i:FREQ:STAR %f' % (self._ci,start))
        self._start = start
        self.get_centerfreq();
        self.get_stopfreq();
        self.get_span();

        self._visainstrument.write('SENS%i:FREQ:STOP %f' % (self._ci,stop))
        self._stop = stop
        self.get_startfreq();
        self.get_centerfreq();
        self.get_span();

    def get_xlim(self):
        return self._start, self._stop

    def get_sweep_time(self):
        """
        Get the time needed for one sweep

        Returns:
            out: float
                time in ms
        """
        return float(self._visainstrument.query(':SENS%i:SWE:TIME?' %(self._ci)))*1e3
    ###
    # SET and GET functions
    ###

    def do_set_nop(self, nop):
        """
        Set Number of Points (nop) for sweep

        Input:
            nop (int) : Number of Points

        Output:
            None
        """
        self._visainstrument.write(':SENS%i:SWE:POIN %i' %(self._ci,nop))
        self._nop = nop
        self.get_freqpoints() #Update List of frequency points

    def do_get_nop(self):
        """
        Get Number of Points (nop) for sweep

        Input:
            None
        Output:
            nop (int)
        """
        if self._zerospan:
          return 1
        else:
            self._nop = int(self._visainstrument.query(':SENS%i:SWE:POIN?' %(self._ci)))
        return self._nop

    def do_set_average(self, status):
        """
        Set status of Average

        Input:
            status (boolean) : True or False of 1 or 0

        Output:
            None
        """
        if status:
            self._visainstrument.write(':AVERage:STAT 1')
        elif status == False:
            self._visainstrument.write(':AVERage:STAT 0')

    def do_get_average(self):
        """
        Get status of Average

        Input:
            None

        Output:
            Status of Averaging ('on' or 'off) (string)
        """
        return bool(int(self._visainstrument.query(":AVERage:STAT?")))

    def do_set_averages(self, av):
        """
        Set number of averages

        Input:
            av (int) : Number of averages

        Output:
            None
        """
        self._visainstrument.write(":AVERage:COUNt  %d"%av)
        if av > 1:
            self.do_set_average(True)
            # self._visainstrument.write('SENS:SWE:GRO:COUN %i'%av)
        else:
            self.do_set_average(False)
            # self._visainstrument.write('SENS:SWE:GRO:COUN 1')

    def do_get_averages(self):
        """
        Get number of averages

        Input:
            None
        Output:
            number of averages
        """
        return self._visainstrument.query(":OBWidth:AVERage:COUNt?")

    def do_set_centerfreq(self, cf):
        """
        Set the center frequency

        Input:
            cf (float) :Center Frequency in Hz

        Output:
            None
        """
        self._visainstrument.write('SENS%i:FREQ:CENT %f' % (self._ci,cf))
        self.get_startfreq();
        self.get_stopfreq();
        self.get_span();

    def do_get_centerfreq(self):
        """
        Get the center frequency

        Input:
            None

        Output:
            cf (float) :Center Frequency in Hz
        """
        return  float(self._visainstrument.query('SENS%i:FREQ:CENT?'%(self._ci)))

    def do_set_span(self,span):
        """
        Set Span

        Input:
            span (float) : Span in KHz

        Output:
            None
        """
        self._visainstrument.write('SENS%i:FREQ:SPAN %i' % (self._ci,span))
        if span==0:
            self._zerospan=True
            self._visainstrument.write("SENS:SWE:TIME 1 us")
        else:
            self._zerospan=False
        self.get_startfreq();
        self.get_stopfreq();
        self.get_centerfreq();


    def do_get_span(self):
        """
        Get Span

        Input:
            None

        Output:
            span (float) : Span in Hz
        """
        span = self._visainstrument.query('SENS%i:FREQ:SPAN?' % (self._ci) ) #float( self.query('SENS1:FREQ:SPAN?'))
        return span

    def setup_list_sweep(self, frequency_list, rbw_list):
        """
        Setup the EXA for the list sweep measurement. See manual for details.

        Parameters:
        -----------
        frequency_list: array-like
            List of the center frequencies for which the EXA will record data
        rbw_list: array-like
            list of the resolution bandwidths to be used for corresponding frequencies
        """
        self._visainstrument.write(":CONFigure:LIST")
        self._list_sweep = True

        freqs_str = "".join(["%f,"%freq for freq in frequency_list])
        self._visainstrument.write(":LIST:FREQ "+freqs_str[:-1])

        rbws_str = "".join(["%f,"%rbw for rbw in rbw_list])
        self._visainstrument.write(":LIST:BAND:RES "+rbws_str[:-1])

        sweep_times = "".join(["%f,"%swt for swt in ones_like(frequency_list)/1e3])
        self._visainstrument.write(":LIST:SWEep:TIME "+sweep_times[:-1])

    def setup_swept_sa(self, center_freq=5e9, span=1e9, nop=1001, rbw=1e6):
        """
        Setup the EXA for the standard swept measurement. See manual for details.

        Parameters:
        -----------
        center_freq: float
            central frequency of the sweep
        span: float
            frequency span around the center_freq
        nop: int
            number of points for the sweep
        rbw: float
            resolution bandwidth for each point
        """
        self._visainstrument.write(":CONFigure:SAN")
        self._visainstrument.write(":DET:trace1 POS")
        self._list_sweep = False
        self.do_set_centerfreq(center_freq)
        self.do_set_span(span)
        self.do_set_nop(nop)
        self.do_set_bandwidth(rbw)

    def sweep_single(self):
        """
        Sweep single
        """
        self.init()
        #self.write("SENSe{0}:SWEep:MODE SINGle".format(self.current_channel))
        #self.write("SENS%i:SWE:MODE SING"%(self._ci))

    def get_averaging_setup(self):
        return self._visainstrument.query("CHP:AVER:TCON?")[:-1]

    def set_averaging_setup(self, av_type='REP'):
        """
        types are "EXP" - continues after specified number (WTF)
                  "REP" - resets average counter
        """
        return self._visainstrument.write("CHP:AVER:TCON %s"%av_type)

    def set_singlesweep_mode(self):
        return self._visainstrument.write("INIT:CONT 0")

    def set_contsweep_mode(self):
        return self._visainstrument.write("INIT:CONT 1")

    def do_set_startfreq(self,val):
        """
        Set Start frequency

        Input:
            span (float) : Frequency in Hz

        Output:
            None
        """
        self._visainstrument.write('SENS%i:FREQ:STAR %f' % (self._ci,val))
        self._start = val
        self.get_centerfreq();
        self.get_stopfreq();
        self.get_span();

    def do_get_startfreq(self):
        """
        Get Start frequency

        Input:
            None

        Output:
            span (float) : Start Frequency in Hz
        """
        self._start = float(self._visainstrument.query('SENS%i:FREQ:STAR?' % (self._ci)))
        return  self._start

    def do_set_stopfreq(self,val):
        """
        Set STop frequency

        Input:
            val (float) : Stop Frequency in Hz

        Output:
            None
        """
        self._visainstrument.write('SENS%i:FREQ:STOP %f' % (self._ci,val))
        self._stop = val
        self.get_startfreq();
        self.get_centerfreq();
        self.get_span();

    def do_get_stopfreq(self):
        """
        Get Stop frequency

        Input:
            None

        Output:
            val (float) : Start Frequency in Hz
        """
        self._stop = float(self._visainstrument.query('SENS%i:FREQ:STOP?' %(self._ci) ))
        return  self._stop

    def do_set_bandwidth(self,band):
        """
        Set Bandwidth

        Input:
            band (float) : Bandwidth in Hz

        Output:
            None
        """
        self._visainstrument.write('SENS%i:BWID:RES %i' % (self._ci,band))

    def do_get_bandwidth(self):
        """
        Get Bandwidth

        Input:
            None

        Output:
            band (float) : Bandwidth in Hz
        """
        # getting value from instrument
        return  float(self._visainstrument.query('SENS%i:BWID:RES?'%self._ci))

    def do_set_zerospan(self,val):
        self._zerospan=val

    def do_get_zerospan(self):
        """
        Check weather the virtual zerospan mode is turned on

        Input:
            None

        Output:
            val (bool) : True or False
        """
        return self._zerospan

    def do_set_trigger_source(self,source):
        """
        Set Trigger Mode

        Input:
            source (string) : AUTO | MANual | EXTernal | REMote

        Output:
            None
        """
        if source.upper() in [AUTO, MAN, EXT, REM]:
            self._visainstrument.write('TRIG:SOUR %s' % source.upper())
        else:
            raise ValueError('set_trigger_source(): must be AUTO | MANual | EXTernal | REMote')

    def do_get_trigger_source(self):
        """
        Get Trigger Mode

        Input:
            None

        Output:
            source (string) : AUTO | MANual | EXTernal | REMote
        """
        return self._visainstrument.query('TRIG:SOUR?')


    def do_set_channel_index(self,val):
        """
        Set the index of the channel to address.

        Input:
            val (int) : 1 .. number of active channels (max 16)

        Output:
            None
        """
        nop = self._visainstrument.read('DISP:COUN?')
        if val < nop:
            self._ci = val
        else:
            raise ValueError('set_channel_index(): index must be < nop channels')

    def do_get_channel_index(self):
        """
        Get active channel

        Input:
            None

        Output:
            channel_index (int) : 1-16
        """
        return self._ci

    def make_sweep_get_data(self):
        self.prepare_for_stb()
        self.sweep_single()
        self.wait_for_stb()
        return self.get_tracedata()

    def prepare_for_stb(self):
        # Clear the instrument's Status Byte
        self._visainstrument.write("*CLS")
        # Enable for the OPC bit (bit 0, which has weight 1) in the instrument's
        # Event Status Register, so that when that bit's value transitions from 0 to 1
        # then the Event Status Register bit in the Status Byte (bit 5 of that byte)
        # will become set.
        self._visainstrument.write("*ESE 1")
        return "OPC bit enabled (*ESE 1)."

    def wait_for_stb(self):
        self._visainstrument.write("*OPC")
        done = False
        while not(done):
            bla = self._visainstrument.query("*STB?")
            try:
                stb_value = int(bla)
            except:
                print("Error in wait(): value returned: {0}".format(bla))
            else:
                done = (2**5 == (2**5 & stb_value))
                sleep(0.01)



    def read(self):
        return self._visainstrument.read()
    def write(self,msg):
        return self._visainstrument.write(msg)
    def query(self,msg):
        return self._visainstrument.query(msg)

def create_window(self,ON=True):
    if(ON):
        self._visainstrument.write("CALCulate%i:PARameter:DEFine:EXT 'Meas1','S21'" )
        self._visainstrument.write("CALCulate%i:PARameter:DEFine:EXT 'Meas2','S21'" )
        self._visainstrument.write("DISPlay:WINDow%i:STATE ON")
        self._visainstrument.write("DISPlay:WINDow%i:TRACe1:FEED 'Meas1'")
        self._visainstrument.write("DISPlay:WINDow%i:TRACe1:FEED 'Meas2'")
        self._visainstrument.write("CALC1:FORM PHAS")
        self._visainstrument.write("SENSe1:FREQuency:STAR %e")
        self._visainstrument.write("SENSe1:FREQuency:STOP %e")
        self._visainstrument.write("CALCulate%i:PARameter:SELect 'Meas1'")
        self._visainstrument.write("CALCulate%i:PARameter:SELect 'Meas2'")
    else:
        self._visainstrument.write("DISPlay:WINDow%i:STATE OFF")
        self._visainstrument.write("CALC:PAR:DEL 'Meas1'")
        self._visainstrument.write("CALC:PAR:DEL 'Meas2'")

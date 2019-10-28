# agilent_PNA_L.py
# Gleb Fedorov <vdrhtc@gmail.com>
# Derived from Agilent PNA X made by Marcus Jerger, 2012
# Derived from Anritsu_VNA.py hacked by Hannes Rotzinger hannes.rotzinger@kit.edu, 2011
# derived from Anritsu_VNA.py (and whatever this is derived from)
# Pascal Macha <pascalmacha@googlemail.com>, 2010
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

from drivers.instrument import Instrument
import visa
import types
import logging
from time import sleep
import numpy

class Agilent_PNA_L(Instrument):
    """
    This is the python driver for the Agilent PNA L Vector Network Analyzer

    Usage:
    Initialise with
    <name> = instruments.create(address='<GPIB address>', reset=<bool>)

    """

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
        self._visainstrument = rm.open_resource(self._address) # no term_chars for GPIB!!!!!
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
        # self.define_S21() # these two lines were uncommented by Shamil 06/26/2017 due to the fact that
        # self.set_S21()  # by using high level measurement child classes it was not possible to continue proper operation
                        # of PNA-L after self._visaintrument.write( "SYST:FPReset" ) command. It seems like without these
                        # lines of code there is no trace selected after self.select_default_trace()
                        # and self.get_all seem do interrupt the program with timeout exception thrown by low-level visa
                        # GPIB drivers. The reason is that PNA-L doesn't have any number of points in sweep (get_all start
                        # by quering number of points in current sweep), because there is no traces defined, hence there
                        # is no number of points parameter available to read
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

    def get_all(self):
        self.get_nop()
        self.get_power()
        self.get_centerfreq()
        self.get_startfreq()
        self.get_stopfreq()
        self.get_span()
        self.get_bandwidth()
        self.get_trigger_source()
        self.get_average()
        self.get_averages()
        self.get_frequencies()
        self.get_channel_index()
        #self.get_zerospan()

    ###
    #Communication with device
    ###

    def init(self):
        if self._zerospan:
          self._visainstrument.write('INIT1;*wai')
        else:
          if self.get_average():
            for i in range(self.get_averages()):
              self._visainstrument.write('INIT1;*wai')
          else:
              self._visainstrument.write('INIT1;*wai')

    def select_default_trace(self):
        available_traces = self._visainstrument.query("CALC:PAR:CAT?").replace('"', "").split(",")
        self._visainstrument.write("CALC:PAR:SEL '%s'"%available_traces[0])


    def set_S21(self):
        """
        calls the defined S21 setting
        """
        self._visainstrument.write("CALC:PAR:SEL 'CH1_S21_1'")

    def del_all_meas(self):
        """
        Deletes all the windows and measurements present in the VNA
        """
        self._visainstrument.write("CALC:PAR:DEL:ALL")

    def define_S21(self):
        """
        defines the S21 measurement in the PNA X
        """
        self._visainstrument.write( "CALCulate:PARameter:DEF:EXT 'CH1_S21_1','S21'")

    def define_Sij(self,i=2,j=1, name=None):
        """
        defines the Sij measurement in the PNA X
        """
        if( name == None ):
            self._visainstrument.write( "CALCulate:PARameter:DEF:EXT "+'CH1_S%s%s_1'%(i,j)+', S%s%s'%(i,j))
        else:
            self._visainstrument.write(  "CALCulate:PARameter:DEF:EXT "+name+', S%s%s'%(i,j))

    def select_S_param(self, S_param):
        self.preset()

        self._visainstrument.write("DISPlay:ARRange SPLit")

        meas_names = ["POLar", "UPHase", "MLOGarithmic"]
        for i, name in enumerate(meas_names):
            # defining new measurement with the same channel number and s-param but different name
            self.define_Sij(i=int(S_param[1]), j=int(S_param[2]), name=name)
            # different names are allowing to feed measurements to different window traces
            # This is NOT the trace number of the channel which appears as the Tr annotation on the Trace Status display
            self._visainstrument.write("DISPlay:WINDow{0}:TRACe1:FEED ".format(i+1) + name)
            self._visainstrument.write("CALCulate1:PARameter:SELect "+name) # selecting measurement
            self._visainstrument.write("CALCulate1:FORMat "+name) # changing format of the selecting measurement

    def get_sweep_type(self):
        return self._visainstrument.query("SENS:SWE:TYPE?")[:-1]

    def autoscale_all(self):
        windows = self._visainstrument.query(" Disp:Cat?").replace('"', "").replace("\n", "").split(",")
        for window in windows:
            self._visainstrument.write("DISP:WIND%s:TRAC:Y:AUTO"%window)

    def reset_windows(self):
        self._visainstrument.write('DISP:WIND Off')
        self._visainstrument.write('DISP:WIND On')

    def set_autoscale(i,self):
        self._visainstrument.write("DISP:WIND:TRAC:Y:AUTO")

    def get_sweep(self):
        self._visainstrument.write( "ABORT; INITiate:IMMediate;*wai")

    def preset(self):
        self._visainstrument.write( "SYST:FPReset" )

    def avg_clear(self):
        self._visainstrument.write(':SENS%i:AVER:CLE' %(self._ci))

    def avg_status(self):
        # this does not work the same way than the VNA:
        # return int(self._visainstrument.query(':SENS%i:AVER:COUN?' %(self._ci))
        pass

    def get_avg_status(self):
        return self._visainstrument.query('STAT:OPER:AVER1:COND?')

    def still_avg(self):
        if int(self.get_avg_status()) == 1: return True
        else: return False

    def get_sdata(self):
        self._visainstrument.write(':FORMAT:DATA REAL,32; :FORMat:BORDer SWAP;')
        data = self._visainstrument.query_binary_values("CALCulate:DATA? SDATA")
        data_size = numpy.size(data)
        datareal = numpy.array(data[0:data_size:2])
        dataimag = numpy.array(data[1:data_size:2])
        return datareal+1j*dataimag

    def get_tracedata(self, format = 'AmpPha'):
        """
        Get the data of the current trace

        Input:
            format (string) : 'AmpPha': Amp in dB and Phase, 'RealImag',

        Output:
            'AmpPha':_ Amplitude and Phase
        """
        self._visainstrument.write(':FORMAT:DATA REAL,32; :FORMat:BORDer SWAP;')
        #data = self._visainstrument.ask_for_values(':FORMAT REAL,32; FORMat:BORDer SWAP;*CLS; CALC:DATA? SDATA;*OPC',format=visa.single)
        #data = self._visainstrument.ask_for_values(':FORMAT REAL,32;CALC:DATA? SDATA;',format=visa.double)
        #data = self._visainstrument.ask_for_values('FORM:DATA REAL; FORM:BORD SWAPPED; CALC%i:SEL:DATA:SDAT?'%(self._ci), format = visa.double)
        #test
        data = self._visainstrument.ask_for_values("CALCulate:DATA? SDATA")
        data_size = numpy.size(data)
        datareal = numpy.array(data[0:data_size:2])
        dataimag = numpy.array(data[1:data_size:2])

        #print datareal,dataimag,len(datareal),len(dataimag)
        if format.upper() == 'REALIMAG':
          if self._zerospan:
            return numpy.mean(datareal), numpy.mean(dataimag)
          else:
            return datareal, dataimag
        elif format.upper() == 'AMPPHA':
          if self._zerospan:
            datareal = numpy.mean(datareal)
            dataimag = numpy.mean(dataimag)
            dataamp = numpy.sqrt(datareal*datareal+dataimag*dataimag)
            datapha = numpy.arctan(dataimag/datareal)
            return dataamp, datapha
          else:
            dataamp = numpy.sqrt(datareal*datareal+dataimag*dataimag)
            datapha = numpy.arctan2(dataimag,datareal)
            return dataamp, datapha
        else:
          raise ValueError('get_tracedata(): Format must be AmpPha or RealImag')


    def get_freqpoints(self, query = False):
        """
        Deprecated
        """
        return self.get_frequencies()

    def get_frequencies(self, query = False):
      #if query == True:
        #self._freqpoints = numpy.array(self._visainstrument.ask_for_values('SENS%i:FREQ:DATA:SDAT?'%self._ci,format=1)) / 1e9
        #self._freqpoints = numpy.array(self._visainstrument.ask_for_values(':FORMAT REAL,32;*CLS;CALC1:DATA:STIM?;*OPC',format=1)) / 1e9
      self._freqpoints = numpy.linspace(self._start,self._stop,self._nop)
      return self._freqpoints

    def set_electrical_delay(self, delay):
        self._visainstrument.write("CALC{0}:CORRection:EDELay:TIME {1}".format(self._ci, delay))

    def get_electrical_delay(self):
        return float(self._visainstrument.query("CALC{0}:CORRection:EDELay:TIME?".format(self._ci)))

    def set_xlim(self, start, stop):
        self.logger.debug(__name__ + ' : setting start freq to %s Hz' % start)
        self._visainstrument.write('SENS%i:FREQ:STAR %f' % (self._ci,start))
        self._start = start
        self.get_centerfreq();
        self.get_stopfreq();
        self.get_span();

        self.logger.debug(__name__ + ' : setting stop freq to %s Hz' % stop)
        self._visainstrument.write('SENS%i:FREQ:STOP %f' % (self._ci,stop))
        self._stop = stop
        self.get_startfreq();
        self.get_centerfreq();
        self.get_span();

    def get_xlim(self):
        return self._start, self._stop

    def get_freq_limits(self):
        return self._start, self._stop

    def set_freq_limits(self, start, stop):
        self.logger.debug(__name__ + ' : setting start freq to %s Hz' % start)
        self._visainstrument.write('SENS%i:FREQ:STAR %f' % (self._ci,start))
        self._start = start
        self.get_centerfreq();
        self.get_stopfreq();
        self.get_span();

        self.logger.debug(__name__ + ' : setting stop freq to %s Hz' % stop)
        self._visainstrument.write('SENS%i:FREQ:STOP %f' % (self._ci,stop))
        self._stop = stop
        self.get_startfreq();
        self.get_centerfreq();
        self.get_span();


    def get_parameters(self):
        """
        Returns a dictionary containing bandwidth, nop, power, averages and
        freq_limits currently used by the VNA
        """
        return {"bandwidth":self.get_bandwidth(),
                  "nop":self.get_nop(),
                  "sweep_type":self.get_sweep_type(),
                  "power":self.get_power(),
                  "averages":self.get_averages(),
                  "freq_limits":self.get_freq_limits()}

    def set_parameters(self, parameters_dict):
        """
        Method allowing to set all or some of the VNA parameters at once
        (bandwidth, nop, power, averages and freq_limits)
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
            if (parameters_dict["sweep_type"] == "CW"):
                self.do_set_CWfreq(numpy.mean(parameters_dict["freq_limits"]))
            else:
                self.set_freq_limits(*parameters_dict["freq_limits"])
        if "span" in parameters_dict.keys():
            self.set_span(parameters_dict["span"])
        if "centerfreq" in parameters_dict.keys():
            self.set_centerfreq(parameters_dict["centerfreq"])
        if "sweep_type" in parameters_dict.keys():
            self.set_sweep_type(parameters_dict["sweep_type"])

        if "aux_num" in parameters_dict.keys():
            self.set_aux_num(parameters_dict["aux_num"])
        if "trig_per_point" in parameters_dict.keys():
            self.set_trig_per_point(parameters_dict["trig_per_point"])
        if "pos" in parameters_dict.keys():
            self.set_pos(parameters_dict["pos"])
        if "bef" in parameters_dict.keys():
            self.set_bef(parameters_dict["bef"])
        if "trig_dur" in parameters_dict.keys():
            self.set_trig_dur(parameters_dict["trig_dur"])

    def do_set_CWfreq(self,freq):
        """
        Set CW frequency valid if sweepind in CW mode.
        """

        self.logger.debug(__name__ + ' : set CW frequency')
        self._visainstrument.write("SENS%i:FOM:RANG:FREQ:CW %.6f" %(self._ci,freq))

    def do_get_CWfreq(self):
        """
        Asking for CW freq
        """
        self.logger.debug(__name__ + ' : getting CW freq')
        return float(self._visainstrument.query('SENS%i:FOM:RANG:FREQ:CW?' % (self._ci)))

    def get_sweep_time(self):
        """
        Get the time needed for one sweep

        Returns:
            out: float
                time in ms
        """
        return float(self._visainstrument.query(':SENS%i:SWE:TIME?' % (self._ci))) * 1e3
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
        self.logger.debug(__name__ + ' : setting Number of Points to %s ' % (nop))
        self._visainstrument.write(':SENS%i:SWE:POIN %i' %(self._ci,nop))
        self._nop = nop
        self.get_frequencies() #Update List of frequency points

    def do_get_nop(self):
        """
        Get Number of Points (nop) for sweep

        Input:
            None
        Output:
            nop (int)
        """
        self.logger.debug(__name__ + ' : getting Number of Points')
        if self._zerospan:
          return 1
        else:
            self._nop = int(self._visainstrument.query(':SENS%i:SWE:POIN?' % (self._ci)))
        return self._nop

    def do_set_average(self, status):
        """
        Set status of Average

        Input:
            status (string) : 'on' or 'off'

        Output:
            None
        """
        self.logger.debug(__name__ + ' : setting Average to "%s"' % (status))
        if status:
            status = 'ON'
            self._visainstrument.write('SENS%i:AVER:STAT %s' % (self._ci,status))
        elif status == False:
            status = 'OFF'
            self._visainstrument.write('SENS%i:AVER:STAT %s' % (self._ci,status))
        else:
            raise ValueError('set_Average(): can only set on or off')

    def do_get_average(self):
        """
        Get status of Average

        Input:
            None

        Output:
            Status of Averaging ('on' or 'off) (string)
        """
        self.logger.debug(__name__ + ' : getting average status')
        return bool(int(self._visainstrument.query('SENS%i:AVER:STAT?' % (self._ci))))

    def do_set_averages(self, av):
        """
        Set number of averages

        Input:
            av (int) : Number of averages

        Output:
            None
        """
        self._visainstrument.write('SENS%i:AVER:COUN %i' % (self._ci,av))
        self._visainstrument.write('SENS:AVER:MODE POIN')
        self.do_set_average(True)
        # if av > 1:
        #     self.do_set_average(True)
        #     self._visainstrument.write('SENS:SWE:GRO:COUN %i'%av)
        # else:
        #     self.do_set_average(False)
        #     self._visainstrument.write('SENS:SWE:GRO:COUN 1')

    def do_get_averages(self):
        """
        Get number of averages

        Input:
            None
        Output:
            number of averages
        """
        self.logger.debug(__name__ + ' : getting Number of Averages')
        if self._zerospan:
            return int(self._visainstrument.query('SWE%i:POIN?' % self._ci))
        else:
            return int(self._visainstrument.query('SENS%i:AVER:COUN?' % self._ci))

    def set_sweep_type(self,sweep_type = "LIN"):
        self._visainstrument.write("SENS:SWE:TYPE "+sweep_type)

    def do_set_power(self,pow):
        """
        Set probe power

        Input:
            pow (float) : Power in dBm

        Output:
            None
        """
        self.logger.debug(__name__ + ' : setting power to %s dBm' % pow)
        self._visainstrument.write('SOUR%i:POW1:LEV:IMM:AMPL %.2f' % (self._ci,pow))
    def do_get_power(self):
        """
        Get probe power

        Input:
            None

        Output:
            pow (float) : Power in dBm
        """
        self.logger.debug(__name__ + ' : getting power')
        return float(self._visainstrument.query('SOUR%i:POW1:LEV:IMM:AMPL?' % (self._ci)))

    def do_set_center(self, f):
        self.do_set_centerfreq(f)

    def do_get_center(self):
        return self.do_get_centerfreq()

    def do_set_centerfreq(self,cf):
        """
        Set the center frequency

        Input:
            cf (float) :Center Frequency in Hz

        Output:
            None
        """
        self.logger.debug(__name__ + ' : setting center frequency to %s' % cf)
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
        self.logger.debug(__name__ + ' : getting center frequency')
        return float(self._visainstrument.query('SENS%i:FREQ:CENT?' % (self._ci)))

    def do_set_span(self,span):
        """
        Set Span

        Input:
            span (float) : Span in KHz

        Output:
            None
        """
        self.logger.debug(__name__ + ' : setting span to %s Hz' % span)
        self._visainstrument.write('SENS%i:FREQ:SPAN %i' % (self._ci,span))
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
        #self.logger.debug(__name__ + ' : getting center frequency')
        span = self._visainstrument.query(
            'SENS%i:FREQ:SPAN?' % (self._ci))  # float( self.query('SENS1:FREQ:SPAN?'))
        return span

    def sweep_hold(self):
        self.write("SENS:SWE:MODE HOLD")

    def sweep_continuous(self):
        self.write("SENS:SWE:MODE CONT")

    def sweep_single(self):
        self.write("SENSe{0}:SWEep:MODE SINGle".format(self._ci))
        # self.write("SENS%i:SWE:MODE GROUPS"%(self._ci))

    def do_set_startfreq(self,val):
        """
        Set Start frequency

        Input:
            span (float) : Frequency in Hz

        Output:
            None
        """
        self.logger.debug(__name__ + ' : setting start freq to %s Hz' % val)
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
        self.logger.debug(__name__ + ' : getting start frequency')
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
        self.logger.debug(__name__ + ' : setting stop freq to %s Hz' % val)
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
        self.logger.debug(__name__ + ' : getting stop frequency')
        self._stop = float(self._visainstrument.query('SENS%i:FREQ:STOP?' % (self._ci)))
        return  self._stop

    def do_set_bandwidth(self,band):
        """
        Set Bandwidth

        Input:
            band (float) : Bandwidth in Hz

        Output:
            None
        """
        self.logger.debug(__name__ + ' : setting bandwidth to %s Hz' % (band))
        self._visainstrument.write('SENS%i:BWID:RES %i' % (self._ci,band))
    def do_get_bandwidth(self):
        """
        Get Bandwidth

        Input:
            None

        Output:
            band (float) : Bandwidth in Hz
        """
        self.logger.debug(__name__ + ' : getting bandwidth')
        # getting value from instrument
        return float(self._visainstrument.query('SENS%i:BWID:RES?' % self._ci))

    def do_set_zerospan(self,val):
        """
        Zerospan is a virtual "zerospan" mode. In Zerospan physical span is set to
        the minimal possible value (2Hz) and "averages" number of points is set.

        Input:
            val (bool) : True or False

        Output:
            None
        """
        #self.logger.debug(__name__ + ' : setting status to "%s"' % status)
        if val not in [True, False]:
            raise ValueError('set_zerospan(): can only set True or False')
        if val:
          self._oldnop = self.get_nop()
          self._oldspan = self.get_span()
          if self.get_span() > 0.002:
            Warning('Setting ZVL span to 2Hz for zerospan mode')
            self.set_span(0.002)

        av = self.get_averages()
        self._zerospan = val
        if val:
            self.set_Average(False)
            self.set_averages(av)
            if av<2:
              av = 2
        else:
          self.set_Average(True)
          self.set_span(self._oldspan)
          self.set_nop(self._oldnop)
          self.get_averages()
        self.get_nop()

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
        self.logger.debug(__name__ + ' : setting trigger source to "%s"' % source)
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
        self.logger.debug(__name__ + ' : getting trigger source')
        return self._visainstrument.query('TRIG:SOUR?')


    def do_set_channel_index(self,val):
        """
        Set the index of the channel to address.

        Input:
            val (int) : 1 .. number of active channels (max 16)

        Output:
            None
        """
        self.logger.debug(__name__ + ' : setting channel index to "%i"' % int)
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
        self.logger.debug(__name__ + ' : getting channel index')
        return self._ci

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
                sleep(0.02)

    def set_output_state(self, state):
        """
        new function, must be checked
        """
        available_states = {'ON', 'OFF'}
        if state not in available_states:
            raise ValueError("state must be 'ON' or 'OFF'")
        self._visainstrument.write("OUTP {}".format(state))

    def read(self):
        return self._visainstrument.read()
    def write(self,msg):
        return self._visainstrument.write(msg)

    def query(self, msg):
        return self._visainstrument.query(msg)

    def do_set_aux_num(self, aux_num):
        self._visainstrument.write("TRIG:CHAN:AUX %i" % (aux_num))

    def do_get_aux_num(self):
        raise NotImplemented

    def do_set_trig_per_point(self, trig_per_point):
        if trig_per_point == True:
            self._visainstrument.write("TRIG:CHAN:AUX:INT POIN")
        else:
            self._visainstrument.write("TRIG:CHAN:AUX:INT SWE")

    def do_get_trig_per_point(self):
        raise NotImplemented

    def do_set_pos(self, pos):
        if pos == True:
            self._visainstrument.write("TRIG:CHAN:AUX:OPOL POS")
        else:
            self._visainstrument.write("TRIG:CHAN:AUX:OPOL NEG")

    def do_get_pos(self):
        raise NotImplemented

    def do_set_bef(self, bef):
        if bef == True:
            self._visainstrument.write("TRIG:CHAN:AUX:POS BEF")
        else:
            self._visainstrument.write("TRIG:CHAN:AUX:POS AFT")

    def do_get_bef(self):
        raise NotImplemented

    def do_set_trig_dur(self, trig_dur):  # > 2e-3 sec (EXG restriction)
        self._visainstrument.write("TRIG:CHAN:AUX:DUR %f" % trig_dur)

    def do_get_trig_dur(self):
        raise NotImplemented

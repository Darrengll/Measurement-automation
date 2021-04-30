import time
import numpy as np
import matplotlib.cm
import matplotlib.colors
from Drivers.Yokogawa_GS200 import Yokogawa_GS210
from Drivers.Agilent_PNA_L import Agilent_PNA_L
from time import sleep
import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
import os
import sys


mypna = Agilent_PNA_L("name_pna","PNA-L1")
mypna._visainstrument.write("CALC:PAR:DEL:ALL")
mypna._visainstrument.write("CALC:PAR:DEF:EXT 'CH1_S21_1',S21")
mypna._visainstrument.write("CALC:PAR:SEL 'CH1_S21_1'")
myyoko = Yokogawa_GS210("GPIB0::5::INSTR")
#myyoko.set_current_limits(-10e-3,10e-3)
myyoko.set_status(1)
sleep(0.01)

dt = datetime.datetime.now().strftime("%d.%m.%Y_%H-%M-%S_")
fname0 = "Data/Bolgar/A2_2/2Q/resonance_2_82_GHz/{0}".format(dt)
electrical_delay = 55e-9

power = int(sys.argv[1])#-55          #Power in dBm
if_bw = np.round(float(sys.argv[2]),1)  #bandwidth (in Hz)

start_freq = float(sys.argv[3])
stop_freq = float(sys.argv[4])
step_freq = (stop_freq-start_freq)/1000

P = int((stop_freq-start_freq)/step_freq+1) #kol-vo tochek by if_freq

current = float(sys.argv[5])/1000


pars = "{:g}-{:g}GHz_{:g}mA_{:}dBm_bw{:}Hz".format(start_freq/1e9,stop_freq/1e9,current*1e3,power,if_bw)
fname = (fname0+pars+"/data")
#print(fname)
if not os.path.exists(fname0+pars):
    os.makedirs(fname0+pars)

overhead_time = 31.4e-3
exp_start_time = datetime.datetime.now()
exp_duration_calc = (P/if_bw + overhead_time)/3600
print("Start_time: ", exp_start_time.ctime(),"\n","expected_duration: {0} ".format(exp_duration_calc),"hours\n")


freq = np.linspace(start_freq, stop_freq, P)



#Presetting the current in the coil to initial current of the sweep****************
# myyoko.set_appropriate_range(max(abs(current)))
curstepabs = 20e-6


#****************************************



mypna.set_power(power)
mypna.set_nop(P)
mypna.set_xlim(start_freq,stop_freq)
mypna.set_bandwidth(if_bw)

mypna.set_electrical_delay(electrical_delay)


myyoko.set_current(current)
# sleep(0.02)
mypna.prepare_for_stb()
mypna.sweep_single()
mypna.wait_for_stb()
data = mypna.get_tracedata("RealImag")
sdata = data[0]+1j*data[1]
phase_data = np.angle(sdata)
amp_data = np.abs(sdata)



np.savez(fname,
    P=P,
    if_bw=if_bw,
    power=power,
    freq=freq,
    sdata=sdata,
    current=current,
    start_freq=start_freq,
    stop_freq=stop_freq
    )

plt.clf()
axx=plt.subplot(111)
axx.grid(True)
axx.set_title("Amp")
axx.plot(freq,amp_data, linewidth=1)  #
#plt.show()

plt.savefig(fname0+pars+'/image.jpg')

exp_stop_time = datetime.datetime.now()
#print(exp_stop_time-exp_start_time, 'executed.')

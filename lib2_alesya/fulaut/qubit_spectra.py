
from numpy import *

def transmon_spectrum(curs, period, sweet_spot_cur, max_freq, d):
    phis = (curs-sweet_spot_cur)/period
    return max_freq*sqrt(abs(cos(pi*phis))*sqrt(1+d**2*tan(pi*phis)**2))

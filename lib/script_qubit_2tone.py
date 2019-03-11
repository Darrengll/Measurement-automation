
from measurements import parametric_sweep as ps

try:
    
    pna.set_average(True)
    pna.set_bandwidth(100)	
    pna.set_averages(3)
    pna.set_nop(1)
    pna.set_power(-35)
    pna.get_all()
    pna.set_centerfreq(5.2682e9)
    pna.set_span(1e6)
    
    lo.set_power(-5)
    lo.set_status(1)
    #lo.set_frequency(13.8477e9)
    #current.set_current(-7231.25e-6)
    
    Frequencies = arange (3e9, 10e9, 30e6)
    currents = arange (-10000e-6, -4000e-6, 30e-6)
    current.set_status(1)
    measurement = ps.sweep2D(pna, Frequencies, lo.set_frequency, currents, current.set_current)
    #measurement = ps.sweep1D(pna, Powers, lo.set_power)
	

finally:
	current.set_current(0)
	current.set_status(0)
	lo.set_status(0)



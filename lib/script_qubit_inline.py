
from measurements import parametric_sweep as ps

try:
    
    pna.set_average(True)
    pna.set_bandwidth(100)
    pna.set_averages(1)
    pna.set_nop(3)
    pna.set_power(-35)
    pna.get_all()
    #pna.set_xlim(5.26e9, 5.28e9)
    pna.set_centerfreq(5.27023e9)
    pna.set_span(5e6)
    
    currents = arange (-10000e-6, 10000e-6, 10e-6)
    #powerss =arange (0, -55, -0.3)
    #lo.set_status(1)
    #lo.set_frequency(14.0515e9)
    
    current.set_status(1)
    measurement = ps.sweep1D(pna, currents, current.set_current)
	

finally:
	current.set_current(0)	
	current.set_status(0)

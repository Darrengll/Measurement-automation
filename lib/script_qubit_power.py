
from measurements import parametric_sweep as ps

try:
    #pna.avg_clear()
    #sleep(pna.get_sweep_time()*pna.get_averages()/1000)

    #res_freq1 = RD.detect_resonator(pna)[0]
    #pna.set_centerfreq(res_freq1)
    #sleep(1)


    pna.set_average(True)
    pna.set_bandwidth(100)	
    pna.set_averages(3)
    pna.set_nop(400)
    pna.get_all()
    pna.set_span(40e6)
    
 
    powers = arange (-10,-60,-1)
    current.set_status(0)
    measurement = ps.sweep1D(pna, powers, pna.set_power)

	

finally:
	current.set_current(0)
	current.set_status(0)



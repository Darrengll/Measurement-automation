
from numpy import *


startfreq, stopfreq = 7e9, 13e9
step = 10e6
try:
	lo.set_status(1)
	mw_src.set_status(1)
	mw_src.set_power(-5)

	S21s = []
	freqs = linspace(startfreq, stopfreq, (stopfreq-startfreq)/step)
	for freq in freqs:
		lo.set_frequency(freq)
		mw_src.set_frequency(freq-25e6)
		a = dso.get_data(1)[0]
		S21s.append(max(a) - min(a))
		print("%.2e\r"%freq, end="")
		
		
finally:
	lo.set_status(0)
	mw_src.set_status(0)
	mw_src.set_power(-20)

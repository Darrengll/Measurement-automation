import numpy as np
from resonator_tools import circuit

def detect_resonator(pna, type="AMP"):
	"""
	Finds a resonator on the screen
	"""

	freq = 1
	if type=="AMP":
		Y = 20*np.log10(abs(pna.get_sdata()))
		idx = np.where(Y==np.min(Y))[0][0]
		freq = pna.get_frequencies()[idx]
		amp = (Y[idx])
		return freq, amp
	elif type=="PHAS":
		Y = np.diff(np.unwrap(np.angle(pna.get_sdata())))
		idx = np.where(Y==np.max(Y))[0][0]
		freq = pna.get_frequencies()[idx]
		phas_derivative = Y[idx]
		return freq, phas_derivative
	elif type=="FIT":
		port = circuit.notch_port(pna.get_frequencies(), pna.get_sdata())
		port.autofit()
		# port.plotall()
		# print(port.fitresults)
		return pna.get_frequencies()[np.argmin(abs(port.z_data_sim))], 20*np.log10(min(abs(port.z_data_sim)))

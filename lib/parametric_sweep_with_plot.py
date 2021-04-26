import numpy as np
import time
from datetime import datetime
from time import sleep
import sys
from lib import data_management as dm
from matplotlib import pyplot as plt, colorbar
from lib.measurement import Measurement
import traceback as tb

def update_2D_plot(X, Y, data, ax_amps, ax_phas, cax_amps, cax_phas):
	XX, YY = np.meshgrid(X, Y)
	ax_amps.reset()
	ax_phas.reset()
	cax_amps.reset()
	cax_phas.reset()
	amps_map = ax_amps.pcolormesh(XX, YY, np.real(data[0].T), vmax=np.real(np.max(data[0][data[0]!=1j])), vmin=np.real(np.min(data[0][data[0]!=1j])), cmap="RdBu_r")
	plt.colorbar(amps_map, cax = cax_amps)
	recorded_phas_data = np.unwrap(np.unwrap(np.real(data[1][data[1]!=1j])).T)
	phas_map = ax_phas.pcolormesh(XX, YY, np.unwrap(np.unwrap(np.real(data[1])).T), vmax=np.max(recorded_phas_data), vmin=np.min(recorded_phas_data), cmap="RdBu_r")
	plt.colorbar(phas_map, cax = cax_phas)
	ax_amps.axis("tight")
	ax_phas.axis("tight")
	ax_phas.set_title("Phase")
	ax_amps.set_title("Amplitude")
	#plt.pause(0.01)


def sweep2D(vna, param1_vals, param1_setter, param2_vals, param2_setter, filename=None, center_freqs = None, averages=None):

	# plt.ion()

	fig, axes = plt.subplots(1,2,figsize=(15,7))
	ax_amps, ax_phas = axes
	ax_amps.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
	ax_phas.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
	ax_amps.grid()
	ax_phas.grid()
	cax_amps, kw = colorbar.make_axes(ax_amps)
	cax_phas, kw = colorbar.make_axes(ax_phas)

	start_datetime = datetime.now()

	print("Started at: ", start_datetime.ctime())

	data_amp = 1j+np.zeros((len(param1_vals), len(param2_vals)))
	data_phas = 1j+np.zeros((len(param1_vals), len(param2_vals)))

	context =  {"vna_power":vna.get_power(), "bandwidth":vna.get_bandwidth(), "averages":vna.get_averages()}

	measurement_type = "pna-p2D-3D" if min(len(param1_vals), len(param2_vals))>1 else "pna-p2D-2D"
	measurement = Measurement(Measurement.TYPES[measurement_type],
	 	(param1_vals, param2_vals, vna.get_frequencies(), data_amp, data_phas), start_datetime, context)

	try:
		nop = vna.get_nop()

		done_sweeps = 0
		total_sweeps = len(param1_vals)*len(param2_vals)

		print("Averages: "+str(vna.get_averages())+", bandwidth: "+str(vna.get_bandwidth())+", power:"+str(vna.get_power()))
		print("Sweeping total: "+str(total_sweeps)+" sweeps "+ "(%ix%i)"%(len(param1_vals), len(param2_vals)) +", each of "+str(vna.get_nop())+" point(s)")

		vna.sweep_hold()


		i = 0
		for value1 in param1_vals:

			if center_freqs != None:
				vna.set_center(center_freqs[i])
			if averages != None:
				vna.set_averages(averages[i])

			j=0
			param1_setter(value1)
			for value2 in param2_vals:
				param2_setter(value2)

				vna.avg_clear()
				vna.prepare_for_stb()
				vna.sweep_single()
				vna.wait_for_stb()
				sdata = vna.get_sdata()
				amps, phas = abs(sdata), np.angle(sdata)

				if nop>1:
					measurement.get_data()[-2][i,j] = 20*np.log10(amps)
					measurement.get_data()[-1][i,j] = phas
				else:
					measurement.get_data()[-2][i,j] = 20*np.log10(amps)[0]
					measurement.get_data()[-1][i,j] = phas[0]

				done_sweeps += 1
				avg_time = (datetime.now() - start_datetime).total_seconds()/done_sweeps

				print("\rTime left: "+format_time_delta( \
					avg_time*(total_sweeps-done_sweeps))+", parameter values (1,2): "+str(("%.3e"%value1, "%.3e"%value2))+\
						", average cycle time: "+str(round(avg_time, 2))+" s          ", end="", flush=True)
				j+=1

			update_2D_plot(param1_vals, param2_vals, measurement.remove_background(-1,"x").get_data()[3:], ax_amps, ax_phas, cax_amps, cax_phas)

			i+=1

		elapsed_time = format_time_delta((datetime.now() - start_datetime).total_seconds())
		print("\nElapsed time: "+elapsed_time)
		measurement.set_recording_time(elapsed_time)
		measurement.get_context()["recording_time"] = elapsed_time
		measurement.get_context()["resolution_xy"] = (len(param1_vals), len(param2_vals))
		curs, freqs, freq, amps, phas = measurement.get_data()
		measurement.set_data((curs, freqs, freq, real(amps), real(phas)))

		if filename!=None:
			directory = dm.save_measurement(measurement, filename)
	except:
		print("\nUnexpected error:", tb.print_exc())
	finally:
		plt.close("all")
		return measurement

def sweep1D(vna, param_vals, param_setter, filename=None):

	# plt.ion()

	fig, axes = plt.subplots(1,2,figsize=(15,7))
	ax_amps, ax_phas = axes
	ax_amps.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
	ax_phas.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))
	ax_amps.grid()
	ax_phas.grid()

	start_datetime = datetime.now()

	print("Started at: ", start_datetime.ctime())

	data_amp = 1j+np.zeros((len(param_vals), vna.get_nop()))
	data_phas = 1j+np.zeros((len(param_vals), vna.get_nop()))

	context =  {"power":vna.get_power(), "bandwidth":vna.get_bandwidth(), "averages":vna.get_averages()}
	measurement = Measurement(Measurement.TYPES["pna-p1D-3D"], (param_vals, vna.get_frequencies(), data_amp, data_phas), start_datetime, context)

	try:
		done_sweeps = 0
		total_sweeps = len(param_vals)
		# nop = vna.get_nop()

		print("Averages: "+str(vna.get_averages())+", bandwidth: "+str(vna.get_bandwidth())+ ", power: "+str(vna.get_power()))
		print("Sweeping total: "+str(total_sweeps)+" sweeps, each of "+str(vna.get_nop())+" point(s)")

		param_setter(param_vals[0])
		vna.sweep_hold()

		for idx, value in enumerate(param_vals):

			param_setter(value)

			vna.avg_clear()
			vna.prepare_for_stb()
			vna.sweep_single()
			vna.wait_for_stb()
			sdata = vna.get_sdata()
			amps, phas = abs(sdata), np.angle(sdata)
			measurement.get_data()[-2][idx] = 20*np.log10(amps)
			measurement.get_data()[-1][idx] = phas

			done_sweeps += 1
			avg_time = (datetime.now() - start_datetime).total_seconds()/done_sweeps

			update_2D_plot(param_vals, vna.get_frequencies(), measurement.get_data()[2:], ax_amps, ax_phas)

			print("\rTime left: "+format_time_delta( \
				avg_time*(total_sweeps-done_sweeps))+", parameter value: "+"%.3e"%value+", average cycle time: "+str(round(avg_time, 2))+" s          ", end="", flush=True)

		elapsed_time = format_time_delta((datetime.now() - start_datetime).total_seconds())
		print("\nElapsed time: "+elapsed_time)
		measurement.set_recording_time(elapsed_time)
		measurement.get_context()["recording_time"] = elapsed_time
		measurement.get_context()["resolution_xy"] = (total_sweeps, vna.get_nop())

		if filename != None:
			dm.save_measurement(measurement, filename)
	except:
		print("\nUnexpected error:", tb.print_exc())
	finally:
		return measurement


def format_time_delta(delta):
	hours, remainder = divmod(delta, 3600)
	minutes, seconds = divmod(remainder, 60)
	return '%s h %s m %s s' % (int(hours), int(minutes), round(seconds, 2))

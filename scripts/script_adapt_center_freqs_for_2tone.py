from lib import ResonatorDetector as RD
from lib import parametric_sweep as ps
from time import sleep
from datetime import datetime as dt
from scipy.interpolate import interp1d as interpolate
from scipy.signal import savgol_filter as savgol

def format_time_delta(delta):
	hours, remainder = divmod(delta, 3600)
	minutes, seconds = divmod(remainder, 60)
	return '%s h %s m %s s' % (int(hours), int(minutes), round(seconds, 2))

try:

    start_center_freq = RD.detect_resonator(znb)[0]

    currents = np.linspace(-12.5e-3, 7e-3, 20)
    current.set_compliance(5)
    current.set_range(max(abs(currents)))
    current.output_on()

    znb.set_nop(100)
    znb.set_bandwidth(50)
    znb.set_averages(2)
    znb.set_power(-60)

    print("Adapting settlement frequencies...", flush=True)
    start = dt.now()
    center_freqs = []
    for idx, current_val in enumerate(currents):
        current.set_current(current_val)
        znb.avg_clear()
        znb.prepare_for_stb()
        znb.sweep_single()
        znb.wait_for_stb()
        center_freq = RD.detect_resonator(znb, type="FIT")
        center_freqs.append(center_freq)
        znb.set_freq_center_span(center_freq, 10e6)

        part_done = (idx+1)/len(currents)
        elapsed_time = (dt.now()-start).total_seconds()
        estimated_time_left = elapsed_time/part_done - elapsed_time
        print("\rTime left:", format_time_delta(estimated_time_left),
              "current %.3e, center frequency %.5e"%(current_val, center_freq),
              end=" %.2f %%"%((idx+1)/len(currents)*100), flush=True)

    center_freq_estimator = interpolate(currents, center_freqs)
    plot(currents, center_freq_estimator(currents), "o")
    plot(linspace(currents[0], currents[-1], 1000), center_freq_estimator(linspace(currents[0], currents[-1], 1000)))

finally:
    current.set_current(0)
    current.output_off()
    mw_src.set_output_state("OFF")
    znb.avg_clear()
    znb.set_averages(10)
    znb.set_nop(200)
    #lo1.set_status(0)
    znb.set_freq_center_span(start_center_freq, 5e6)
    znb.set_bandwidth(1000)
    znb.set_power(-50)
    znb.sweep_single()

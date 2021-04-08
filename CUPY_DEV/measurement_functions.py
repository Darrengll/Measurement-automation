import cupy as cp
import numpy as np
import asyncio
import cupyx.scipy.ndimage as cpndimage
import scipy.signal as sg
from drivers.Spectrum_m4x import SPCM_MODE, SPCM_TRIGGER
from drivers.pyspcm import *
import scripts.photon_wave_mixing.devices as dev
from lib2.IQPulseSequence import IQPulseBuilder
from concurrent.futures import ThreadPoolExecutor
import threading
import asyncio
from tqdm.notebook import tqdm


def measure_ADC_sync(dig, cpu_buffer=None):
    dig.start_card()
    dig.wait_for_card()  # wait till the end of a measurement

    if cpu_buffer is None:
        cpu_buffer = (int8 * dig._bufsize)()

    # define Card -> PC transfer buffer
    res = dig._def_simp_transfer(cpu_buffer)
    if res is not 0:
        print("Error: %d" % res)

    # Start the transfer and wait till it's completed
    dig._write_to_reg_32(SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)
    # Explicitly stop DMA transfer
    dig._write_to_reg_32(SPC_M2CMD, M2CMD_DATA_STOPDMA)
    dig._invalidate_buffer()  # Invalidate the buffer
    gpu_data = (dig.ch_amplitude / 1000 / 128) \
               * cp.asarray(pData, dtype=cp.int8).astype(cp.float64) \
                   .reshape(n_seg, dig._segment_size, 2)
    VI_segs = gpu_data[:, :, 0]
    VQ_segs = gpu_data[:, :, 1]
    return VI_segs, VQ_segs


def meas_gpu_P1(dig, Z):
    cpu_buffer = (int8 * dig._bufsize)()
    VI_segs, VQ_segs = measure_ADC_sync(dig, cpu_buffer)
    # 1000/Z convert V^2(t) to mW
    gpu_res = (1000 / Z) * cp.mean(VI_segs ** 2 + VQ_segs ** 2, axis=0)
    cpu_res = gpu_res.get()
    del gpu_res
    return cpu_res


def calc_gpu_P1_spectrum(dig):
    pass


def calc_gpu_P2(dig, Z):
    gpu_res = (1000 / Z) * cp.mean(
        (VI_segs - cp.mean(VI_segs, axis=0)[cp.newaxis, :]) ** 2 + (
                VQ_segs - cp.mean(VQ_segs, axis=0)[cp.newaxis, :]) ** 2,
        axis=0
    )
    cpu_res = gpu_res.get()
    del gpu_res
    return cpu_res


def calc_gpu_g1(VI_segs, VQ_segs, Z):
    pass


def gpu_mem_free():
    with cp.cuda.Device(0):
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()


def gpu_memory_report():
    with cp.cuda.Device(0):
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        print("GPU mempool", mempool.used_bytes())
        print("GPU total mempool", mempool.total_bytes())
        print("GPU pinned mempool", pinned_mempool.n_free_blocks())


async def get_data_from_adc_safe(devices_dict):
    dig = devices_dict['dig']
    dig.start_card()

    if "cpu_buffer" not in globals():
        global cpu_buffer
        cpu_buffer = (int8 * dig._bufsize)()

    try:
        while True:
            await asyncio.sleep(0.032768)
            if dig.is_ready():
                break
    except KeyboardInterrupt:
        dig.stop_card()
        print("Card was interrupted")
        return

    # define Card -> PC transfer buffer
    res = dig._def_simp_transfer(cpu_buffer)
    if res is not 0:
        print("Error: %d" % res)

    # Start the transfer and wait till it's completed
    dig._write_to_reg_32(SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)

    # Explicitly stop DMA transfer
    dig._write_to_reg_32(SPC_M2CMD, M2CMD_DATA_STOPDMA)
    # dig._invalidate_buffer()  # Invalidate the buffer


def get_data_from_adc(devices_dict):
    dig = devices_dict['dig']
    dig.start_card()
    dig.wait_for_card()  # wait till the end of a measurement

    # if "cpu_buffer" not in globals()
    cpu_buffer = (int8 * dig._bufsize)()

    # define Card -> PC transfer buffer
    res = dig._def_simp_transfer(cpu_buffer)
    if res is not 0:
        print("Error: %d" % res)

    # Start the transfer and wait till it's completed
    dig._write_to_reg_32(SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA)

    # Explicitly stop DMA transfer
    dig._write_to_reg_32(SPC_M2CMD, M2CMD_DATA_STOPDMA)
    dig._invalidate_buffer()  # Invalidate the buffer
    return cpu_buffer


def transfer_to_gpu(devices_dict, params, data):
    segment_size = devices_dict['dig']._segment_size
    gpu_data = (params['dig_amp'] / 128) * \
               cp.asarray(data, dtype=cp.int8).astype(np.float32) \
                   .reshape((params['segments'], segment_size, 2))
    return gpu_data[:, :, 0], gpu_data[:, :, 1]


def downconvert(traces_i, traces_q, params):
    traces = traces_i + 1j * traces_q
    traces *= params["downconversion"]
    return cp.real(traces), cp.imag(traces)


def filter(traces, params):
    cpndimage.convolve1d(traces, params['filter'], output=traces,
                         axis=-1, mode="wrap")


def measure_adc_sync_gpu(devices_dict, params):
    # with params['lock']:
    cpu_buffer = get_data_from_adc(devices_dict)
    # print('Measured')

    # convert data to float32 in mV
    i_segs, q_segs = transfer_to_gpu(devices_dict, params, cpu_buffer)

    i_segs, q_segs = down_calibrate_on_gpu(i_segs, q_segs,
                                           params['downconv_coefficients'])
    i_segs, q_segs = downconvert(i_segs, q_segs, params)
    if 'filter' in params:
        filter(i_segs, params)
        filter(q_segs, params)
    return i_segs, q_segs


def calculate_p1(i_segs, q_segs):
    return cp.sum(i_segs ** 2 + q_segs ** 2, axis=0, dtype=cp.float32) / 5e7


def calculate_p2(i_segs, q_segs, average_i, average_q):
    return (1000 / 50) * cp.sum((i_segs - average_i) ** 2
                                + (q_segs - average_q) ** 2,
                                axis=0, dtype=cp.float32)


def meas_p1_gpu(devices_dict, params, cpu_buffer=None):
    i_segs, q_segs = measure_adc_sync_gpu(devices_dict, params, cpu_buffer)
    # 1000/Z convert V^2(t) to mW
    gpu_res = (1000 / 50) * cp.sum(i_segs ** 2 + q_segs ** 2,
                                   axis=0, dtype=cp.float32)
    # cpndimage.convolve(gpu_res, params['filter'],
    #                      output=gpu_res, mode="wrap")
    # cpu_res = gpu_res.get()
    return gpu_res


def down_calibrate_on_gpu(I, Q, coeffs):
    I1 = coeffs[0, 0] * I + coeffs[0, 2]
    Q1 = coeffs[1, 0] * I + coeffs[1, 1] * Q + coeffs[1, 2]
    return I1, Q1


def meas_p2_gpu(dig, Z, cpu_buffer=None):
    VI_segs, VQ_segs = measure_adc_sync_gpu(dig, cpu_buffer)
    gpu_res = (1000 / Z) * cp.mean(
        (VI_segs - cp.mean(VI_segs, axis=0)[cp.newaxis, :]) ** 2
        + (VQ_segs - cp.mean(VQ_segs, axis=0)[cp.newaxis, :]) ** 2,
        axis=0
    )
    cpu_res = gpu_res.get()
    del gpu_res
    return cpu_res


def meas_avg_fft_squared_gpu(dig, Z, cpu_buffer=None):
    return VI_segs_fft_squared_avg, VQ_segs_fft_squared_avg


def meas_p1_spectrum_gpu(dig, Z, cpu_buffer=None):
    VI_segs, VQ_segs = measure_adc_sync_gpu(dig, cpu_buffer)
    VI = cp.fft.fftshift(
        cp.mean(
            cp.abs(cp.fft.fft(VI_segs, axis=1)) ** 2,
            axis=0
        )
    )
    VQ = cp.fft.fftshift(
        cp.mean(
            cp.abs(cp.fft.fft(VQ_segs, axis=1)) ** 2,
            axis=0
        )
    )

    VI, VQ = VI.get(), VQ.get()
    return (1 / Z) * (VI + VQ)


def meas_p2_spectrum_gpu(dig, Z, cpu_buffer=None):
    VI_segs, VQ_segs = measure_adc_sync_gpu(dig, cpu_buffer)
    VI_fft = cp.fft.fft(VI_segs, axis=1)
    del VI_segs
    VI_fft -= cp.mean(VI_fft, axis=0)[cp.newaxis, :]

    VQ_fft = cp.fft.fft(VQ_segs, axis=1)
    del VQ_segs
    VQ_fft -= cp.mean(VQ_fft, axis=0)[cp.newaxis, :]
    VI = cp.fft.fftshift(
        cp.mean(
            cp.abs(VI_fft) ** 2,
            axis=0
        )
    )
    del VI_fft

    VQ = cp.fft.fftshift(
        cp.mean(
            cp.abs(VQ_fft) ** 2,
            axis=0
        )
    )
    del VQ_fft
    # transfer to GPU
    VI, VQ = VI.get(), VQ.get()
    del VI, VQ

    return (1 / Z) * (VI + VQ)


async def acquire(devices_dict, params, queue):
    print("aquisition started")
    for i in range(params['iterations']):
        print(f"Measurement {i}")
        await get_data_from_adc_safe(devices_dict)
        print("got data")
        # await queue.put(np.frombuffer(cpu_buffer, dtype=np.int8))
        # await queue.put(transfer_to_gpu(devices_dict, params))
    print("aquisition finished")


async def process_p1(devices_dict, params, queue):
    res = cp.zeros(devices_dict['dig'].get_segment_size(), dtype=cp.float32)
    print("processing started")
    for i in range(params['iterations']):
        print(f"Iteration {i}")
        data = await queue.get()
        i_segs, q_segs = transfer_to_gpu(devices_dict, params, data)
        i_segs, q_segs = down_calibrate_on_gpu(i_segs, q_segs,
                                               params['downconv_coefficients'])
        i_segs, q_segs = downconvert(i_segs, q_segs, params)
        if 'filter' in params:
            await filter(i_segs, params)
            await filter(q_segs, params)
        power_1 = await calculate_p1(i_segs, q_segs)
        cp.add(power_1, res, out=res)
    queue.task_done()
    print('processing finished')
    return res


async def measure_p1_on_pulse_sync(devices_dict, params):
    """
    Measurement of the average field reflected from the single-photon source.

    Parameters
    ----------
    devices_dict: dict
        Dictionary with devices. It contains an awg that we use to supply
        pulses to the single-photon source and a digitizer that measures the
        transmitted field
    params: dict
        Parameters used for setup
    Returns
    -------
        Measurement class
    """
    # Reset AWG
    dev.turn_off_awg(devices_dict)

    output_pulse_sequence(devices_dict, params)
    setup_digitizer(devices_dict, params)

    res = cp.zeros(devices_dict['dig'].get_segment_size(), dtype=cp.float32)
    queue = asyncio.Queue()

    acquisition_task = asyncio.create_task(acquire(devices_dict,
                                                   params, queue))
    processing_task = asyncio.create_task(process_p1(devices_dict,
                                                     params, queue))

    # await acquisition_task
    res = await processing_task

    # params['lock'] = threading.Lock()
    #
    # def measurement(idx):
    #     i_segs, q_segs = measure_adc_sync_gpu(devices_dict, params)
    #     power_1 = calculate_p1(i_segs, q_segs)
    #     return power_1
    #     # np.add(power_1, res, out=res)
    #
    # # Measure the signal
    # print("Measuring signal")
    # with ThreadPoolExecutor(max_workers=2) as executor:
    #     powers = list(executor.map(measurement, range(params['iterations'])))
    # res = cp.sum(cp.asarray(powers, dtype=cp.float32))

    # for i in range(params["iterations"]):
    #     i_segs, q_segs = measure_adc_sync_gpu(devices_dict, params)
    #     power_1 = calculate_p1(i_segs, q_segs)
    #     np.add(power_1, res, out=res)
    #
    # # Measure the noise
    # noise = cp.zeros(devices_dict['dig'].get_segment_size(),
    #                  dtype=cp.float32)
    # devices_dict['iqawg'].output_zero(trigger_sync_every=params['period'])
    # print("Measuring noise level")
    # for i in range(params["iterations"]):
    #     i_segs, q_segs = measure_adc_sync_gpu(devices_dict, params)
    #     power_1 = calculate_p1(i_segs, q_segs)
    #     np.add(power_1, noise, out=noise)

    devices_dict['dig']._invalidate_buffer()
    global cpu_buffer
    del cpu_buffer
    print("buffer is freed")
    # gpu_mem_free()
    print("memory freed")
    return res
    # print(noise)
    # return (res.get() - noise.get()) \
    #        / (params["iterations"] * params["segments"])


def measure_p1_on_pulse(devices_dict, params):
    """
    Measurement of the average field reflected from the single-photon source.

    Parameters
    ----------
    devices_dict: dict
        Dictionary with devices. It contains an awg that we use to supply
        pulses to the single-photon source and a digitizer that measures the
        transmitted field
    params: dict
        Parameters used for setup
    Returns
    -------
        Measurement class
    """
    # Reset AWG
    dev.turn_off_awg(devices_dict)

    output_pulse_sequence(devices_dict, params)
    setup_digitizer(devices_dict, params)

    res = cp.zeros(devices_dict['dig'].get_segment_size(), dtype=cp.float32)

    for i in range(params["iterations"]):
        i_segs, q_segs = measure_adc_sync_gpu(devices_dict, params)
        power_1 = calculate_p1(i_segs, q_segs)
        np.add(power_1, res, out=res)

    # Measure the noise
    noise = cp.zeros(devices_dict['dig'].get_segment_size(),
                     dtype=cp.float32)
    # devices_dict['iqawg'].output_zero(trigger_sync_every=params['period'])
    # print("Measuring noise level")
    # for i in range(params["iterations"]):
    #     i_segs, q_segs = measure_adc_sync_gpu(devices_dict, params)
    #     power_1 = calculate_p1(i_segs, q_segs)
    #     np.add(power_1, noise, out=noise)

    devices_dict['dig']._invalidate_buffer()
    global cpu_buffer
    del cpu_buffer
    gpu_mem_free()
    # print(noise)
    return (res.get() - noise.get()) \
           / (params["iterations"] * params["segments"])


def measure_p2_on_pulse(devices_dict, params):
    """
    Measurement of the average field reflected from the single-photon source.

    Parameters
    ----------
    devices_dict: dict
        Dictionary with devices. It contains an awg that we use to supply
        pulses to the single-photon source and a digitizer that measures the
        transmitted field
    params: dict
        Parameters used for setup
    Returns
    -------
        Measurement class
    """
    # Reset AWG
    dev.turn_off_awg(devices_dict)

    output_pulse_sequence(devices_dict, params)

    devices_dict['coil'].set_current(params['shifted_current'])

    average_trace_i, average_trace_q = get_averaged_signal(devices_dict,
                                                           params)
    average_trace_i = np.reshape(average_trace_i, (1, len(average_trace_i)))
    average_trace_q = np.reshape(average_trace_q, (1, len(average_trace_q)))
    average_trace_i = cp.asarray(np.repeat(average_trace_i,
                                           params['segments'], axis=0),
                                 dtype=cp.float32)
    average_trace_q = cp.asarray(np.repeat(average_trace_q,
                                           params['segments'], axis=0),
                                 dtype=cp.float32)

    devices_dict['coil'].set_current(params['main_current'])

    setup_digitizer(devices_dict, params)

    res = cp.empty(devices_dict['dig'].get_segment_size(), dtype=cp.float32)

    for i in range(params["iterations"]):
        i_segs, q_segs = measure_adc_sync_gpu(devices_dict, params)
        power_2 = calculate_p2(i_segs, q_segs,
                               average_trace_i, average_trace_q)
        np.add(power_2, res, out=res)

    # Measure the noise
    noise = cp.zeros(devices_dict['dig'].get_segment_size(),
                     dtype=cp.float32)
    devices_dict['iqawg'].output_zero(trigger_sync_every=params['period'])
    print("Measuring noise level")
    for i in range(params["iterations"]):
        i_segs, q_segs = measure_adc_sync_gpu(devices_dict, params)
        power_1 = calculate_p1(i_segs, q_segs)
        np.add(power_1, noise, out=noise)

    print(noise)

    devices_dict['dig']._invalidate_buffer()
    global cpu_buffer
    del cpu_buffer

    data = (res.get() - noise.get()) / (params["iterations"]
                                        * params["segments"])
    gpu_mem_free()

    return data


def output_pulse_sequence(devices_dict, params):
    repetition_period = params['period']  # ns

    se_sequence_parameters = {
        "start_delay": params['awg_delay'],
        "digitizer_delay": params['dig_delay'],
        "after_pulse_delay": 0,  # ns
        "readout_duration": params['readout_duration'],  # ns
        "repetition_period": repetition_period,  # ns
        "modulating_window": params['window'],
        "window_parameter": params['window_parameter'],
        "excitation_durations": [params['pulse_length']],  # ns
        "excitation_amplitudes": [params['amplitude']],
        "pulse_sequence": ["0"],
        "periods_per_segment": 1,
        "phase_shifts": [0],
        "d_freq": 0,  # Hz, not used if `pulse_sequence` includes only zeros
    }
    pb = devices_dict['iqawg'].get_pulse_builder()
    seq = IQPulseBuilder.build_stimulated_emission_sequence(
        se_sequence_parameters, **{"q_pbs": [pb]})
    devices_dict['iqawg'].output_pulse_sequence(seq['q_seqs'][0])


def setup_digitizer(devices_dict, params):
    dig_params = {
        "channels": [0, 1],  # a list of channels to measure
        "ch_amplitude": params['dig_amp'],  # mV, amplitude for every
        # channel (allowed values are 200, 500, 1000, 2500 mV)
        "dur_seg": params['period'] - 100,  # duration of a segment in ns
        "n_seg": params['segments'],  # number of segments
        "oversampling_factor": 1,
        # sample_rate = max_sample_rate / oversampling_factor
        "n_avg": 1,
        "pretrigger": 32,  # samples
        "mode": SPCM_MODE.MULTIPLE,
        "trig_source": SPCM_TRIGGER.EXT0
    }
    devices_dict['dig'].set_parameters(dig_params)


def get_averaged_signal(devices_dict, params):
    dig_params = {
        "channels": [0, 1],  # a list of channels to measure
        "ch_amplitude": params['dig_amp'],  # mV, amplitude for every
        # channel (allowed values are 200, 500, 1000, 2500 mV)
        "dur_seg": params['period'] - 100,  # duration of a segment in ns
        "n_seg": 1,  # number of segments
        "oversampling_factor": 1,
        # sample_rate = max_sample_rate / oversampling_factor
        "n_avg": params['segments'] * params['iterations'],
        "pretrigger": 32,  # samples
        "mode": SPCM_MODE.AVERAGING,
        "trig_source": SPCM_TRIGGER.EXT0
    }
    devices_dict['dig'].set_parameters(dig_params)
    data = devices_dict['dig'].measure()
    trace = devices_dict['downconv_cal'].apply(data[0::2] + 1j * data[1::2])
    shift = params['downconversion'][0, :].get()
    trace *= shift
    trace = sg.convolve(trace, params['filter'].get(), mode="same")
    return np.real(trace), np.imag(trace)


def measure_noise_level(devices_dict, params):
    """
    Measurement of the average field reflected from the single-photon source.

    Parameters
    ----------
    devices_dict: dict
        Dictionary with devices. It contains an awg that we use to supply
        pulses to the single-photon source and a digitizer that measures the
        transmitted field
    params: dict
        Parameters used for setup
    Returns
    -------
        Measurement class
    """
    # Reset AWG
    dev.turn_off_awg(devices_dict)

    # Noise power trace
    noise = cp.zeros(devices_dict['dig'].get_segment_size(),
                     dtype=cp.float32)

    # Standard deviation of the noise power
    noise_std = np.zeros(params['iterations'], dtype=np.float32)

    # Output nothing
    devices_dict['iqawg'].output_zero(trigger_sync_every=params['period'])
    setup_digitizer(devices_dict, params)

    # Measurement
    for i in tqdm(range(params["iterations"])):
        i_segs, q_segs = measure_adc_sync_gpu(devices_dict, params)
        # print(f'here {i}')
        power_1 = calculate_p1(i_segs, q_segs)
        # print(f"calculated {i}")
        cp.add(power_1, noise, out=noise)
        noise_std[i] = (noise[40:-40] / (i + 1) / params['segments']).std()

    # Clean the buffer
    # devices_dict['dig']._invalidate_buffer()
    # global cpu_buffer
    # del cpu_buffer

    # Clean the GPU memory
    gpu_mem_free()

    return noise.get() / (params["iterations"] * params["segments"]), noise_std
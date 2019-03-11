# Measurement automation

[![CircleCI](https://circleci.com/gh/vdrhtc/Measurement-automation/tree/master.svg?style=svg)](https://circleci.com/gh/vdrhtc/Measurement-automation/tree/master) [![codecov](https://codecov.io/gh/vdrhtc/Measurement-automation/branch/master/graph/badge.svg)](https://codecov.io/gh/vdrhtc/Measurement-automation)

This is the system designed to perform various experiments with superconducting quantum circuits. It was developed in response to the increasing number and diversity of tasks and provides a common framework for all of them.

It is fully open-source and is based on pyvisa, matplotlib, scipy, ipython (jupyter) and some other open-source packages (i.e. [resonator tools](https://github.com/sebastianprobst/resonator_tools) by S. Probst)

Features:
- Percievable and scalable object-oriented code (with encapsulation, polymorphism and inheritance)
- Distinct self-sufficient classes for each of the primitive experiments and their results with common interface
- Eye-pleasing visualization using matplotlib
- Real-time plotting and data collection in separate threads
- Image fitting modules to extract parameters of flux-tunable qubits from single-tone and two-tone heatmap spectra
- Real-time robust simultaneous real-imag curve fitting in time-resolved experiments
- Human-friendly storage and loading of raw data and measurement result objects (with pickle, high-res .png and .pdf plots)
- No GUI <img src=https://user-images.githubusercontent.com/3819012/42594391-bddafb04-8557-11e8-8565-1504d9e0f3de.png width=20>


The key goal of this project is to develop measurement protocols and fully automate processes to reduce overall human participation in experiments

## Gallery

Here some of the dispersive measurement results are presented

### Single-tone spectroscopy
```python
STS = SingleToneSpectroscopy(measurement_name, sample_name, vna = ['vna1'], src=[current_src])
vna_parameters = {"bandwidth":1000, "freq_limits":(8e9, 9e9), "nop":101, "power":-15, "averages":1}
STS.set_fixed_parameters(vna = [vna_parameters])
currents = linspace(-6e-3, 8e-3, 101)
STS.set_swept_parameters({'Current': (STS._src.set_current, currents)})
sts_result = STS.launch()  # live plotting
sts_result.visualize();  # static plotting
sts_result.save()
```

![8-9ghz-random-anticrossings-shifted-w-coil](https://user-images.githubusercontent.com/3819012/42591823-8051e542-8550-11e8-8c11-3b7f65febb09.png)

 <p align="center"><i>Various anticrossings in a multimode sample</i></p>



### Two-tone spectroscopy

```python
TTS = FluxTwoToneSpectroscopy("%s-two-tone"%qubit_name, sample_name, vna ="vna4", 
                              mw_src="psg2",  current_src="yok6")
vna_parameters = {"bandwidth":100, "freq_limits":res_limits, "nop":10, "power":-10, "averages":1}
mw_src_parameters = {"power":0}
mw_src_frequencies = linspace(4.5e9, 5.8e9, 101)
currents = linspace(-0.5e-5, 1.5e-5, 101) # comment out to use fluxes from STS
TTS.set_fixed_parameters(vna = [vna_parameters], mw_src = [mw_src_parameters], adaptive=True)
TTS.set_swept_parameters(mw_src_frequencies, current_values = currents)
tts_result = TTS.launch()  # live plotting
tts_result.visualize();
tts_result.save()
```

![default](https://user-images.githubusercontent.com/3819012/42592612-b2cc0ec4-8552-11e8-9031-dd6f66b5dfcf.png)
 <p align="center"><i>Two coupled transmons</i></p>

### Time-resolved experiments

```python
awg = Tektronix_AWG5014("TCPIP::192.168.137.7::INSTR")
ro_awg = IQAWG(AWGChannel(awg, 3), AWGChannel(awg, 4))
q_awg = IQAWG(AWGChannel(awg, 1), AWGChannel(awg, 2))
DR = DispersiveRamsey("%s-ramsey"%qubit_name, sample_name, vna=["vna4"], 
                      ro_awg=[ro_awg], q_awg=[q_awg], q_lo=['psg2'])
vna_parameters = {"res_find_nop":401, "bandwidth":10, "freq_limits":res_limits, 
                  "nop":10, "averages":1}
ramsey_delays = linspace(0, 5000, 301)
exc_frequency = q_freq - 5e6
pulse_sequence_parameters = {"awg_trigger_reaction_delay":0, "readout_duration":5000, 
                             "repetition_period":15000, "half_pi_pulse_duration":pi_pulse_duration/2}
ro_awg_params =  {"calibration":ro_cal}
q_awg_params = {"calibration":q_cal}
DR.set_fixed_parameters([vna_parameters], [ro_awg_params], [q_awg_params], 
                        exc_frequency, pulse_sequence_parameters)
DR.set_swept_parameters(ramsey_delays)
dr_result = DR.launch()  # live plotting and fitting
dr_result.visualize();
dr_result.save()
```

<p align="center">
<img src=https://user-images.githubusercontent.com/3819012/42593084-20bb8c38-8554-11e8-8153-50706e86065d.png width=400><img src=https://user-images.githubusercontent.com/3819012/42593239-8ee7aaac-8554-11e8-9e5c-2a74fb548a43.png width=400></p>
 <p align="center"><i>Relaxation and dephasing</i></p>


![tomo_arb_state pdf](https://user-images.githubusercontent.com/3819012/42593726-ebdb4074-8555-11e8-86bf-956dfb715197.png)
 <p align="center"><i>Quantum state tomography (experiment and LSQ fit) for the preparation sequence +Y/4, +X/3</i></p>

# Q_Chain

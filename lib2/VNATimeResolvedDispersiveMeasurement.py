from lib2.Measurement import *
from lib2.MeasurementResult import *
from lib2.IQPulseSequence import *

from scipy.optimize import curve_fit


class VNATimeResolvedDispersiveMeasurementContext(ContextBase):

    def __init__(self):
        super().__init__()
        self._pulse_sequence_parameters = {}

    def get_pulse_sequence_parameters(self):
        return self._pulse_sequence_parameters

    def to_string(self):
        return "Pulse sequence parameters:\n" + \
               str(self._pulse_sequence_parameters) + "\n" + \
               super().to_string()


class VNATimeResolvedDispersiveMeasurement(Measurement):

    def __init__(self, name, sample_name, devs_aliases_map, plot_update_interval=1):

        super().__init__(name, sample_name, devs_aliases_map,
                         plot_update_interval=plot_update_interval)
        self._sequence_generator = None
        self._basis = None
        self._ult_calib = False
        self._pulse_sequence_parameters = \
            {"modulating_window": "rectangular", "excitation_amplitude": 1,
             "z_smoothing_coefficient": 0}

    def set_fixed_parameters(self, pulse_sequence_parameters,
                             detect_resonator=True, plot_resonator_fit=True,
                             **dev_params):
        """
        :param dev_params:
            Minimum expected keys and elements expected in each:
                'vna': 0
                'q_awg': 0
                'ro_awg': 0
        """

        # TODO check carefully. All single device functions should be deleted?
        self._pulse_sequence_parameters.update(pulse_sequence_parameters)
        self._measurement_result.get_context() \
            .get_pulse_sequence_parameters() \
            .update(pulse_sequence_parameters)

        dev_params['vna'][0]["trigger_type"] = "single"
        freq_limits = dev_params['vna'][0]["freq_limits"]

        if detect_resonator and freq_limits[0] != freq_limits[-1]:
            q_z_cal = dev_params['q_z_awg'][0]["calibration"] if \
                'q_z_awg' in dev_params.keys() else None
            res_freq = self._detect_resonator(dev_params['vna'][0],
                                              dev_params['ro_awg'][0]["calibration"],
                                              dev_params['q_awg'][0]["calibration"],
                                              q_z_cal, plot_resonator_fit=plot_resonator_fit)
            dev_params['vna'][0]["freq_limits"] = (res_freq, res_freq)

        super().set_fixed_parameters(**dev_params)

    def set_basis(self, basis):
        d_real = abs(real(basis[0] - basis[1]))
        d_imag = abs(imag(basis[0] - basis[1]))
        relation = d_real / d_imag
        if relation > 5:
            # Imag quadrature is not oscillating, ignore it by making imag
            # distance equal to ten real distances so that new normalized values
            # obtained via that component will be small
            ground_state = real(basis[0]) - 1j * 5 * d_real
            excited_state = real(basis[1]) + 1j * 5 * d_real
            basis = (ground_state, excited_state)
        elif relation < 0.2:
            # Real quadrature is not oscillating, ignore it
            ground_state = -5 * d_imag + 1j * imag(basis[0])
            excited_state = 5 * d_imag + 1j * imag(basis[1])
            basis = (ground_state, excited_state)

        self._basis = basis

    def set_ult_calib(self, value=False):
        self._ult_calib = value

    def _recording_iteration(self):
        vna = self._vna[0]
        q_lo = self._q_lo[0]
        vna.avg_clear()
        vna.prepare_for_stb()
        vna.sweep_single()
        vna.wait_for_stb()
        data = vna.get_sdata()
        if self._ult_calib:
            q_lo.set_output_state("OFF")
            vna.avg_clear()
            vna.prepare_for_stb()
            vna.sweep_single()
            vna.wait_for_stb()
            bg = vna.get_sdata()
            q_lo.set_output_state("ON")
            mean_data = mean(data) / mean(bg)
        else:
            mean_data = mean(data)
        if self._basis is None:
            return mean_data
        basis = self._basis
        p_r = (real(mean_data) - real(basis[0])) / (real(basis[1]) - real(basis[0]))
        p_i = (imag(mean_data) - imag(basis[0])) / (imag(basis[1]) - imag(basis[0]))
        return p_r + 1j * p_i

    def _detect_resonator(self, vna_parameters, ro_calibration, q_calibration,
                          q_z_calibration=None, plot_resonator_fit=True):

        self._q_lo[0].set_output_state("OFF")
        print("Detecting a resonator within provided frequency range of the VNA %s\
                    " % (str(vna_parameters["freq_limits"])))

        self._vna[0].set_nop(vna_parameters["res_find_nop"])
        self._vna[0].set_freq_limits(*vna_parameters["freq_limits"])
        self._vna[0].set_power(vna_parameters["power"])
        self._vna[0].set_bandwidth(vna_parameters["bandwidth"] * 10)
        self._vna[0].set_averages(vna_parameters["averages"])

        rep_period = self._pulse_sequence_parameters["repetition_period"]
        ro_duration = self._pulse_sequence_parameters["readout_duration"]

        ro_pb = IQPulseBuilder(ro_calibration)
        q_pb = IQPulseBuilder(q_calibration)
        self._ro_awg[0].output_pulse_sequence(ro_pb
                                              .add_dc_pulse(ro_duration).add_zero_until(rep_period).build())
        self._q_awg[0].output_pulse_sequence(q_pb.add_zero_until(rep_period).build())

        # TODO: 'and (q_z_calibration is not None)'  hotfix by Shamil (below)
        # I intend to declare all possible device attributes of the measurement class in it's child class definitions.
        # So hasattr(self, "_q_z_awg") is True
        # due to the fact that I had declared this parameter and initialized it with "[None]" in RabiFromFrequency.py
        if hasattr(self, "_q_z_awg") and (q_z_calibration is not None):
            q_z_pb = PulseBuilder(q_z_calibration)
            self._q_z_awg[0].output_pulse_sequence(q_z_pb.add_zero_until(rep_period).build())

        res_freq, res_amp, res_phase = super()._detect_resonator(plot_resonator_fit)
        print("Detected frequency is %.5f GHz, at %.2f mU and %.2f degrees" % \
              (res_freq / 1e9, res_amp * 1e3, res_phase / pi * 180))
        self._q_lo[0].set_output_state("ON")
        return res_freq

    def _output_pulse_sequence(self):

        q_pbs = [q_awg.get_pulse_builder() for q_awg in self._q_awg]
        ro_pbs = [ro_awg.get_pulse_builder() for ro_awg in self._ro_awg]

        # TODO: 'and (self._q_z_awg[0] is not None)'  hotfix by Shamil (below)
        # I intend to declare all possible device attributes of the measurement class in it's child class definitions.
        # So hasattr(self, "_q_z_awg") is True
        # due to the fact that I had declared this parameter and initialized it with "[None]" in RabiFromFrequency.py
        if hasattr(self, '_q_z_awg') and (self._q_z_awg[0] is not None):
            q_z_pbs = [q_z_awg.get_pulse_builder() for q_z_awg in self._q_z_awg]
        else:
            q_z_pbs = [None]

        pbs = {'q_pbs': q_pbs,
               'ro_pbs': ro_pbs,
               'q_z_pbs': q_z_pbs}

        seqs = self._sequence_generator(self._pulse_sequence_parameters, **pbs)

        for (seq, dev) in zip(seqs['q_seqs'], self._q_awg):
            dev.output_pulse_sequence(seq)
        for (seq, dev) in zip(seqs['ro_seqs'], self._ro_awg):
            dev.output_pulse_sequence(seq, asynchronous=False)
        if 'q_z_seqs' in seqs.keys():
            for (seq, dev) in zip(seqs['q_z_seqs'], self._q_z_awg):
                dev.output_pulse_sequence(seq, asynchronous=False)


class VNATimeResolvedDispersiveMeasurementResult(MeasurementResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._context = \
            VNATimeResolvedDispersiveMeasurementContext()
        self._is_finished = False
        self._fit_params = None
        self._fit_errors = None
        self._phase_units = "rad"
        self._data_formats = {
            "imag": (imag, "$\mathfrak{Im}[S_{21}]$ [a.u.]"),
            "real": (real, "$\mathfrak{Re}[S_{21}]$ [a.u.]"),
            "phase": (self._unwrapped_phase, \
                      r"$\angle S_{21}$ [%s]" % self._phase_units),
            "abs": (abs, r"$\left.|S_{21}|\right.$ [a.u.]")}

    def _generate_fit_arguments(self):
        """
        Should be implemented in child classes.

        Returns:
        p0: array
            Initial parameters
        scale: tuple
            characteristic scale of the parameters
        bounds: tuple of 2 arrays
            See scipy.optimize.least_squares(...) documentation
        """
        pass

    def _model(self, *params):
        """
        Fit theoretical function. Should be implemented in child classes
        """
        return None

    def _unwrapped_phase(self, sdata):
        try:
            unwrapped_phase = unwrap(angle(sdata))
            unwrapped_phase[sdata == 0] = 0
            return unwrapped_phase
        except:
            return angle(sdata)

    def _prepare_figure(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 7), sharex=True)
        fig.canvas.set_window_title(self._name)
        axes = ravel(axes)
        return fig, axes, (None, None)

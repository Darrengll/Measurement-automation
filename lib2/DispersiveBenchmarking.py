
from lib2.IQPulseSequence import *
from lib2.VNATimeResolvedDispersiveMeasurement import *
from lib2.InterleavedBenchmarkingSequenceGenerator import *

class DispersiveRandomizedInterleavedBenchmarking(VNATimeResolvedDispersiveMeasurement):

    def __init__(self, name, sample_name, vna_name, ro_awg, q_awg,
                q_lo_name, line_attenuation_db = 60):
        super().__init__(name, sample_name, vna_name, ro_awg, q_awg,
                    q_lo_name, line_attenuation_db)
        self._measurement_result = \
        DispersiveRandomizedInterleavedBenchmarkingResult(name, sample_name)

        self._sequence_generator = \
                PulseBuilder.build_interleaved_benchmarking_sequence

    def set_fixed_parameters(self, vna_parameters,
        ro_awg_parameters, q_awg_parameters, pulse_sequence_parameters,
        excitation_frequency, detect_resonator=True, basis=None):

        vna_parameters["power"] = ro_awg_parameters["calibration"]\
            .get_radiation_parameters()["lo_power"]

        q_if_frequency = q_awg_parameters["calibration"] \
            .get_radiation_parameters()["if_frequency"]

        q_lo_parameters = {"power":q_awg_parameters["calibration"]\
            .get_radiation_parameters()["lo_power"],
            "frequency":excitation_frequency+q_if_frequency}

        super().set_fixed_parameters(vna_parameters, q_lo_parameters,
            ro_awg_parameters, q_awg_parameters, pulse_sequence_parameters,
            detect_resonator)

        self._number_of_sequences = pulse_sequence_parameters["number_of_sequences"]
        self._max_sequence_length = pulse_sequence_parameters["max_sequence_length"]
        self._gate_to_benchmark = pulse_sequence_parameters["gate_to_benchmark"]
        self._rb_sequence_generator = \
            InterleavedBenchmarkingSequenceGenerator(self._number_of_sequences, \
                            self._max_sequence_length, self._gate_to_benchmark)
        self._rb_sequence_generator.generate_full_sequences()

        self._measurement_result._basis = basis

    def set_swept_parameters(self, subsequence_lengths):
        swept_pars = {"subsequence_length":\
                        (self._set_sequence_length,
                            subsequence_lengths),
                      "random_sequence_idx":\
                            (self._set_random_sequence_idx,
                            range(0, self._number_of_sequences)),
                      "is_interleaved":\
                            (self._output_pulse_sequence,
                            [False, True])}
        super().set_swept_parameters(**swept_pars)

    def _set_sequence_length(self, subsequence_length):
        self._reference_subsequences, self._interleaved_subsequences = \
            self._rb_sequence_generator.generate_partial_sequences(subsequence_length)

    def _set_random_sequence_idx(self, idx):
        self._reference_sequence = self._reference_subsequences[idx]
        self._interleaved_sequence = self._interleaved_subsequences[idx]

    def _output_pulse_sequence(self, is_interleaved):
        self._pulse_sequence_parameters["benchmarking_sequence"] =\
            self._interleaved_sequence if is_interleaved else self._reference_sequence
        super()._output_pulse_sequence()


class DispersiveRandomizedInterleavedBenchmarkingResult(VNATimeResolvedDispersiveMeasurementResult):

    def _theoretical_function(self, t, A, T_R, Omega_R, offset):
        return A*exp(-1/T_R*t)*cos(Omega_R*t)+offset

    def _generate_fit_arguments(self, x, data):
        bounds =([-10, 0, 0, -10], [10, 100, 1e3, 10])
        p0 = [-(max(data)-min(data))/2, 1, 10*2*pi, mean((max(data), min(data)))]
        return p0, bounds

    def _generate_annotation_string(self, opt_params, err):
        return "$T_R=%.2f\pm%.2f \mu$s\n$\Omega_R/2\pi = %.2f\pm%.2f$ MHz"%\
                (opt_params[1], err[1], opt_params[2]/2/pi, err[2]/2/pi)

    def _prepare_data_for_plot(self, data):
        return data[self._parameter_names[0]], data["data"]

    def _plot(self, axes, caxes):
        axes = dict(zip(self._data_formats.keys(), axes))

        data = self.get_data()
        if "data" not in data.keys():
            return

        X, Y_3D_raw = self._prepare_data_for_plot(data)


        for idx, name in enumerate(self._data_formats.keys()):
            ax = axes[name]
            ax.clear()

            for is_interleaved in range(0,2):
                for subseq_idx in range(Y_3D_raw.shape[0]):
                    samples_data =\
                        self._data_formats[name][0](Y_3D_raw[subseq_idx,
                                                            :, is_interleaved])
                    samples_data = samples_data[samples_data!=0]
                    if len(samples_data) > 0:
                        offset = 0.1 if is_interleaved else -0.1
                        ax.plot(ones_like(samples_data)*X[subseq_idx]+offset,
                            samples_data, "C%d."%is_interleaved, alpha=0.25,
                            ms=4, zorder=1)

                Y_raw = nan_to_num(true_divide(Y_3D_raw.sum(1),
                                (Y_3D_raw!=0).sum(1)))[:, is_interleaved]

                Y = self._data_formats[name][0](Y_raw)
                Y = Y[Y!=0]
                ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
                ax.plot(X[:len(Y)], Y, "C%d"%is_interleaved, ls=":", marker="o",
                                                markerfacecolor='none', zorder=2)

            if self._basis is not None:
                ax.plot(X, ones_like(X)*self\
                    ._data_formats[name][0](self._basis[0]), '--', color="black")
                ax.plot(X, ones_like(X)*self\
                    ._data_formats[name][0](self._basis[1]), "--", color="black")

            ax.set_xlim(X[0], X[-1])
            ax.set_ylabel(self._data_formats[name][1])
            ax.grid()

        xlabel = self._parameter_names[0][0].upper()+\
                    self._parameter_names[0][1:].replace("_", " ")

        axes["phase"].set_xlabel(xlabel)
        axes["imag"].set_xlabel(xlabel)

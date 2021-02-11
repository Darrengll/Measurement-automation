from numpy import *
from lib2.QuantumState import *
from random import choice


class InterleavedBenchmarkingSequenceGenerator():

    def __init__(self, number_of_sequences=2, max_sequence_length=10,
                 gate_to_benchmark="X/2"):

        self._gate_to_benchmark = gate_to_benchmark
        self._number_of_sequences = number_of_sequences
        self._cliffords = ["I", "X/2", "Y/2"]
        self._max_sequence_length = max_sequence_length

    def generate_full_sequences(self):
        self._reference_sequences = \
            [self._generate_reference_sequence() \
             for i in range(self._number_of_sequences)]
        self._interleaved_sequences = \
            [self._generate_interleaved_sequence(sequence) \
             for sequence in self._reference_sequences]

    def generate_partial_sequences(self, subsequence_length):
        recovered_reference_sequences = \
            [self._calculate_and_insert_recovery_gate(sequence[:subsequence_length]) \
             for sequence in self._reference_sequences]
        recovered_interleaved_sequences = \
            [self._calculate_and_insert_recovery_gate(sequence[:subsequence_length * 2]) \
             for sequence in self._interleaved_sequences]

        return recovered_reference_sequences, recovered_interleaved_sequences

    def _generate_reference_sequence(self):
        """
        generates a random sequence of pi/2 gates of desirable length.
        output is in the format ["-X", "+X/2", "-Z", "-Y", ...]
        """
        signs = ["+", "-"]
        return [choice(signs) + choice(self._cliffords) for i in range(self._max_sequence_length)]

    def _generate_interleaved_sequence(self, sequence):
        """
        Places the gate which was passed to the constructor after each gate in
        the sequence and returns the result.
        """
        interleaved_sequence = []
        for gate in sequence:
            interleaved_sequence.append(gate)
            interleaved_sequence.append(self._gate_to_benchmark)
        return interleaved_sequence

    def _calculate_and_insert_recovery_gate(self, sequence):
        """
        Calculates the final quantum state of a qubit after all the pulses in the
        given sequence are applied to |0> state. Then defines which gate has to
        be applied in order to return the qubit to |0> state. Adds this gate
        to the end of the given sequence and returns the result.

        """
        qs = QuantumState("pulses", sequence)
        qs.change_represent("dens_mat")
        dm = qs._coords
        projections = array([trace(dm.dot(sig_x)),
                             trace(dm.dot(sig_y)),
                             trace(dm.dot(sig_z))])

        max_projection_idx = argmax(abs(projections))
        projection_axis = ["X", "Y", "Z"][max_projection_idx]
        if projection_axis == 'Z':
            if projections[max_projection_idx] > 0:
                return sequence + [choice(["+X", "-X", "+Y", "-Y"])]
            else:
                return sequence + ["+I"]
        elif projection_axis == "X":
            sign = "-" if projections[max_projection_idx] < 0 else "+"
            return sequence + [sign + "Y" + "/2"]
        elif projection_axis == "Y":
            sign = "+" if projections[max_projection_idx] < 0 else "-"
            return sequence + [sign + "X" + "/2"]

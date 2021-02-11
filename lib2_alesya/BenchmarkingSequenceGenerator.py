from numpy import *
from lib2.QuantumState import *
from random import choice

class BenchmarkingSequenceGenerator():

    def __init__(self, N_seqs=2, lk_array=linspace(1,10,10), N_e=10, gate="X/2"):
        self._gate = gate
        self._N_seqs = N_seqs
        self._axis_string = ["I","I/2","X","X/2","Y","Y/2","Z","Z/2"]
        self._lk_array = lk_array
        self._seq_length = int(max(self._lk_array))
        self._reference_sequences =\
            [self._generate_reference_sequence() for i in range(N_seqs)]
        self._interleaved_sequences = [self._generate_interleaved_sequence(sequence)\
                for sequence in self._reference_sequences]

    def _generate_reference_sequence(self):
        """
        generates a random sequence of pi/2 gates of desirable length.
        output is in the format ["-X", "+X/2", "-Z", "-Y", ...]
        """
        signs = ["+", "-"]
        return [choice(signs)+choice(self._axis_string) for i in range(self._seq_length)]

    def _generate_interleaved_sequence(self, sequence):
        """
        """
        interleaved_sequence = []
        for gate in sequence:
            interleaved_sequence.append(gate)
            interleaved_sequence.append(self._gate)
        return interleaved_sequence

    def _calc_and_insert_recovery_gate(self, sequence):
        """
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
            if projections[max_projection_idx]>0:
                return sequence+[choice(["+X", "-X", "+Y", "-Y"])]
            else:
                return sequence+[choice(["+I", "-I", "+I/2", "-I/2"])]
        elif projection_axis == "X":
            sign = "-" if projections[max_projection_idx]<0 else "+"
            return sequence+[sign+"Y"+"/2"]
        elif projection_axis == "Y":
            sign = "+" if projections[max_projection_idx]<0 else "-"
            return sequence+[sign+"X"+"/2"]

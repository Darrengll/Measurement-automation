# This file is to implement the preparation and tomography of arbitrary quantum states.
from numpy import *
from scipy.linalg import expm

eye = matrix([[1, 0], [0, 1]])
sig_x = matrix([[0, 1], [1, 0]])
sig_y = matrix([[0, -1j], [1j, 0]], dtype=complex)
sig_z = matrix([[1, 0], [0, -1]])

operators = {"I": eye, "X": sig_x, "Y": sig_y, "Z": sig_z}
signs = {"+": 1, "-": -1}


def matrix_from_gate(gate):
    angle = pi * eval(gate.replace(gate[1], "1"))
    return expm(-1j * operators[gate[1]] * angle / 2)


class QuantumState():
    """
    The instances of that class are quantum states to be prepared on
    the Bloch sphere.

    Methods:
    --------------
        __init__(self, represent='spherical', coords = [90,90]):
                Initialization of the state

        check_norm(coords):
                The normalizarion is checked before creation of quantum state.
                The method is static.
    """

    def __init__(self, represent='spherical', coords=[1.0, pi / 2, 0]):
        """
        The creation of QuantumState instance.

        Parameters:
        -----------------
        represent:  (defined by) "spherical", "pulses", "dens_mat" or "bloch"
            defines  how to encode a quantum state.

            "spherical" are in format [r, phi, teta] - list of floats
            where phi is from X axis and teta is from |0> state, in radians:
            Zero state is [1,pi,0] or [1,pi, 0], sigma_x state is [1,pi/2,0], sigma_y state is
            [1,pi/2, pi], sigma_z state is [1,pi,0]. Angles are not restricted and are to be (-inf, inf)
            for future applications.

            "pulses" are in terms of pulse which is required to prepare the state from zero state.
            Examples: [('X', 90), ('Y', 180)]. The best representation for constructing a pulse sequence.

            "dens_mat" is the density matrix (numpy.matrix 2x2) of a given state. It is related with the bloch components as:
                rho = 1/2(Unity + x*sigma_x + y*sigma_y + z*sigma_z)

            "bloch" are bloch components of a quantum state on sigma axes.
            format [sig_x, sig_y, sig_z], where sig_i lays in [-1,1]. - a list from three floats.

        coords: depends on the represent (see higher)

        """
        self._norm = None
        self._coords = coords
        self._represent = represent
        self._calc_norm()
        self._check_norm()

    def _calc_norm(self):
        if self._represent == 'bloch':
            self._norm = float(sqrt(sum(array(self._coords) ** 2)))
        elif self._represent == "dens_mat":
            self._norm = 2 * float((self._coords ** 2).trace) - 1
        elif self._represent == "spherical":
            self._norm = self._coords[0]
        elif self._represent == 'pulses':
            self._norm = 1.0
            self._check_pulse_axis()
        else:
            raise ValueError("Used representation of quantum state is not allowed")

    def _check_norm(self, digit_error=1e-4):
        if self._norm > 1.0 + digit_error:
            raise ValueError("Your norm is >1.0 for your state. Look for a mistake.")

    def is_on_sphere(self, digit_error=1e-4):
        """
        Raises an exception if the quantum state is not on the Bloch Sphere (if needed).
        (only needed for bloch and dens_mat representation)
        """

        if self._norm <= 1.0 - digit_error:
            return True
        else:
            return False

    def extend_to_sphere(self):
        pass

    def _change_state(self, new_coords=None, new_represent=None):
        """

        """
        if (new_coords is not None) and (new_represent is not None):
            self._coords = new_coords
            self._represent = new_represent
        else:
            pass

    def _check_pulse_axis(self):
        """
        Checks for proper axis naming for pulses representation. (X,Y or Z)
        """
        for pulse in self._coords:
            if pulse[1] not in "XYZI":
                raise ValueError("For one of the pulses axis is not valid. Check your state.")

    def change_represent(self, repr_to="pulses"):
        """
        Change the representation and accordingly, coordinates
        of a given quantum state (not creating new object)
        """
        repr_from = self._represent
        new_represent = repr_to
        if repr_from == "bloch":
            old_x, old_y, old_z = self._coords
            if repr_to == "dens_mat":
                new_coords = 0.5 * (eye + old_x * sig_x + old_y * sig_y + old_z * sig_z)
            elif repr_to == "spherical":
                new_r = self._norm
                new_phi = arctan2(real(old_y), real(old_x))
                new_theta = arctan2(real(old_z), sqrt(real(old_y) ** 2 + real(old_x) ** 2))
                new_coords = [new_r, new_theta, new_phi]
            elif repr_to == "pulses":
                self.change_represent("spherical")
                self.change_represent("pulses")
                return True
            else:
                raise ValueError("New representation is invalid.")
        if repr_from == "spherical":
            old_r, old_theta, old_phi = self._coords
            if repr_to == "bloch":
                new_x = old_r * cos(old_theta) * cos(old_phi)
                new_y = old_r * cos(old_theta) * sin(old_phi)
                new_z = old_r * sin(old_theta)
                new_coords = [new_x, new_y, new_z]
            if repr_to == "dens_mat":
                new_dm = 0.5 * (eye + cos(old_phi) * cos(old_theta) * sig_x + \
                                sin(old_phi) * cos(old_theta) * sig_y + \
                                sin(old_theta) * sig_z)
                new_coords = new_dm
            if repr_to == "pulses":
                if self.is_on_sphere() == False:
                    self.extend_to_sphere()
                # new_coords = [old_theta,old_theta,('Z',old_phi)]
        if repr_from == "dens_mat":
            old_dm = self._coords
            if repr_to == "bloch":
                new_x = 2 * real(old_dm[0, 1])
                new_y = 2 * imag(old_dm[1, 0])
                new_z = old_dm[0, 0] - old_dm[1, 1]
                new_coords = [new_x, new_y, new_z]
            if repr_to == "spherical":
                pass
        if repr_from == "pulses":
            if repr_to == "dens_mat":
                state = array([[0], [1]])
                for gate in self._coords:
                    state = matrix_from_gate(gate).dot(state)
                new_coords = state.dot(state.conj().T)
            if repr_to == 'spherical':
                self.change_represent('dens_mat')
                self.change_represent('bloch')
                self.change_represent('spherical')
                return True
        if self._represent != new_represent:
            self._change_state(new_coords, repr_to)
            self._represent = new_represent

    def __str__(self):
        return "A QuantumState object: \"" + self._represent + "\" representation: " + \
               str(self._coords) + "."

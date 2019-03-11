"""
Possible types are:
pna-p1D-2D : 1D parameter set with 1 point sweep of PNA ---------------------------------------- 1
pna-p2D-2D : 2D parameter set with 1 point sweep of PNA and 1 point of one of the parameters --- 4
pna-p1D-3D : 1D parameter set with many-point sweep of PNA ---------------------------------------- 2
pna-p2D-3D : 2D parameter set with 1 point sweep of PNA ---------------------------------------- 3

Numbers needed for backwards compatibility with Python 3 pickle and because of lack of enum class
"""

class Measurement():

    TYPES = {"pna-p1D-2D":1, "pna-p1D-3D":2, "pna-p2D-3D":3, "pna-p2D-2D":4}

    def __init__(self, type = None, data=None, datetime=None, context={}, recording_time=0):
        self.__type__ = type
        self.__data__ = data
        self.__context__ = context
        self.__datetime__ = datetime
        self.__recording_time__ = recording_time


    def get_type_str(self):
        if self.__type__ == 1:
            return "pna-p1D-2D"
        elif self.__type__ == 2:
            return "pna-p1D-3D"
        elif self.__type__ == 3:
            return "pna-p2D-3D"
        elif self.__type__ == 4:
            return "pna-p2D-2D"

    def get_datetime(self):
        return self.__datetime__

    def set_datetime(self, datetime):
        self.__datetime__ = datetime

    def get_recording_time(self):
        return self.__recording_time__

    def set_recording_time(self, recording_time):
        self.__recording_time__ = recording_time

    def get_data(self):
        return self.__data__

    def set_data(self, data):
        self.__data__ = data

    def get_type(self):
        return self.__type__

    def set_type(self, type):
        self.__type__ = type

    def set_context(self, context):
        """
        Sets the measurement's context, i.e. power, averages and bandwidth
        """
        self.__context__ = context

    def get_context(self):
        """
        Returns the measurement's context, i.e. power, averages and bandwidth
        """
        return self.__context__

    def copy(self):
        """
        Returns a shallow copy!
        """
        copy = Measurement()
        copy.set_data(self.__data__)
        copy.set_type(self.__type__)
        copy.set_datetime(self.__datetime__)
        copy.set_context(self.__context__)
        return copy

    def normalize(self):
        """
        For each value on y axis divides each z value for that y value by the average of all
        z values for that y value (in absolute scale)

        Returns: measurement : Measurement
                                    normalized measurement
        """
        new = self.copy()
        if self.__type__ in [2,3]:
            new_amps = new.get_data()[2].copy()
            new_phas = new.get_data()[3].copy()
            for i in range(0, new_amps.shape[1]):
                new_amps[:,i] = new_amps[:,i]-new_amps[:,i].mean()
                new_phas[:,i] = new_phas[:,i]-new_phas[:,i].mean()
            new.set_data((self.__data__[0], self.__data__[1], new_amps, new_phas))
            return new
        else:
            print("Measurement type is not supported")

    def remove_background(self, sweep_number, axis="y"):
        """
        Subtracts a specified sweep from every sweep and returns a measurement with new data
        WARNING Now it works only for NxM plots where N!=M

        Parameters: sweep_number : int
                                    specifies the number of parameter
                                    in a parameter list corresponding to the needed sweep
				    axis : string
				                    the axis subtracted sweep is parallel to, "x" or "y"
        Returns:    measurement : Measurement
                                    measurement with no background
        """
        new = self.copy()

        if self.__type__ == 2:
            if axis == "y":
                new_data = self.__data__[0], self.__data__[1], self.__data__[2] - self.__data__[2][sweep_number], self.__data__[3] - self.__data__[3][sweep_number]
                new.set_data(new_data)
            elif axis=="x":
                new_data = self.__data__[0], self.__data__[1], (self.__data__[2].T - self.__data__[2][:, sweep_number]).T, (self.__data__[3].T - self.__data__[3][:, sweep_number]).T
                new.set_data(new_data)
        elif self.__type__ == 3:
            if axis == "y":
                new_data = self.__data__[0], self.__data__[1], self.__data__[2], \
                    self.__data__[3] - self.__data__[3][sweep_number], self.__data__[4] - self.__data__[4][sweep_number]
                new.set_data(new_data)
            elif axis == "x":
                new_data = self.__data__[0], self.__data__[1], self.__data__[2], \
                    (self.__data__[3].T - self.__data__[3][:, sweep_number]).T, (self.__data__[4].T - self.__data__[4][:, sweep_number]).T
                new.set_data(new_data)
        return new

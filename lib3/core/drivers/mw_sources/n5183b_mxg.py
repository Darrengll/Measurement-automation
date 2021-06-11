from .n5173b_exg import N5173B


class N1583B(N5173B):
    def set_power(self, power_dbm):
        if (power_dbm >= -20) & (power_dbm <= 19):
            self.write(":SOURce:POWer {0}DBM".format(power_dbm))
        else:
            print("Error: power must be between -20 and 19 dBm")

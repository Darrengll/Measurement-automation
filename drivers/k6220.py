from  drivers import instr
from time import sleep

class K6220(instr.Instr):
    def __init__(self, visa_name):
        super(K6220, self).__init__(visa_name)
        self._visainstrument.read_termination = '\n'
        self._visainstrument.write_termination = '\n'
        self._visainstrument.baud_rate = 9600
        self._visainstrument.chunk_size = 2048*8

        self.current_range = 0   # TO READ FROM DEVICE AT INIT

        self.min_request_delay = 0.01   # minium time between two requests in a loop with a sleep()
        self.sweep_rate = 1.e-3   # A/s
        self.sweep_min_delay = 0.001
        self.sweep_nb_points = 101
        self.last_sweep_current_init = None
        self.last_sweep_current_final = None
        self.last_sweep_time = None
        self.last_sweep_delay = None
        self.last_sweep_nb_points = None
        self.last_sweep_step = None
        self.last_sweep_finished = True


    def get_range(self):
        return float(self.query("SOUR:CURR:RANG?"))

    def get_compliance(self):
        return float(self.query("SOUR:CURR:COMP?"))


    def set_compliance(self, voltage):
        if abs(voltage) <= 10:
            self.write("SOUR:CURR:COMP {0}".format(voltage))
        else:
            print("Error: compliance voltage should be <= 10V.")

    def set_range(self, current_range): # use 0 for AUTO RANGE
        if current_range == 0:
            self.write("SOUR:CURR:RANG:AUTO ON")
            self.current_range = 0
        elif abs(current_range) > 0 and abs(current_range) < 105e-3:
            self.write("SOUR:CURR:RANG:AUTO OFF")
            self.write("SOUR:CURR:RANG {0}".format(current_range))
            self.current_range = current_range

    def output_off(self):
        self.write("OUTPut OFF")

    def output_on(self):
        self.write("OUTPut ON")

    def set_current(self, current):
        if (abs(current) <= self.get_range()):
            self.write("SOUR:CURR:AMPL {0}".format(current))
        else:
            print("Given current %f is out of range %f"%(current, self.get_range()))

    def get_current(self):
        bla = self.query("SOUR:CURR:AMPL?")
        try:
            output = float(bla)
        except:
            print("Error in get_current. Value read: {0}".format(bla))
            output = None
        return output

    def get_last_error(self):
        return self.query("SYST:ERR?")



    def reset(self):
        self.write("*RST")
        self.write("*CLS")


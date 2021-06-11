from time import sleep

class K6220(object):
    def __init__(self, pna_instance, gpib_address):
        self.pna = pna_instance
        self.gpib_address = gpib_address
        self.handle = self.pna.gpib_bridge_open(self.gpib_address)
        self.current_range = 0   # TO READ FROM DEVICE AT INIT
        self.min_current = -105.e-3  # A
        self.max_current =  105.e-3  # A
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

    def __del__(self):
        self.pna.gpib_bridge_close(self.handle)

    def ask(self, command):
        return self.pna.gpib_bridge_ask(self.handle, command)
        # print command

    def write(self, command):
        self.pna.gpib_bridge_write(self.handle, command)
        # print command

    def read(self):
        return self.pna.gpib_bridge_read(self.handle)

    def set_limits(self, min_current, max_current):
        if max_current >= min_current:
            self.min_current = min_current
            self.max_current = max_current
        else:
            print "Error: upper limit should be greater than lower limit."

    def get_limits(self):
        return [self.min_current, self.max_current]

    def get_range(self):
        pass    # TODO

    def set_compliance_voltage(self, voltage):
        if abs(voltage) <= 10:
            self.write("SOUR:CURR:COMP {0}".format(voltage))
        else:
            print "Error: compliance voltage should be <= 10V."

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

    def set_current_instant(self, current):
        if current <= self.max_current and current >= self.min_current:
            self.write("SOUR:CURR:AMPL {0}".format(current))
        else:
            print "Error: bias out of range. use set_limits(min,max)."

    def get_current(self):
        bla = self.ask("SOUR:CURR:AMPL?")
        try:
            output = float(bla)
        except:
            print "Error in get_current. Value read: {0}".format(bla)
            output = None
        return output

    def get_last_error(self):
        return self.ask("SYST:ERR?")

    def set_sweep_rate(self, sweep_rate):
        if sweep_rate > 0 and sweep_rate <= 0.1:
            self.sweep_rate = sweep_rate
        else:
            print "If you want a sweep rate higher than 0.1 A/s, zero or negative, please change the program..."

    def set_current(self, current):
        if current <= self.max_current and current >= self.min_current:
            # nb_steps = 50
            self.last_sweep_current_final = current
            self.last_sweep_current_init = self.get_current()
            if self.last_sweep_current_init != self.last_sweep_current_final:
                self.last_sweep_time = (self.last_sweep_current_final - self.last_sweep_current_init) / self.sweep_rate
                step = (self.last_sweep_current_final - self.last_sweep_current_init) / (self.sweep_nb_points - 1)
                delay = self.last_sweep_time / (self.sweep_nb_points - 1)
                if delay > self.sweep_min_delay:
                    self.last_sweep_delay = delay
                    self.last_sweep_nb_points = self.sweep_nb_points
                    self.last_sweep_step = step
                else:
                    self.last_sweep_delay = self.sweep_min_delay
                    self.last_sweep_nb_points = self.last_sweep_time / self.last_sweep_delay
                    self.last_sweep_step = (self.last_sweep_current_final - self.last_sweep_current_init) / self.last_sweep_nb_points
                self.write("SOUR:SWE:SPAC LIN")
                self.write("SOUR:CURR:STAR {0}".format(self.last_sweep_current_init))
                self.write("SOUR:CURR:STOP {0}".format(self.last_sweep_current_final))
                self.write("SOUR:CURR:STEP {0}".format(self.last_sweep_step))
                self.write("SOUR:DEL {0}".format(self.last_sweep_delay))
                self.write("SOUR:SWE:RANG FIX")
                self.write("SOUR:SWE:COUN 1")
                self.write("SOUR:SWE:ARM")
                self.write("INIT")
                self.last_sweep_finished = False
        else:
            print "Error: Current must be in the range {0} to {1} mA".format(self.min_current, self.max_current)

    def wait_for_sweep(self):
        while not self.last_sweep_finished:
            if self.last_sweep_current_final == self.get_current():
                self.last_sweep_finished = True
            else:
                sleep(min(self.last_sweep_delay, self.min_request_delay))

    def abort_sweep(self):
        self.write("SOUR:SWE:ABOR")
        self.last_sweep_finished = True

    def reset(self):
        self.write("*RST")
        self.write("*CLS")
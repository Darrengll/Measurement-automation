from  drivers import instr
from time import sleep

class K2400(instr.Instr):
    def __init__(self, visa_name):
        super(K2400, self).__init__(visa_name)
        self.visa_instr.read_termination = '\n'
        self.visa_instr.write_termination = '\n'
        self.visa_instr.baud_rate = 9600
        self.visa_instr.chunk_size = 2048*8

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

    def set_current_compliance(self, current):
        self.write("SENSE:CURR:PROT {0}".format(current))

    def set_voltage_compliance(self, voltage):
        self.write("SENSE:VOLT:PROT {0}".format(voltage))

    def get_current_compliance(self):
        return self.query("SENSE:CURR:PROT?")

    def get_voltage_compliance(self):
        return self.query("SENSE:VOLT:PROT?")


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

    #def output_stat(self):
     #   return self.query("OUTPut:STAT?")
        #return self.query("OUTPut:[STAT]?")
        #return self.query("SOUR:OUTPut:STAT?")
        #return self.query("OUTPut:STAT?")
        #self.query("OUTPut:STAT?")
        #self.write("OUTPut:STAT?")
        #self.write("OUTPut_STAT?")
    #OUTPut [:STAT]?

    def set_mode(self, mode):
        """
        Sets the mode of the device.

        Parameters:
        mode : string
            VOLT for V-source or CURR for I-source
        """
        self.write("SOUR:FUNC %s"%mode)

    def get_mode(self):
        """
        Gets the mode of the device.
        """
        return self.query("SOUR:FUNC?")

    def set_current(self, current):
        if (abs(current) <= self.get_range()):
            self.write("SOUR:CURR {0}".format(current))
        else:
            print("Given bias %f is out of range %f"%(current, self.get_range()))

    def get_current(self):
        bla = self.query("SOUR:CURR?")
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

    #def goto_current(current_aim):
    #mykei = k6220.K6220("GPIB0::12::INSTR")
    #self.set_limits(-0.04,0.04)
    #self.write("SOUR:CURR:COMP {0}".format(10))

    #current_now = mykei.get_current()
    #step = 1e-6    # шаг, с которым пойдем по току = 10мкА
    ######################
    ## проверка на значение задаваемого тока
    #if abs(current_aim) > 10e-3:
    #    print("error: Current, that you try to set is too much")
    #    return 0
    ######################
    #if mykei.get_current() == 0:
    #    mykei.output_on()
    #    sleep(0.1)
    #if mykei.output_stat == 0:
        #mykei.set_current_instant(0)
        #mykei.output_on()
    ######################
    # вдруг, разница небольшая и можно сразу задать
    #if abs(current_aim - current_now) < 3*step:
    #    mykei.set_current_instant(current_aim)
    #    if current_aim == 0:
    #        #mykei.output_off()
    #    return 1
    ######################
    ##число шагов до цели
    #num_step = int(np.round((abs(current_aim - current_now) / step),0))
    #if current_aim < current_now:
    #    step = -step #направление движения по току
    ## пошли менять ток
    #for i in range(1,num_step):
    #    current_now = np.round((current_now + step),9)
    #    mykei.set_current_instant(current_now)
    #    print(mykei.get_current())
    #    #print("Current: {0:.2f} A".format(mykei.get_current()))
    #    sleep(0.2)
    ## последний штрих, на случай, если чуть-чуть не попали
    #if abs(mykei.get_current() - current_aim) < 3*abs(step):
    #    mykei.set_current_instant(current_aim)
    #    # если ток = 0, стоит вырубить прибор
    #else: # паника, если сильно промахнулись
    #    print("error: miss!")
    #    if abs(mykei.get_current()) > 3e-3:
    #        #mykei.output_off()
    #    return 0
    #if mykei.get_current() == 0:
    #    #mykei.output_off()
    #    return 1

import pyvisa
from matplotlib._pylab_helpers import Gcf

from drivers import *
from datetime import datetime as dt
from threading import Thread

from lib2.MeasurementResult import MeasurementResult
from lib2.ResonatorDetector import *
from lib2.GlobalParameters import *
from itertools import product
from functools import reduce
from operator import mul
from matplotlib import pyplot as plt
import sys
from numpy import zeros, complex_
from lib2.GlobalParameters import *

from loggingserver import LoggingServer


class Measurement:
    """
    Any inheritance?
    The class contains methods to help with the implementation of measurement classes.

    """
    _actual_devices = {}
    _log = []
    _devs_dict = \
        {'vna1': [["PNA-L", "PNA-L1"], [Agilent_PNA_L, "Agilent_PNA_L"]],
         'vna2': [["PNA-L-2", "PNA-L2"], [Agilent_PNA_L, "Agilent_PNA_L"]],
         'vna3': [["pna"], [Agilent_PNA_L, "Agilent_PNA_L"]],
         'vna4': [["ZNB"], [znb, "Znb"]],
         'exa': [["EXA"], [Agilent_EXA, "Agilent_EXA_N9010A"]],
         'exg': [["EXG"], [E8257D, "EXG"]],
         'psg2': [['PSG'], [E8257D, "EXG"]],
         'mxg': [["MXG"], [E8257D, "MXG"]],
         'psg1': [["psg1"], [E8257D, "EXG"]],
         'awg1': [["AWG", "AWG1"], [KeysightAWG, "KeysightAWG"]],
         'awg2': [["AWG_Vadik", "AWG2"], [KeysightAWG, "KeysightAWG"]],
         'awg3': [["AWG3"], [KeysightAWG, "KeysightAWG"]],
         'awg4': [["TEK1"], [Tektronix_AWG5014, "Tektronix_AWG5014"]],
         'dso': [["DSO"], [Keysight_DSOX2014, "Keysight_DSOX2014"]],
         'yok1': [["GS210_1"], [Yokogawa_GS200, "Yokogawa_GS210"]],
         'yok2': [["GS210_2"], [Yokogawa_GS200, "Yokogawa_GS210"]],
         'yok3': [["GS210_3"], [Yokogawa_GS200, "Yokogawa_GS210"]],
         'yok4': [["gs210"], [Yokogawa_GS200, "Yokogawa_GS210"]],
         'yok5': [["GS_210_3"], [Yokogawa_GS200, "Yokogawa_GS210"]],
         'yok6': [["YOK1"], [Yokogawa_GS200, "Yokogawa_GS210"]],
         'k6220': [["k6220"], [k6220, "K6220"]]
         }

    def __init__(self, name, sample_name, devs_aliases_map, plot_update_interval=5):
        """
        Parameters:
        --------------------
        name: string
            name of the measurement
        sample_name: string
            the name of the sample that is measured
        devs_aliases_map: dictionary
            with devices' standard names(if it`s not virtual device) or
             object from a device class. A key is a string that will be an object field
        --------------------

        Constructor creates variables for devices passed to it and initialises all devices.

        Standard names of devices within this driver are:

            'vna1',vna2','exa','exg','mxg','awg1','awg2','awg3','dso','yok1','yok2','yok3'

        with _ added in front for a variable of a class

        if key is not recognised, do not return an error

        """

        self._logger = LoggingServer.getInstance('manual_meas')

        self._logger.debug("Measurement " + name + " init, devs: "+ str(devs_aliases_map))

        self._interrupted = False
        self._name = name
        self._sample_name = sample_name
        self._plot_update_interval = plot_update_interval
        self._resonator_detector = ResonatorDetector()
        if GlobalParameters().resonator_types['reflection'] == True:
            self._resonator_detector = ResonatorDetector(type= 'reflection')
        else:
            self._resonator_detector = ResonatorDetector(type = 'transmission')

        self._devs_aliases_map = devs_aliases_map
        self._list = ""
        try:
            rm = pyvisa.ResourceManager()
            # returns list of tuples: (IP Address string, alias) for all
            # devices present in VISA
            temp_list = list(rm.list_resources_info().values())
            self._devs_info = [item[4] for item in list(temp_list)]
        except ValueError:
            print("NI Visa implementation not found; automatic device discovery unavailable")
        
        for field_name, dev_list in self._devs_aliases_map.items():
            atr_name = "_" + field_name
            self.__setattr__(atr_name, [None] * len(dev_list))
            for index, value in enumerate(dev_list):
                if isinstance(value, str):
                    name = value
                    if name in Measurement._actual_devices.keys():
                        print(name + ' is already initialized')
                        device_object = Measurement._actual_devices[name]
                        self.__getattribute__(atr_name)[index] = device_object
                        continue
                    if name in Measurement._devs_dict.keys():
                        for device_address in self._devs_info:
                            if device_address in Measurement._devs_dict[name][0]:
                                # print(name, device_address)
                                device_object = getattr(*Measurement._devs_dict[name][1])(device_address)
                                Measurement._actual_devices[name] = device_object
                                print("The device %s is detected as %s" % (name, device_address))
                                self.__getattribute__(atr_name)[index] = device_object
                                break
                    else:
                        print("Device", name, "is unknown!")
                else:
                    self.__getattribute__(atr_name)[index] = value

    @staticmethod
    def close_devs(devs_to_close):
        for name in devs_to_close:
            if name in Measurement._actual_devices.keys():
                Measurement._actual_devices.pop(name)._visainstrument.close()

    def _load_fixed_parameters_into_devices(self):
        """
        exa_parameters
        fixed_pars: {'dev1': {'par1': value1, 'par2': value2},
                     'dev2': {par1: value1, par2: ...}...}
        """
        for dev_name in self._fixed_pars.keys():
            dev_list = getattr(self, '_' + dev_name)
            for pars, dev in zip(self._fixed_pars[dev_name], dev_list):
                dev.set_parameters(pars)


    def set_fixed_parameters(self, **fixed_pars):
        """
        fixed_pars: {'dev1': {'par1': value1, 'par2': value2},
                     'dev2': {par1: value1, par2: ...},...}
        """
        self._fixed_pars = fixed_pars
        for dev_name in self._fixed_pars.keys():
            self._measurement_result.get_context().get_equipment()[dev_name] = fixed_pars[dev_name]
        self._load_fixed_parameters_into_devices()

    def set_swept_parameters(self, **swept_pars):
        """
        swept_pars :{'par1': (setter1, [value1, value2, ...]),
                     'par2': (setter2, [value1, value2, ...]), ...}
        """
        self._swept_pars = swept_pars
        self._swept_pars_names = list(swept_pars.keys())
        self._measurement_result.set_parameter_names(self._swept_pars_names)
        self._last_swept_pars_values = \
            {name: None for name in self._swept_pars_names}

    def _call_setters(self, values_group):
        for name, value in zip(self._swept_pars_names, values_group):
            if self._last_swept_pars_values[name] != value:
                self._last_swept_pars_values[name] = value
                self._swept_pars[name][0](value)  # this is setter call, look carefully

    def launch(self):

        self._interrupted = False  # ensure

        self._measurement_result.set_start_datetime(dt.now())
        if self._measurement_result.is_finished():
            print("Starting with a result from a previous launch")

        print("Started at: ", self._measurement_result.get_start_datetime())
        t = Thread(target=self.measure)
        t.start()

        self._measurement_result.visualize_dynamic()
        self.join()

        return self._measurement_result

    def join(self):
        stop_messages = {KeyboardInterrupt: "\nMeasurement interrupted!",
                         AttributeError: "\nPlot has been closed, aborting!"}
        figure_number = self._measurement_result.get_figure_number()
        try:
            # wait for the measurement to end or for an interrupt
            while not self._measurement_result.is_finished():
                # check if the window is still there
                manager = Gcf.get_fig_manager(figure_number)
                manager.canvas.start_event_loop(.1)
        except (KeyboardInterrupt, AttributeError) as e:
            print(stop_messages[type(e)])
            self._interrupted = True
        finally:
            self._measurement_result.finalize()
            plt.close(figure_number)

    def measure(self):
        self._measurement_result.set_is_finished(False)  # ensure

        try:
            self._record_data()
        except Exception:
            self._measurement_result.set_exception_info(sys.exc_info())
        finally:
            self._measurement_result.set_is_finished(True)

    def _record_data(self):

        par_names = self._swept_pars_names
        done_iterations = 0
        start_time = self._measurement_result.get_start_datetime()

        parameters_values = \
            [self._swept_pars[parameter_name][1] for parameter_name in par_names]
        parameters_idxs = \
            [list(range(len(self._swept_pars[parameter_name][1]))) for parameter_name in par_names]
        raw_data_shape = \
            [len(indices) for indices in parameters_idxs]
        total_iterations = reduce(mul, raw_data_shape, 1)

        for idx_group, values_group in zip(product(*parameters_idxs), product(*parameters_values)):

            self._call_setters(values_group)

            # This should be implemented in child classes:
            data = self._recording_iteration()

            if done_iterations == 0:
                try:
                    self._raw_data = zeros(raw_data_shape + [len(data)], dtype=complex_)
                except TypeError:  # data has no __len__ attribute
                    self._raw_data = zeros(raw_data_shape, dtype=complex_)
            self._raw_data[idx_group] = data

            # This may need to be extended in child classes:
            measurement_data = \
                self._prepare_measurement_result_data(par_names, parameters_values)
            self._measurement_result.set_data(measurement_data)

            done_iterations += 1

            avg_time = (dt.now() - start_time).total_seconds() / done_iterations
            time_left = self._format_time_delta(avg_time * (total_iterations - done_iterations))

            formatted_values_group = \
                '[' + "".join(["%s: %.2e, " % (par_names[idx], value)
                               for idx, value in enumerate(values_group)])[:-2] + ']' \
                    if isinstance(values_group[0], (float, int)) else \
                    '[' + "".join(["%s: %s, " % (par_names[idx], str(value))
                                   for idx, value in enumerate(values_group)])[:-2] + ']'

            print("\rTime left: " + time_left + ", %s" % formatted_values_group +
                  ", average cycle time: " + str(round(avg_time, 2)) + " s       ",
                  end="", flush=True)

            if self._interrupted:
                return

        self._measurement_result.set_recording_time(dt.now() - start_time)
        print("\nElapsed time: %s" % self._format_time_delta((dt.now() - start_time)
                                                             .total_seconds()))

    def set_measurement_result(self, measurement_result: MeasurementResult):
        self._measurement_result = measurement_result

    def _recording_iteration(self):
        """
        This method must be overridden for each new measurement type.

        Should contain the recording logic and set the data of the
        corresponding MeasurementResult object.
        See lib2.SingleToneSpectroscopy.py as an example implementation
        """
        pass

    def _prepare_measurement_result_data(self, parameter_names, parameter_values):
        """
        This method MAY be overridden for a new measurement type.

        An override is needed if you have _recording_iteration(...) that returns
        an array, so effectively you have an additional parameter that is swept
        automatically. You will be able to pass its values and name in the
        overridden method (see lib2.SingleToneSpectroscopy.py).
        """
        measurement_data = self._measurement_result.get_data()
        measurement_data.update(zip(parameter_names, parameter_values))
        measurement_data["data"] = self._raw_data
        return measurement_data

    def _detect_resonator(self, plot=False, tries_number=10):
        """
        Finds frequency of the resonator visible on the VNA screen
        """
        vna = self._vna[0]
        init_averages = vna.get_averages()
        for i in range(1, tries_number+1):
            vna.set_averages(init_averages*i)
            vna.avg_clear()
            vna.prepare_for_stb()
            vna.sweep_single()
            vna.wait_for_stb()
            frequencies, sdata = vna.get_frequencies(), vna.get_sdata()
            # vna.autoscale_all()
            self._resonator_detector.set_data(frequencies, sdata*exp(2*pi*1j*frequencies*50e-9))
            self._resonator_detector.set_plot(plot)
            result = self._resonator_detector.detect()

            if result is not None:
                break
            else:
                print("\rFit was inaccurate (try #%d), retrying" % i, end="")
        # if result is None:
        # print(frequencies, sdata)
        vna.set_averages(init_averages)
        return result

    def _detect_qubit(self):
        """
        To find a peak/dip from a qubit in line automatically (to be implemented)
        """
        pass

    def _write_to_log(self, line='Unknown measurement', parameters=''):
        """
        A method writes line with the name of measurement
        (probably with formatted parameters) to log list
        """
        self._log += str(dt.now().replace(microsecond=0)) + "  " + line + parameters + '\n'

    def return_log(self):
        """
        Returns string of log containing all adressed measurements in chronological order.
        """
        return self._log

    def _construct_fixed_parameters(self):

        self._fixed_params = {}

        yn = input('Do you want to set the dictionary of fixed parameters interactively: yes/no \n')

        if yn == 'yes':
            while True:
                dev_name = input(
                    'Enter name of device : "exa", "vna", etc.\n' + 'If finished enter whatever else you want \n')
                if dev_name in self._actual_devices.keys():
                    self._fixed_params[dev_name] = {}
                    print('Enter parameter and value as: "frequency 5e9" and press Enter)\n' + \
                          'If finished with this device enter "stop next"\n')
                    while True:
                        par_name, vs = input().split()
                        if par_name == 'stop':
                            print('\n')
                            break
                        else:
                            value = float(vs)
                            self._fixed_params.get(dev_name)[par_name] = value
                else:
                    return self._fixed_params
        elif yn == 'no':
            return self._fixed_params

        else:
            return self._fixed_params

    def _format_time_delta(self, delta):
        hours, remainder = divmod(delta, 3600)
        minutes, seconds = divmod(remainder, 60)
        return '%s h %s m %s s' % (int(hours), int(minutes), round(seconds, 2))

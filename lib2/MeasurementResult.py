import fnmatch
import os
import pickle
import traceback
from datetime import datetime
from threading import Lock

from IPython.display import clear_output
import matplotlib
from matplotlib import animation, pyplot as plt
from matplotlib._pylab_helpers import Gcf
from numpy import array, where
import copy
import shutil

import locale

def find(pattern, path):
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name))
    return result


class ContextBase():

    def __init__(self):
        self._equipment = {}
        self._comment = ""

    def get_equipment(self):
        return self._equipment

    def to_string(self):
        return "Equipment with parameters:\n" + str(self._equipment) + \
               "\nComment:\n" + self._comment

    def update(self, equipment={}, comment=""):
        self._equipment.update(equipment)
        self._comment.join(comment)


class MeasurementResult:

    def __init__(self, name, sample_name):
        self._name = name
        self._sample_name = sample_name
        self._data_lock = Lock()
        self._data = {}
        self._context = ContextBase()

        self._parameter_names = None

        # visualization fields, see _prepare_figure(...) docstring below
        self._figure = None  # the figure that will be dynamically updated
        self._axes = None  # axes of the subplots contained inside it
        self._caxes = None  # colorbar axes for heatmaps
        self._anim = None

        self._exception_info = None

    def set_parameter_names(self, parameter_names):
        self._parameter_names = parameter_names

    @staticmethod
    def delete(sample_name, name, date='', delete_all=False):
        """
        Finds all files with matching result name within the file structure of ./data/
        folder, prompts user to resolve any ambiguities. Then deletes selected
        measurement data.

        Example usage:
        >>> from lib2.MeasurementResult import MeasurementResult
        >>> result = MeasurementResult.delete("<sample_name>", "<name>")

        If the user hits EOF (*nix: Ctrl-D, Windows: Ctrl-Z+Return),
        raise EOFError. On *nix systems, readline is used if available.
        """
        paths = MeasurementResult._find_paths_by(sample_name, name,
                                                 ".pkl", date, delete_all)

        time_locations = set()
        for path in paths:
            time_location = os.path.join(*(path.split(os.sep)[:-1]))
            time_locations.add(time_location)

        print(time_location)
        for time_location in time_locations:
            shutil.rmtree(time_location, ignore_errors=True)

    @staticmethod
    def load(sample_name, name, date='', subfolder="", return_all=False):
        """
        Finds all files with matching result name within the file structure
        of ./data/ folder and optionally prompts user to resolve any ambiguities.

        Returns:
            an instance of the child class containing the specific measurement
            result

        Example usage:
        >>> from lib2.MeasurementResult import MeasurementResult
        >>> result = MeasurementResult.load("<sample_name>", "<name>")

        If the user hits EOF (*nix: Ctrl-D, Windows: Ctrl-Z+Return), raise EOFError.
        On *nix systems, readline is used if available.
        """

        paths = MeasurementResult._find_paths_by(sample_name, name, ".pkl", date, subfolder, return_all)

        if paths is None:
            return

        results = []
        for idx, path in enumerate(paths):
            try:
                with open(path, "rb") as f:
                    results.append(pickle.load(f))
            except pickle.UnpicklingError as e:
                results.append(e)

        return results[0] if len(results) == 1 and not return_all else results

    @staticmethod
    def _find_paths_by(sample_name, name, extension, date, subfolder, return_all=False):
        paths = find(name + extension, os.path.join('data', sample_name, subfolder, date))

        if len(paths) == 0:
            print("Measurement result '%s' for the sample '%s' not found" % (name, sample_name))
            return

        locale.setlocale(locale.LC_TIME, "C")
        dates = [datetime.strptime(path.split(os.sep)[-3], "%b %d %Y")
                 for path in paths]
        z = zip(dates, paths)
        sorted_dates, sorted_paths = zip(*sorted(z))

        if not return_all and len(paths)>1:
            return MeasurementResult._prompt_user_to_choose(sorted_paths)

        return sorted_paths

    @staticmethod
    def _prompt_user_to_choose(paths):
        for idx, file in enumerate(paths):
            print(idx, file)
        print("More than one file found. Enter an index from listed above:")
        index = input()
        return [paths[int(index)]]

    def get_save_path(self):

        if not os.path.exists("data"):
            os.makedirs("data")

        sample_directory = os.path.join('data', self._sample_name)
        if not os.path.exists(sample_directory):
            os.makedirs(sample_directory)

        locale.setlocale(locale.LC_TIME, "C")
        date_directory = os.path.join(sample_directory,
                                      self.get_start_datetime().strftime("%b %d %Y"))
        if not os.path.exists(date_directory):
            os.makedirs(date_directory)

        time_directory = os.path.join(date_directory,
                                      self.get_start_datetime().strftime("%H-%M-%S")
                                      + " - " + self._name)

        if not os.path.exists(time_directory):
            os.makedirs(time_directory)

        return time_directory

    def __getstate__(self):
        d = dict(self.__dict__)
        d['_data_lock'] = None
        d['_anim'] = None
        d['_figure'] = None
        d['_axes'] = None
        d['_caxes'] = None
        d['_exception_info'] = None
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._data_lock = Lock()

    def save(self, plot_maximized = True):
        """
        This method may be overridden in a child class but super().save()
        must be called in the beginning of the overridden method.

        Saves the MeasurementResult object using pickle, creating the folder
        structure if necessary.

        The path is structured as follows:
            data/<sample name>/DD MM YYYY/HH-MM-SS - <name>/

        At least <name>.pkl with serialized object, <name>_raw_data.pkl with raw
        data only and human-readable context will be stored, though
        child methods should save additional files in their overridden methods,
        i.e. plot pictures
        """
        fig, axes, caxes = self.visualize(plot_maximized)

        with self._data_lock:
            with open(os.path.join(self.get_save_path(), self._name + '.pkl'), 'w+b') as f:
                pickle.dump(self, f)
            with open(os.path.join(self.get_save_path(), self._name + '_raw_data.pkl'), 'w+b') as f:
                pickle.dump(self._data, f)
            with open(os.path.join(self.get_save_path(), self._name + '_context.txt'), 'w+') as f:
                f.write(self.get_context().to_string())

        plt.savefig(os.path.join(self.get_save_path(), self._name + ".png"), bbox_inches='tight')
        plt.savefig(os.path.join(self.get_save_path(), self._name + ".pdf"), bbox_inches='tight')
        plt.close(fig)

    def visualize(self, maximized=True):
        """
        Generates the required plots to visualize the measurement result. Should
        be implemented for each subclass.
        """
        fig, axes, caxes = self._prepare_figure()
        self._figure = fig
        self._axes = axes
        self._caxes = caxes
        self._plot(self.get_data())
        figManager = plt.get_current_fig_manager()
        if maximized:
            try:
                try:
                    figManager.window.showMaximized()
                except:
                    figManager.window.state('zoomed')
            except:
                fig.set_size_inches(10, 5)
        else:
            fig.set_size_inches(10, 5)
        return fig, axes, caxes

    def _yield_data(self):
        while not self.is_finished():
            yield self.get_data()

    def visualize_dynamic(self):
        """
        Dynamically visualizes the measurement data. To be used in the recording
        scripts.
        """

        fig, axes, caxes = self._prepare_figure()
        self._dynamic = True
        self._figure = fig
        self._axes = axes
        self._caxes = caxes
        fig_manager = Gcf.get_fig_manager(fig.number)

        try:
            try:
                fig_manager.window.showMaximized()
            except:
                fig_manager.window.state('zoomed')
        except:
            # we are probably in the notebook regime
            fig.set_size_inches(10, 5)

        self._anim = animation.FuncAnimation(fig, self._plot,
                                             frames=self._yield_data,
                                             repeat=False, interval=100)

    def _prepare_figure(self):
        """
        This method must be implemented for each new measurement type.

        See lib2.SingleToneSpectroscopy.py for an example implementation
        Should return:
        figure: matplotlib figure
            figure window
        axes: array of matplotlib.Axes objects
            axes of the subplots contained inside the figure
        caxes: array of colorbar axes
            these axes are obtained by calling matplotlib.colorbar.make_axes(ax)
            and then may be used to updated colorbars for each subplot
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
        caxes = None
        return fig, axes, caxes

    def _plot(self, data):
        """
        This method must be implemented for each new measurement type.

        The axes and caxes are those created by _prepare_figure(...) method and
        should be used here to visualize the data

        The data to plot is passed as an argument by FuncAnimation
        """
        pass

    def finalize(self):
        """
        This method may be overridden in a child class but super().finalize()
        must be called in the beginning of the overridden method.

        Method that should be called FROM THE MAIN THREAD upon the end of the
        measurement recording.

        Should at least close the dynamically updated figure (implemented)
        """
        self._dynamic_figure = None
        self._dynamic_axes = None
        self._dynamic_caxes = None
        self._dynamic = False

        if self._exception_info is not None:
            clear_output()
            traceback.print_tb(self._exception_info[-1])
            print(*self._exception_info[:2])

    def set_is_finished(self, is_finished):
        self._is_finished = is_finished

    def is_finished(self):
        return self._is_finished

    def get_start_datetime(self):
        return self._datetime

    def set_start_datetime(self, datetime):
        self._datetime = datetime

    def get_recording_time(self):
        return self._recording_time

    def set_recording_time(self, recording_time):
        self._recording_time = recording_time

    def get_data(self):
        with self._data_lock:
            return copy.deepcopy(self._data)

    def get_context(self):
        return self._context

    def set_data(self, data):
        """
        Data should consist only of built-in data types to be easy to use on
        other computers without the whole measurement library.
        """
        with self._data_lock:
            self._data = copy.deepcopy(data)

    def _latex_float(self, f):
        float_str = "{0:.2e}".format(f)
        base, exponent = float_str.split("e")
        if int(exponent) != 0:
            return r"${0} \times 10^{{{1}}}$".format(base, int(exponent))
        else:
            return base

    def get_figure_number(self):
        return self._figure.number

    @staticmethod
    def close_figure_by_window_name(window_name):
        try:
            idx = int(where(array([manager.canvas.figure.canvas.get_window_title() \
                                   for manager in matplotlib._pylab_helpers.Gcf \
                                  .get_all_fig_managers()]) == window_name)[0][0])
            plt.close(plt.get_fignums()[idx])
        except IndexError:
            print("Figure with window name '%s' not found" % window_name)

    def copy(self):
        with self._data_lock:
            return copy.deepcopy(self)

    def set_exception_info(self, exception_info):
        self._exception_info = exception_info

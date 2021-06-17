from matplotlib import pyplot as plt, colorbar
from lib2.VNATimeResolvedDispersiveMeasurement import *


class VNATimeResolvedDispersiveMeasurement2D(VNATimeResolvedDispersiveMeasurement):

    def __init__(self, name, sample_name, devs_aliases_map):
        super().__init__(name, sample_name, devs_aliases_map,
                         plot_update_interval=5)


class VNATimeResolvedDispersiveMeasurement2DResult(VNATimeResolvedDispersiveMeasurementResult):

    def __init__(self, name, sample_name):
        super().__init__(name, sample_name)
        self._maps = [None]*2
        self._cbs = [None]*2

    def _prepare_figure(self):
        fig, axes, caxes = super()._prepare_figure()
        plt.tight_layout(pad=2, h_pad=5, w_pad=0)
        caxes = []
        for ax in axes:
            caxes.append(colorbar.make_axes(ax)[0])
        return fig, axes, caxes

    def _prepare_data_for_plot(self, data):
        """
        Should be implemented in child classes
        """
        pass

    def _annotate_axes(self, axes):
        """
        Should be implemented in child classes
        """
        pass

    def _plot(self, data):

        axes = self._axes
        caxes = self._caxes
        if "data" not in data.keys():
            return

        X, Y, Z_raw = self._prepare_data_for_plot(data)
        extent = [X[0], X[-1], Y[0], Y[-1]]


        for idx, name in enumerate(self._data_formats.keys()):
            ax = axes[idx]
            Z = self._data_formats[name][0](Z_raw)
            max_Z = max(Z[Z != 0])
            min_Z = min(Z[Z != 0])
            if self._maps[idx] is None or not self._dynamic:
                ax.grid()
                self._maps[idx] = ax.imshow(Z, origin='lower', cmap="RdBu_r",
                                                aspect='auto', vmax=max_Z, vmin=min_Z,
                                                extent=extent)
                self._cbs[idx] = plt.colorbar(self._maps[idx], cax=caxes[idx])
                self._cbs[idx].set_label(self._data_formats[name][1])
                self._cbs[idx].formatter.set_scientific(True)
                self._cbs[idx].formatter.set_powerlimits((0, 0))
                self._cbs[idx].update_ticks()
                self._annotate_axes(axes)
            else:
                self._maps[idx].set_data(Z)
                self._maps[idx].set_clim(min_Z, max_Z)
                self._cbs[idx].set_clim(min_Z, max_Z)


        plt.draw()

    def __getstate__(self):
        d = super().__getstate__()
        d['_maps'] = [None]*2
        d['_cbs'] = [None]*2
        return d
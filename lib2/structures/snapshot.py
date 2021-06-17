import numpy as np
from copy import deepcopy
from scipy.signal import correlate
from scipy.interpolate import interp1d

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colorbar
from matplotlib.ticker import LinearLocator, FormatStrFormatter

# pyCharm is not resolving this package due to the fact that only cv2.pyd file
# is present in the Python packges dir but not cv2.py (the latter is simply does not exist)
import cv2


'''
Non documented functions are still in development
'''


def _inplace(method_to_decorate):
    """
    @brief: decorator for functions that either modifies or returns modified copy of the object
    @params: class method to decorate
    @return: decorated class method

    Note: this is the possible replacement for self._handle_inplace() function
    The issue: jupyter notebook is not able to determine arguments of the decorated function
    but can find docstring related to this function. Now I doubt the usefulness of this decorator.
    """
    def _inplace_wrapper(self, inplace=False, *args, **kwargs):
        if inplace is True:
            return method_to_decorate(self, *args, **kwargs)
        else:
            return method_to_decorate(deepcopy(self), *args, **kwargs)

    return _inplace_wrapper


class Snapshot:

    # value that is assigned to all self.data entries
    # that are below relative threshold
    # during the call of the self._preproc_data(...)
    MACROPARAM_NULLIFY = 0

    def __init__(self, data_ma=None, phase=False):
        ## data structure attributes ##
        data_ma = deepcopy(data_ma) # force to copy during debugging
        self.x = None
        self.y = None
        self.data = None
        self.dx = None
        self.dy = None
        if isinstance(data_ma, dict):
            if "Current [A]" in data_ma:
                self.x = data_ma["Current [A]"]
            elif "Voltage [V]" in data_ma:
                self.x = data_ma["Voltage [V]"]

            self.y = data_ma["Frequency [Hz]"] / 1e9
            self.data = data_ma["data"].T
            self._init_dependent_attributes()

        self._phase = phase # determines whether to work with phase space
        self._exctract = np.abs if phase is False else np.angle

        ## attributes utilized by various algorithms ##
        self._connectivity_result = None
        self._preprocessed_self = None

        # '_target...' is applied to data that is referred to the curve
        # with index 'target_label_i' in self._connectivity_result[2] aka stats
        self._target_label_i = None
        self._target_component_mask = None
        self._target_x_idxs = None
        self._target_y_idxs = None
        self._target_y_func = None


    def _handle_inplace(self,inplace):
        if inplace is True:
            return self
        else:
            return deepcopy(self)

    def _init_dependent_attributes(self):
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

    def resize(self, new_pixel_size: tuple, inplace=False):
        """
        @brief: box physical dimensions is not changed. Only number of pixels (data points) is changed
        @params
            new_pixel_size : tuple
                (x_pixels_N, y_pixels_N) - tuple of integers
                                           specifying the amount of pixels
                                           along x and y direction respectively
            inplace : bool
                True - modifies self
                False - return modified copy of self
        """
        if inplace is False:
            tmp = Snapshot()
        else:
            tmp = self

        xlims = (self.x[0], self.x[-1])
        ylims = (self.y[0], self.y[-1])

        # INTER_AREA to downscale
        tmp.data = cv2.resize(np.real(self.data), new_pixel_size, interpolation=cv2.INTER_AREA) + \
                   1j * cv2.resize(np.imag(self.data), new_pixel_size, interpolation=cv2.INTER_AREA)

        tmp.x = np.linspace(*xlims, new_pixel_size[0])
        tmp.y = np.linspace(*ylims, new_pixel_size[1])
        tmp._init_dependent_attributes()  # updates tmp.dx and tmp.dy

        return tmp

    def crop(self, xlims=None, ylims=None, inplace=False):
        """
        @params:
            inplace : bool
                True - modifies self
                False - return modified copy of self
        """
        tmp = self._handle_inplace(inplace)

        x_idxs = (self.x > xlims[0]) & (self.x < xlims[1]) if (xlims is not None) else \
            np.full_like(self.x, True, dtype=np.bool)
        y_idxs = (self.y > ylims[0]) & (self.y < ylims[1]) if (ylims is not None) else \
            np.full_like(self.y, True, dtype=np.bool)
        tmp.x = self.x[x_idxs]
        tmp.y = self.y[y_idxs]
        tmp.data = self.data[np.ix_(y_idxs, x_idxs)]
        tmp._init_dependent_attributes()

        return tmp

    def intersection(self, snapshot):
        return ((max(self.x[0], snapshot.x[0]), min(self.x[-1], snapshot.x[-1])),
                (max(self.y[0], snapshot.y[0]), min(self.y[-1], snapshot.y[-1])))

    def match_ylims(self, snapshot, inplace=False):
        """
        @brief: return Snapshot that is contained in 'self' as a mathamtical set
                and results from intersection of the 'self' and 'snapshot' in 'y' direction
        @params:
            inplace : bool
                True - modifies self
                False - return modified copy of self
        """
        common_ylims = self.intersection(snapshot)[1]
        return self.crop(ylims=common_ylims, inplace=inplace)

    def match_resolution(self, snapshot, inplace=False):
        """
        @brief: resizes self in order to provide the same self.dx and self.dy
                that is coinciding with snapshot.dx and snapshot.dy
        @params:
            inplace : bool
                True - modifies self
                False - return modified copy of self
        """

        new_dx = snapshot.dx
        new_dy = snapshot.dy
        new_pixel_size = (int(round((self.x[-1] - self.x[0]) / new_dx + 1)),
                          int(round((self.y[-1] - self.y[0]) / new_dy + 1)))
        return self.resize(new_pixel_size, inplace=inplace)

    @_inplace
    def calibrate_phase(self):
        """
        @brief: Substracts singal's phases of the lowest if_freq from all the if_freq phases
        It turn all phases at the lowest if_freq to be equal to zero.
        @params:
        @return: None
        """
        self.data /= np.exp(1j*np.angle(self.data[0]))
        return self

    def _nullify_rel_threshold(self, relative_threshold=0.4, inplace=False):
        """
        @params:
            relative_threshold : float from interval (0.0 ; 1.0)
                defines the threshold in the normalized data interval in a relative way
                normalized data interval is (0.0 ; 1.0)
                all the data values below threshold is set to 0.0

            inplace : bool
                True - modifies self
                False - return modified copy of self
        """

        tmp = self._handle_inplace(inplace)

        abs_threshold = self._threshold_rel2abs(relative_threshold)
        tmp._nullify_abs_threshold(abs_threshold, inplace=True)

        return tmp

    def _threshold_rel2abs(self, rel_threshold):
        """
        @brief: converts relative threshold to the absolute value of the data
        @params:
                rel_threshold : f float from interval (0.0 ; 1.0)
                    defines the threshold in the normalized data interval in a relative way
                    normalized data interval is (0.0 ; 1.0)
                    all the data values below threshold is set to 0.0
        @return: float
                absolute threshold value in the data range (min(data) ; max(data))
        """
        maximum = np.amax(self._exctract(self.data))
        minimum = np.amin(self._exctract(self.data))
        return minimum * (1 - rel_threshold) + rel_threshold * maximum

    def _nullify_abs_threshold(self, abs_threshold, inplace=False):
        """TODO: utilize cv2.inRange & cv2.threshold
        @brief: set all the data entries from x with values
                below 'threshold' to fixed value close to zero
        @params:
            inplace : bool
                True - modifies self
                False - return modified copy of self
            abs_threshold : float
                all self.data values which absolute value is
                below threshold will be set to Snapshot.MACROPARAM_NULLIFY
        """
        tmp = self._handle_inplace(inplace)

        tmp.data[self._exctract(self.data) < abs_threshold] = Snapshot.MACROPARAM_NULLIFY

        return tmp

    def _normalize(self, inplace=False):
        """
        @brief: Transforms x and y to (0.0;1.0) interval
                by linear transformation (side points included).
                Transforms data absolute value to (0.0;1.0) interval
                by dividing all the self.data by its max value.
        @params:
            inplace : bool
                True - modifies self
                False - return modified copy of self
        """
        tmp = self._handle_inplace(inplace)

        min_amp = np.amin(np.abs(tmp.data))
        max_amp = np.amax(np.abs(tmp.data))
        d_amp = max_amp - min_amp
        max_phase = np.amax(np.angle(tmp.data))
        min_phase = np.amin(np.angle(tmp.data))
        d_phas = max_phase - min_phase

        tmp.data = (np.abs(tmp.data) - min_amp)/d_amp * np.exp(1j*(np.angle(tmp.data) - min_phase)/d_phas)

        tmp.y = (tmp.y - tmp.y[0])/(tmp.y[-1] - tmp.y[0])
        tmp.x = (tmp.x - tmp.x[0])/(tmp.x[-1] - tmp.x[0])

        tmp._init_dependent_attributes()

        return tmp

    def _correlate_overX(self, snapshot2find):
        """
        @brief:     Calculates correlation function over the horizontal (axis = 1) variable
                    between the self and snapshot2find.
                    Correlation is calculated only on the XY box that is intersection
                    between 'self' and 'snapshot2find'.
                    Returns normalized correlation function and normalization factor array
                    in the denominator.
        @desc:      The algorithm is analogous to the opencv.matchTemplate function
                    except that image that's location is to be found only displaced along
                    the horizontal axis.
                    See more in documentation of 'opencv.matchTemplate(..)' method:
                    https://docs.opencv.org/2.4/doc/tutorials/imgproc/histograms/template_matching/template_matching.html
                    method = CV_TM_CCORR_NORMED - match function implemented in this function
        @params:
                    snapshot2find : class Snapshot
                        represents an image that participates as a second variable in scipy.trace.correlate method
        @return:    (res,res_norm)
                    res : 1D numpy array.
                        len(res) = self.shape[1] + snapshot2find.shape[1] - 1
                        autocorrelation function of the two images across horizontal axis
                    res_norm : 1D numpy array
                        len(res_norm) = len(res)
                        normalization factor in denumenator of the matching function formula.
                        See more via the link provided in @desc section.

        """
        # constructing 2 new snapshots with ylims
        # that are contained in both initial snapshots
        cropped_self = self.match_ylims(snapshot2find)
        cropped_2find = snapshot2find.match_ylims(self)

        # print(snapshot.dx,snapshot.dy)
        # print(snapshot_ymatch.dx,snapshot_ymatch.dy)
        # print(self_matched.data.shape,self_matched.dx,self_matched.dy)
        # transform cropped_self to the same resolution as a cropped_2find
        cropped_2find.match_resolution(self, inplace=True)
        # print(self_matched.data.shape,self_matched.dx,self_matched.dy)

        # plot_snapshot(cropped_self)
        # plot_snapshot(cropped_2find)

        n_y = min(cropped_self.data.shape[0], cropped_2find.data.shape[0])
        n_cross_corr = cropped_self.data.shape[1] + cropped_2find.data.shape[1] - 1  # number of correlation output

        cross_corr_res = np.zeros((n_y, n_cross_corr), dtype=np.complex128)  # cross-correlation
        cross_corr_norm = np.zeros((n_y, n_cross_corr), dtype=np.complex128)  # cross correlation norm

        # print(cropped_self.data.shape)
        # print(cropped_2find.data.shape)
        l = cropped_2find.data.shape[1]
        r = cropped_self.data.shape[1]
        for i, (y1_i, y2_i) in enumerate(zip(cropped_self.data, cropped_2find.data)):  # equals to len(snap2crop)
            # correct phase for each measurement
            y1_i /= np.exp(np.angle(y1_i[0]))
            y2_i /= np.exp(np.angle(y2_i[0]))

            cross_corr_res[i] = correlate(y1_i, y2_i, mode="full")
            for j in range(n_cross_corr):
                # normalization of the correlation norm
                # I will never explain this indexes origin without piece of paper
                cross_corr_norm[i, j] = np.linalg.norm(cropped_self.data[:, max(1 - l + j, 0):min(r, 1 + j)]) * \
                                        np.linalg.norm(cropped_2find.data[:, max(0, l - 1 - j):min(l, l + r - 1 - j)])

        cross_corr_res = cross_corr_res.sum(axis=0)
        cross_corr_norm = cross_corr_norm.sum(axis=0)

        cross_corr_res /= cross_corr_norm

        return cross_corr_res, cross_corr_norm

    def find_shift_overX(self, snapshot):
        '''
        TODO: add description
        '''
        res, res_norm = self._correlate_overX(snapshot)
        pixel_shift = np.argmax(res)
        xvar_shift = self.dx * (pixel_shift - (len(snapshot.x) - 1))
        return xvar_shift

    def apply_gauss(self, kernel_x=0.1, kernel_y=0.1, inplace=False):
        '''
        TODO: add description
        '''
        tmp = self._handle_inplace(inplace)

        kernel_x_px = int(kernel_x / tmp .dx)
        if kernel_x_px%2 == 0:
            kernel_x_px += 1
        kernel_y_px = int(kernel_y / tmp .dy)
        if kernel_y_px%2 == 0:
            kernel_y_px += 1

        # tmp.data = cv2.GaussianBlur(np.real(tmp .data), (kernel_x_px, kernel_y_px), 0) + \
        #                   1j * cv2.GaussianBlur(np.imag(tmp .data), (kernel_x_px, kernel_y_px), 0)

        # separate amplitude and phase gaussian transform
        tmp.data = cv2.GaussianBlur(np.abs(tmp.data), (kernel_x_px, kernel_y_px), 0) * \
                   np.exp(1j*cv2.GaussianBlur(np.angle(tmp.data), (kernel_x_px, kernel_y_px), 0))

        return tmp

    def _connected_components(self, rel_threshold=0.5,
                              kernel_x=0.1, kernel_y=0.1,
                              connectivity=8):
        """
        @brief: Returns cv2.connectedComponentsWithStats result
                with the preprocessed self.data.
                Data is normalized with self._normalize()
                Gauss filter is applied.
                Data values that lies below rel_threshold are set to fixed value.
        @params:
            rel_threshold : float
                relative threshold from interval (0; 1.0)
            kernel_x : float
                size of the gauss filter in 'y' directions for gauss filter
                applied to normalized data
            kernel_y : float
                size of the gauss filter in 'y' directions for gauss filter
                applied to normalized data
            connectivity : int
                This parameter is simply passed to
                the opencv.connectedComponents(...).
                Allowed values: 4,8.
                See opencv.connectedComponents(...)
                for more info on this parameter
        @return:
            return cv2.connectedComponentsWithStats()
            tuple of 4 elements respectively:
            (retval, labeled_img, stats, centroids)
            retval : int
                number of connected components
            labeled_img : 2D np.array with shape equal to self.data.shape
                contains pixels marked to the corresponding connectivity
                component starting from 0 (uint16 or int32 datatype)
            stats : 2D numpy array
                Contains data about each connectivity component
                that have been aggregated during the components construction
                see more in opencv.connectedComponents(...) docs.
            centroid : float tuple (x,y)
                contains centroids coordinates
                to the corresponding connectivity components
        """
        # duplicating and performing series of transofmations
        # on self attributes
        self._preprocessed_self = self._normalize()
        self._preprocessed_self.apply_gauss(kernel_x, kernel_y, inplace=True)
        self._preprocessed_self._normalize(inplace=True)
        self._preprocessed_self._nullify_rel_threshold(rel_threshold, inplace=True)

        # convert to binary image for opencv.connectedComponents
        img = (self._exctract(self._preprocessed_self.data) > rel_threshold).astype(np.uint8)
        self._connectivity_result = cv2.connectedComponentsWithStats(img,
                                                                     connectivity=connectivity)
        return self._connectivity_result

    def visualize_connectivity_map(self):
        """
        @brief: You should construct connected components map
                self._connectivity_result by calling self._connected_components(...)
                On details see of self._connected_components structure
                see self._connected_components() docstring
        """
        fig, ax = plt.subplots(1, 1, figsize=(16, 9))

        # preparing z data
        img_data = self._connectivity_result[1]

        # choosing color map object
        colors_n = self._connectivity_result[0]
        cmap = cm.get_cmap("tab20", colors_n)

        # defying rules by which scalar values in img_data
        # will be transformed into RGB colours from colormap
        # chosen above
        norm = matplotlib.colors.BoundaryNorm(np.arange(colors_n+1)-0.5, colors_n)

        # making colorbar axes
        cax, kw = matplotlib.colorbar.make_axes(ax, ticks=np.arange(1, colors_n+1), aspect="auto")
        # making axes with image itself
        extent = [self.x[0] - self.dx / 2, self.x[-1] + self.dx / 2, \
                  self.y[0] - self.dy / 2, self.y[-1] + self.dy / 2, ]
        img = ax.imshow(img_data, cmap=cmap, norm=norm, origin="lower", aspect="auto", extent=extent)

        # reforging image data, color axes and data axes into
        # colorbar object
        cb = fig.colorbar(img, cax, ax, ticks=np.arange(colors_n))
        cb.ax.set_yticklabels(np.arange(colors_n))

    def make_and_visualize_connectivity_map(self,
                                            rel_threshold=0.5,
                                            kernel_x=0.1, kernel_y=0.1,
                                            connectivity=8):
        self._connected_components(rel_threshold=rel_threshold,
                                   kernel_x=kernel_x, kernel_y=kernel_y,
                                   connectivity=connectivity)
        self.visualize_connectivity_map()

    def select_connectivity_component(self, label_i):
        self._target_label_i = label_i
        return self._make_target_component_mask()

    def _make_target_component_mask(self):
        """
        @brief: constructs binary mask of the connected component
                based on its label index 'label_i'
                and previously constructed connected components map
                using self._connected_components
        TODO: add description
        """
        connectivity_map = self._connectivity_result[1]
        self._target_component_mask = deepcopy(connectivity_map)
        self._target_component_mask[ connectivity_map == self._target_label_i ] = 1
        self._target_component_mask[ connectivity_map != self._target_label_i ] = 0

        return deepcopy(self._target_component_mask)

    def interpolate_yx_curve(self):
        self._exctract_target_yx_points()
        self._target_y_func = interp1d(self.x[self._target_x_idxs], self.y[self._target_y_idxs],
                                       kind="cubic", copy=False, assume_sorted=True)
        return self._target_y_func

    def _exctract_target_yx_points(self):
        stats = self._connectivity_result[2]
        x_left_i = stats[self._target_label_i, cv2.CC_STAT_LEFT]
        x_right_i = x_left_i + stats[self._target_label_i, cv2.CC_STAT_WIDTH] - 1
        #print( x_left_i, x_right_i, stats[self._target_label_i, cv2.CC_STAT_WIDTH], self._target_label_i )
        x_mask_idxs = np.arange(x_left_i, x_right_i)

        # target_mask_y_idxs[x_idx] - list of y_idxs that belong to mask for a given x_idx
        # see np.where(..) documentation for the additional [0] inside list comprehension
        y_mask_idxs = [np.where(self._target_component_mask[:, x_idx] > 0)[0] for x_idx in x_mask_idxs]

        # y_idxs_max_aliased[x_idx] - contains index of the maximum value in target_mask_y_idxs[x_idx]
        # TODO: shorten following cycles into list comprehensions
        x_idxs_aliased = x_mask_idxs - x_mask_idxs[0]
        y_idxs_max_aliased = []
        for x_idx_aliased, x_mask_idx in zip(x_idxs_aliased, x_mask_idxs):
            y_idx_of_max = np.argmax(self._exctract(self._preprocessed_self.data[y_mask_idxs[x_idx_aliased], x_mask_idx]))
            y_idxs_max_aliased.append(y_idx_of_max)

        y_max_idxs = []
        for x_idx_aliased in x_idxs_aliased:
            y_idx_aliased = y_idxs_max_aliased[x_idx_aliased]
            y_idx_back = y_mask_idxs[x_idx_aliased][y_idx_aliased]
            y_max_idxs.append(y_idx_back)

        self._target_y_idxs, self._target_x_idxs = (y_max_idxs, x_mask_idxs)
        return self._target_y_idxs, self._target_x_idxs

    def plot(self):
        fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True, sharex=True)
        ax_amps, ax_phas = axes
        ax_amps.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_amps.set_ylabel("Frequency [GHz]")
        xlabel = "Current, A"
        ax_amps.set_xlabel(xlabel)
        ax_phas.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax_phas.set_xlabel(xlabel)
        plt.tight_layout(pad=2, h_pad=-10)
        cax_amps, kw = matplotlib.colorbar.make_axes(ax_amps, aspect=40)
        cax_phas, kw = matplotlib.colorbar.make_axes(ax_phas, aspect=40)
        cax_amps.set_title("$|S_{21}|$", position=(0.5, -0.05))
        cax_phas.set_title("$\\angle S_{21}$\n [deg]",
                           position=(0.5, -0.1))
        ax_amps.grid()
        ax_phas.grid()
        extent = [self.x[0] - self.dx / 2, self.x[-1] + self.dx / 2, \
                  self.y[0] - self.dy / 2, self.y[-1] + self.dy / 2, ]
        amps_map = ax_amps.imshow(abs(self.data), aspect='auto', origin='lower', cmap="RdBu_r", extent=extent)
        cb_amps = plt.colorbar(amps_map, cax=cax_amps)
        phas_map = ax_phas.imshow(np.angle(self.data), aspect='auto', origin='lower', cmap="RdBu_r", extent=extent)
        cb_phas = plt.colorbar(phas_map, cax=cax_phas)
        #print("reloaded 3")
        return fig, axes

    def plot3d(self, plot_phase=False):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        # Make data.
        x, y = np.meshgrid(self.x, self.y)
        z = self._exctract(self.data)

        surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        ax_cb = fig.colorbar(surf, shrink=0.5, aspect=5)

        return fig, (ax, ax_cb)
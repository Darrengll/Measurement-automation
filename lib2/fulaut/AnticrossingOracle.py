
from matplotlib import pyplot as plt
from scipy import *
from scipy.signal import *
from scipy.optimize import *
from IPython.display import clear_output
from lib2.fulaut.qubit_spectra import *
from lib2.ResonatorDetector import *
from lib2.LoggingServer import *

class AnticrossingOracle():

    '''
    This class automatically processes anticrossing spectral data for
    different types of qubits and frequency arrangements between the qubits and
    resonators
    '''

    qubit_spectra = {"transmon":transmon_spectrum}

    def __init__(self, qubit_type, sts_result, plot=False, fast = False):
        self._qubit_spectrum = AnticrossingOracle.qubit_spectra[qubit_type]
        self._sts_result = sts_result
        self._plot = plot
        self._minimum_points_between_zeroes = 5
        self._minimum_points_around_zero = 5
        self._distance_from_intersection = 2
        self._logger = LoggingServer.getInstance()
        self._fast = fast

        self._extract_data()

    def launch(self):

        intersections = self.find_resonator_intersections()

        if intersections is None or len(intersections) <= 2:
            idx_1 = argmin(self._res_points[:,1])
            idx_2 = argmax(self._res_points[:,1])
            res_freq_1 = self._res_points[idx_1,1]
            res_freq_2 = self._res_points[idx_2,1]
            period = 2*abs(self._res_points[idx_2, 0]-self._res_points[idx_1, 0])

        elif len(intersections) > 2:
            idx_1 = (intersections[0]+intersections[1])//2
            idx_2 = (intersections[1]+intersections[2])//2
            res_freq_1 = self._freqs[argmin(abs(self._data)[idx_1, :])]
            res_freq_2 = self._freqs[argmin(abs(self._data)[idx_2, :])]
            res_freq_1, res_freq_2 = sorted((res_freq_1, res_freq_2))
            period = self._res_points[intersections[2],0]-\
                                        self._res_points[intersections[0],0]

        d_range = slice(0.,0.9,0.1)
        mean_freq = mean((res_freq_1, res_freq_2))
        res_freq_range = slice(mean_freq-2e6, mean_freq+2e6, 4e6/10)
        q_freq_range = slice(4e9,12e9, 100e6)
        g_range = slice(20e6, 40e6, 10e6)
        Ns = 3
        args = (self._res_points[:,0], self._res_points[:,1])
        # self._logger.debug(str(res_freq)

        best_fit_loss = 1e10
        fitresult = None
        # We are not sure where the sweet spot is, so let's choose the best
        # fit among two possibilities:
        for sweet_spot_cur in [self._res_points[idx_1,0],\
                               self._res_points[idx_2,0]]:

            def brute_cost_function(params, curs, res_freqs):
                f_res, g, q_max_freq, d = list(params)
                full_params = [f_res, g, period, sweet_spot_cur, q_max_freq, d]
                return self._cost_function(full_params, curs, res_freqs,
                                                        freqs_fine_number = 100)

            result = brute(brute_cost_function,
                    (res_freq_range, g_range, q_freq_range, d_range), Ns=Ns,
                        args = args, finish=None)

            f_res, g, q_max_freq, d = list(result)
            full_params = [f_res, g, period, sweet_spot_cur, q_max_freq, d]

            result = minimize(self._cost_function, full_params,
                                        args=args, method="Nelder-Mead")

            loss =\
                self._cost_function(result.x, *args)/len(self._res_points)*1000
            if loss<best_fit_loss:
                best_fit_loss = loss
                best_fitresult = result

        res_freq, g, period, sweet_spot_cur, q_freq, d = best_fitresult.x

        if self._plot:
            plt.figure()
            plt.plot(self._res_points[:,0], self._res_points[:,1], '.',
                        label="Data")
            plt.plot([sweet_spot_cur], [mean(self._res_points[:,1])], '+')

            p0 = [mean((res_freq_2, res_freq_2)), 30e6, period,
                                                    sweet_spot_cur, 10e9, 0.6]
            # plt.plot(self._curs, self._model(self._curs, p0), "o")
            plt.plot(self._res_points[:,0],
                    self._model(self._res_points[:,0], best_fitresult.x, False),
                                "orange", ls="", marker=".", label="Model")
            plt.legend()
            plt.gcf().set_size_inches(15,5)

        best_fitresult.x[0] = best_fitresult.x[0]
        best_fitresult.x[4] = best_fitresult.x[4]
        return best_fitresult.x, best_fit_loss

    def _extract_data(self, plot=False):
        data = self._sts_result.get_data()
        try:
            curs, freqs, self._data =\
                data["Current [A]"], data["frequency"], data["data"]
        except:
            curs, freqs, self._data =\
                data["current"], data["frequency"], data["data"]
        data = self._data
        res_freqs = []

        # Taking peaks deeper than half of the distance between the median
        # transmission level and the deepest point
        threshold = abs(data).min()+0.5*(median(abs(data))-abs(data).min())

        res_points = []
        for idx, row in enumerate(data):
            row = abs(row)
            extrema = argrelextrema(row, less, order=10)[0]
            extrema = extrema[row[extrema]<threshold]

            if len(extrema)>0:
                RD = ResonatorDetector(freqs, data[idx], plot=False)
                result = RD.detect()
                if result is not None:
                    res_points.append((curs[idx], result[0]))
                else:
                    smallest_extremum = extrema[argmin(row[extrema])]
                    res_points.append((curs[idx], freqs[smallest_extremum]))

        self._res_points = array(res_points)
        self._freqs = freqs
        self._curs = curs

        if self._plot:
            plt.figure()
            plt.plot(self._res_points[:,0], self._res_points[:,1], 'C1.',
                        label="Extracted points")
            plt.pcolormesh(curs, freqs, abs(data).T)
            plt.legend()
            plt.gcf().set_size_inches(15,5)

    def find_resonator_intersections(self):
        plt.plot(self._res_points[:,0],
                    ones_like(self._res_points[:,1])\
                        *mean(self._res_points[:,1]), 'C2',
                    label="Mean")
        data = (mean(self._res_points[:,1])-self._res_points[:,1])

        raw_intersections = where((data[:-1]*data[1:])<0)[0]+1

        refined_intersections = []
        current_intersection = raw_intersections[0]
        refined_intersections.append(current_intersection)

        for raw_intersection in raw_intersections[1:]:
            distance = raw_intersection - current_intersection
            if distance > self._minimum_points_between_zeroes:
                current_intersection = raw_intersection
                refined_intersections.append(current_intersection)

        fine_intersections = []

        points_around_zero = self._minimum_points_around_zero
        distance_from_intersection = self._distance_from_intersection
        for intersection in refined_intersections:
            if intersection<=distance_from_intersection\
                        or  intersection>=(len(data)-distance_from_intersection):
                continue

            if points_around_zero-1>=intersection:
                left_edge = 0
                right_edge = intersection+points_around_zero

            elif points_around_zero-1<\
                    intersection<\
                        (len(data)-points_around_zero):
                left_edge = intersection-points_around_zero
                right_edge = intersection+points_around_zero
            else:
                right_edge = len(data)
                left_edge = intersection-points_around_zero

            left_points = data[left_edge:intersection-distance_from_intersection]
            right_points = data[intersection+distance_from_intersection:right_edge]

            self._logger.debug("Intersection: "+str(intersection)\
                                +", points: "+str((left_points, right_points)))

            if len(set(left_points<0))==1\
                and len(set(right_points<0))==1\
                and left_points[0]*right_points[0]<0:
                fine_intersections.append(intersection)

        self._logger.debug("Fine intersections: "+str(fine_intersections))
        return fine_intersections

    @staticmethod
    def _transmission(f_q, f_probe, f_r, g):
        κ = 1e6
        γ = 1e6
        return abs(1/(κ+1j*(f_r-f_probe)+(g**2)/(γ+1j*(f_q-f_probe))))**2

    @staticmethod
    def _eigenlevels(f_q, f_r, g):
        E0 = (f_r - f_q)/2
        E1 = f_r-1/2*sqrt(4*g**2+(f_q-f_r)**2)
        E2 = f_r+1/2*sqrt(4*g**2+(f_q-f_r)**2)
        return array([E0-E0, E1-E0, E2-E0])


    def _model(self, curs, params, plot_colours = False, freqs_fine_number=5e3):

        f_r, g = params[:2]
        qubit_params = params[2:]
    #     phis_fine = linspace(phis[0],phis[-1], 1000)
        f_qs = self._qubit_spectrum(curs, *qubit_params)

        span = ptp(self._freqs)
        freqs_fine = linspace(self._freqs[0]-span*0.5,
                                self._freqs[-1]+span*0.5,
                                    freqs_fine_number)

        XX, YY = meshgrid(f_qs, freqs_fine)
        transmissions = self._transmission(XX, YY, f_r, g)

        if plot_colours:
            plt.pcolormesh(curs, freqs_fine, transmissions)
            plt.colorbar()

        res_freqs_model = []
        for row in transmissions.T:
            res_freqs_model.append(freqs_fine[argmax(abs(row))])
        return array(res_freqs_model)

    def _model_fast(self, curs, params, plot = False):
        f_r, g = params[:2]
        qubit_params = params[2:]
    #     phis_fine = linspace(phis[0],phis[-1], 1000)
        f_qs = self._qubit_spectrum(curs, *qubit_params)

        span = ptp(self._freqs)
        levels = self._eigenlevels(f_qs, f_r, g)

        if plot:
            plt.plot(curs, levels[0,:])
            plt.plot(curs, levels[1,:])
            plt.plot(curs, levels[2,:])
            plt.ylim(self._freqs[0], self._freqs[-1])

        res_freqs_model = zeros_like(curs)+mean(self._freqs)
        idcs1 = where(logical_and(self._freqs[0]<levels[1,:],
                        levels[1,:]<self._freqs[-1]))
        idcs2 = where(logical_and(self._freqs[0]<levels[2,:],
                        levels[2,:]<self._freqs[-1]))
        res_freqs_model[idcs1] = levels[1,:][idcs1]
        res_freqs_model[idcs2] = levels[2,:][idcs2]

        return array(res_freqs_model)

    def _cost_function(self, params, curs, res_freqs, freqs_fine_number = 5e3):

        if self._fast:
            cost = abs(self._model_fast(curs, params) - res_freqs)
        else:
            cost = abs(self._model(curs, params,
                        freqs_fine_number=freqs_fine_number) - res_freqs)
        # clear_output(wait=True)
        # print((("{:.4e}, "*len(params))[:-2]).format(*params), "loss:",
        #         "%.2f"%(sum(cost)/len(curs)*1000), "MHz")
        return sum(cost)

    def get_res_points(self):
        return self._res_points*1e9

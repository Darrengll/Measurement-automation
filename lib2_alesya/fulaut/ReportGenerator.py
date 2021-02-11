import os
import subprocess

from loggingserver import LoggingServer
from numpy import ptp, real, imag, mean, pi

from lib2.MeasurementResult import MeasurementResult
from matplotlib import pyplot as plt

from lib2.fulaut import qubit_spectra
from lib2.fulaut.AnticrossingOracle import AnticrossingOracle
from lib2.fulaut.SpectrumOracle import SpectrumOracle


class ReportGenerator:


    def __init__(self, sample_name, qubit_names, measurement_names):

        self._sample_name = sample_name
        self._qubit_names = qubit_names
        self._measurement_names = measurement_names
        self._results = []
        self._res_freqs = {}
        self._f_q_maxs = {}
        self._ds = {}
        self._T1s = {}
        self._T2s = {}
        self._logger = LoggingServer.getInstance("report_gen")
        self._only_pdflatex = False

    def generate(self, only_pdflatex = False):

        try:
            os.mkdir("data/" + self._sample_name + "/Report")
        except FileExistsError:
            pass

        self._only_pdflatex = only_pdflatex

        self._load_results()

        self._save_sts_results()
        self._save_tts_results()
        self._save_rabi_results()
        self._save_ramsey_results()
        self._save_decay_results()

        self._generate_tex()



    def _load_results(self):

        self._results = []
        for qubit_name in self._qubit_names:
            qubit_resuts = []
            for measurement_name in self._measurement_names:
                qubit_resuts.append(
                    MeasurementResult.load(self._sample_name,
                                           qubit_name + "-" + measurement_name,
                                           return_all=True))
            self._results.append(qubit_resuts)

    def _save_sts_results(self):

        for idx, qubit_results in enumerate(self._results):
            sts = qubit_results[0][-1]

            self._res_freqs[self._qubit_names[idx]] = mean(sts.get_data()["Frequency [Hz]"])/1e9

            if self._only_pdflatex:
                continue

            fig, axes, caxes = sts.visualize()
            axes[0].set_title(self._qubit_names[idx], fontsize=16, loc="left")
            axes[1].set_visible(False)
            caxes[1].set_visible(False)
            fig.set_size_inches(10, 2.5)
            plt.savefig("data/%s/Report/%s-sts.pdf" % (self._sample_name,
                                                       self._qubit_names[idx]),
                        bbox_inches="tight")
            plt.close("all")

    def _save_tts_results(self):
        for idx_q, qubit_results in enumerate(self._results):
            best_two_tone = None
            best_resolution = 100
            for two_tone in qubit_results[1]:
                resolution = len(two_tone.get_data()["data"].ravel())
                if resolution >= best_resolution:
                    best_resolution = resolution
                    best_two_tone = two_tone

            if not hasattr(best_two_tone, "_fit_params"):
                ao = AnticrossingOracle("transmon",
                                        sts_result = qubit_results[0][-1],
                                        plot=False,
                                        fast_res_detect=False,
                                        hints=["fqmax_below"])
                params, loss = ao.launch()
                so =SpectrumOracle("transmon", best_two_tone,
                                   params[2:], plot=False)
                best_two_tone._fit_params = so.launch()
                best_two_tone.save()


            self._f_q_maxs[self._qubit_names[idx_q]] = best_two_tone._fit_params[-3]/1e9
            self._ds[self._qubit_names[idx_q]] = best_two_tone._fit_params[-2]

            if self._only_pdflatex:
                continue

            fig, axes, caxes = best_two_tone.visualize()
            axes[0].set_title(self._qubit_names[idx_q], fontsize=16, loc="left")

            bias_values = best_two_tone.get_data()[best_two_tone._parameter_names[0]]
            axes[0].plot(bias_values, 1e-9*qubit_spectra.transmon_spectrum(bias_values,
                                                                      *best_two_tone._fit_params[:-1]),
                         ls = "--", color="black")

            axes[1].set_visible(False)
            caxes[1].set_visible(False)
            fig.set_size_inches(10, 2.5)
            plt.savefig("data/%s/Report/%s-tts.pdf" % (self._sample_name,
                                                       self._qubit_names[idx_q]),
                        bbox_inches="tight")
            plt.close("all")

    def _save_rabi_results(self):
        for idx, qubit_results in enumerate(self._results):
            best_rabi_result = None
            smallest_fit_error = 1000
            for rabi_result in qubit_results[2]:
                if rabi_result._fit_errors[3] < smallest_fit_error:
                    smallest_fit_error = rabi_result._fit_errors[3]
                    best_rabi_result = rabi_result

            if self._only_pdflatex:
                continue

            fig, axes, caxes = best_rabi_result.visualize()

            data = (best_rabi_result.get_data()["data"])
            self._truncate_td_plot(fig, axes, data, best_rabi_result._name)

            plt.savefig("data/%s/Report/%s-rabi.pdf" % (self._sample_name,
                                                        self._qubit_names[idx]),
                        bbox_inches="tight")
            plt.close("all")


    def _save_ramsey_results(self):

        for idx, qubit_results in enumerate(self._results):
            best_ramsey_result = None
            best_T2 = 0
            #     print(qubit_names[idx])

            for ramsey_result in qubit_results[3]:
                #         print("%.2f"%ramsey_result._fit_params[2])
                max_ramsey_delay = max(ramsey_result.get_data()["ramsey_delay"]) / 1e3
                if max_ramsey_delay > ramsey_result._fit_params[2] > best_T2:
                    best_T2 = ramsey_result._fit_params[2]
                    best_ramsey_result = ramsey_result

            self._T2s[self._qubit_names[idx]] = best_T2

            if self._only_pdflatex:
                continue

            fig, axes, caxes = best_ramsey_result.visualize()

            data = (best_ramsey_result.get_data()["data"])

            self._truncate_td_plot(fig, axes, data, best_ramsey_result._name)

            plt.savefig("data/%s/Report/%s-ramsey.pdf" % (self._sample_name,
                                                          self._qubit_names[idx]),
                        bbox_inches="tight")
            plt.close("all")


    def _save_decay_results(self):
        for idx, qubit_results in enumerate(self._results):
            best_decay = None
            best_T1 = 0
            #     print(qubit_names[idx])

            for decay in qubit_results[4]:
                #         print("%.2f"%ramsey_result._fit_params[2])
                max_decay_delay = max(decay.get_data()["readout_delay"]) / 1e3
                if max_decay_delay/2 > decay._fit_params[2] > best_T1:
                    best_T1 = decay._fit_params[2]
                    best_decay = decay

            self._T1s[self._qubit_names[idx]] = best_T1

            if self._only_pdflatex:
                continue

            fig, axes, caxes = best_decay.visualize()

            data = (best_decay.get_data()["data"])
            self._truncate_td_plot(fig, axes, data, best_decay._name+
                                   " ("+str(best_decay.get_start_datetime())+")")
            plt.savefig("data/%s/Report/%s-decay.pdf" % (self._sample_name,
                                                         self._qubit_names[idx]),
                        bbox_inches="tight")
            plt.close("all")

    def _truncate_td_plot(self, fig, axes, data, fig_name):
        working_ax = None
        if ptp(real(data)) > ptp(imag(data)):
            axes[1].set_visible(False)
            working_ax = axes[0]
            axes[0].xaxis.set_tick_params(labelbottom=True)
            axes[0].set_xlabel(axes[1].get_xlabel())
        else:
            axes[0].set_visible(False)
            working_ax = axes[1]

        working_ax.set_title(fig_name)
        fig.set_size_inches(5, 5)


    def _center(self, string):
        return "\\begin{center} %s \\end{center}" % string


    def _generate_tex(self):
        sts_string = r""
        for idx, qubit_name in enumerate(self._qubit_names):
            if idx % 3 == 0:
                sts_string += "\n\n"
            sts_string += r"\includegraphics[width=.3\linewidth]{%s-sts}\quad" % qubit_name
        # print(sts_string)

        tts_string = r""
        for idx, qubit_name in enumerate(self._qubit_names):
            if idx % 3 == 0:
                tts_string += "\n\n"
            tts_string += r"\includegraphics[width=.3\linewidth]{%s-tts}\quad" % qubit_name

        td_string = r""
        for idx, qubit_name in enumerate(self._qubit_names):
            for idx2, meas_name in enumerate(self._measurement_names[-3:]):
                td_string += r"\includegraphics[width=.3\linewidth]{%s-%s}\quad" % (qubit_name, meas_name)
            td_string += "\n\n"

        table_string = "Name & $f_{res}$ & $f_q^{max}$ & d & $T_1$ @ SWS & $T_2$ @ SWS & Q \\\\ \n"

        for idx, qubit_name in enumerate(self._qubit_names):
            Q_factor_str = "%.2e"%(1e9*self._f_q_maxs[qubit_name]*2*pi*(self._T1s[qubit_name]*1e-6))
            Q_factor_base, Q_factor_power = Q_factor_str.split("e")

            table_string += "&".join([qubit_name] + ["%.2f"%param_dict[qubit_name] for param_dict in [self._res_freqs,
                                                                                            self._f_q_maxs,
                                                                                            self._ds,
                                                                                            self._T1s,
                                                                                            self._T2s]] +
                                     [r"$%s \cdot 10^{%s}$"%(Q_factor_base, Q_factor_power[2:])])
            if idx != len(self._qubit_names)-1:
                table_string += "\\\\ \n"

        print(table_string)

        tex_contents = r'''
        \documentclass{article}
        \usepackage{graphicx}
        \usepackage[margin=.1in]{geometry}
        \graphicspath{{data/%s/Report/}}
        
        \title{%s}
        \begin{document}
        \maketitle
        
        \section{Resonators}
        
        \includegraphics[width=\linewidth]{1910-103-5_Q_factors_and_freqs}
    
        \section{Single-tone spectroscopy}
        %s
    
        \section{Two-tone spectroscopy}
        %s
    
        \section{Time-domain}
        %s
        
        \section{Summary}
        
        \centering
        \begin{tabular}{ccccccc}
        %s
        \end{tabular}
    
        \end{document}
        ''' % (self._sample_name, self._sample_name,
               self._center(sts_string), self._center(tts_string), self._center(td_string),
               table_string)

        with open("data/%s/Report/report.tex" % self._sample_name, "w") as f:
            f.write(tex_contents)

        process = subprocess.Popen(["C:/Program Files/MiKTeX 2.9/miktex/bin/x64/pdflatex",
                                    "data/%s/Report/report.tex" % self._sample_name,
                                    "--output-directory=data/%s/Report/build" % self._sample_name],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(str(stdout, "utf-8"))



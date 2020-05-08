"""Main script for  controlling the calculation method of the IS spectrum.
"""

import os
import time
import datetime
import itertools
# The start method of the multiprocessing module was changed from python3.7 to python3.8.
# Instead of using 'fork', 'spawn' is the new default. To be able to use global
# variables across all parallel processes, the start method must be reset to 'fork'.
# See https://docs.python.org/3/library/multiprocessing.html for more info.
import multiprocessing as mp
mp.set_start_method('fork')

import matplotlib  # pylint: disable=C0413
import matplotlib.gridspec as grid_spec  # pylint: disable=C0413
import matplotlib.pyplot as plt  # pylint: disable=C0413
from matplotlib.backends.backend_pdf import PdfPages  # pylint: disable=C0413
import numpy as np  # pylint: disable=C0413
import scipy.signal as signal  # pylint: disable=C0413
import si_prefix as sip  # pylint: disable=C0413

from inputs import config as cf  # pylint: disable=C0413
from utils import spectrum_calculation as isr  # pylint: disable=C0413

# Customize matplotlib
matplotlib.rcParams.update({
    'text.usetex': True,
    'font.family': 'Ovo',
    'axes.unicode_minus': False,
    'pgf.texsystem': 'pdflatex'
})


class PlotClass:
    """Create a plot object that automatically will show the data created.
    """

    def __init__(self, plasma=False):
        """Make plots of an IS spectrum based on a variety of VDFs.

        Keyword Arguments:
            plasma {bool} -- choose to plot only the part of the spectrum where the plasma line is found (default: {False})
        """
        self.save = input('Press "y/yes" to save plot, any other key to dismiss.\t').lower()
        self.page = 1
        self.pdffig = None
        self.save_path = str
        self.plasma = plasma
        self.correct_inputs()
        self.line_styles = ['-', '--', ':', '-.',
                            (0, (3, 5, 1, 5, 1, 5)),
                            (0, (3, 1, 1, 1, 1, 1))]

    def correct_inputs(self):
        """Extra check suppressing the parameters that was given but is not necessary.
        """
        if self.plasma:
            if self.find_p_line(None, None, None, check=True):
                print(f"F_MAX (= {cf.I_P['F_MAX']}) is not high enough to look at the plasma line. Overrides option.")
                self.plasma = False

    def save_it(self, version, params):
        """Save the figure as a multi page pdf with all parameters saved in the meta data.

        The date and time is used in the figure name, in addition to it ending with which method was used.
        The settings that was used in config as inputs to the plot object is saved in the metadata of the figure.
        """
        params.insert(0, {'F_MAX': cf.I_P['F_MAX'], 'F0': cf.I_P['F0'], 'V:MAX': cf.V_MAX, 'F_N_POINTS': cf.F_N_POINTS,
                       'Y_N_POINTS': cf.Y_N_POINTS, 'V_N_POINTS': cf.V_N_POINTS})
        tt = time.localtime()
        the_time = f'{tt[0]}_{tt[1]}_{tt[2]}_{tt[3]}--{tt[4]}--{tt[5]}'
        os.makedirs('../../../report/master-thesis/figures', exist_ok=True)
        self.save_path = f'../../../report/master-thesis/figures/{the_time}_{version}'
        self.pdffig = PdfPages(str(self.save_path) + '.pdf')
        metadata = self.pdffig.infodict()
        metadata['Title'] = f'ISR Spectrum w/ {version}'
        metadata['Author'] = 'Eirik R. Enger'
        metadata['Subject'] = f"IS spectrum made using a {version} distribution and Simpson's integration rule."
        metadata['Keywords'] = f'{params}'
        metadata['ModDate'] = datetime.datetime.today()

    def plot_normal(self, f, Is, func_type, l_txt):
        """Make a plot using f as x-axis scale and Is as values.

        Is may be either a (N,) or (N,1) np.ndarray or a list of such arrays.

        Arguments:
            f {np.ndarray} -- variable along x-axis
            Is {list} -- y-axis values along x-axis
            func_type {str} -- attribute of the matplotlib.pyplot object
        """
        try:
            getattr(plt, func_type)
        except Exception:
            print(
                f'{func_type} is not an attribute of the matplotlib.pyplot object. Using "plot".')
            func_type = 'plot'
        if len(Is) != len(l_txt):
            print('Warning: The number of spectra does not match the number of labels.')
        Is = Is.copy()
        # Linear plot show only ion line (kHz range).
        if func_type == 'plot' and not self.plasma:
            f, Is = self.only_ionline(f, Is)
        p, freq, exp = self.scale_f(f)
        plt.figure(figsize=(6, 3))
        if self.plasma:
            # Clip the frequency axis around the plasma frequency.
            mask = self.find_p_line(freq, Is, exp)
            freq = freq[mask]
        if func_type == 'semilogy':
            # Rescale the y-axis to a dB scale.
            plt.xlabel(f'Frequency [{p}Hz]')
            plt.ylabel(
                '10*log10(Power) [dB]')
            for i, _ in enumerate(Is):
                Is[i] = 10 * np.log10(Is[i])
        else:
            plt.xlabel(f'Frequency [{p}Hz]')
            plt.ylabel('Power')
        for st, s, lab in zip(itertools.cycle(self.line_styles), Is, l_txt):
            if self.plasma:
                s = s[mask]
            if func_type == 'semilogy':
                plt.plot(freq, s, 'r', linestyle=st,
                         linewidth=.8, label=lab)
            else:
                plot_object = getattr(plt, func_type)
                plot_object(freq, s, 'r', linestyle=st,
                            linewidth=.8, label=lab)
        plt.legend()
        plt.minorticks_on()
        plt.grid(True, which="both", ls="-", alpha=0.4)
        plt.tight_layout()

        if self.save in ['y', 'yes']:
            self.pdffig.attach_note(func_type)
            plt.savefig(self.pdffig, bbox_inches='tight', format='pdf', dpi=600)
            plt.savefig(str(self.save_path) + f'_page_{self.page}.pgf', bbox_inches='tight')
            self.page += 1

    def plot_ridge(self, frequency, multi_parameters, func_type, l_txt, ridge_txt=None):
        # Inspired by https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
        # To make sure not to alter any list objects, they are copied.
        try:
            getattr(plt, func_type)
        except Exception:
            print(
                f'{func_type} is not an attribute of the matplotlib.pyplot object. Using "plot".')
            func_type = 'plot'
        if len(multi_parameters) != len(ridge_txt):
            print('Warning: The list if spectra lists is not of the same length as the length of "ridge_txt"')
        f_original = frequency.copy()
        multi_params = multi_parameters.copy()
        multi_params.reverse()
        if ridge_txt is None:
            ridge_txt = ['' for _ in multi_params]
        else:
            ridge_txt.reverse()
        gs = grid_spec.GridSpec(len(multi_params), 1)
        fig = plt.figure(figsize=(7, 9))
        ax_objs = []
        Rgb = np.linspace(0, 1, len(multi_params))
        # If you want equal scaling of the y axis as well
        # y_min, y_max = self.scaling_y(multi_params)
        for j, params in enumerate(multi_params):
            if len(params) != len(l_txt):
                print('Warning: The number of spectra does not match the number of labels.')
            # f is reset due to the scaling of 'plot' immediately below.
            f = f_original
            # Linear plot show only ion line (kHz range).
            if func_type == 'plot' and not self.plasma:
                f, params = self.only_ionline(f, params)
            p, freq, exp = self.scale_f(f)
            if self.plasma:
                mask = self.find_p_line(freq, params, exp)
                freq = freq[mask]
            ax_objs.append(fig.add_subplot(gs[j:j + 1, 0:]))
            # if list...
            first = 0
            for st, s, lab in zip(itertools.cycle(self.line_styles), params, l_txt):
                if self.plasma:
                    s = s[mask]
                plot_object = getattr(ax_objs[-1], func_type)
                plot_object(freq, s, color=(Rgb[j], 0., 1 - Rgb[j]), linewidth=1, label=lab, linestyle=st)
                if first == 0:
                    idx = np.argwhere(freq > ax_objs[-1].viewLim.x0)[0]
                    legend_pos = (ax_objs[-1].viewLim.x1, np.max(s))
                    y0 = s[idx]
                    ax_objs[-1].text(freq[idx], s[idx], ridge_txt[j],
                                        fontsize=14, ha="right", va='bottom')
                first += 1
                # ax_objs[-1].fill_between(freq, s, alpha=1, color=(Rgb[j], 0., 1 - Rgb[j]))
                if j == 0:
                    plt.legend(loc='upper right', bbox_to_anchor=legend_pos, bbox_transform=ax_objs[-1].transData)
            # else:
            #     if self.plasma:
            #         params = params[mask]
            #     plot_object = getattr(ax_objs[-1], func_type)
            #     plot_object(freq, params, color=(Rgb[j], 0., 1 - Rgb[j]), linewidth=1)
            #     idx = np.argwhere(freq > ax_objs[-1].viewLim.x0)[0]
            #     y0 = params[idx]
            #     ax_objs[-1].text(freq[idx], params[idx], ridge_txt[j],
            #                      fontsize=14, ha="right", va='bottom')
            #     # ax_objs[-1].fill_between(freq, params, alpha=1, color=(Rgb[j], 0., 1 - Rgb[j]))

            # plt.ylim([y_min, y_max])
            if func_type == 'plot':
                # Make a vertical line of comparable size in all plots.
                self.match_box(f_original, freq, multi_params, [y0, j])

            self.remove_background(ax_objs[-1], multi_params, j, p)

        gs.update(hspace=-0.6)
        # plt.tight_layout()
        if self.save in ['y', 'yes']:
            self.pdffig.attach_note(func_type)
            plt.savefig(self.pdffig, bbox_inches='tight', format='pdf', dpi=600)
            plt.savefig(str(self.save_path) + f'_page_{self.page}.pgf', bbox_inches='tight')
            self.page += 1

    @staticmethod
    def remove_background(plt_obj, multi_params, j, p):
        # make background transparent
        rect = plt_obj.patch
        rect.set_alpha(0)
        # remove borders, axis ticks and labels
        plt_obj.set_yticklabels([])
        plt.tick_params(axis='y', which='both', left=False,
                        right=False, labelleft=False)
        if j == len(multi_params) - 1:
            plt.xlabel(f'Frequency [{p}Hz]')
        else:
            plt.tick_params(axis='x', which='both', bottom=False,
                            top=False, labelbottom=False)

        spines = ["top", "right", "left", "bottom"]
        for sp in spines:
            plt_obj.spines[sp].set_visible(False)

    @staticmethod
    def rename_labels(kappa):
        if any([isinstance(kappa_i, int) for kappa_i in kappa]):
            for v, kappa_i in enumerate(kappa):
                kappa[v] = r'$\kappa = {}$'.format(kappa_i)
        if 'Maxwellian' not in kappa:
            kappa.insert(0, 'Maxwellian')

    @staticmethod
    def scale_f(frequency):
        """Scale the axis and add the appropriate SI prefix.

        Arguments:
            frequency {np.ndarray} -- the variable along an axis

        Returns:
            str, np.ndarray, int -- the prefix, the scaled variables, the exponent corresponding to the prefix
        """
        freq = np.copy(frequency)
        exp = sip.split(np.max(freq))[1]
        freq /= 10**exp
        pre = sip.prefix(exp)
        return pre, freq, exp

    @staticmethod
    def find_p_line(freq, spectrum, scale, check=False):
        """Find the frequency that is most likely the peak of the plasma line
        and return the lower and upper bounds for an interval around the peak.

        Arguments:
            freq {np.ndarray} -- sample points of frequency parameter
            spectrum {np.ndarray} -- values of spectrum at the sampled frequencies
            scale {int} -- exponent corresponding to the prefix of the frequency scale
            temp {int} -- electron temperature

        Returns:
            np.ndarray -- array with boolean elements
        """
        # if isinstance(spectrum, list):
        spec = spectrum[0]
        # else:
        #     spec = spectrum
        p, _ = signal.find_peaks(spec)[-1]
        f = freq[p]
        # if isinstance(cf.I_P['NE'], list):
        #     n_e = cf.I_P['NE'][0]
        # else:
        #     n_e = cf.I_P['NE']
        # w_p = np.sqrt(n_e * const.elementary_charge**2
        #               / (const.m_e * const.epsilon_0))
        # f = w_p * (1 + 3 * cf.K_RADAR**2 *
        #            temp * const.k / (const.m_e * w_p**2))**.5 / (2 * np.pi)
        if check:
            upper = f + 1e6
            return bool(upper > cf.I_P['F_MAX'])
        fr = np.copy(freq)
        sp = np.copy(spec)
        lower, upper = (f - 1e6) / 10**scale, (f + 1e6) / 10**scale
        m = (fr > lower) & (fr < upper)
        fr_n = fr[m]
        sp = sp[m]
        av = fr_n[np.argmax(sp)]
        lower, upper = av - 2e6 / 10**scale, av + 2e6 / 10**scale
        # Don't want the ion line to ruin the scaling
        if lower < 1e5 / 10**scale:
            lower = 1e5 / 10**scale
        return (freq > lower) & (freq < upper)

    @staticmethod
    def only_ionline(f, Is):
        Is = Is.copy()
        idx = np.argwhere(abs(f) < 4e4)
        f = f[idx].reshape((-1,))
        if isinstance(Is, list):
            for i, _ in enumerate(Is):
                Is[i] = Is[i][idx].reshape((-1,))
        else:
            Is = Is[idx].reshape((-1,))
        return f, Is

    @staticmethod
    def scaling_y(multi_params):
        y_min, y_max = np.inf, - np.inf
        for params in multi_params:
            if isinstance(params, list):
                for s in params:
                    if y_min > np.min(s):
                        y_min = np.min(s)
                    if y_max < np.max(s):
                        y_max = np.max(s)
            else:
                if y_min > np.min(params):
                    y_min = np.min(params)
                if y_max < np.max(params):
                    y_max = np.max(params)
        return y_min, y_max

    def match_box(self, freq_original, freq, multi_parameters, args):
        multi_params = multi_parameters.copy()
        v_line_x = np.linspace(.04, .2, len(multi_params))
        if self.plasma:
            f = freq_original.copy()
            if isinstance(multi_params, list):
                spec = multi_params[0]
            else:
                spec = multi_params
            mask = self.find_p_line(f, spec, 0)
        diff = np.inf
        for params in multi_params:
            if isinstance(params, list):
                for s in params:
                    if self.plasma:
                        s = s[mask]
                    difference = np.max(s) - np.min(s)
                    if difference < diff:
                        diff = difference
            else:
                if self.plasma:
                    params = params[mask]
                difference = np.max(params) - np.min(params)
                if difference < diff:
                    diff = difference

        x0 = np.min(freq) + (np.max(freq) - np.min(freq)) * v_line_x[args[1]]
        plt.vlines(x=x0, ymin=args[0],
                   ymax=args[0] + int(np.ceil(diff / 10) * 5), color='k', linewidth=3)
        plt.text(x0, args[0] + int(np.ceil(diff / 10) * 5) / 2,
                 r'${}$'.format(int(np.ceil(diff / 10) * 5)), rotation=90, ha='right', va='center')


class Simulation:
    def __init__(self):
        # === Input parameters ===
        # B -- Magnetic field strength [T]
        # F0 -- Radar frequency [Hz]
        # F_MAX -- Range of frequency domain [Hz]
        # MI -- Ion mass in atomic mass units [u]
        # NE -- Electron number density [m^(-3)]
        # NU_E -- Electron collision frequency [Hz]
        # NU_I -- Ion collision frequency [Hz]
        # T_E -- Electron temperature [K]
        # T_I -- Ion temperature [K]
        # T_ES -- Temperature of suprathermal electrons in the gauss_shell VDF [K]
        # THETA -- Pitch angle [1]
        # Z -- Height of real data [100, 599] [km]
        # mat_file -- Important when using real data and decides the time of day
        self.f = np.ndarray([])
        self.data = []
        self.meta_data = []
        self.legend_txt = []
        self.ridge_txt = []
        self.plot = PlotClass()

    def create_data(self):
        # Message for temperature
        # ridge_txt = [r'$T_e = {}$'.format(j) + ' K' for j in cf.I_P['T_E']]
        # Message for height
        # ridge_txt = [r'${}$'.format(j) + ' km' for j in cf.I_P['Z']]
        # Message for ToD
        # the_time = [8 + (int(j.split('-')[-1].split('.')[0]) + 1) / 2 for j in cf.I_P['mat_file']]
        # ridge_txt = [f"ToD: {int(j):02d}:{int(j * 60 % 60):02d} UT" for j in the_time]
        self.ridge_txt.append('One')

        self.legend_txt.append('Maxwellian')
        sys_set = {'B': 5e-4, 'MI': 16, 'NE': 2e11, 'NU_E': 0, 'NU_I': 0, 'T_E': 5000, 'T_I': 2000, 'T_ES': 90000,
                   'THETA': 40 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-01.mat'}
        params = {'kappa': 3, 'vdf': 'kappa', 'area': False}
        self.f, s, meta_data = isr.isr_spectrum('maxwell', sys_set, **params)
        self.data.append(s)
        # Add data parameters to meta_data
        self.meta_data.append(meta_data)

        self.legend_txt.append('Kappa')
        sys_set = {'B': 5e-4, 'MI': 16, 'NE': 2e10, 'NU_E': 0, 'NU_I': 0, 'T_E': 5000, 'T_I': 2000, 'T_ES': 90000,
                   'THETA': 40 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-01.mat'}
        _, s, meta_data = isr.isr_spectrum('kappa', sys_set, **params)
        self.data.append(s)
        # Add data parameters to meta_data
        self.meta_data.append(meta_data)

    def save_handle(self, mode):
        if self.plot.save in ['y', 'yes']:
            if mode == 'setUp':
                self.plot.save_it('m_k', self.meta_data)
            elif mode == 'tearDown':
                self.plot.pdffig.close()
                plt.show()

    def plot_data(self):
        self.plot.plot_normal(self.f, self.data, 'plot', self.legend_txt)
        self.plot.plot_ridge(self.f, [self.data], 'plot', self.legend_txt, self.ridge_txt)

    def run(self):
        self.create_data()
        self.save_handle('setUp')
        self.plot_data()
        self.save_handle('tearDown')


if __name__ == '__main__':
    sim = Simulation()
    sim.run()

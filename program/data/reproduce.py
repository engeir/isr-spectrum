from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const

from utils import spectrum_calculation as isr
from inputs import config as cf


class ReproduceS(ABC):
    """Abstract base class for reproducible figures.

    Arguments:
        ABC {class} -- make it an abstract base class that all VDF objects should inherit from
    """

    @abstractmethod
    def create_it(self):
        """Method that create needed data.
        """

    @abstractmethod
    def plot_it(self):
        """Method that plot relevant plots.
        """


class PlotTestNumerical(ReproduceS):
    """Reproduce figure with ridge plot over different temperatures."""

    def __init__(self, p):
        self.p = p

    @classmethod
    def setUpClass(cls):
        F0 = 430e6
        K_RADAR = - 2 * F0 * 2 * np.pi / const.c  # Radar wavenumber
        cls.sys_set = {'K_RADAR': K_RADAR, 'B': 35000e-9, 'MI': 16, 'NE': 1e11, 'NU_E': 100, 'NU_I': 100, 'T_E': 2000, 'T_I': 1500, 'T_ES': 90000,
                       'THETA': 30 * np.pi / 180, 'Z': 300, 'mat_file': 'fe_zmuE-07.mat', 'pitch_angle': 'all'}
        cls.params = {'kappa': 3, 'vdf': 'maxwell', 'area': False}
        cls.f = None
        cls.s1 = None
        cls.s2 = None
        cls.maxwell = True
        cls.meta_data = []

    def tearDown(self):
        # In config, set 'F_MIN': - 2e6, 'F_MAX': 9e6
        # Also, using
        #     F_N_POINTS = 1e3
        # is sufficient.
        plot = plt.loglog
        xlim = [1e3, self.f[-1]]
        d = self.s1 - self.s2
        rd = d / self.s1
        plt.figure()
        plt.subplot(1, 3, 1)
        if self.maxwell == True:
            plt.title('Maxwell')
            self.maxwell = False
        else:
            plt.title('Kappa')
        plot(self.f, self.s1, 'k', label='Semi-analytic (SA)')
        plot(self.f, self.s2, 'r--', label='Numerical (N)')
        plt.legend()
        plt.xlim(xlim)
        plt.subplot(1, 3, 2)
        plt.title('Difference (SA - N)')
        plot(self.f, d, 'k', label='Positive')
        plot(self.f, - d, 'r', label='Negative')
        plt.legend()
        plt.xlim(xlim)
        plt.subplot(1, 3, 3)
        plt.title('Difference relative to semi-analytic ([SA - N] / SA)')
        plot(self.f, rd, 'k', label='Positive')
        plot(self.f, - rd, 'r', label='Negative')
        plt.legend()
        plt.xlim(xlim)

    def test_numerical_maxwell(self):
        self.f, self.s1, _ = isr.isr_spectrum('maxwell', self.sys_set, **self.params)
        _, self.s2, _ = isr.isr_spectrum('a_vdf', self.sys_set, **self.params)

    def test_numerical_kappa(self):
        self.params['vdf'] = 'kappa'
        _, self.s1, _ = isr.isr_spectrum('kappa', self.sys_set, **self.params)
        _, self.s2, _ = isr.isr_spectrum('a_vdf', self.sys_set, **self.params)

    def create_it(self):
        self.setUpClass()
    
    def plot_it(self):
        self.test_numerical_maxwell()
        self.tearDown()
        self.test_numerical_kappa()
        self.tearDown()
    
    def run(self):
        self.setUpClass()
        self.test_numerical_maxwell()
        self.tearDown()
        self.test_numerical_kappa()
        self.tearDown()
        plt.show()


class PlotTestDebye(ReproduceS):
    """Reproduce figure with ridge plot over different temperatures."""
    def __init__(self, p):
        # cf.I_P = {'F_MIN': - 2e6, 'F_MAX': 2e6}
        self.f = np.ndarray([])
        self.data = []
        self.meta_data = []
        self.legend_txt = []
        self.ridge_txt = []
        self.p = p

    def create_it(self):
        # In config, set 'F0': 430e6, 'F_MIN': - 2e6, 'F_MAX': 2e6
        # Also, using
        #     F_N_POINTS = 5e5
        # is sufficient.
        F0 = 430e6
        K_RADAR = - 2 * F0 * 2 * np.pi / const.c  # Radar wavenumber
        # Change the value of kappa in params (and in legend_txt) to obtain plots of different kappa value.
        self.legend_txt = [r'$\lambda_{\mathrm{D}} = \lambda_{\mathrm{D},\kappa}$', r'$\lambda_{\mathrm{D}} = \lambda_{\mathrm{D,M}}$']
        sys_set = {'K_RADAR': K_RADAR, 'B': 35000e-9, 'MI': 29, 'NE': 2e10, 'NU_E': 0, 'NU_I': 0, 'T_E': 200, 'T_I': 200, 'T_ES': 90000,
                   'THETA': 45 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-07.mat'}
        params = {'kappa': 3, 'vdf': 'real_data', 'area': False}
        self.f, s, meta_data = isr.isr_spectrum('kappa', sys_set, **params)
        self.data.append(s)
        self.meta_data.append(meta_data)
        params['debye'] = 'maxwell'
        self.f, s, meta_data = isr.isr_spectrum('kappa', sys_set, **params)
        self.data.append(s)
        self.meta_data.append(meta_data)

    def plot_it(self):
        self.p.plot_normal(self.f, self.data, 'semilogy', self.legend_txt)


class PlotMaxwell(ReproduceS):
    """Reproduce figure with ridge plot over different temperatures."""
    def __init__(self, p):
        self.f = np.ndarray([])
        self.data = []
        self.meta_data = []
        self.legend_txt = []
        self.ridge_txt = []
        self.p = p

    def create_it(self):
        F0 = 430e6
        K_RADAR = - 2 * F0 * 2 * np.pi / const.c  # Radar wavenumber
        # In config, set 'F_MIN': - 2e6, 'F_MAX': 2e6
        # Also, using
        #     F_N_POINTS = 5e5
        # is sufficient.
        self.legend_txt = ['Maxwellian']
        sys_set = {'K_RADAR': K_RADAR, 'B': 35000e-9, 'MI': 29, 'NE': 2e10, 'NU_E': 0, 'NU_I': 0, 'T_E': 200, 'T_I': 200, 'T_ES': 90000,
                   'THETA': 45 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-07.mat'}
        params = {'kappa': 8, 'vdf': 'real_data', 'area': False}
        self.f, s, meta_data = isr.isr_spectrum('maxwell', sys_set, **params)
        self.data.append(s)
        self.meta_data.append(meta_data)

    def plot_it(self):
        self.p.plot_normal(self.f, self.data, 'semilogy', self.legend_txt)


class PlotKappa(ReproduceS):
    """Reproduce figure with ridge plot over different temperatures."""
    def __init__(self, p):
        self.f = np.ndarray([])
        self.data = []
        self.meta_data = []
        self.legend_txt = []
        self.ridge_txt = []
        self.p = p

    def create_it(self):
        F0 = 430e6
        K_RADAR = - 2 * F0 * 2 * np.pi / const.c  # Radar wavenumber
        # In config, set 'F_MIN': - 2e6, 'F_MAX': 2e6
        # Also, using
        #     F_N_POINTS = 5e5
        # is sufficient.
        # Change the value of kappa in params (and in legend_txt) to obtain plots of different kappa value.
        self.legend_txt = [r'$\kappa = 20$']
        sys_set = {'K_RADAR': K_RADAR, 'B': 35000e-9, 'MI': 29, 'NE': 2e10, 'NU_E': 0, 'NU_I': 0, 'T_E': 200, 'T_I': 200, 'T_ES': 90000,
                   'THETA': 45 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-07.mat'}
        params = {'kappa': 20, 'vdf': 'real_data', 'area': False}
        self.f, s, meta_data = isr.isr_spectrum('kappa', sys_set, **params)
        self.data.append(s)
        self.meta_data.append(meta_data)

    def plot_it(self):
        self.p.plot_normal(self.f, self.data, 'semilogy', self.legend_txt)


class PlotSpectra(ReproduceS):
    """Reproduce figure with ridge plot over different temperatures."""
    def __init__(self, p):
        self.f = np.ndarray([])
        self.data = []
        self.meta_data = []
        self.legend_txt = []
        self.ridge_txt = []
        self.p = p

    def create_it(self):
        F0 = 430e6
        K_RADAR = - 2 * F0 * 2 * np.pi / const.c  # Radar wavenumber
        # In config, set 'F_MIN': - 2e6, 'F_MAX': 2e6
        # Also, using
        #     F_N_POINTS = 1e5
        # is sufficient.
        self.legend_txt = ['Maxwellian', r'$\kappa = 3$', r'$\kappa = 5$', r'$\kappa = 20$']
        kappa = [3, 5, 20]
        sys_set = {'K_RADAR': K_RADAR, 'B': 35000e-9, 'MI': 29, 'NE': 2e10, 'NU_E': 0, 'NU_I': 0, 'T_E': 200, 'T_I': 200, 'T_ES': 90000,
                   'THETA': 45 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-07.mat'}
        params = {'kappa': 20, 'vdf': 'real_data', 'area': False}
        self.f, s, meta_data = isr.isr_spectrum('maxwell', sys_set, **params)
        self.data.append(s)
        for k in kappa:
            params['kappa'] = k
            self.f, s, meta_data = isr.isr_spectrum('kappa', sys_set, **params)
            self.data.append(s)
        meta_data['version'] = 'both'
        self.meta_data.append(meta_data)

    def plot_it(self):
        self.p.plot_normal(self.f, self.data, 'semilogy', self.legend_txt)


class PlotIonLine(ReproduceS):
    """Reproduce figure with ridge plot over different temperatures."""
    def __init__(self, p):
        self.f = np.ndarray([])
        self.data = []
        self.meta_data = []
        self.legend_txt = []
        self.ridge_txt = []
        self.p = p

    def create_it(self):
        # In config, set 'F0': 430e6, 'F_MIN': - 3e3, 'F_MAX': 3e3
        # Also, using
        #     F_N_POINTS = 1e3
        # is sufficient.
        self.legend_txt = ['Maxwellian', r'$\kappa = 3$', r'$\kappa = 5$', r'$\kappa = 8$', r'$\kappa = 20$']
        kappa = [3, 5, 8, 20]
        sys_set = {'B': 35000e-9, 'MI': 29, 'NE': 2e10, 'NU_E': 0, 'NU_I': 0, 'T_E': 200, 'T_I': 200, 'T_ES': 90000,
                   'THETA': 45 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-07.mat'}
        params = {'kappa': 20, 'vdf': 'real_data', 'area': False}
        self.f, s, meta_data = isr.isr_spectrum('maxwell', sys_set, **params)
        self.data.append(s)
        for k in kappa:
            params['kappa'] = k
            self.f, s, meta_data = isr.isr_spectrum('kappa', sys_set, **params)
            self.data.append(s)
        meta_data['version'] = 'both'
        self.meta_data.append(meta_data)

    def plot_it(self):
        self.p.plot_normal(self.f, self.data, 'plot', self.legend_txt)


class PlotPlasmaLine(ReproduceS):
    """Reproduce figure with ridge plot over different temperatures."""
    def __init__(self, p):
        self.f = np.ndarray([])
        self.data = []
        self.meta_data = []
        self.legend_txt = []
        self.ridge_txt = []
        self.p = p

    def create_it(self):
        # In config, set 'F0': 933e6, 'F_MIN': 3.5e6, 'F_MAX': 7e6
        # Also, using
        #     F_N_POINTS = 1e3
        # is sufficient.
        self.legend_txt = ['Maxwellian', r'$\kappa = 3$', r'$\kappa = 5$', r'$\kappa = 8$', r'$\kappa = 20$']
        kappa = [3, 5, 8, 20]
        sys_set = {'B': 50000e-9, 'MI': 16, 'NE': 2e11, 'NU_E': 0, 'NU_I': 0, 'T_E': 5000, 'T_I': 2000, 'T_ES': 90000,
                   'THETA': 0 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-07.mat'}
        params = {'kappa': 20, 'vdf': 'real_data', 'area': False}
        self.f, s, meta_data = isr.isr_spectrum('maxwell', sys_set, **params)
        self.data.append(s)
        for k in kappa:
            params['kappa'] = k
            self.f, s, meta_data = isr.isr_spectrum('kappa', sys_set, **params)
            self.data.append(s)
        meta_data['version'] = 'both'
        self.meta_data.append(meta_data)

    def plot_it(self):
        self.p.plot_normal(self.f, self.data, 'plot', self.legend_txt)


class PlotTemperature(ReproduceS):
    """Reproduce figure with ridge plot over different temperatures."""
    def __init__(self, p):
        self.f = np.ndarray([])
        self.data = []
        self.meta_data = []
        self.legend_txt = []
        self.ridge_txt = []
        self.f_list = [[], [], []]
        self.p = p

    def create_it(self):
        # In config, set 'F0': 933e6, 'F_MIN': 3.5e6, 'F_MAX': 7.5e6
        # Also, using
        #     F_N_POINTS = 5e3
        # is sufficient.
        T = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        self.ridge_txt = [r'$T_{\mathrm{e}} = %d \mathrm{K}$' % j for j in T]
        print(self.ridge_txt[0])
        self.legend_txt = ['Maxwellian', r'$\kappa = 3$', r'$\kappa = 20$']
        sys_set = {'B': 50000e-9, 'MI': 16, 'NE': 2e11, 'NU_E': 0, 'NU_I': 0, 'T_E': 2000, 'T_I': 2000, 'T_ES': 90000,
                   'THETA': 0 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-07.mat'}
        params = {'kappa': 8, 'vdf': 'real_data', 'area': False}
        kappa = [3, 20]
        for t in T:
            ridge = []
            sys_set['T_E'] = t
            self.f, s, meta_data = isr.isr_spectrum('maxwell', sys_set, **params)
            ridge.append(s)
            for k in kappa:
                params['kappa'] = k
                self.f, s, meta_data = isr.isr_spectrum('kappa', sys_set, **params)
                ridge.append(s)
            self.data.append(ridge)
        self.meta_data.append(meta_data)
        
        for r in self.data:
            self.f_list[0].append(self.f[int(np.argwhere(r[0]==np.max(r[0])))])
            self.f_list[1].append(self.f[int(np.argwhere(r[1]==np.max(r[1])))])
            self.f_list[2].append(self.f[int(np.argwhere(r[2]==np.max(r[2])))])

    def plot_it(self):
        self.p.plot_ridge(self.f, self.data, 'plot', self.legend_txt, self.ridge_txt)
        
        T = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        plt.figure()
        plt.plot(T, self.f_list[0], 'k', label='Maxwellian')
        plt.plot(T, self.f_list[1], 'k--', label=r'$\kappa = 3$')
        plt.plot(T, self.f_list[2], 'k:', label=r'$\kappa = 20$')
        plt.legend()


class PlotHKExtremes(ReproduceS):
    """Reproduce figure with ridge plot over different temperatures."""
    def __init__(self, p):
        self.f = np.ndarray([])
        self.data = []
        self.meta_data = []
        self.legend_txt = []
        self.ridge_txt = []
        self.p = p

    def create_it(self):
        F0 = 430e6
        K_RADAR = - 2 * F0 * 2 * np.pi / const.c  # Radar wavenumber
        # In config, set 'F_MIN': 2.5e6, 'F_MAX': 9.5e6
        # Also, using
        #     F_N_POINTS = 1e4
        # is sufficient.
        sys_set = {'K_RADAR': K_RADAR, 'B': 35000e-9, 'MI': 16, 'NE': 1e11, 'NU_E': 100, 'NU_I': 100, 'T_E': 2000, 'T_I': 1500, 'T_ES': 90000,
                   'THETA': 30 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-07.mat', 'pitch_angle': list(range(10))}
        params = {'kappa': 8, 'vdf': 'real_data', 'area': False}
        ridge = []
        self.f, s, meta_data = isr.isr_spectrum('a_vdf', sys_set, **params)
        ridge.append(s)
        self.meta_data.append(meta_data)
        # f = (sys_set['NE'] * const.elementary_charge**2 / (const.m_e * const.epsilon_0))**.5 / (2 * np.pi)
        # print(f)
        sys_set['NE'] = 1e12
        self.f, s, meta_data = isr.isr_spectrum('a_vdf', sys_set, **params)
        ridge.append(s)
        self.data.append(ridge)
        self.meta_data.append(meta_data)
        ridge = []
        # f = (sys_set['NE'] * const.elementary_charge**2 / (const.m_e * const.epsilon_0))**.5 / (2 * np.pi)
        # print(f)
        sys_set['THETA'] = 60 * np.pi / 180
        sys_set['NE'] = 1e11
        self.f, s, meta_data = isr.isr_spectrum('a_vdf', sys_set, **params)
        ridge.append(s)
        self.meta_data.append(meta_data)
        # f = (sys_set['NE'] * const.elementary_charge**2 / (const.m_e * const.epsilon_0))**.5 / (2 * np.pi)
        # print(f)
        sys_set['NE'] = 1e12
        self.f, s, meta_data = isr.isr_spectrum('a_vdf', sys_set, **params)
        ridge.append(s)
        self.data.append(ridge)
        self.meta_data.append(meta_data)
        # f = (sys_set['NE'] * const.elementary_charge**2 / (const.m_e * const.epsilon_0))**.5 / (2 * np.pi)
        # print(f)

        self.legend_txt = ['2e10', '2e11']
        self.ridge_txt = ['30', '60']

    def plot_it(self):
        self.p.plot_ridge(self.f, self.data, 'semilogy', self.legend_txt, self.ridge_txt)


# if __name__ == '__main__':
#     PlotTestNumerical().run()

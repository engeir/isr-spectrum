from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np

from utils import spectrum_calculation as isr


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


class PlotTestNumerical():
    """Reproduce figure with ridge plot over different temperatures."""

    @classmethod
    def setUpClass(cls):
        cls.sys_set = {'B': 35000e-9, 'MI': 16, 'NE': 2e11, 'NU_E': 100, 'NU_I': 100, 'T_E': 2000, 'T_I': 1500, 'T_ES': 90000,
                       'THETA': 60 * np.pi / 180, 'Z': 300, 'mat_file': 'fe_zmuE-07.mat', 'pitch_angle': 'all'}
        cls.params = {'kappa': 3, 'vdf': 'maxwell', 'area': False}
        cls.f = None
        cls.s1 = None
        cls.s2 = None

    def tearDown(self):
        d = self.s1 - self.s2
        plt.figure()
        plt.plot(self.f, d)
        plt.show()

    def test_vdf_maxwell(self):
        self.f, self.s1, _ = isr.isr_spectrum('maxwell', self.sys_set, **self.params)
        self.f, self.s2, _ = isr.isr_spectrum('a_vdf', self.sys_set, **self.params)

    def test_vdf_kappa(self):
        self.params['vdf'] = 'kappa'
        self.f, self.s1, _ = isr.isr_spectrum('kappa', self.sys_set, **self.params)
        self.f, self.s2, _ = isr.isr_spectrum('a_vdf', self.sys_set, **self.params)

    def run(self):
        self.setUpClass()
        self.test_vdf_maxwell()
        self.tearDown()
        self.test_vdf_maxwell()
        self.tearDown()


class PlotTestDebye(ReproduceS):
    """Reproduce figure with ridge plot over different temperatures."""
    def __init__(self, p):
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
        #     Y_N_POINTS = 4e4
        #     V_N_POINTS = 1e5
        # is sufficient.
        # Change the value of kappa in params (and in legend_txt) to obtain plots of different kappa value.
        self.legend_txt = [r'$\kappa = 20$']
        sys_set = {'B': 35000e-9, 'MI': 29, 'NE': 2e10, 'NU_E': 0, 'NU_I': 0, 'T_E': 200, 'T_I': 200, 'T_ES': 90000,
                   'THETA': 45 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-07.mat'}
        params = {'kappa': 20, 'vdf': 'real_data', 'area': False}
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
        # In config, set 'F0': 430e6, 'F_MIN': - 2e6, 'F_MAX': 2e6
        # Also, using
        #     F_N_POINTS = 5e5
        #     Y_N_POINTS = 4e4
        #     V_N_POINTS = 1e5
        # is sufficient.
        self.legend_txt = ['Maxwellian']
        sys_set = {'B': 35000e-9, 'MI': 29, 'NE': 2e10, 'NU_E': 0, 'NU_I': 0, 'T_E': 200, 'T_I': 200, 'T_ES': 90000,
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
        # In config, set 'F0': 430e6, 'F_MIN': - 2e6, 'F_MAX': 2e6
        # Also, using
        #     F_N_POINTS = 5e5
        #     Y_N_POINTS = 4e4
        #     V_N_POINTS = 1e5
        # is sufficient.
        # Change the value of kappa in params (and in legend_txt) to obtain plots of different kappa value.
        self.legend_txt = [r'$\kappa = 20$']
        sys_set = {'B': 35000e-9, 'MI': 29, 'NE': 2e10, 'NU_E': 0, 'NU_I': 0, 'T_E': 200, 'T_I': 200, 'T_ES': 90000,
                   'THETA': 45 * np.pi / 180, 'Z': 599, 'mat_file': 'fe_zmuE-07.mat'}
        params = {'kappa': 20, 'vdf': 'real_data', 'area': False}
        self.f, s, meta_data = isr.isr_spectrum('kappa', sys_set, **params)
        self.data.append(s)
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
        #     Y_N_POINTS = 6e4
        #     V_N_POINTS = 1e5
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
        #     Y_N_POINTS = 6e4
        #     V_N_POINTS = 1e5
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
        self.p = p

    def create_it(self):
        # In config, set 'F0': 933e6, 'F_MIN': 3.5e6, 'F_MAX': 7.5e6
        # Also, using
        #     F_N_POINTS = 5e3
        #     Y_N_POINTS = 6e4
        #     V_N_POINTS = 1e5
        # is sufficient.
        T = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
        self.ridge_txt = [r'$T_{\mathrm{e}} = {}$'.format(j) + ' K' for j in T]
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

    def plot_it(self):
        self.p.plot_ridge(self.f, self.data, 'plot', self.legend_txt, self.ridge_txt)


if __name__ == '__main__':
    PlotTestNumerical().run()

"""Script containing the calculation of the power density spectrum
and other plasma parameters.
"""

import os
import sys
from typing import Tuple

import numpy as np
import scipy.constants as const
import scipy.integrate as si

from isr_spectrum.inputs import config as cf
from isr_spectrum.utils import integrand_functions as intf
from isr_spectrum.utils.njit import gordeyev_njit
from isr_spectrum.utils.parallel import gordeyev_int_parallel


class SpectrumCalculation:
    """Class containing the calculation of the power density spectrum."""

    def __init__(self):
        self.i_int_func = intf.INT_MAXWELL()
        self.e_int_func = intf.INT_MAXWELL()
        self.params: dict
        self.numba = True

    def set_params(self, params) -> None:
        self.params = params

    def set_electron_integrand_function(self, int_func) -> None:
        self.e_int_func = int_func

    def set_ion_integrand_function(self, int_func) -> None:
        self.i_int_func = int_func

    def calculate_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, "params"):
            raise ValueError("No parameters set.")
        m_i = self.params["MI"] * (const.m_p + const.m_n) / 2
        wc_e = w_gyro(np.linalg.norm(self.params["B"], 2), const.m_e)
        wc_i = w_gyro(np.linalg.norm(self.params["B"], 2), m_i)

        fi, fe = self._calulate_f_functions()

        if hasattr(self.i_int_func, "kappa"):
            kappa_i = getattr(self.i_int_func, "kappa")
        else:
            kappa_i = 1
        if hasattr(self.e_int_func, "kappa"):
            kappa_e = getattr(self.e_int_func, "kappa")
        else:
            kappa_e = 1
        xp_i = self._susceptibility(self.i_int_func.the_type, kappa_i)
        xp_e = self._susceptibility(self.e_int_func.the_type, kappa_e)

        with np.errstate(divide="ignore", invalid="ignore"):
            denominator = abs(1 + 2 * xp_e ** 2 * fe + 2 * xp_i ** 2 * fi) ** 2
            numerator1 = np.imag(-fe) * abs(1 + 2 * xp_i ** 2 * fi) ** 2
            numerator2 = 4 * xp_e ** 4 * np.imag(-fi) * abs(fe) ** 2
            numerator = numerator1 + numerator2
            isr = self.params["NE"] / (np.pi * cf.w) * numerator / denominator
        return cf.f, isr

    def _calulate_f_functions(self) -> Tuple[np.ndarray, np.ndarray]:
        m_i = self.params["MI"] * (const.m_p + const.m_n) / 2
        y_i = (
            np.linspace(
                0, cf.Y_MAX_i ** (1 / cf.ORDER), int(cf.Y_N_POINTS), dtype=np.double
            )
            ** cf.ORDER
        )
        self.i_int_func.initialize(y_i, self.params)
        kappa_i = (
            1
            if not hasattr(self.i_int_func, "kappa")
            else getattr(self.i_int_func, "kappa")
        )
        y_e = (
            np.linspace(
                0, cf.Y_MAX_e ** (1 / cf.ORDER), int(cf.Y_N_POINTS), dtype=np.double
            )
            ** cf.ORDER
        )
        self.e_int_func.initialize(y_e, self.params)
        kappa_e = (
            1
            if not hasattr(self.e_int_func, "kappa")
            else getattr(self.e_int_func, "kappa")
        )
        if self.numba:
            fi = gordeyev_njit.integrate(
                m_i,
                self.params["T_I"],
                self.params["NU_I"],
                y_i,
                function=self.i_int_func,  # .integrand(),
                the_type=self.i_int_func.the_type,
                kappa=kappa_i,
            )
            fe = gordeyev_njit.integrate(
                const.m_e,
                self.params["T_E"],
                self.params["NU_E"],
                y_e,
                function=self.e_int_func,
                the_type=self.e_int_func.the_type,
                kappa=kappa_e,
            )
        else:
            fi = gordeyev_int_parallel.integrate(
                m_i,
                self.params["T_I"],
                self.params["NU_I"],
                y_i,
                function=self.i_int_func,
                kappa=kappa_i,
            )
            fe = gordeyev_int_parallel.integrate(
                const.m_e,
                self.params["T_E"],
                self.params["NU_E"],
                y_e,
                function=self.e_int_func,
                kappa=kappa_e,
            )
        return fi, fe

    def _susceptibility(self, the_type, kappa) -> np.ndarray:
        if the_type == "maxwell":
            debye_length = L_Debye(self.params["NE"], self.params["T_E"])
            xp = np.sqrt(1 / (2 * debye_length ** 2 * self.params["K_RADAR"] ** 2))
        elif the_type == "kappa":
            debye_length = L_Debye(self.params["NE"], self.params["T_E"], kappa=kappa)
            xp = np.sqrt(1 / (2 * debye_length ** 2 * self.params["K_RADAR"] ** 2))
        elif the_type == "a_vdf":
            debye_length = L_Debye(
                self.params["NE"], self.params["T_E"], char_vel=self.e_int_func.char_vel
            )
            xp = np.sqrt(1 / (2 * debye_length ** 2 * self.params["K_RADAR"] ** 2))
        else:
            raise ValueError("Unknown function type.")
        return xp


def isr_spectrum(version, system_set, kappa=None, vdf=None, area=False, debye=None):
    """Calculate an ISR spectrum using the theory
    presented by Hagfors [1961] and Mace [2003].

    Arguments:
        version {str} -- decide which integral to use when
        calculating ISR spectrum
        system_set {dict} -- all plasma parameters and other parameters
        needed in the different calculation methods

    Keyword Arguments:
        kappa {int} -- kappa index used in any kappa distribution
        (default: {None})
        vdf {str} -- gives the VDF used in the a_vdf calculation
        (default: {None})
        area {bool} -- if True, calculates the area under the ion line
        (default: {False})
        debye {str} -- if set to `maxwell`, the Maxwellian Debye length
        is used (default: {None})

    Returns:
        f {np.ndarray} -- 1D array giving the frequency axis
        Is {np.ndarray} -- 1D array giving the spectrum at
        the sampled frequencies
        meta_data {dict} -- all parameters used to calculate
        the returned spectrum
    """
    sys_set, p = correct_inputs(
        version, system_set.copy(), {"kappa": kappa, "vdf": vdf}
    )
    kappa, vdf = p["kappa"], p["vdf"]
    func = version_check(version, vdf, kappa)
    w_c = w_e_gyro(np.linalg.norm([sys_set["B"]], 2))
    M_i = sys_set["MI"] * (const.m_p + const.m_n) / 2
    W_c = w_gyro(np.linalg.norm([sys_set["B"]], 2), M_i)

    # Ions
    params = {
        "K_RADAR": sys_set["K_RADAR"],
        "THETA": sys_set["THETA"],
        "nu": sys_set["NU_I"],
        "m": M_i,
        "T": sys_set["T_I"],
        "w_c": W_c,
    }
    y = (
        np.linspace(
            0, cf.Y_MAX_i ** (1 / cf.ORDER), int(cf.Y_N_POINTS), dtype=np.double
        )
        ** cf.ORDER
    )
    f_ion = intf.INT_MAXWELL()
    f_ion.initialize(y, params)
    if kappa is None:
        kappa = 1
    else:
        pass
    if cf.NJIT:
        Fi = gordeyev_njit.integrate(
            M_i,
            sys_set["T_I"],
            sys_set["NU_I"],
            y,
            function=f_ion,  # .integrand(),
            the_type=f_ion.the_type,
            # char_vel=char_vel,
            kappa=kappa,
        )
    else:
        Fi = gordeyev_int_parallel.integrate(
            M_i, sys_set["T_I"], sys_set["NU_I"], y, function=f_ion, kappa=kappa
        )

    # Electrons
    params = {
        "K_RADAR": sys_set["K_RADAR"],
        "THETA": sys_set["THETA"],
        "nu": sys_set["NU_E"],
        "m": const.m_e,
        "T": sys_set["T_E"],
        "T_ES": sys_set["T_ES"],
        "w_c": w_c,
        "kappa": kappa,
        "vdf": vdf,
        "Z": sys_set["Z"],
        "mat_file": sys_set["mat_file"],
        "pitch_angle": sys_set["pitch_angle"],
    }
    y = (
        np.linspace(
            0, cf.Y_MAX_e ** (1 / cf.ORDER), int(cf.Y_N_POINTS), dtype=np.double
        )
        ** cf.ORDER
    )
    func.initialize(y, params)
    if cf.NJIT:
        Fe = gordeyev_njit.integrate(
            const.m_e,
            sys_set["T_E"],
            sys_set["NU_E"],
            y,
            function=func,
            the_type=func.the_type,
            # char_vel=char_vel,
            kappa=kappa,
        )
    else:
        Fe = gordeyev_int_parallel.integrate(
            const.m_e, sys_set["T_E"], sys_set["NU_E"], y, function=func, kappa=kappa
        )

    Xp_i = np.sqrt(
        1
        / (
            2
            * L_Debye(sys_set["NE"], sys_set["T_E"], kappa=None) ** 2
            * sys_set["K_RADAR"] ** 2
        )
    )
    if func.the_type == "maxwell" or debye == "maxwell":
        Xp_e = np.sqrt(
            1
            / (
                2
                * L_Debye(sys_set["NE"], sys_set["T_E"]) ** 2
                * sys_set["K_RADAR"] ** 2
            )
        )
    elif func.the_type == "kappa":
        Xp_e = np.sqrt(
            1
            / (
                2
                * L_Debye(sys_set["NE"], sys_set["T_E"], kappa=kappa) ** 2
                * sys_set["K_RADAR"] ** 2
            )
        )
    elif func.the_type == "a_vdf":
        Xp_e = np.sqrt(
            1
            / (
                2
                * L_Debye(sys_set["NE"], sys_set["T_E"], char_vel=func.char_vel) ** 2
                * sys_set["K_RADAR"] ** 2
            )
        )

    # In case we have $ \omega = 0 $ in our frequency array, we just ignore this warning message $\label{lst:is_spectrum}$
    with np.errstate(divide="ignore", invalid="ignore"):
        Is = (
            sys_set["NE"]
            / (np.pi * cf.w)
            * (
                np.imag(-Fe) * abs(1 + 2 * Xp_i ** 2 * Fi) ** 2
                + (4 * Xp_e ** 4 * np.imag(-Fi) * abs(Fe) ** 2)
            )
            / abs(1 + 2 * Xp_e ** 2 * Fe + 2 * Xp_i ** 2 * Fi) ** 2
        )

    if area:
        if cf.I_P["F_MAX"] < 1e4:
            area = si.simps(Is, cf.f)
            print("The area under the ion line is %1.6e." % area)
        else:
            print("F_MAX is set too high. The area was not calculated.")

    sys_set["THETA"] = round(params["THETA"] * 180 / np.pi, 1)
    sys_set["version"] = version
    return cf.f, Is, dict(sys_set, **p)


def L_Debye(*args, kappa=None, char_vel=None):
    """Calculate the Debye length.

    Input args may be
        n_e -- electron number density
        T_e -- electron temperature
        T_i -- ion temperature

    Returns:
        float -- the Debye length
    """
    nargin = len(args)
    if nargin == 1:
        n_e = args[0]
    elif nargin == 2:
        n_e = args[0]
        T_e = args[1]
    elif nargin == 3:
        n_e = args[0]
        T_e = args[1]
        T_i = args[2]

    Ep0 = 1e-09 / 36 / np.pi

    if nargin < 3:
        if kappa is not None:
            LD = np.sqrt(Ep0 * const.k * T_e / (max(0, n_e) * const.e ** 2)) * np.sqrt(
                (kappa - 3 / 2) / (kappa - 1 / 2)
            )
        elif char_vel is not None:
            LD = np.sqrt(Ep0 * const.k * T_e / (max(0, n_e) * const.e ** 2)) * np.sqrt(
                char_vel
            )
        else:
            LD = np.sqrt(Ep0 * const.k * T_e / (max(0, n_e) * const.e ** 2))
    else:
        LD = np.sqrt(
            Ep0 * const.k / ((max(0, n_e) / T_e + max(0, n_e) / T_i) / const.e ** 2)
        )

    return LD


def w_gyro(B, m):
    """Gyro frequency as a function of magnetic field strength and particle mass.

    Arguments:
        B {float} -- magnetic field strength
        m {float} -- particle mass

    Returns:
        float -- ion gyro frequency
    """
    w = const.e * B / m

    return w


def w_e_gyro(B):
    """Electron gyro frequency as a function of magnetic field strength.

    Arguments:
        B {float} -- magnetic field strength

    Returns:
        float -- electron gyro frequency
    """
    w_e = const.e * B / const.m_e

    return w_e


def correct_inputs(version, sys_set, params):
    """Extra check suppressing the parameters
    that was given but is not necessary.
    """
    if version != "kappa" and not (
        version == "a_vdf" and params["vdf"] in ["kappa", "kappa_vol2"]
    ):
        params["kappa"] = None
    if version != "a_vdf":
        params["vdf"] = None
    if version != "a_vdf" or params["vdf"] != "gauss_shell":
        sys_set["T_ES"] = None
    if version != "a_vdf" or params["vdf"] != "real_data":
        sys_set["Z"] = None
        sys_set["mat_file"] = None
        sys_set["pitch_angle"] = None
    return sys_set, params


def version_check(version, vdf, kappa):
    """Check if the parameters given are complete.

    Args:
        version {str} -- which Gordeyev integrand to use
        vdf {str} -- which distribution to use
        kappa {int or float} -- kappa index

    Returns:
        object -- an integrand object from `integrand_functions.py`
    """
    versions = ["kappa", "maxwell", "a_vdf"]
    try:
        if version not in versions:
            raise SystemError
        print(f'Using version "{version}"', flush=True)
    except SystemError:
        sys.exit(version_error(version, versions))
    if version == "maxwell":
        func = intf.INT_MAXWELL()
    elif version == "kappa":
        kappa_check(kappa)
        func = intf.INT_KAPPA()
    elif version == "a_vdf":
        vdfs = ["maxwell", "kappa", "kappa_vol2", "gauss_shell", "real_data"]
        try:
            if vdf not in vdfs:
                raise SystemError
            print(f'Using VDF "{vdf}"', flush=True)
        except Exception:
            sys.exit(version_error(vdf, vdfs, element="VDF"))
        if vdf in ["kappa", "kappa_vol2"]:
            kappa_check(kappa)
        func = intf.INT_LONG()
    return func


def version_error(version, versions, element="version"):
    exc_type, _, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(f"{exc_type} error in file {fname}, line {exc_tb.tb_lineno}")
    print(f'The {element} is wrong: "{version}" not found in {versions}')


def kappa_check(kappa):
    try:
        kappa = int(kappa)
    except SystemError:
        sys.exit(print("You did not send in a valid kappa index."))

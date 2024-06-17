"""Calculate the power density spectrum and other plasma parameters."""

from typing import Callable, Optional

import numpy as np
import scipy.constants as const

from inscar import config, integrand_functions, numba_integration


class SpectrumCalculation:
    """Class containing the calculation of the power density spectrum."""

    def __init__(self):
        """Create the basic spectrum calculation object with empty attributes."""
        self._calculate_f = self._calculate_f_function
        self._susceptibility = self._susceptibility_function

    def set_params(self, params: config.Parameters) -> None:
        """Set the plasma parameters.

        Parameters
        ----------
        params : Parameters
            An instance of the ``Parameters`` class.
        """
        self.params = params

    def set_ion(self, ion: config.Particle) -> None:
        """Set the ion particles to use.

        Parameters
        ----------
        ion : Particle
            An instance of the ``Particle`` class, representing ions.
        """
        self.ion = ion

    def set_electron(self, electron: config.Particle) -> None:
        """Set the electron particles to use.

        Parameters
        ----------
        electron : Particle
            An instance of the ``Particle`` class, representing electrons.
        """
        self.electron = electron

    def set_ion_integration_function(
        self, function: integrand_functions.Integrand
    ) -> None:
        """Set the ion integration method to use.

        Parameters
        ----------
        function : Integrand
            An object of type ``Integrand``, representing the ions.
        """
        self.ion_integration_function = function

    def set_electron_integration_function(
        self, function: integrand_functions.Integrand
    ) -> None:
        """Set the electron integration method to use.

        Parameters
        ----------
        function : Integrand
            An object of type ``Integrand``, representing the electrons.
        """
        self.electron_integration_function = function

    def calculate_spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate a spectrum based on the given parameters and calculation methods.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The frequency axis in the first position and the spectrum values along the
            axis in the second position.

        Raises
        ------
        ValueError
            If any of the needed attribute objects are not set.
        """
        if not hasattr(self, "params"):
            raise ValueError("No parameters set. Use set_params().")
        if not hasattr(self, "ion"):
            raise ValueError("No ion particle set. Use set_ion().")
        if not hasattr(self, "electron"):
            raise ValueError("No electron particle set. Use set_electron().")
        if not hasattr(self, "ion_integration_function"):
            raise ValueError(
                "No ion integration function set. Use set_ion_integration_function()."
            )
        if not hasattr(self, "electron_integration_function"):
            raise ValueError(
                "No electron integration function set. "
                + "Use set_electron_integration_function()."
            )

        fi = self._calculate_f(self.ion, self.ion_integration_function)
        fe = self._calculate_f(self.electron, self.electron_integration_function)

        xp_i = self._susceptibility(self.ion, self.ion_integration_function)
        xp_e = self._susceptibility(self.electron, self.electron_integration_function)

        with np.errstate(divide="ignore", invalid="ignore"):
            numerator1 = np.imag(-fe) * np.abs(1 + 2 * xp_i**2 * fi) ** 2
            numerator2 = 4 * xp_e**4 * np.imag(-fi) * np.abs(fe) ** 2
            numerator = numerator1 + numerator2
            denominator = np.abs(1 + 2 * xp_e**2 * fe + 2 * xp_i**2 * fi) ** 2
            spectrum = (
                self.electron.number_density
                / (np.pi * self.params.angular_frequency)
                * numerator
                / denominator
            )
        return self.params.linear_frequency, spectrum

    def set_calculate_f_function(
        self,
        f_func: Callable[[config.Particle, integrand_functions.Integrand], np.ndarray],
    ) -> None:
        """Set what function to use to calculate the :math:`F` function.

        Parameters
        ----------
        f_func : Callable[[Particle, Integrand], np.ndarray]
            A function that take a particle and an integrand function as input, and that
            calculates the :math:`F` function based on these, returning a numpy array.
            By default, the ``integrate`` function from the ``numba_integration`` module
            is used.

        See Also
        --------
        inscar.numba_integration.integrate
        """
        self._calculate_f = f_func  # type: ignore

    def _calculate_f_function(
        self, particle: config.Particle, int_func: integrand_functions.Integrand
    ) -> np.ndarray:
        int_func.initialize(self.params, particle)
        the_type = int_func.the_type
        integrand = int_func.integrand()
        characteristic_velocity = getattr(int_func, "char_vel", None)
        return numba_integration.integrate(
            self.params,
            particle,
            integrand,
            the_type,
            characteristic_velocity,
        )

    def set_susceptibility_function(
        self, func: Callable[[config.Particle, integrand_functions.Integrand], float]
    ) -> None:
        """Set what function to use to calculate the susceptibility function.

        Parameters
        ----------
        func : Callable[[Particle, Integrand], float]
            A function that take a particle and an integrand function as input, and that
            calculates the susceptibility function based on these, returning a single
            float.
        """
        self._susceptibility = func  # type: ignore

    def _susceptibility_function(
        self, particle: config.Particle, int_func: integrand_functions.Integrand
    ) -> float:
        kappa = getattr(particle, "kappa", 1)
        temp = particle.temperature
        if int_func.the_type == "maxwell":
            debye_length = get_debye_length(particle.number_density, temp)
            xp = np.sqrt(1 / (2 * debye_length**2 * self.params.radar_wavenumber**2))
        elif int_func.the_type == "kappa":
            debye_length = get_debye_length(particle.number_density, temp, kappa=kappa)
            xp = np.sqrt(1 / (2 * debye_length**2 * self.params.radar_wavenumber**2))
        elif int_func.the_type == "a_vdf":
            char_vel = getattr(int_func, "char_vel", 1)
            debye_length = get_debye_length(
                particle.number_density, temp, char_vel=char_vel
            )
            xp = np.sqrt(1 / (2 * debye_length**2 * self.params.radar_wavenumber**2))
        else:
            raise ValueError("Unknown function type.")
        return xp


def get_debye_length(
    number_density: float,
    electron_temperature: float,
    ion_temperature: Optional[float] = None,
    kappa: Optional[float] = None,
    char_vel: Optional[float] = None,
) -> float:
    """Calculate the Debye length.

    Parameters
    ----------
    number_density : float
        The number density of the plasma.
    electron_temperature : float
        The electron temperature.
    ion_temperature : float, optional
        The ion temperature.
    kappa : float, optional
        Kappa parameter.
    char_vel : float, optional
        Characteristic velocity.

    Returns
    -------
    float : float
        Debye length.
    """
    vacuum_permittivity = 1e-09 / 36 / np.pi

    if ion_temperature is None:
        if kappa is not None:
            length = np.sqrt(
                vacuum_permittivity
                * const.k
                * electron_temperature
                / (max(0, number_density) * const.e**2)
            ) * np.sqrt((kappa - 3 / 2) / (kappa - 1 / 2))
        elif char_vel is not None:
            length = np.sqrt(
                vacuum_permittivity
                * const.k
                * electron_temperature
                / (max(0, number_density) * const.e**2)
            ) * np.sqrt(char_vel)
        else:
            length = np.sqrt(
                vacuum_permittivity
                * const.k
                * electron_temperature
                / (max(0, number_density) * const.e**2)
            )
    else:
        length = np.sqrt(
            vacuum_permittivity
            * const.k
            / (
                (
                    max(0, number_density) / electron_temperature
                    + max(0, number_density) / ion_temperature
                )
                / const.e**2
            )
        )

    return length

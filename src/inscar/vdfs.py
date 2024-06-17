"""Velocity distribution function to be used by the integrand class ``a_vdf``.

One of the integrands available for use in the Gordeyev integral.
"""

from abc import ABC, abstractmethod

import numpy as np
import scipy.constants as const
import scipy.special as sps

from inscar import config


class Vdf(ABC):
    """Base class for a VDF object.

    Parameters
    ----------
    ABC:
        Abstract base class that all Vdf objects inherit from
    """

    @abstractmethod
    def normalize(self) -> None:
        """Calculate the normalization for the VDF."""

    @abstractmethod
    def f_0(self) -> np.ndarray:
        """Return the values along the velocity axis of a VDF.

        Returns
        -------
        np.ndarray
            1D array with the VDF values at the sampled points
        """


class VdfMaxwell(Vdf):
    """Create an object that make Maxwellian distribution functions.

    Parameters
    ----------
    VDF : ABC
        Abstract base class to make VDF objects
    """

    def __init__(self, params: config.Parameters, particle: config.Particle):
        """Initialize VDF parameters.

        Parameters
        ----------
        params : Parameters
            `Parameters` object with the parameters of the simulation.
        particle : Particle
            `Particle` object with the parameters of the particle.
        """
        self.params = params
        self.particle = particle
        self.normalize()

    def normalize(self) -> None:
        """Normalize the distribution function."""
        self.A = (
            2 * np.pi * self.particle.temperature * const.k / self.particle.mass
        ) ** (-3 / 2)

    def f_0(self) -> np.ndarray:
        """Return the values along the velocity axis of a VDF."""
        return self.A * np.exp(
            -(self.particle.velocity_axis**2)
            / (2 * self.particle.temperature * const.k / self.particle.mass)
        )


class VdfKappa(Vdf):
    """Create an object that make kappa distribution functions.

    Notes
    -----
    Kappa VDF used in Gordeyev paper by Mace [5]_.

    References
    ----------
    .. [5] R. L. Mace, "A Gordeyev integral for electrostatic waves in a magnetized
       plasma with a kappa velocity distribution," Physics of plasmas, vol. 10, no. 6,
       pp. 2101-2193, 2003.
    """

    def __init__(self, params: config.Parameters, particle: config.Particle):
        """Initialize VDF parameters.

        Parameters
        ----------
        params : Parameters
            `Parameters` object with the parameters of the simulation.
        particle : Particle
            `Particle` object with the parameters of the particle.
        """
        self.params = params
        self.particle = particle
        self.normalize()

    def normalize(self) -> None:
        """Normalize the distribution function."""
        self.theta_2 = (
            2
            * ((self.particle.kappa - 3 / 2) / self.particle.kappa)
            * self.particle.temperature
            * const.k
            / self.particle.mass
        )
        self.A = (
            (np.pi * self.particle.kappa * self.theta_2) ** (-3 / 2)
            * sps.gamma(self.particle.kappa + 1)
            / sps.gamma(self.particle.kappa - 1 / 2)
        )

    def f_0(self) -> np.ndarray:
        """Return the values along the velocity axis of a VDF."""
        return self.A * (
            1 + self.particle.velocity_axis**2 / (self.particle.kappa * self.theta_2)
        ) ** (-self.particle.kappa - 1)

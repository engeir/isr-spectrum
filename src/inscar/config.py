"""Objects used to configure the physical parameters of the system."""

from typing import Union

import attr
import numpy as np
import scipy.constants as const


def is_odd(_, attribute, value):
    """Verify that a value is odd."""
    if value % 2 == 0:
        raise ValueError(f"{attribute} must be odd")


def is_positive(_, attribute, value):
    """Verify that a value is positive."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{attribute} must be a positive number")
    if value <= 0:
        raise ValueError(f"{attribute} must be positive")


def is_nonnegative(_, attribute, value):
    """Verify that a value is non-negative."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{attribute} must be a non-negative number")
    if value < 0:
        raise ValueError(f"{attribute} must be non-negative")


def is_range_tuple(_, attribute, value):
    """Verify that a value is a tuple of two numbers."""
    two_tuple = 2
    if not isinstance(value, tuple):
        raise ValueError(f"{attribute} must be a tuple")
    if len(value) != two_tuple:
        raise ValueError(f"{attribute} must be a tuple of length 2")
    if isinstance(value[0], (int, float)) and isinstance(value[1], (int, float)):
        if value[0] >= value[1]:
            raise ValueError(f"{attribute} must be a tuple of increasing values")
    else:
        raise ValueError(f"{attribute} must be a tuple of int/floats")


def to_radians(value) -> float:
    """Convert degrees to radians."""
    return value * np.pi / 180


@attr.s(auto_attribs=True)
class Particle:
    """Object used to configure the physical parameters of a particle.

    See Also
    --------
    Parameters
    """  # noqa: D410, D407, D411, D414

    gordeyev_upper_lim: Union[float, int] = attr.ib(
        default=1.5e-4,
        validator=is_positive,
        on_setattr=attr.setters.validate,
    )
    gordeyev_size: int = attr.ib(
        default=int(8e4 + 1),
        validator=is_odd,
        on_setattr=[attr.setters.convert, attr.setters.validate],
        converter=int,
    )
    gordeyev_exp: int = attr.ib(
        default=3,
        validator=is_odd,
        on_setattr=[attr.setters.convert, attr.setters.validate],
        converter=int,
    )
    velocity_upper_lim: Union[float, int] = attr.ib(
        default=6e6,
        validator=is_positive,
        on_setattr=attr.setters.validate,
    )
    velocity_size: int = attr.ib(
        default=int(4e4 + 1),
        validator=is_odd,
        on_setattr=[attr.setters.convert, attr.setters.validate],
        converter=int,
    )
    velocity_exp: int = attr.ib(
        default=3,
        validator=is_odd,
        on_setattr=[attr.setters.convert, attr.setters.validate],
        converter=int,
    )
    temperature: float = attr.ib(
        default=5000,
        validator=is_positive,
        on_setattr=attr.setters.validate,
    )
    collision_frequency: float = attr.ib(
        default=0,
        validator=is_nonnegative,
        on_setattr=attr.setters.validate,
    )
    mass: float = attr.ib(
        default=const.m_e,
        validator=is_positive,
        on_setattr=attr.setters.validate,
    )
    number_density: float = attr.ib(
        default=2e11,
        validator=is_positive,
        on_setattr=attr.setters.validate,
    )
    kappa: float = attr.ib(
        default=20,
        validator=is_positive,
        on_setattr=attr.setters.validate,
    )

    @property
    def velocity_axis(self) -> np.ndarray:
        """Return the axis for the velocity integral."""
        return (
            np.linspace(
                0,
                self.velocity_upper_lim ** (1 / self.velocity_exp),
                self.velocity_size,
            )
        ) ** self.velocity_exp

    @property
    def gordeyev_axis(self) -> np.ndarray:
        """Return the axis for the Gordeyev integral."""
        return (
            np.linspace(
                0,
                self.gordeyev_upper_lim ** (1 / self.gordeyev_exp),
                self.gordeyev_size,
                dtype=np.double,
            )
        ) ** self.gordeyev_exp


@attr.s(auto_attribs=True)
class Parameters:
    """Object used to configure the physical parameters of the system.

    See Also
    --------
    Particle

    Examples
    --------
    >>> from inscar import config

    Setting aspect angle to 45 degrees.

    >>> p = config.Parameters(aspect_angle=45)
    >>> print(f"p.aspect_angle = {p.aspect_angle:.4f}")
    p.aspect_angle = 0.7854

    It's automatically converted to radians! Similarly, setting radar frequency to 430
    MHz will automatically update radar wave number:

    >>> print(f"p.radar_wavenumber = {p.radar_wavenumber:.4f}")
    p.radar_wavenumber = -18.0243
    >>> # Changing radar frequency...
    >>> p.radar_frequency = 430e5
    >>> print(f"p.radar_wavenumber = {p.radar_wavenumber:.4f}")
    p.radar_wavenumber = -1.8024

    Setting aspect angle to 360.5 degrees.

    >>> p.aspect_angle = 360.5
    >>> print(f"p.aspect_angle = {p.aspect_angle:.4f}")
    p.aspect_angle = 6.2919
    """

    radar_frequency: float = attr.ib(
        default=430e6,
        validator=is_positive,
        on_setattr=attr.setters.validate,
    )
    frequency_range: tuple = attr.ib(
        default=(-2e6, 2e6),
        validator=is_range_tuple,
        on_setattr=attr.setters.validate,
    )
    frequency_size: int = attr.ib(
        default=int(1e4 + 1),
        validator=is_odd,
        on_setattr=[attr.setters.convert, attr.setters.validate],
        converter=int,
    )
    frequency_exp: int = attr.ib(
        default=1,
        validator=is_odd,
        on_setattr=[attr.setters.convert, attr.setters.validate],
        converter=int,
    )
    magnetic_field_strength: float = attr.ib(
        default=35000e-9,
        validator=is_positive,
        on_setattr=attr.setters.validate,
    )
    aspect_angle: Union[int, float] = attr.ib(
        default=45,
        validator=attr.validators.instance_of((int, float)),
        on_setattr=[attr.setters.convert, attr.setters.validate],
        converter=to_radians,
    )

    @property
    def linear_frequency(self) -> np.ndarray:
        """Return the linear frequency used for the power spectrum."""
        return (
            np.linspace(
                self.frequency_range[0], self.frequency_range[1], self.frequency_size
            )
            / self.frequency_range[1]
        ) ** self.frequency_exp * self.frequency_range[1]

    @property
    def angular_frequency(self) -> np.ndarray:
        """Return the angular frequency used for the power spectrum."""
        return self.linear_frequency * 2 * np.pi

    @property
    def radar_wavenumber(self) -> float:
        """Return the radar wave number."""
        return -2 * self.radar_frequency * 2 * np.pi / const.c

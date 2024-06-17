"""Example usage of the library."""

from typing import Union

import attr
import inscar as isr
import matplotlib.pyplot as plt
import numpy as np
import read
import scipy.constants as const
import scipy.integrate as si


@attr.s
class RealDataParticle(isr.Particle):
    """Create a particle object for the data sets."""

    mat_file: str = attr.ib(default="fe_zmuE-07.mat")
    pitch_angle: Union[int, str, list] = attr.ib(default="all")
    z: int = attr.ib(default=300)


class VdfRealData(isr.Vdf):
    """Create an object that make distribution functions from a 1D array."""

    def __init__(self, params: isr.Parameters, particle: RealDataParticle):
        """Initialize VDF parameters.

        Parameters
        ----------
        params : isr.Parameters
            Parameters object.
        particle : RealDataParticle
            Particle object.
        """
        self.params = params
        self.particle = particle
        self.normalize()

    def normalize(self):
        """Normalize the distribution function."""
        v = self.particle.velocity_axis
        func = read.interpolate_data(self.particle)
        f = func * v**2 * 4 * np.pi
        self.A = 1 / si.simps(f, v)

    def f_0(self):
        """Return the distribution function."""
        return self.A * read.interpolate_data(self.particle)


def _info():
    print("Setting aspect angle to 45 degrees.")
    p = isr.Parameters(aspect_angle=45)
    print(f"\tp.aspect_angle = {p.aspect_angle:.4f}")
    print("It's automatically converted to radians!\n")
    print(
        "Similarly, setting radar frequency to 430 MHz will "
        + "automatically update radar wave number:"
    )
    print(f"\tp.radar_wavenumber = {p.radar_wavenumber:.4f}")
    print("Changing radar frequency...")
    p.radar_frequency = 430e5
    print(f"\tp.radar_wavenumber = {p.radar_wavenumber:.4f}\n")
    print("Setting aspect angle to 360.5 degrees.")
    p.aspect_angle = 360.5
    print(f"\tp.aspect_angle = {p.aspect_angle:.4f}")


def _ion_line():
    e = isr.Particle(temperature=200, kappa=8)
    m_i = 29 * (const.m_p + const.m_n) / 2
    i = isr.Particle(
        gordeyev_upper_lim=1.5e-2,
        temperature=200,
        mass=m_i,
    )
    params = isr.Parameters(
        frequency_range=(-3e3, 3e3),
        frequency_size=int(1e3 + 1),
        radar_frequency=430e6,
        magnetic_field_strength=35000e-9,
        aspect_angle=45,
    )

    sim = isr.SpectrumCalculation()
    sim.set_params(params)
    sim.set_ion(i)
    sim.set_electron(e)
    sim.set_ion_integration_function(isr.IntMaxwell())
    sim.set_electron_integration_function(isr.IntMaxwell())
    x, y = sim.calculate_spectrum()

    plt.figure()
    plt.title("Ion line")
    plt.plot(x, y)


def _ion_line_long():
    e = isr.Particle(temperature=200, kappa=8)
    m_i = 29 * (const.m_p + const.m_n) / 2
    i = isr.Particle(
        gordeyev_upper_lim=1.5e-2,
        temperature=200,
        mass=m_i,
    )
    params = isr.Parameters(
        frequency_range=(-3e3, 3e3),
        frequency_size=int(1e3 + 1),
        radar_frequency=430e6,
        magnetic_field_strength=35000e-9,
        aspect_angle=45,
    )

    sim = isr.SpectrumCalculation()
    sim.set_params(params)
    sim.set_ion(i)
    sim.set_electron(e)
    sim.set_ion_integration_function(isr.IntMaxwell())
    sim.set_electron_integration_function(isr.IntLong())
    x, y = sim.calculate_spectrum()

    plt.figure()
    plt.title("Ion line long")
    plt.plot(x, y)


def _plasma_line():
    e = isr.Particle(temperature=5000, kappa=8)
    m_i = 16 * (const.m_p + const.m_n) / 2
    i = isr.Particle(
        gordeyev_upper_lim=1.5e-2,
        temperature=2000,
        mass=m_i,
    )
    params = isr.Parameters(
        frequency_range=(3.5e6, 7e6),
        frequency_size=int(1e3 + 1),
        radar_frequency=933e6,
        magnetic_field_strength=50000e-9,
        aspect_angle=0,
    )

    sim = isr.SpectrumCalculation()
    sim.set_params(params)
    sim.set_ion(i)
    sim.set_electron(e)
    sim.set_ion_integration_function(isr.IntMaxwell())
    sim.set_electron_integration_function(isr.IntMaxwell())
    x, y = sim.calculate_spectrum()

    plt.figure()
    plt.title("Plasma line")
    plt.plot(x, y)


def _gyro_line():
    e = isr.Particle(temperature=200, kappa=8, number_density=2e10)
    m_i = 16 * (const.m_p + const.m_n) / 2
    i = isr.Particle(
        gordeyev_upper_lim=1.5e-2,
        temperature=200,
        mass=m_i,
    )
    params = isr.Parameters(
        frequency_range=(-2e6, 2e6),
        frequency_size=int(1e4 + 1),
        radar_frequency=430e6,
        magnetic_field_strength=3.5e-5,
        aspect_angle=(360 + 135),
    )

    sim = isr.SpectrumCalculation()
    sim.set_params(params)
    sim.set_ion(i)
    sim.set_electron(e)
    sim.set_ion_integration_function(isr.IntMaxwell())
    sim.set_electron_integration_function(isr.IntMaxwell())
    x, y = sim.calculate_spectrum()

    plt.figure()
    plt.title("Gyro-line")
    plt.semilogy(x, y)


def _real_data_custom_vdf():
    e = RealDataParticle(
        temperature=200, kappa=8, velocity_size=10001, gordeyev_size=10001
    )
    m_i = 29 * (const.m_p + const.m_n) / 2
    i = isr.Particle(
        gordeyev_upper_lim=1.5e-2,
        temperature=200,
        mass=m_i,
        gordeyev_size=10001,
        velocity_size=10001,
    )
    params = isr.Parameters(
        frequency_range=(-3e3, 3e3),
        frequency_size=int(1e3 + 1),
        radar_frequency=430e6,
        magnetic_field_strength=35000e-9,
        aspect_angle=45,
    )

    sim = isr.SpectrumCalculation()
    sim.set_params(params)
    sim.set_ion(i)
    sim.set_electron(e)
    sim.set_ion_integration_function(isr.IntMaxwell())
    electron_int_func = isr.IntLong()
    electron_int_func.set_vdf(VdfRealData)
    sim.set_electron_integration_function(electron_int_func)
    x, y = sim.calculate_spectrum()

    plt.figure()
    plt.title("Real data")
    plt.plot(x, y)


if __name__ == "__main__":
    _info()
    _ion_line()
    _ion_line_long()
    _plasma_line()
    _gyro_line()
    _real_data_custom_vdf()
    plt.show()

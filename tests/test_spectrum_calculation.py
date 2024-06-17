"""Test cases for the spectrum calculation module."""

import inscar as isr
import pytest
import scipy.constants as const


def test_missing_objects() -> None:
    """Test the initialisation of a new spectrum calculation."""
    sim = isr.SpectrumCalculation()

    params = isr.Parameters()
    e = isr.Particle(temperature=200, kappa=8)
    m_i = 29 * (const.m_p + const.m_n) / 2
    i = isr.Particle(
        gordeyev_upper_lim=1.5e-2,
        temperature=200,
        mass=m_i,
    )

    with pytest.raises(ValueError):
        sim.calculate_spectrum()
    sim.set_params(params)
    with pytest.raises(ValueError):
        sim.calculate_spectrum()
    sim.set_ion(i)
    with pytest.raises(ValueError):
        sim.calculate_spectrum()
    sim.set_electron(e)
    with pytest.raises(ValueError):
        sim.calculate_spectrum()
    sim.set_ion_integration_function(isr.IntMaxwell())
    with pytest.raises(ValueError):
        sim.calculate_spectrum()

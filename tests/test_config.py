"""Tests for the `config` module."""

import inscar as isr
import numpy as np
import scipy.constants as const


def test_parameters() -> None:
    """Test the `parameters` configuration."""
    p = isr.Parameters(aspect_angle=45)
    assert p.aspect_angle == (45 * np.pi / 180)
    p.radar_frequency = 430e5
    assert p.radar_wavenumber == -2 * 430e5 * 2 * np.pi / const.c
    p.aspect_angle = 360.5
    assert p.aspect_angle == (360.5 * np.pi / 180)

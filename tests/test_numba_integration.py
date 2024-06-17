"""Test cases for the numba integration module."""

import inscar as isr
import numpy as np
from inscar import numba_integration as nb_int


def test_trapz() -> None:
    """Test the `trapz` function."""
    axis = np.linspace(0, 20, 20)
    values = np.linspace(0, 1, 20)
    integral_np = np.trapezoid(values, axis)  # type: ignore[attr-defined]
    integral_nb = nb_int.trapz(values, axis)
    int_value = 10
    assert integral_np == int_value
    assert round(integral_np, 1) == round(integral_nb, 1)


def _inner_int(w: np.ndarray, x: np.ndarray, function: np.ndarray) -> np.ndarray:
    array = np.zeros_like(w, dtype=np.complex128)
    for idx in range(len(w)):
        array[idx] = np.trapezoid(np.exp(-1j * w[idx] * x) * function, x)  # type: ignore[attr-defined]
    return array


def test_inner_int() -> None:
    """Test the `inner_int` function."""
    w = np.linspace(0, 10, 10)
    x = np.linspace(0, 10, 100)
    function = np.linspace(0, 1, 100)
    array_np = _inner_int(w, x, function)
    array_nb = nb_int.inner_int(w, x, function)
    assert np.allclose(array_np, array_nb)


def test_integrate() -> None:
    """Test the `integrate` function."""
    # TODO: Check numerical precision
    e = isr.Particle()
    params = isr.Parameters(
        frequency_range=(3.5e6, 7e6),
        frequency_size=int(1e3 + 1),
        radar_frequency=933e6,
        magnetic_field_strength=50000e-9,
        aspect_angle=0,
    )

    int_func = isr.IntMaxwell()
    int_func.initialize(params, e)
    nb_int.integrate(params, e, int_func.integrand(), int_func.the_type)

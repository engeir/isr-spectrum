"""Compare time of numpy implementation vs custom numba implementation."""

import timeit

import numpy as np
from inscar import numba_integration as nb_int


def _print_time(t1, t2):
    v = "Numpy" if t2 > t1 else "Mine"
    print(f"{v} is {t2 / t1 if t2 > t1 else t1/t2}x faster. (numpy: {t1}, numba: {t2})")


def _trapz() -> None:
    print("Trapz")
    print("Medium both")
    axis = np.linspace(0, 20, int(4e5))
    values = np.linspace(0, 1, int(4e5))
    t1 = timeit.timeit(lambda: np.trapezoid(values, axis), number=1000)  # type: ignore
    t2 = timeit.timeit(lambda: nb_int.trapz(values, axis), number=1000)
    _print_time(t1, t2)

    print("Single large array")
    axis = np.linspace(0, 20, int(4e7))
    values = np.linspace(0, 1, int(4e7))
    t1 = timeit.timeit(lambda: np.trapezoid(values, axis), number=10)  # type: ignore
    t2 = timeit.timeit(lambda: nb_int.trapz(values, axis), number=10)
    _print_time(t1, t2)

    print("Many calls to func")
    axis = np.linspace(0, 20, int(4e1))
    values = np.linspace(0, 1, int(4e1))
    t1 = timeit.timeit(lambda: np.trapezoid(values, axis), number=100000)  # type: ignore
    t2 = timeit.timeit(lambda: nb_int.trapz(values, axis), number=100000)
    _print_time(t1, t2)
    print("")


def _np_inner_int(w: np.ndarray, x: np.ndarray, function: np.ndarray) -> np.ndarray:
    array = np.zeros_like(w, dtype=np.complex128)
    for idx in range(len(w)):
        array[idx] = np.trapezoid(np.exp(-1j * w[idx] * x) * function, x)  # type: ignore
    return array


def _inner_int() -> None:
    print("Many small arrays/many calls")
    w = np.linspace(0, 10, 10)
    x = np.linspace(0, 10, 100)
    function = np.linspace(0, 1, 100)
    t1 = timeit.timeit(lambda: _np_inner_int(w, x, function), number=10000)
    t2 = timeit.timeit(lambda: nb_int.inner_int(w, x, function), number=10000)
    _print_time(t1, t2)

    print("Single large integration axis")
    w = np.linspace(0, 10, 1000)
    x = np.linspace(0, 10, 100000)
    function = np.linspace(0, 1, 100000)
    t1 = timeit.timeit(lambda: _np_inner_int(w, x, function), number=1)
    t2 = timeit.timeit(lambda: nb_int.inner_int(w, x, function), number=1)
    _print_time(t1, t2)

    print("Many integrations")
    w = np.linspace(0, 10, 100000)
    x = np.linspace(0, 10, 1000)
    function = np.linspace(0, 1, 1000)
    t1 = timeit.timeit(lambda: _np_inner_int(w, x, function), number=1)
    t2 = timeit.timeit(lambda: nb_int.inner_int(w, x, function), number=1)
    _print_time(t1, t2)


if __name__ == "__main__":
    _trapz()
    _inner_int()

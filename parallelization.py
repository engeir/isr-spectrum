import ctypes
import multiprocessing as mp
from functools import partial

import scipy.special as sps
import numpy as np
from tqdm import tqdm

import config as cf
import integrand_functions as intf
import tool


def integrate(w_c, m, T, Lambda_s, T_MAX, function):
    idx = [x for x in enumerate(cf.w)]
    func = partial(parallel, w_c, m, T, Lambda_s, T_MAX, function)
    pool = mp.Pool()
    for _ in tqdm(pool.imap(func, idx)):
        pass
    if function == intf.kappa_gordeyev:
        a = array / (2**(cf.KAPPA - 1 / 2) * sps.gamma(cf.KAPPA + 1 / 2))
    else:
        a = array
    F = 1 - (1j * cf.w + Lambda_s * w_c) * a
    return F


def parallel(w_c, m, T, Lambda_s, T_MAX, function, index):
    # print(index[0], end='\r')
    array[index[0]] = tool.simpson(
        function, index[1], w_c, m, T, Lambda_s, T_MAX)


def shared_array(shape):
    """
    Form a shared memory numpy array.

    http://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-python-multiprocessing
    """

    shared_array_base = mp.Array(ctypes.c_double, 2 * shape[0])
    shared_arr = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_arr = shared_arr.view(np.complex128).reshape(*shape)
    return shared_arr


array = shared_array((len(cf.w),))

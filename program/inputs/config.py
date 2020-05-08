"""Constants used system wide.

The physical properties of the plasma is changed here, described below 'Input parameters'.
Only one dictionary containing plasma parameters should be uncommented at any instance in time.
"""

import os
import sys

import numpy as np
import scipy.constants as const


# Check if a test is running. Potential paths are
# ['pytest.py', 'pytest', 'test_ISR.py', '__main__.py', 'python3.7 -m unittest']
# or check if 'main.py' was used.
if os.path.basename(os.path.realpath(sys.argv[0])) != 'main.py':
    # DO NOT EDIT
    F_N_POINTS = 1e1
    Y_N_POINTS = 1e1
    V_N_POINTS = 1e1
else:
    F_N_POINTS = 1e3  # Number of sample points in frequency
    Y_N_POINTS = 6e4  # Number of sample points in integral variable
    V_N_POINTS = 1e4  # Number of sample points in velocity integral variable
# Adds one sample to get an even number of bins, which in
# turn give better precision in the Simpson integration.
Y_N_POINTS += 1
V_N_POINTS += 1
Y_MAX_e = 1.5e-4  # Upper limit of integration (= infinity)
Y_MAX_i = 1.5e-2
# When using real data, E_max = 110 eV -> 6.22e6 m/s
# V_MAX = 3e7
V_MAX = 6e6
ORDER = 3

# DO NOT EDIT
I_P = {'F0': 933e6, 'F_MAX': 8e6}
K_RADAR = - 2 * I_P['F0'] * 2 * np.pi / const.c  # Radar wavenumber
# If 'plasma' == True, might as well set f_min ≈ 1e6
f = np.linspace(- I_P['F_MAX'], I_P['F_MAX'], int(F_N_POINTS))
f = (f / I_P['F_MAX'])**3 * I_P['F_MAX']
w = 2 * np.pi * f  # Angular frequency

"""Constants used system wide.
"""

import numpy as np

# Physical constants
C_0 = 299792458  # Speed of light
M_E = 9.10938291e-31  # electron rest mass [kg]
M_P = 1.672621778e-27  # proton rest mass [kg]
M_N = 1.674927352e-27  # neutron rest mass [kg]
Q_E = 1.602176565e-19  # elementary charge [C]

K_B = 1.380662e-23  # Boltzmann constant [J/K]
MY_0 = 4 * np.pi * 1e-7  # Permeability [Vs/Am]

N_POINTS = 1e5  # Number of sample points in integral variable
T_MAX_e = 1e-4  # Upper limit to integration (= infinity)
T_MAX_i = 1e-3
SCALING = 1e2

# === Input parameters ===
# For seeing gyro lines
B = 35000e-9  # Magnetic field strength [T]
F0 = 430e6  # Radar frequency [Hz]
F_MAX = 2e6  # Range of frequency domain
MI = 16  # Ion mass in atomic mass units [u]
NE = 2e11  # Electron number density [m^(-3)]  (1.5e6)^2/(8.98^2)
NU_E = 0  # Electron collision frequency [Hz]
NU_I = 0  # Ion collision frequency [Hz]
T_E = 200  # Electron temperature [K]
T_I = 200  # Ion temperature [K]
THETA = 45 * np.pi / 180  # Pitch angle

# For same plots as Hagfors
# B = 35000e-9  # Magnetic field strength [T]
# F0 = 430e6  # Radar frequency [Hz]
# F_MAX = 6e6  # Range of frequency domain
# MI = 16  # Ion mass in atomic mass units [u]
# NE = 2e10  # Electron number density [m^(-3)]  (1.5e6)^2/(8.98^2)
# NU_E = 0  # Electron collision frequency [Hz]
# NU_I = 0  # Ion collision frequency [Hz]
# T_E = 1000  # Electron temperature [K]
# T_I = 1000  # Ion temperature [K]
# THETA = 0 * np.pi / 180  # Pitch angle

# For kappa distribution
KAPPA = 5 / 2
NU = - KAPPA - 1 / 2

# DO NOT EDIT
K_RADAR = F0 * 2 * np.pi / C_0  # Radar wavenumber
F_N_POINTS = 1e3
f = np.arange(F_N_POINTS / 2) * (F_MAX - 0) / (F_N_POINTS / 2)  # Frequency
dW = 2 * np.pi * (F_MAX - 0) / (F_N_POINTS / 2)  # Step size angular frequency
w = np.arange(F_N_POINTS / 2) * dW  # Angular frequency

# This give a clear indication that the sampling is too low.
# y1, y2, y3 represent the peaks of the electron, gyro and ion lines.
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = np.array([5e6, 1e7, 2e7, 3e7, 4e7, 5e7, 1e8])
    y1 = np.array([2.078, 2.692, 3.322, 3.620, 3.7787, 3.8703, 3.8536])
    y2 = np.array([.679, .6812, .6819, .6820, .6820, .6820, .6821])
    y3 = np.array([6.288, 6.35, 6.384, 6.392, 6.390, 6.393, 6.403]) * 1e-4
    plt.figure()
    plt.plot(x, y1)
    plt.figure()
    plt.plot(x, y2)
    plt.figure()
    plt.plot(x, y3)
    # plt.legend(['y1', 'y2', 'y3'])
    plt.show()


# phi = np.linspace(0, 2 * np.pi, 1000)
# a = [.9, .1, .4 * np.sin(3 * phi), .1 * np.cos(20 * phi)]
# func = 0
# for c, v in enumerate(a):
#     func += v * abs(np.cos(phi))**(c + 1)
# # plt.figure()
# _, axs = plt.subplots()
# axs.set_aspect('equal', 'box')
# plt.plot(func * np.cos(phi), func * np.sin(phi))
# plt.show()

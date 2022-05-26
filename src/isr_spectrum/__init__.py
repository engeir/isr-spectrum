"""ISR Spectrum."""

from importlib_metadata import version

from isr_spectrum.config import *  # noqa: F401, F403
from isr_spectrum.integrand_functions import *  # noqa: F401, F403
from isr_spectrum.spectrum_calculation import SpectrumCalculation  # noqa: F401
from isr_spectrum.vdfs import *  # noqa: F401, F403

__version__ = version(__package__)

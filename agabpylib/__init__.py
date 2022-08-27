"""
Python utilities.

Anthony Brown Oct 2015 - Aug 2022
"""

__version__ = "0.2.0"

try:
    import numpy
except ImportError:
    raise ImportError("NumPy does not seem to be installed.")

try:
    import scipy
except ImportError:
    raise ImportError("SciPy does not seem to be installed.")

try:
    import matplotlib
except ImportError:
    raise ImportError("matplotlib does not seem to be installed.")

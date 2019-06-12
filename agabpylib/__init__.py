"""
Python utilities.

Anthony Brown Oct 2015 - June 2019
"""

__version__ = "0.1.8"

try:
    import numpy
except ImportError:
    raise ImportError('NumPy does not seem to be installed.')

try:
    import scipy
except ImportError:
    raise ImportError('SciPy does not seem to be installed.')

try:
    import matplotlib
except ImportError:
    raise ImportError('matplotlib does not seem to be installed.')

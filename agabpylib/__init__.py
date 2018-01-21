"""
Python utilities.

Anthony Brown Oct 2015 - Jan 2018
"""

__version__ = "0.1.5"

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

#__modules__ = ['blah', 'foo']

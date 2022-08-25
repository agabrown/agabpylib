"""
Star cluster simulation tools.

This package provides functions for simple simulations of star clusters.

.. note:: 
    The cluster simulations are focused on generating realistic observables and
    are not intended for the simulation of dynamically self-consistent clusters
    (i.e. where the mass distribution, potential, and kinematics are
    consistent).

Anthony Brown Jul 2019 - Aug 2022
"""

from .cluster import *
from .kinematics import *
from .observables import *
from .spacedistributions import *

"""
Posterior distributions.

This package provides functions for the calculation of various posterior
probability densities resulting from Bayesian analyses applied to astronomical
problems.
"""

from .distancefromparallax import *
from .gaussian import *
from .magnitudefromflux import *

__all__ = [s for s in dir() if not s.startswith("_")]

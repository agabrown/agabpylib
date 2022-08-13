"""
Simulation tools.

This package provides functions for very simple simulations of astronomical
surveys etc.
"""

from .parallaxsurveys import *
from .imf import *

__all__ = [s for s in dir() if not s.startswith("_")]

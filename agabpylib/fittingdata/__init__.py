"""
Tools for fitting models to data.

This package provides functions fitting models to data.
"""

from .fitpolynomials import *

__all__ =  [s for s in dir() if not s.startswith('_')]
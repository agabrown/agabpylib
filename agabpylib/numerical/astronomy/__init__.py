"""
astronomy.

This module provides simple numerical algorithms for astronomy.
"""

from .celestialmechanics import *

__all__ = [s for s in dir() if not s.startswith("_")]

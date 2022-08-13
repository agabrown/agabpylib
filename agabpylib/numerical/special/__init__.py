"""
Special functions.

This module provides special functions not available from scipy.
"""

from .fermidirac import *

__all__ = [s for s in dir() if not s.startswith("_")]

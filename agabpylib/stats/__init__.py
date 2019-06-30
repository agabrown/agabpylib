"""
Statistical functions.

Anthony Brown June 2019 - June 2019
"""

from .distributions import *

__all__ = [s for s in dir() if not s.startswith('_')]
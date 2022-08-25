"""
Posterior distributions.

This package provides functions for the calculation of posterior probability
densities as examples of Bayesian data analyses.
"""

from .meanvarnormal import *

__all__ = [s for s in dir() if not s.startswith("_")]

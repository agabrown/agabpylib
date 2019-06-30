"""
Custom quadrature methods which are not available from scipy.

Anthony Brown Aug 2017 -Jun 2019
"""

import numpy as np
from scipy.special import roots_laguerre

__all__ = ['fixed_quad_laguerre']


def _cached_roots_laguerre(n):
    """
    Cache roots_laguerre results to speed up calls of the fixed_quad_laguerre function.

    2017.08.09: Code adapted from https://github.com/scipy/scipy/blob/master/scipy/integrate/quadrature.py

    NOTE that the weights resulting from a call to roots_laguerre(n) are multiplied by exp(abscissa)
    because it is assumed that the function to be integrated includes the exp(-x) factor.
    """
    if n in _cached_roots_laguerre.cache:
        return _cached_roots_laguerre.cache[n]

    x, w, _ = roots_laguerre(n)
    # Multiply the weights by exp(x)
    _cached_roots_laguerre.cache[n] = (x, w * np.exp(x))
    return _cached_roots_laguerre.cache[n]


_cached_roots_laguerre.cache = dict()


def fixed_quad_laguerre(func, a, args=(), n=5):
    """
    Compute a definite integral using fixed-order Gauss-Laguerre quadrature.
    Code adapted from fixed_quad in
    https://github.com/scipy/scipy/blob/master/scipy/integrate/quadrature.py.

    Parameters
    ----------
    func : callable
        A Python function or method to integrate (must accept vector inputs).
        If integrating a vector-valued function, the returned array must have
        shape ``(..., len(x))``.
    a : float
        Lower limit of integration. THE UPPER LIMIT IS ASSUMED TO BE +INF.
    args : tuple, optional
        Extra arguments to pass to function, if any.
    n : int, optional
        Order of quadrature integration. Default is 5.

    Returns
    -------
    val : float
        Gaussian quadrature approximation to the integral
    none : None
        Statically returned value of None
    """
    x, w = _cached_roots_laguerre(n)
    x = np.real(x)
    y = a + x
    return np.sum(w * func(y, *args), axis=-1), None

"""
Optimization utility methods not available from scipy.optimize.

Anthony Brown Aug 2017 - Aug 2022
"""

import numpy as np

__all__ = ["bracket_root"]


def bracket_root(func, xa=0.0, xb=1.0, args=(), grow_factor=1.6, maxiter=50):
    """
    Bracket the root of a function. 

    This is done by considering increasingly large intervals within which the root
    might lie. Implicitly assumes a monotonic function.

    Parameters
    ----------
    func : Callable f(x,*args)
        Objective function to minimize.
    xa, xb : float, optional
        Bracketing interval. Defaults xa to 0.0, and xb to 1.0.
    args : tuple, optional
        Additional arguments (if present), passed to func.
    grow_factor : float, optional
        Factor by which to increase search interval, defaults to 1.6.
    maxiter : int, optional
        Maximum number of tries to bracket the root, defaults to 50

    Returns
    -------
    x1, x1 : float
        The interval brackteing the root.
    converged : Boolean
        If false the bracketing of the root failed.
    """
    x1 = xa
    x2 = xb
    f1 = func(x1, *args)
    f2 = func(x2, *args)
    for i in range(maxiter):
        if f1 * f2 < 0.0:
            return x1, x2, True
        else:
            if np.abs(f1) < np.abs(f2):
                x1 = x1 + grow_factor * (x1 - x2)
                f1 = func(x1, *args)
            else:
                x2 = x2 + grow_factor * (x2 - x1)
                f2 = func(x2, *args)
    return x1, x2, False

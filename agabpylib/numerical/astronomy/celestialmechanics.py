"""
Provides useful functions for basic celestial mechanics applications.

Anthony Brown May 2020 - May 2020
"""

import numpy as np
from scipy.optimize import toms748

__all__ = ['kepler_equation_solver']

_kepler_equation = lambda EE, ee, MM: EE - ee * np.sin(EE) - MM


def kepler_equation_solver(e, M):
    """
    Solve the Kepler equation for the input values of eccentricity and mean anomaly. Use a root-finding method.

    :param e: scalar or 1D array
        Values of the eccentricity
    :param M: scalar or 1D array
        Values of the mean anomaly
    :return:
        Array of eccentric anomaly values.
    """
    if np.ndim(e) > 1 or np.ndim(M) > 1:
        raise RuntimeError("Only scalars or 1D arrays allowed as input.")

    if np.ndim(e)==0 and np.ndim(M)==0:
        if (e == 0 or np.mod(M, np.pi) == 0):
            return np.mod(M, 2*np.pi)
        else:
            return toms748(_kepler_equation, 0, 2 * np.pi, args=(e, np.mod(M,2*np.pi)))

    if np.ndim(e) == 0 and np.ndim(M) == 1:
        ecc = np.repeat(e, M.size)
        meanano = M
    elif np.ndim(e) == 1 and np.ndim(M) == 0:
        meanano = np.repeat(M, e.size)
        ecc = e
    else:
        if e.size != M.size:
            raise RuntimeError("The two input arrays must be of the same size.")
        ecc = e
        meanano = M

    E = []
    for el, Ml in zip(ecc, meanano):
        if (el == 0 or np.mod(Ml, np.pi) == 0):
            E.append(np.mod(Ml, 2*np.pi))
        else:
            E.append(toms748(_kepler_equation, 0, 2 * np.pi, args=(el, np.mod(Ml,2*np.pi))))

    return np.squeeze(np.asarray(E))

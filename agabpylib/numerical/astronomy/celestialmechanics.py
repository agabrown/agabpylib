"""
Provides useful functions for basic celestial mechanics applications.

Anthony Brown May 2020 - May 2020
"""

import numpy as np
from scipy.optimize import toms748

__all__ = ['kepler_equation']

def kepler_equation(e, M):
    """
    Solve the Kepler equation for the input values of eccentricity and mean anomaly. Use a root-finding method.

    :param e: array-like
        Values of the eccentricity
    :param M: array-like
        Values of the mean anomaly
    :return:
        Array of eccentric anomaly values.
    """

    f = lambda E, e, M: E-e*np.sin(E)-M
    E = []
    for ecc, meanA in zip(e, M):
        if (ecc==0 or np.mod(meanA,np.pi)==0):
            E.append(meanA)
        else:
            E.append(toms748(f, 0, 2*np.pi, args=(ecc,meanA)))
    if (len(E)=1):
        return E[0]
    else:
        return np.asarray(E)
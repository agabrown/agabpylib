"""
Provides various methods for robust estimates of simple statistics such as the mean and variance, which
in this case are estimated through the median and RSE.

Anthony Brown May 2015 - Aug 2022
"""

import numpy as np
import scipy as sp

__all__ = ["rse", "robust_stats"]

_rse_constant = 1.0 / (np.sqrt(2) * 2 * sp.special.erfinv(0.8))


def rse(x):
    """
    Calculate the Robust Scatter Estimate for an array of values (see GAIA-C3-TN-ARI-HL-007).

    Parameters
    ----------
    x : float array
        Array of input values (can be of any dimension)

    Returns
    -------
    rse : float
        The Robust Scatter Estimate (RSE), defined as 0.390152 * (P90-P10),
        where P10 and P90 are the 10th and 90th percentile of the distribution
        of x.
    """
    return _rse_constant * (
        sp.stats.scoreatpercentile(x, 90) - sp.stats.scoreatpercentile(x, 10)
    )


def robust_stats(x):
    """
    Provide robust statistics of the values in array x (which can be of any dimension).

    Parameters
    ----------
    x : float array
        Input array (numpy array is assumed)

    Returns
    -------
    stats : dict
        Dictionary {'median':median, 'rse':RSE, 'lowerq':lower quartile,
        'upperq':upper quartile, 'min':minimum value, 'max':maximum value}
    """

    med = np.median(x)
    therse = rse(x)
    lowerq = sp.stats.scoreatpercentile(x, 25)
    upperq = sp.stats.scoreatpercentile(x, 75)
    lowerten = sp.stats.scoreatpercentile(x, 10)
    upperten = sp.stats.scoreatpercentile(x, 90)

    return {
        "median": med,
        "rse": therse,
        "lowerq": lowerq,
        "upperq": upperq,
        "lower10": lowerten,
        "upper10": upperten,
        "min": x.min(),
        "max": x.max(),
        "ndata": x.size,
    }

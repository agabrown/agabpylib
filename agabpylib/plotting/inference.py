"""
Provides several functions that are useful for plotting in the context of inference problems.

Anthony Brown Mar 2015 - Jan 2018
"""

import numpy as np
from matplotlib.patches import Ellipse
from scipy.special import erf
from math import atan2

def error_ellipses(mu, covmat, sigma_levels, **kwargs):
    """
    Given a covariance matrix for a 2D Normal distribution calculate the error-ellipses and return
    matplotlib patches for plotting.

    Parameters
    ----------

    mu : float array
        Mean of Normal distribution (2-vector)
    covmat : float array
        Covariance matrix stored as [sigma_x^2, sigma_y^2, sigma_xy]
    sigma_levels : float or 1-D array
        Equivalent n-sigma levels to draw

    Keywords
    --------
    **kwargs : 
        Extra arguments for matplotlib.patches.Ellipse

    Returns
    -------

    List of matplotlib.patches.Ellipse objects
    """

    sigmaLevels2D = -2.0*np.log(1.0-erf(np.array([sigma_levels]).flatten()/np.sqrt(2.0)))

    eigvalmax = 0.5*(covmat[0]+covmat[1]+np.sqrt((covmat[0]-covmat[1])**2+4*covmat[2]**2))
    eigvalmin = 0.5*(covmat[0]+covmat[1]-np.sqrt((covmat[0]-covmat[1])**2+4*covmat[2]**2))
    angle = atan2((covmat[0]-eigvalmax), -covmat[2])/np.pi*180
    errEllipses = []
    for csqr in sigmaLevels2D:
        errEllipses.append( Ellipse(mu,2*np.sqrt(csqr*eigvalmax), 2*np.sqrt(csqr*eigvalmin), angle, **kwargs) )

    return errEllipses

def convert_to_stdev_nan(logL):
    """
    Given a grid of log-likelihood values, convert them to cumulative
    standard deviation. This is useful for drawing contours from a
    grid of likelihoods.

    Code from astroML: https://github.com/astroML/astroML

    THIS version is robust to a logL array that contains NaNs.

    Parameters
    ----------

    logL : float array
        Input array of log-likelihood (or log-probability) values. Can contain NaNs.

    Returns
    -------

    Array of values that represent the cumulative probability for logL.
    """
    sigma = np.exp(logL)

    shape = sigma.shape
    sigma = sigma.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(sigma)[::-1]
    i_unsort = np.argsort(i_sort)

    sigma_sorted = sigma[i_sort]
    notnan = np.logical_not(np.isnan(sigma_sorted))
    sigma_cumsum = np.empty(sigma.size)
    sigma_cumsum[:] = np.nan
    sigma_cumsum[notnan] = sigma_sorted[notnan].cumsum()
    sigma_cumsum /= sigma_cumsum[-1]

    return sigma_cumsum[i_unsort].reshape(shape)

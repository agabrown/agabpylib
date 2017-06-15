"""
Provides several functions that are useful for plotting in the inference context.

Anthony Brown 2015-03-03
"""

from matplotlib.patches import Ellipse
from scipy.special import erf
from numpy import sqrt, array, zeros, pi, log
from math import atan2

def error_ellipses(mu, covmat, sigma_levels, **kwargs):
    """
    Given a covariance matrix for a 2D Normal distribution calculate the error-ellipses and return
    matplotlib patches for plotting.

    Parameters
    ----------

    mu - Mean of Normal distribution (2-vector)
    covmat - Covariance matrix stored as [sigma_x^2, sigma_y^2, sigma_xy]
    sigma_levels - Equivalent n-sigma levels to draw
    **kwargs - Extra arguments for matplotlib.patches.Ellipse

    Returns
    -------

    List of matplotlib.patches.Ellipse objects
    """

    sigmaLevels2D = -2.0*log(1.0-erf(sigma_levels/sqrt(2.0)))

    eigvalmax = 0.5*(covmat[0]+covmat[1]+sqrt((covmat[0]-covmat[1])**2+4*covmat[2]**2))
    eigvalmin = 0.5*(covmat[0]+covmat[1]-sqrt((covmat[0]-covmat[1])**2+4*covmat[2]**2))
    angle = atan2((covmat[0]-eigvalmax), -covmat[2])/pi*180
    errEllipses = []
    for csqr in sigmaLevels2D:
        errEllipses.append( Ellipse(mu,2*sqrt(csqr*eigvalmax), 2*sqrt(csqr*eigvalmin),angle, **kwargs) )

    return errEllipses

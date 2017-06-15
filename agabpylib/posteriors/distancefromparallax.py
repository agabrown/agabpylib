"""
Provides functions for the evaluation of posteriors on the distance to a star, given its observed
parallax and parallax error.

Anthony Brown 2015-05-04
"""

from numpy import log,arctan,pi

def distance_improper_uniform_prior(observed_parallax, relative_error, d):
    """
    Calculate the posterior on the distance d_true=1/parallax_true, given the observed parallax and the
    relative error on the parallax, and assuming a uniform improper prior P(d) = 1 for d>0 and zero
    otherwise.

    Parameters
    ----------

    observed_parallax - The value of the observed parallax.
    relative_error - The value of the relative error sigma_parallax/observed_parallax.
    d - Array of true distances for which the posterior is to be evaluated.

    Returns
    -------

    Values of unnormalized ln(posterior) at each value of d. Note that the mode of this unnormalized
    posterior is always 1.
    """
    lnP = - 0.5*((1.0/(d*observed_parallax)-1.0)/relative_error)**2
    pctoau = 180*3600/pi
    #lnP = - 0.5*((arctan(1.0/(d*pctoau))/(observed_parallax/pctoau)-1.0)/relative_error)**2
    return lnP

def distance_uniform_density_prior(observed_parallax, relative_error, d):
    """
    Calculate the posterior on the distance d_true=1/parallax_true, given the observed parallax and the
    relative error on the parallax, and assuming a uniform density improper prior P(d) \propto d^2 for
    d>0 and zero otherwise.

    Parameters
    ----------

    observed_parallax - The value of the observed parallax.
    relative_error - The value of the relative error sigma_parallax/observed_parallax.
    d - Array of true distances for which the posterior is to be evaluated.

    Returns
    -------

    Values of unnormalized ln(posterior) at each value of d.
    """
    z=1.0/(d*observed_parallax)
    lnP = -2.0*log(z) - 0.5*((z-1.0)/relative_error)**2
    return lnP

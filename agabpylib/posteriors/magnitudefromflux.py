"""
Provides functions for the evaluation of posteriors on the magnitude of a star, given its observed
flux and flux error.

Anthony Brown 2015-09-26
"""

from numpy import log, log10, pi, power


def magnitude_improper_uniform_prior(observed_flux, relative_error, m, c):
    """
    Calculate the posterior on the magnitude m=-2.5*log10(flux)+C, given the observed flux and the
    relative error on the flux, and assuming a uniform improper prior P(d) = 1 for m.

    Parameters
    ----------

    observed_flux - The value of the observed flux.
    relative_error - The value of the relative error sigma_flux/observed-flux.
    m - Array of true magnitudes for which the posterior is to be evaluated.
    c - Magnitude zero point.

    Returns
    -------

    Values of unnormalized ln(posterior) at each value of d. Note that the mode of this unnormalized
    posterior is always 1.
    """
    flux = power(10.0,-0.4*(m-c))
    lnP = - 0.5*((flux/observed_flux-1.0)/relative_error)**2
    return lnP


def magnitude_uniform_density_prior(observed_flux, relative_error, m, c):
    """
    Calculate the posterior on the magnitude m=-2.5*log10(flux)+C, given the observed flux and the
    relative error on the flux, and assuming a uniform density improper prior 

    Parameters
    ----------

    observed_flux - The value of the observed flux.
    relative_error - The value of the relative error sigma_flux/observed_flux.
    m - Array of true magnitudes for which the posterior is to be evaluated.
    c - Magnitude zero point.

    Returns
    -------

    Values of unnormalized ln(posterior) at each value of d.
    """
    lnP = 0.6*log(10.0)*m- 0.5*((flux/observed_flux-1.0)/relative_error)**2
    return lnP

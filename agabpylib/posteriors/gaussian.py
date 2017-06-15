"""
Functions for calculating the posterior distributions for the mean (mu) and variance (sigma^2) of a
Gaussian.

Anthony Brown 2015-01-17
"""

from numpy import log, zeros_like, sum, pi
from scipy.special import gammaln

def unknownMeanAndVarianceFlatPrior(n, xbar, V, mu, tau):
    """
    Calculate the joint posterior distribution for mu and =>tau=sigma^2<= over the input grid in (mu,
    sigma) and for the data characterized by the number of samples n, the mean xbar, and the variance v.

    Parameters
    ----------

    n - The number of data points x_i
    xbar - The mean of the data points x_i
    V - The variance of the data points x_i
    mu - 2D-array of mu values (obtained through numpy.meshgrid for example)
    tau - 2D-array of tau values

    Returns
    -------

    The value of ln(posterior) at each grid point.
    """

    lnP = log(2.0) + 0.5*(n-3)*log(n*V) - gammaln((n-3)/2.0) \
            + 0.5*(log(n)-log(pi)) - n/2.0*log(2*tau) - 0.5*(n*(V+(mu-xbar)**2))/tau 
    return lnP

def marginalMeanUMVFlatPrior(n, xbar, V, mu):
    """
    Calculate the marginal posterior distribution of mu for the case of unknown mean and unknown
    variance.

    Parameters
    ----------

    n - The number of data points x_i
    xbar - The mean of the data points x_i
    v - The variance of the data points x_i
    mu - 1D-array of mu values

    Returns
    -------
    
    The value of ln(posterior) at each grid point.
    """
    lnP = gammaln((n-2)/2.0) - gammaln((n-3)/2.0) - 0.5*(log(pi)+log(V)) \
            - (n-2)/2.0*log(1+(mu-xbar)**2/V)
    return lnP

def marginalTauUMVFlatPrior(n, xbar, V, tau):
    """
    Calculate the marginal posterior distribution of tau for the case of unknown mean and unknown
    variance.

    Parameters
    ----------

    n - The number of data points x_i
    xbar - The mean of the data points x_i
    V - The variance of the data points x_i
    tau - 1D-array of tau values

    Returns
    -------
    
    The value of ln(posterior) at each grid point.
    """
    lnP = 0.5*(n-3)*(log(n*V)-log(2*tau)) - gammaln((n-3)/2.0) - log(tau) - n*V/(2*tau)
    return lnP

def unknownMeanAndVarianceUnInfPrior(n, xbar, V, mu, tau):
    """
    Calculate the joint posterior distribution for mu and =>tau sigma^2<= over the input grid in (mu,
    sigma) and for the data characterized by the number of samples n, the mean xbar, and the variance v.
    This is for the case when non-informative priors on mu and sigma are included.

    Parameters
    ----------

    n - The number of data points x_i
    xbar - The mean of the data points x_i
    V - The variance of the data points x_i
    mu - 2D-array of mu values (obtained through numpy.meshgrid for example)
    tau - 2D-array of tau values

    Returns
    -------

    The value of ln(posterior) at each grid point.
    """

    lnP = (n-1)/2.0*(log(n)+log(V)) - n/2.0*log(2.0) - gammaln((n-1)/2.0) \
            + 0.5*(log(n)-log(pi)) - (n+2)/2.0*log(tau) - 0.5*(n*(V+(mu-xbar)**2))/tau
    return lnP

def marginalMeanUMVUnInfPrior(n, xbar, V, mu):
    """
    Calculate the marginal posterior distribution of mu for the case of unknown mean and unknown
    variance and with an uninformative prior on tau..

    Parameters
    ----------

    n - The number of data points x_i
    xbar - The mean of the data points x_i
    V - The variance of the data points x_i
    mu - 1D-array of mu values

    Returns
    -------
    
    The value of ln(posterior) at each grid point.
    """
    lnP = gammaln(n/2.0) - gammaln((n-1)/2.0) - 0.5*(log(pi)+log(V)) \
            - n/2.0*log(1+(mu-xbar)**2/V)
    return lnP

def marginalTauUMVUnInfPrior(n, xbar, V, tau):
    """
    Calculate the marginal posterior distribution of tau for the case of unknown mean and unknown
    variance and with an uninformative prior on tau.

    Parameters
    ----------

    n - The number of data points x_i
    xbar - The mean of the data points x_i
    V - The variance of the data points x_i
    tau - 1D-array of tau values

    Returns
    -------
    
    The value of ln(posterior) at each grid point.
    """
    lnP = 0.5*(n-1)*(log(n*V)-log(2*tau)) - gammaln((n-1)/2.0) - log(tau) - n*V/(2*tau)
    return lnP

def unknownMeanAndStddevFlatPrior(n, xbar, V, mu, sigma):
    """
    Calculate the joint posterior distribution for mu and sigma over the input grid in (mu, sigma) and
    for the data characterized by the number of samples n, the mean xbar, and the variance V.

    Parameters
    ----------

    n - The number of data points x_i
    xbar - The mean of the data points x_i
    V - The variance of the data points x_i
    mu - 2D-array of mu values (obtained through numpy.meshgrid for example)
    sigma - 2D-array of sigma values

    Returns
    -------

    The value of ln(posterior) at each grid point.
    """

    lnP = 1.5*log(2.0) + (n-2)/2.0*(log(n)+log(V)) - n/2.0*log(2.0) - gammaln((n-2)/2.0) \
            + 0.5*(log(n)-log(pi)) - n*log(sigma) - 0.5*(n*(V+(mu-xbar)**2))/(sigma*sigma)
    return lnP

def marginalMeanUMSFlatPrior(n, xbar, V, mu):
    """
    Calculate the marginal posterior distribution of mu for the case of unknown mean and unknown
    variance.

    Parameters
    ----------

    n - The number of data points x_i
    xbar - The mean of the data points x_i
    V - The variance of the data points x_i
    mu - 1D-array of sigma values

    Returns
    -------
    
    The value of ln(posterior) at each grid point.
    """
    lnP = gammaln((n-1)/2.0) - gammaln((n-2)/2.0) - 0.5*(log(pi)+log(V)) \
            - (n-1)/2.0*log(1+(mu-xbar)**2/V)
    return lnP

def marginalSigmaUMSFlatPrior(n, xbar, V, sigma):
    """
    Calculate the marginal posterior distribution of sigma for the case of unknown mean and unknown
    variance.

    Parameters
    ----------

    n - The number of data points x_i
    xbar - The mean of the data points x_i
    V - The variance of the data points x_i
    sigma - 1D-array of sigma values

    Returns
    -------
    
    The value of ln(posterior) at each grid point.
    """
    lnP = 0.5*(n-1)*(log(n*V)-log(2)-2*log(sigma)) - gammaln((n-2)/2.0) \
            -0.5*log(n*V) + 1.5*log(2) - n*V/(2*sigma*sigma)
    return lnP

def unknownMeanAndStddevUnInfPrior(n, xbar, V, mu, sigma):
    """
    Calculate the joint posterior distribution for mu and sigma over the input grid in (mu, sigma) and
    for the data characterized by the number of samples n, the mean xbar, and the variance v. This is
    for the case when non-informative priors on mu and sigma are included.

    Parameters
    ----------

    n - The number of data points x_i
    xbar - The mean of the data points x_i
    V - The variance of the data points x_i
    mu - 2D-array of mu values (obtained through numpy.meshgrid for example)
    sigma - 2D-array of sigma values

    Returns
    -------

    The value of ln(posterior) at each grid point.
    """

    lnP = 1.5*log(2.0) - log(n*V) + (n+1)/2.0*(log(n)+log(V)) - (n+1)/2.0*log(2.0) \
            - gammaln((n-1)/2.0) \
            + 0.5*(log(n)-log(pi)) - (n+1)*log(sigma) - 0.5*(n*(V+(mu-xbar)**2))/(sigma*sigma)
    return lnP

def marginalMeanUMSUnInfPrior(n, xbar, V, mu):
    """
    Calculate the marginal posterior distribution of mu for the case of unknown mean and unknown
    variance. This is for the case when non-informative priors on mu and sigma are included.

    Parameters
    ----------

    n - The number of data points x_i
    xbar - The mean of the data points x_i
    V - The variance of the data points x_i
    mu - 1D-array of sigma values

    Returns
    -------
    
    The value of ln(posterior) at each grid point.
    """
    lnP = gammaln(n/2.0) - gammaln((n-1)/2.0) + 0.5*(log(n)-log(pi)) - 0.5*log(n*V) \
            - n/2.0*log(1+(mu-xbar)**2/V)
    return lnP

def marginalSigmaUMSUnInfPrior(n, xbar, V, sigma):
    """
    Calculate the marginal posterior distribution of sigma for the case of unknown mean and unknown
    variance. This is for the case when non-informative priors on mu and sigma are included.

    Parameters
    ----------

    n - The number of data points x_i
    xbar - The mean of the data points x_i
    V - The variance of the data points x_i
    sigma - 1D-array of sigma values

    Returns
    -------
    
    The value of ln(posterior) at each grid point.
    """
    lnP = 0.5*(n+1)*(log(n*V)-log(2)-2*log(sigma)) - gammaln((n-1)/2.0) \
            + log(4*sigma) - log(n*V) - n*V/(2*sigma*sigma)
    return lnP

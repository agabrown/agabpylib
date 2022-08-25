r"""
Functions for calculating the posterior distributions for the mean (:math:`\mu`) and variance 
(:math:`\tau=\sigma^2`) or standard deviation (:math:`\sigma`) of a normal
distribution, inferred from :math:`n` data points :math:`x_i`.

Anthony Brown Jan 2015 - Aug 2022
"""

import numpy as np
import scipy as sp


def umv_flat_prior(n, xbar, V, mu, tau):
    r"""
    Calculate the joint posterior distribution for the unknown mean :math:`\mu`
    and unknown variance :math:`\tau=\sigma^2`, for a flat prior in both
    parameters.

    The posterior is calculated over the input grid in :math:`(\mu, \tau)` for
    the data characterized by the number of samples :math:`n`, the mean
    :math:`\bar{x}`, and the variance :math:`V`.

    Parameters
    ----------
    n : int
        The number of data points :math:`x_i`
    xbar : float
        The mean of the data points :math:`x_i`
    V : float
        The data variance :math:`\sum(x_i-\bar{x})^2/n`
    mu : float array
        2D-array of :math:`\mu` values (obtained through numpy.meshgrid for example)
    tau : float array
        2D-array of :math:`\tau` values

    Returns
    -------
    lnP : float array
        The value of ln(posterior) at each grid point.
    """

    lnP = (
        np.log(2.0)
        + 0.5 * (n - 3) * np.log(n * V)
        - sp.gammaln((n - 3) / 2.0)
        + 0.5 * (np.log(n) - np.log(np.pi))
        - n / 2.0 * np.log(2 * tau)
        - 0.5 * (n * (V + (mu - xbar) ** 2)) / tau
    )
    return lnP


def marginal_mean_umv_flat_prior(n, xbar, V, mu):
    r"""
    Calculate the marginal posterior distribution of mu for the case of unknown mean and unknown
    variance and flat priors.

    Parameters
    ----------
    n : int
        The number of data points :math:`x_i`
    xbar : float
        The mean of the data points :math:`x_i`
    V : float
        The data variance :math:`\sum(x_i-\bar{x})^2/n`
    mu : float array
        1D-array of :math:`\mu` values

    Returns
    -------
    lnP : float array
        The value of ln(posterior) at each grid point.
    """
    lnP = (
        sp.gammaln((n - 2) / 2.0)
        - sp.gammaln((n - 3) / 2.0)
        - 0.5 * (np.log(np.pi) + np.log(V))
        - (n - 2) / 2.0 * np.log(1 + (mu - xbar) ** 2 / V)
    )
    return lnP


def marginal_tau_umv_flat_prior(n, V, tau):
    r"""
    Calculate the marginal posterior distribution of tau for the case of unknown mean and unknown
    variance for flat priors.

    Parameters
    ----------
    n : int
        The number of data points :math:`x_i`
    V : float
        The data variance :math:`\sum(x_i-\bar{x})^2/n`
    tau : float array
        1D-array of :math:`\tau` values

    Returns
    -------
    lnP : float array
        The value of ln(posterior) at each grid point.
    """
    lnP = (
        0.5 * (n - 3) * (np.log(n * V) - np.log(2 * tau))
        - sp.gammaln((n - 3) / 2.0)
        - np.log(tau)
        - n * V / (2 * tau)
    )
    return lnP


def umv_uninf_prior(n, xbar, V, mu, tau):
    r"""
    Calculate the joint posterior distribution for the unknown mean :math:`\mu` and unknown variance :math:`\tau=\sigma^2`,
    for an uninformative prior in both parameters.

    The prior on :math:`\mu` is flat while the the prior on :math:`\tau` is :math:`p(\tau)\propto 1/\tau`.
    The posterior is calculated over the input grid in :math:`(\mu, \tau)` for the data characterized by
    the number of samples :math:`n`, the mean :math:`\bar{x}`, and the variance :math:`V`.

    Parameters
    ----------
    n : int
        The number of data points :math:`x_i`
    xbar : float
        The mean of the data points :math:`x_i`
    V : float
        The data variance :math:`\sum(x_i-\bar{x})^2/n`
    mu : float array
        2D-array of :math:`\mu` values (obtained through numpy.meshgrid for example)
    tau : float array
        2D-array of :math:`\tau` values

    Returns
    -------
    lnP : float array
        The value of ln(posterior) at each grid point.
    """

    lnP = (
        (n - 1) / 2.0 * (np.log(n) + np.log(V))
        - n / 2.0 * np.log(2.0)
        - sp.gammaln((n - 1) / 2.0)
        + 0.5 * (np.log(n) - np.log(np.pi))
        - (n + 2) / 2.0 * np.log(tau)
        - 0.5 * (n * (V + (mu - xbar) ** 2)) / tau
    )
    return lnP


def marginal_mean_umv_uninf_prior(n, xbar, V, mu):
    r"""
    Calculate the marginal posterior distribution of :math`\mu` for the case of unknown mean and unknown
    variance and with an uninformative prior on :math:`\tau`.

    Parameters
    ----------
    n : int
        The number of data points :math:`x_i`
    xbar : float
        The mean of the data points :math:`x_i`
    V : float
        The data variance :math:`\sum(x_i-\bar{x})^2/n`
    mu : float array
        1D-array of :math:`\mu` values

    Returns
    -------
    lnP : float array
        The value of ln(posterior) at each grid point.
    """
    lnP = (
        sp.gammaln(n / 2.0)
        - sp.gammaln((n - 1) / 2.0)
        - 0.5 * (np.log(np.pi) + np.log(V))
        - n / 2.0 * np.log(1 + (mu - xbar) ** 2 / V)
    )
    return lnP


def marginal_tau_umv_uninf_prior(n, V, tau):
    r"""
    Calculate the marginal posterior distribution of :math:`\tau` for the case of unknown mean and unknown
    variance and with an uninformative prior on :math:`\tau`.

    Parameters
    ----------
    n : int
        The number of data points :math:`x_i`
    V : float
        The data variance :math:`\sum(x_i-\bar{x})^2/n`
    tau : float array
        1D-array of :math:`\tau` values

    Returns
    -------
    lnP : float array
        The value of ln(posterior) at each grid point.
    """
    lnP = (
        0.5 * (n - 1) * (np.log(n * V) - np.log(2 * tau))
        - sp.gammaln((n - 1) / 2.0)
        - np.log(tau)
        - n * V / (2 * tau)
    )
    return lnP


def ums_flat_prior(n, xbar, V, mu, sigma):
    r"""
    Calculate the joint posterior distribution for  the unknown mean :math:`\mu`
    and unknown standard deviation :math:`\sigma` for a flat prior in both
    parameters.

    The posterior is calculated over the input grid in :math:`(\mu, \sigma)` and for the data characterized by
    the number of samples i:math:`n`, the mean :math:`\bar{x}`, and the data variance :math:`V`.

    Parameters
    ----------
    n : int
        The number of data points :math:`x_i`
    xbar : float
        The mean of the data points :math:`x_i`
    V : float
        The data variance :math:`\sum(x_i-\bar{x})^2/n`
    mu : float array
        2D-array of :math:`\mu` values (obtained through numpy.meshgrid for example)
    sigma : float array
        2D-array of :math:`\sigma` values

    Returns
    -------
    lnP : float array
        The value of ln(posterior) at each grid point.
    """

    lnP = (
        1.5 * np.log(2.0)
        + (n - 2) / 2.0 * (np.log(n) + np.log(V))
        - n / 2.0 * np.log(2.0)
        - sp.gammaln((n - 2) / 2.0)
        + 0.5 * (np.log(n) - np.log(np.pi))
        - n * np.log(sigma)
        - 0.5 * (n * (V + (mu - xbar) ** 2)) / (sigma * sigma)
    )
    return lnP


def marginal_mean_ums_flat_prior(n, xbar, V, mu):
    r"""
    Calculate the marginal posterior distribution of :math:`\mu` for the case of unknown mean and unknown
    standard deviation.

    Parameters
    ----------
    n : int
        The number of data points :math:`x_i`
    xbar : float
        The mean of the data points :math:`x_i`
    V : float
        The data variance :math:`\sum(x_i-\bar{x})^2/n`
    mu : float array
        1D-array of :math:`\mu` values

    Returns
    -------
    lnP : float array
        The value of ln(posterior) at each grid point.
    """
    lnP = (
        sp.gammaln((n - 1) / 2.0)
        - sp.gammaln((n - 2) / 2.0)
        - 0.5 * (np.log(np.pi) + np.log(V))
        - (n - 1) / 2.0 * np.log(1 + (mu - xbar) ** 2 / V)
    )
    return lnP


def marginal_sigma_ums_flat_prior(n, V, sigma):
    r"""
    Calculate the marginal posterior distribution of :math:`\sigma` for the case of unknown mean and unknown
    standard deviation.

    Parameters
    ----------
    n : int
        The number of data points :math:`x_i`
    V : float
        The data variance :math:`\sum(x_i-\bar{x})^2/n`
    sigma : float array
        1D-array of :math:`\sigma` values

    Returns
    -------
    lnP : float array
        The value of ln(posterior) at each grid point.
    """
    lnP = (
        0.5 * (n - 1) * (np.log(n * V) - np.log(2) - 2 * np.log(sigma))
        - sp.gammaln((n - 2) / 2.0)
        - 0.5 * np.log(n * V)
        + 1.5 * np.log(2)
        - n * V / (2 * sigma * sigma)
    )
    return lnP


def ums_uninf_prior(n, xbar, V, mu, sigma):
    r"""
    Calculate the joint posterior distribution for the unknown mean :math:`\mu`
    and unknown standard deviation :math:`\sigma`, for an uninformative prior in
    both parameters.

    The prior on :math:`\mu` is flat while the the prior on :math:`\sigma` is :math:`p(\sigma)\propto 1/\sigma`.
    The posterior is calculated over the input grid in :math:`(\mu, \sigma)` for the data characterized by
    the number of samples :math:`n`, the mean :math:`\bar{x}`, and the variance :math:`V`.

    Parameters
    ----------
    n : int
        The number of data points :math:`x_i`
    xbar : float
        The mean of the data points :math:`x_i`
    V : float
        The data variance :math:`\sum(x_i-\bar{x})^2/n`
    mu : float array
        2D-array of :math:`\mu` values (obtained through numpy.meshgrid for example)
    sigma : float array
        2D-array of :math:`\sigma` values

    Returns
    -------
    lnP : float array
        The value of ln(posterior) at each grid point.
    """
    lnP = (
        1.5 * np.log(2.0)
        - np.log(n * V)
        + (n + 1) / 2.0 * (np.log(n) + np.log(V))
        - (n + 1) / 2.0 * np.log(2.0)
        - sp.gammaln((n - 1) / 2.0)
        + 0.5 * (np.log(n) - np.log(np.pi))
        - (n + 1) * np.log(sigma)
        - 0.5 * (n * (V + (mu - xbar) ** 2)) / (sigma * sigma)
    )
    return lnP


def marginal_mean_ums_uninf_prior(n, xbar, V, mu):
    r"""
    Calculate the marginal posterior distribution of :math:`\mu` for the case of unknown mean and unknown
    standard deviation. This is for the case when non-informative priors on :math:'\mu' and :math:`\sigma` are included.

    Parameters
    ----------
    n : int
        The number of data points :math:`x_i`
    xbar : float
        The mean of the data points :math:`x_i`
    V : float
        The data variance :math:`\sum(x_i-\bar{x})^2/n`
    mu : float array
        1D-array of :math:`\mu` values

    Returns
    -------
    lnP : float array
        The value of ln(posterior) at each grid point.
    """
    lnP = (
        sp.gammaln(n / 2.0)
        - sp.gammaln((n - 1) / 2.0)
        + 0.5 * (np.log(n) - np.log(np.pi))
        - 0.5 * np.log(n * V)
        - n / 2.0 * np.log(1 + (mu - xbar) ** 2 / V)
    )
    return lnP


def marginal_sigma_ums_uninf_prior(n, V, sigma):
    r"""
    Calculate the marginal posterior distribution of :math:`\sigma` for the case of unknown mean and unknown
    standard deviation. This is for the case when non-informative priors on :math:`\mu` and :math:`\sigma` are included.

    Parameters
    ----------
    n : int
        The number of data points :math:`x_i`
    V : float
        The data variance :math:`\sum(x_i-\bar{x})^2/n`
    sigma : float array
        1D-array of :math:`\sigma` values

    Returns
    -------
    lnP : float array
        The value of ln(posterior) at each grid point.
    """
    lnP = (
        0.5 * (n + 1) * (np.log(n * V) - np.log(2) - 2 * np.log(sigma))
        - sp.gammaln((n - 1) / 2.0)
        + np.log(4 * sigma)
        - np.log(n * V)
        - n * V / (2 * sigma * sigma)
    )
    return lnP

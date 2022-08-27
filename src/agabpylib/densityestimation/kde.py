"""
Provides wrappers around scikit-learn kernel density estimation methods.

Anthony Brown May 2015 - Aug 2022
"""

import numpy as np
import sklearn


def kde_scikitlearn(
    data, N=100, lims=None, evalOnData=False, kde_bandwidth=1.0, **kwargs
):
    """
    Provide a kernel density estimate for a set of data points (d_i).

    Make use of the sklearn.neighbours.KernelDensity class.

    Parameters
    ----------
    data : float array
        1D array of values of d_i
    lims : tuple
        Limits on data to use (dmin, dmax)
    N : int
        Number of KDE samples in d (regular grid between dmin and dmax)
    evalOndata : boolean
        If true return the log(density) evaluated on the data (instead of the regular grid)
    kde_bandwidth : float
        Bandwith for density estimator

    Returns
    -------
    Dsamples, log_dens : float array
        Dsamples, log_dens: The log(density) evaluated on the regular grid Dsamples (both shape (N,))
        If evalOnData is True return only log_dens evaluated for the data points (shape (data.size,)).

    Other parameters
    ----------------
    **kwargs : dict
        Extra arguments for KernelDensity class initializer
    """
    if lims == None:
        dmin = data.min()
        dmax = data.max()
    else:
        dmin = lims[0]
        dmax = lims[1]

    kde = sklearn.neighbours.KernelDensity(bandwidth=kde_bandwidth, **kwargs)
    kde.fit(data[:, None])
    if not (evalOnData):
        Dsamples = np.linspace(dmin, dmax, N)[:, None]
        log_dens = kde.score_samples(Dsamples)
        return Dsamples, log_dens
    else:
        log_dens = kde.score_samples(data[:, None])
        return log_dens


def kde2d_scikitlearn(
    xdata,
    ydata,
    Nx=100,
    Ny=100,
    xeval=None,
    yeval=None,
    xlims=None,
    ylims=None,
    evalOnData=False,
    kde_bandwidth=1.0,
    **kwargs
):
    """
    Provide a 2D kernel density estimate for a set of data points (x_i, y_i). Make use of the
    scikit-learn scikitlearn.neighbours.KernelDensity class.

    Parameters
    ----------
    xdata : float array
        1D array of values of x_i
    ydata : float array
        1D array of values of y_i
    xlims : tuple
        limits in x to use (xmin, xmax)
    ylims : tuple
        Limits in y to use (ymin, ymax)
    Nx : int
        Number of KDE samples in X (regular grid between xmin and xmax)
    Ny : int
        Number of KDE samples in Y (regular grid between ymin and ymax)
    xeval : float array
        Evaluate on this set of x coordinates (takes precedence over regular grid)
    yeval : float array
        Evaluate on this set of y coordinates (takes precedence over regular grid)
    evalOndata : boolean
        If true return the log(density) evaluated on the data (instead of the regular grid)
    kde_bandwidth : float
        Bandwith for density estimator

    Returns
    -------
    log_dens : float array
        The log(density) evaluated on the regular grid (shape (Nx,Ny)), or the log(density) evaluated for the
        data points (shape (xdata.size,)).

    Other parameters
    ----------------
    **kwargs : dict
        Extra arguments for KernelDensity class initializer
    """

    if xlims == None:
        xmin = xdata.min()
        xmax = xdata.max()
    else:
        xmin = xlims[0]
        xmax = xlims[1]
    if ylims == None:
        ymin = ydata.min()
        ymax = ydata.max()
    else:
        ymin = ylims[0]
        ymax = ylims[1]

    data_values = np.vstack([xdata, ydata]).T
    # First scale the input data to unit variance and zero mean (to handle the possibly very different
    # input units), then estimate the KDE bandwidth and carry out the KDE.
    scaler = sklearn.preprocessing.StandardScaler().fit(data_values)
    scaled_values = scaler.transform(data_values)
    kde = sklearn.neighbours.KernelDensity(bandwidth=kde_bandwidth, **kwargs)
    kde.fit(scaled_values)
    if not (evalOnData):
        if not (xeval == None):
            positions = np.vstack([xeval.T.ravel(), yeval.T.ravel()]).T
            log_dens = kde.score_samples(scaler.transform(positions))
        else:
            Xsamples = np.linspace(xmin, xmax, Nx)
            Ysamples = np.linspace(ymin, ymax, Ny)
            X, Y = np.meshgrid(Xsamples, Ysamples)
            positions = np.vstack([X.T.ravel(), Y.T.ravel()]).T
            log_dens = (
                kde.score_samples(scaler.transform(positions)).reshape((Nx, Ny)).T
            )
    else:
        log_dens = kde.score_samples(scaled_values)

    return log_dens

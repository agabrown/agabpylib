"""
Provides wrappers around existing kernel density estimation methods. In addition some utility methods are
provided.

Anthony Brown May 2015 - Jun 2019
"""

import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
from agabpylib.tools.robuststats import rse


def kde_scikitlearn(data, N=100, lims=None, evalOnData=False, kde_bandwidth=1.0, **kwargs):
    """
    Provide a kernel density estimate for a set of data points (d_i). Make use of the scikit-learn
    scikitlearn.neighbours.KernelDensity class.

    Parameters
    ----------

    data - 1D array of values of d_i

    Keyword Arguments
    -----------------

    lims - Tuple with limits on data to use (dmin, dmax)
    N - Number of KDE samples in d (regular grid between dmin and dmax)
    evalOndata - If true returns the log(density) evaluated on the data (instead of the regular grid)
    kde_bandwidth - Bandwith for density estimator
    **kwargs - Extra arguments for KernelDensity class initializer

    Returns
    -------

    Dsamples, log_dens: The log(density) evaluated on the regular grid Dsamples (both shape (N,))
    
    OR 
    
    log_dens: The log(density) evaluated for the data points (shape (data.size,)).
    """
    if lims == None:
        dmin = data.min()
        dmax = data.max()
    else:
        dmin = lims[0]
        dmax = lims[1]

    kde = KernelDensity(bandwidth=kde_bandwidth, **kwargs)
    kde.fit(data[:, None])
    if not (evalOnData):
        Dsamples = np.linspace(dmin, dmax, N)[:, None]
        log_dens = kde.score_samples(Dsamples)
        return Dsamples, log_dens
    else:
        log_dens = kde.score_samples(data[:, None])
        return log_dens


def kde2d_scikitlearn(xdata, ydata, Nx=100, Ny=100, xeval=None, yeval=None, xlims=None, ylims=None, evalOnData=False,
                      kde_bandwidth=1.0, **kwargs):
    """
    Provide a 2D kernel density estimate for a set of data points (x_i, y_i). Make use of the
    scikit-learn scikitlearn.neighbours.KernelDensity class.

    Parameters
    ----------

    xdata - 1D array of values of x_i
    ydata - 1D array of values of y_i

    Keyword Arguments
    -----------------

    xlims - Tuple with limits in x to use (xmin, xmax)
    ylims - Tuple with limits in y to use (ymin, ymax)
    Nx - Number of KDE samples in X (regular grid between xmin and xmax)
    Ny - Number of KDE samples in Y (regular grid between ymin and ymax)
    xeval - evaluate on this set of x coordinates (takes precedence over regular grid)
    yeval - evaluate on this set of y coordinates (takes precedence over regular grid) 
    evalOndata - If true returns the log(density) evaluated on the data (instead of the regular grid)
    kde_bandwidth - Bandwith for density estimator
    **kwargs - Extra arguments for KernelDensity class initializer

    Returns
    -------

    The log(density) evaluated on the regular grid (shape (Nx,Ny)), or the log(density) evaluated for the
    data points (shape (xdata.size,)).
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
    scaler = StandardScaler().fit(data_values)
    scaled_values = scaler.transform(data_values)
    kde = KernelDensity(bandwidth=kde_bandwidth, **kwargs)
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
            log_dens = kde.score_samples(scaler.transform(positions)).reshape((Nx, Ny)).T
    else:
        log_dens = kde.score_samples(scaled_values)

    return log_dens

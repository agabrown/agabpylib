"""
Provides methods to plot the distribution of data. This can be univariate distributions, such as
histograms, or multivariate distributions, such as 2D histograms.

Anthony Brown May 2015 - Jul 2019
"""

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib import rc, cm

from matplotlib.ticker import NullFormatter
from agabpylib.plotting.inference import convert_to_stdev_nan

from agabpylib.densityestimation.kde import kde2d_scikitlearn, kde_scikitlearn
from agabpylib.plotting.distinct_colours import get_distinct

line_colours = get_distinct(4)

# Configure matplotlib
rc('text', usetex=True)
rc('font', family='serif', size=10)
rc('xtick.major', size='6')
rc('xtick.minor', size='4')
rc('ytick.major', size='6')
rc('ytick.minor', size='4')
rc('lines', linewidth=1.5)
rc('axes', linewidth=1)
rc('axes', facecolor='f0f0f0')
rc('axes', axisbelow=True)
rc('axes', prop_cycle=cycler('color', line_colours))
rc('xtick', direction='out')
rc('ytick', direction='out')
rc('grid', color='cbcbcb')
rc('grid', linestyle='-')
rc('grid', linewidth=0.5)
rc('grid', alpha=1.0)
rc('figure', facecolor='ffffff')


def plot_joint_kde_and_marginals(xdata, ydata, xname=None, yname=None, xunit=None, yunit=None,
                                 xlims=None, ylims=None, nx=100, ny=100, lnpmin=-5, contour_only=False,
                                 show_data=False):
    """
    Plot the joint distribution of two variables, X and Y, for which the data set {(x_i, y_i)} is
    available. The joint and marginal distributions and show in one plot, where the distributions are
    estimated using Kernel Density Estimation (KDE).

    Parameters
    ----------
    xdata : array_like
        1D array of values of x_i
    ydata : array_like
        1D array of values of y_i
    xname : str, optional
        Name of the X variable
    yname : str, optional
        Name of Y variable
    xunit : str, optional
        Units for the X variable.
    yunit : str, optional
        Units for the Y variable.
    xlims : tuple, optional
        Limits in x to use (min, max).
    ylims : tuple, optional
        Limits in y to use (min, max).
    nx : int, optional
        Number of KDE samples in X.
    ny : int, optional
        Number of KDE samples in Y.
    lnpmin : float, optional
        minimum value of the log of the KDE density (relative to maximum) to include in colour image.
    contour_only : bool, optional
        If true plot only the contours enclosing constant levels of cumulative probability.
    show_data : bool, optional
        If true plot the data points.

    Returns
    -------
    Figure object with the plot.
    """

    if xname is None:
        xname = r'$X$'
    if yname is None:
        yname = r'$Y$'
    if xlims is None:
        xmin = xdata.min()
        xmax = xdata.max()
    else:
        xmin = xlims[0]
        xmax = xlims[1]
    if ylims is None:
        ymin = ydata.min()
        ymax = ydata.max()
    else:
        ymin = ylims[0]
        ymax = ylims[1]

    # Set up the rectangles for the three panels of the plot
    ax_joint_rect = [0.12, 0.12, 0.5, 0.5]
    padding = 0.02
    marginal_h = 0.2
    ax_marginal_x_rect = [ax_joint_rect[0], ax_joint_rect[1] + ax_joint_rect[3] + padding, ax_joint_rect[2],
                          marginal_h]
    ax_marginal_y_rect = [ax_joint_rect[0] + ax_joint_rect[2] + padding, ax_joint_rect[1], marginal_h,
                          ax_joint_rect[2]]
    nullfmt = NullFormatter()

    fig = plt.figure(figsize=(6, 6), dpi=144)

    plot_joint_kde(xdata, ydata, xname=xname, yname=yname, xunit=xunit, yunit=yunit, xlims=xlims,
                   ylims=ylims, nx=nx, ny=ny, lnpmin=lnpmin, contour_only=contour_only, show_data=show_data)

    # Marginal over X
    ax_marginal_x = fig.add_axes(ax_marginal_x_rect)
    ax_marginal_x.xaxis.set_major_formatter(nullfmt)
    xsamples, log_dens = kde_scikitlearn(xdata, lims=(xmin, xmax), N=200)
    dens_kde = np.exp(log_dens)
    ax_marginal_x.fill_between(xsamples[:, 0], dens_kde, y2=0, alpha=0.5, facecolor=line_colours[0],
                               edgecolor='none')
    ax_marginal_x.plot(xsamples[:, 0], dens_kde, '-')
    ax_marginal_x.set_ylabel("$P($" + xname + "$)$")
    ax_marginal_x.set_xlim(xmin, xmax)
    ax_marginal_x.set_ylim(ymin=0.0)
    ax_marginal_x.grid()
    ax_marginal_x.spines['left'].set_position(('outward', 5))
    ax_marginal_x.spines['bottom'].set_visible(False)
    ax_marginal_x.spines['right'].set_visible(False)
    ax_marginal_x.spines['top'].set_visible(False)
    ax_marginal_x.yaxis.set_ticks_position('left')
    ax_marginal_x.xaxis.set_ticks_position('none')

    # Marginal over Y
    ax_marginal_y = fig.add_axes(ax_marginal_y_rect)
    ax_marginal_y.yaxis.set_major_formatter(nullfmt)
    ysamples, log_dens = kde_scikitlearn(ydata, lims=(ymin, ymax), N=200)
    dens_kde = np.exp(log_dens)
    ax_marginal_y.fill_betweenx(ysamples[:, 0], 0, dens_kde, alpha=0.5, facecolor=line_colours[0],
                                edgecolor='none')
    ax_marginal_y.plot(dens_kde, ysamples[:, 0], '-')
    ax_marginal_y.set_xlabel("$P($" + yname + "$)$")
    ax_marginal_y.set_ylim(ymin, ymax)
    ax_marginal_y.set_xlim(xmin=0.0)
    ax_marginal_y.grid()
    ax_marginal_y.spines['left'].set_visible(False)
    ax_marginal_y.spines['bottom'].set_position(('outward', 5))
    ax_marginal_y.spines['right'].set_visible(False)
    ax_marginal_y.spines['top'].set_visible(False)
    ax_marginal_y.yaxis.set_ticks_position('none')
    ax_marginal_y.xaxis.set_ticks_position('bottom')
    for xticklab in ax_marginal_y.get_xticklabels():
        xticklab.set_rotation(-90)

    return fig


def plot_joint_kde(xdata, ydata, xname=None, yname=None, xunit=None, yunit=None, xlims=None, ylims=None,
                   nx=100, ny=100, lnpmin=-5, contour_only=False, show_data=False):
    """
    Plot the joint distribution of two variables, X and Y, for which the data set {(x_i, y_i)} is
    available. The joint distribution are estimated using Kernel Density Estimation (KDE). The plot is
    produced using the currently active Axes object.

    Parameters
    ----------
    xdata : array_like
        1D array of values of x_i
    ydata : array_like
        1D array of values of y_i
    xname : str, optional
        Name of the X variable
    yname : str, optional
        Name of Y variable
    xunit :str, optional
        Units for the X variable
    yunit : str, optional
        Units for the Y variable
    xlims : tuple, optional
        Tuple with limits in x to use (min, max)
    ylims : tuple, optional
        Tuple with limits in y to use (min, max)
    nx : int, optional
        Number of KDE samples in X
    ny : int, optional
        Number of KDE samples in Y
    lnpmin : float, optional
        minimum value of the log of the KDE density (relative to maximum) to include in colour image
    contour_only : bool, optional
        If true plot only the contours enclosing constant levels of cumulative probability
    show_data : bool, optional
        If true plot the data points

    Returns
    -------
    Axis object with the plot.
    """

    if xname is None:
        xname = r'$X$'
    if yname is None:
        yname = r'$Y$'
    if xlims is None:
        xmin = xdata.min()
        xmax = xdata.max()
    else:
        xmin = xlims[0]
        xmax = xlims[1]
    if ylims is None:
        ymin = ydata.min()
        ymax = ydata.max()
    else:
        ymin = ylims[0]
        ymax = ylims[1]

    xsamples = np.linspace(xmin, xmax, nx)
    ysamples = np.linspace(ymin, ymax, ny)
    log_dens = kde2d_scikitlearn(xdata, ydata, xlims=(xmin, xmax), ylims=(ymin, ymax))
    log_dens = log_dens - log_dens.max()

    ax_joint = plt.gca()

    if np.logical_not(contour_only):
        kde_image = ax_joint.imshow(log_dens, aspect='auto', cmap=cm.Blues, extent=[xmin,
                                                                                    xmax, ymin, ymax], origin='lower')
        kde_image.set_clim(lnpmin, 0)
    ax_joint.contour(xsamples, ysamples, convert_to_stdev_nan(log_dens), levels=(0.01, 0.683, 0.955, 0.997),
                     colors='k')  # line_colours[0])
    if show_data:
        ax_joint.plot(xdata, ydata, '+k')
    if xunit is None:
        ax_joint.set_xlabel(xname)
    else:
        ax_joint.set_xlabel(xname + " [" + xunit + "]")
    if yunit is None:
        ax_joint.set_ylabel(yname)
    else:
        ax_joint.set_ylabel(yname + " [" + yunit + "]")
    ax_joint.grid()
    # Move left and bottom spines outward by 10 points
    ax_joint.spines['left'].set_position(('outward', 5))
    ax_joint.spines['bottom'].set_position(('outward', 5))
    # Hide the right and top spines
    ax_joint.spines['right'].set_visible(False)
    ax_joint.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax_joint.yaxis.set_ticks_position('left')
    ax_joint.xaxis.set_ticks_position('bottom')

    return ax_joint

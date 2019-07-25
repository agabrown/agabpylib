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
rc('axes', prop_cycle=cycler('color',line_colours))
rc('xtick', direction='out')
rc('ytick', direction='out')
rc('grid', color='cbcbcb')
rc('grid', linestyle='-')
rc('grid', linewidth=0.5)
rc('grid', alpha=1.0)
rc('figure', facecolor='ffffff')


def plot_joint_kde_and_marginals(xdata, ydata, xname=None, yname=None, xunit=None, yunit=None,
        xlims=None, ylims=None, Nx=100, Ny=100, lnpmin=-5, contourOnly=False, showData=False):
    """
    Plot the joint distribution of two variables, X and Y, for which the data set {(x_i, y_i)} is
    available. The joint and marginal distributions and show in one plot, where the distributions are
    estimated using Kernel Density Estimation (KDE).

    Parameters
    ----------

    xdata - 1D array of values of x_i
    ydata - 1D array of values of y_i

    Keyword Arguments
    -----------------

    xname - Name of the X variable
    yname - Name of Y variable
    xunit - Units for the X variable
    yunit - Units for the Y variable
    xlims - Tuple with limits in x to use (min, max)
    ylims - Tuple with limits in y to use (min, max)
    Nx - Number of KDE samples in X
    Ny - Number of KDE samples in Y
    lnpmin - minimum value of the log of the KDE density (relative to maximum) to include in colour image
    contourOnly - If true plot only the contours enclosing constant levels of cumulative probability
    showData - If true plot the data points

    Returns
    -------

    Figure object with the plot.
    """

    if xname==None:
        xname = r'$X$'
    if yname==None:
        yname = r'$Y$'
    if xlims==None:
        xmin = xdata.min()
        xmax = xdata.max()
    else:
        xmin = xlims[0]
        xmax = xlims[1]
    if ylims==None:
        ymin = ydata.min()
        ymax = ydata.max()
    else:
        ymin = ylims[0]
        ymax = ylims[1]

    # Set up the rectangles for the three panels of the plot
    axJoint_rect = [0.12, 0.12, 0.5, 0.5]
    padding =  0.02
    marginal_h = 0.2
    axMarginalX_rect = [axJoint_rect[0], axJoint_rect[1]+axJoint_rect[3]+padding, axJoint_rect[2],
            marginal_h]
    axMarginalY_rect = [axJoint_rect[0]+axJoint_rect[2]+padding, axJoint_rect[1], marginal_h,
            axJoint_rect[2]]
    nullfmt = NullFormatter()

    Xsamples = np.linspace(xmin,xmax,Nx)
    Ysamples = np.linspace(ymin,ymax,Ny)

    fig = plt.figure(figsize=(6,6), dpi=144)

    axJoint = fig.add_axes(axJoint_rect)
    plot_joint_kde(xdata, ydata, xname=xname, yname=yname, xunit=xunit, yunit=yunit, xlims=xlims,
            ylims=ylims, Nx=Nx, Ny=Ny, lnpmin=lnpmin, contourOnly=contourOnly, showData=showData)

    # Marginal over X
    axMarginalX = fig.add_axes(axMarginalX_rect)
    axMarginalX.xaxis.set_major_formatter(nullfmt)
    Xsamples, log_dens = kde_scikitlearn(xdata, lims=(xmin,xmax), N=200)
    dens_kde = np.exp(log_dens)
    axMarginalX.fill_between(Xsamples[:,0], dens_kde, y2=0, alpha=0.5, facecolor=line_colours[0],
            edgecolor='none')
    axMarginalX.plot(Xsamples[:,0], dens_kde, '-')
    axMarginalX.set_ylabel("$P($"+xname+"$)$")
    axMarginalX.set_xlim(xmin,xmax)
    axMarginalX.set_ylim(ymin=0.0)
    axMarginalX.grid()
    axMarginalX.spines['left'].set_position(('outward', 5))
    axMarginalX.spines['bottom'].set_visible(False)
    axMarginalX.spines['right'].set_visible(False)
    axMarginalX.spines['top'].set_visible(False)
    axMarginalX.yaxis.set_ticks_position('left')
    axMarginalX.xaxis.set_ticks_position('none')

    # Marginal over Y
    axMarginalY = fig.add_axes(axMarginalY_rect)
    axMarginalY.yaxis.set_major_formatter(nullfmt)
    Ysamples, log_dens = kde_scikitlearn(ydata, lims=(ymin,ymax), N=200)
    dens_kde = np.exp(log_dens)
    axMarginalY.fill_betweenx(Ysamples[:,0], 0, dens_kde, alpha=0.5, facecolor=line_colours[0],
            edgecolor='none')
    axMarginalY.plot(dens_kde, Ysamples[:,0], '-')
    axMarginalY.set_xlabel("$P($"+yname+"$)$")
    axMarginalY.set_ylim(ymin,ymax)
    axMarginalY.set_xlim(xmin=0.0)
    axMarginalY.grid()
    axMarginalY.spines['left'].set_visible(False)
    axMarginalY.spines['bottom'].set_position(('outward', 5))
    axMarginalY.spines['right'].set_visible(False)
    axMarginalY.spines['top'].set_visible(False)
    axMarginalY.yaxis.set_ticks_position('none')
    axMarginalY.xaxis.set_ticks_position('bottom')
    for xticklab in axMarginalY.get_xticklabels():
        xticklab.set_rotation(-90)

    return fig


def plot_joint_kde(xdata, ydata, xname=None, yname=None, xunit=None, yunit=None, xlims=None, ylims=None,
        Nx=100, Ny=100, lnpmin=-5, contourOnly=False, showData=False):
    """
    Plot the joint distribution of two variables, X and Y, for which the data set {(x_i, y_i)} is
    available. The joint distribution are estimated using Kernel Density Estimation (KDE). The plot is
    produced using the currently active Axes object.

    Parameters
    ----------

    xdata - 1D array of values of x_i
    ydata - 1D array of values of y_i

    Keyword Arguments
    -----------------

    xname - Name of the X variable
    yname - Name of Y variable
    xunit - Units for the X variable
    yunit - Units for the Y variable
    xlims - Tuple with limits in x to use (min, max)
    ylims - Tuple with limits in y to use (min, max)
    Nx - Number of KDE samples in X
    Ny - Number of KDE samples in Y
    lnpmin - minimum value of the log of the KDE density (relative to maximum) to include in colour image
    contourOnly - If true plot only the contours enclosing constant levels of cumulative probability
    showData - If true plot the data points

    Returns
    -------

    Axis object with the plot.
    """

    if xname==None:
        xname = r'$X$'
    if yname==None:
        yname = r'$Y$'
    if xlims==None:
        xmin = xdata.min()
        xmax = xdata.max()
    else:
        xmin = xlims[0]
        xmax = xlims[1]
    if ylims==None:
        ymin = ydata.min()
        ymax = ydata.max()
    else:
        ymin = ylims[0]
        ymax = ylims[1]

    Xsamples = np.linspace(xmin,xmax,Nx)
    Ysamples = np.linspace(ymin,ymax,Ny)
    log_dens = kde2d_scikitlearn(xdata, ydata, xlims=(xmin,xmax), ylims=(ymin,ymax))
    log_dens = log_dens-log_dens.max()

    axJoint = plt.gca()

    if np.logical_not(contourOnly):
        kde_image = axJoint.imshow(log_dens, aspect='auto', cmap=cm.Blues, extent=[xmin,
            xmax, ymin, ymax], origin='lower')
        kde_image.set_clim(lnpmin,0)
    axJoint.contour(Xsamples, Ysamples, convert_to_stdev_nan(log_dens), levels=(0.01, 0.683, 0.955, 0.997),
            colors='k')#line_colours[0])
    if showData:
        axJoint.plot(xdata, ydata, '+k')
    if xunit==None:
        axJoint.set_xlabel(xname)
    else:
        axJoint.set_xlabel(xname+" ["+xunit+"]")
    if yunit==None:
        axJoint.set_ylabel(yname)
    else:
        axJoint.set_ylabel(yname+" ["+yunit+"]")
    axJoint.grid()
    # Move left and bottom spines outward by 10 points
    axJoint.spines['left'].set_position(('outward', 5))
    axJoint.spines['bottom'].set_position(('outward', 5))
    # Hide the right and top spines
    axJoint.spines['right'].set_visible(False)
    axJoint.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    axJoint.yaxis.set_ticks_position('left')
    axJoint.xaxis.set_ticks_position('bottom')

    return axJoint

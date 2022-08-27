"""
Provides methods to visualize 1D or 2D distributions of data.

Anthony Brown May 2015 - Aug 2022

.. note::
    The functionalities offered here are covered much better by other tools for visualization
    of distributions such as `corner <https://github.com/dfm/corner.py>`_ or `arViz <https://www.arviz.org/en/latest/>`_.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cycler

from matplotlib.ticker import NullFormatter
import agabpylib.plotting.inference as infr

import agabpylib.densityestimation.kde as abkde
import agabpylib.plotting.distinct_colours as dc

__all__ = ["plot_joint_kde_and_marginals", "plot_joint_kde"]

line_colours = dc.get_distinct(4)

# Configure matplotlib
mpl.rc("text", usetex=True)
mpl.rc("font", family="serif", size=10)
mpl.rc("xtick.major", size="6")
mpl.rc("xtick.minor", size="4")
mpl.rc("ytick.major", size="6")
mpl.rc("ytick.minor", size="4")
mpl.rc("lines", linewidth=1.5)
mpl.rc("axes", linewidth=1)
mpl.rc("axes", facecolor="f0f0f0")
mpl.rc("axes", axisbelow=True)
mpl.rc("axes", prop_cycle=cycler.cycler("color", line_colours))
mpl.rc("xtick", direction="out")
mpl.rc("ytick", direction="out")
mpl.rc("grid", color="cbcbcb")
mpl.rc("grid", linestyle="-")
mpl.rc("grid", linewidth=0.5)
mpl.rc("grid", alpha=1.0)
mpl.rc("figure", facecolor="ffffff")


def plot_joint_kde_and_marginals(
    xdata,
    ydata,
    xname=None,
    yname=None,
    xunit=None,
    yunit=None,
    xlims=None,
    ylims=None,
    nx=100,
    ny=100,
    lnpmin=-5,
    contour_only=False,
    show_data=False,
):
    """
    Plot the joint distribution of two variables, X and Y, for which the data set {(x_i, y_i)} is
    available.

    The joint and marginal distributions are show in one plot, where the distributions are
    estimated using Kernel Density Estimation (KDE).

    Parameters
    ----------
    xdata : array_like
        1D array of values of x_i
    ydata : array_like
        1D array of values of y_i
    xname : str
        Name of the X variable
    yname : str
        Name of Y variable
    xunit : str
        Units for the X variable.
    yunit : str
        Units for the Y variable.
    xlims : tuple
        Limits in x to use (min, max).
    ylims : tuple
        Limits in y to use (min, max).
    nx : int
        Number of KDE samples in X.
    ny : int
        Number of KDE samples in Y.
    lnpmin : float
        minimum value of the log of the KDE density (relative to maximum) to include in colour image.
    contour_only : bool
        If true plot only the contours enclosing constant levels of cumulative probability.
    show_data : bool
        If true plot the data points.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object with the plot.
    """

    if xname is None:
        xname = r"$X$"
    if yname is None:
        yname = r"$Y$"
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
    ax_marginal_x_rect = [
        ax_joint_rect[0],
        ax_joint_rect[1] + ax_joint_rect[3] + padding,
        ax_joint_rect[2],
        marginal_h,
    ]
    ax_marginal_y_rect = [
        ax_joint_rect[0] + ax_joint_rect[2] + padding,
        ax_joint_rect[1],
        marginal_h,
        ax_joint_rect[2],
    ]
    nullfmt = NullFormatter()

    fig = plt.figure(figsize=(6, 6), dpi=144)

    plot_joint_kde(
        xdata,
        ydata,
        xname=xname,
        yname=yname,
        xunit=xunit,
        yunit=yunit,
        xlims=xlims,
        ylims=ylims,
        nx=nx,
        ny=ny,
        lnpmin=lnpmin,
        contour_only=contour_only,
        show_data=show_data,
    )

    # Marginal over X
    ax_marginal_x = fig.add_axes(ax_marginal_x_rect)
    ax_marginal_x.xaxis.set_major_formatter(nullfmt)
    xsamples, log_dens = abkde.kde_scikitlearn(xdata, lims=(xmin, xmax), N=200)
    dens_kde = np.exp(log_dens)
    ax_marginal_x.fill_between(
        xsamples[:, 0],
        dens_kde,
        y2=0,
        alpha=0.5,
        facecolor=line_colours[0],
        edgecolor="none",
    )
    ax_marginal_x.plot(xsamples[:, 0], dens_kde, "-")
    ax_marginal_x.set_ylabel("$P($" + xname + "$)$")
    ax_marginal_x.set_xlim(xmin, xmax)
    ax_marginal_x.set_ylim(ymin=0.0)
    ax_marginal_x.grid()
    ax_marginal_x.spines["left"].set_position(("outward", 5))
    ax_marginal_x.spines["bottom"].set_visible(False)
    ax_marginal_x.spines["right"].set_visible(False)
    ax_marginal_x.spines["top"].set_visible(False)
    ax_marginal_x.yaxis.set_ticks_position("left")
    ax_marginal_x.xaxis.set_ticks_position("none")

    # Marginal over Y
    ax_marginal_y = fig.add_axes(ax_marginal_y_rect)
    ax_marginal_y.yaxis.set_major_formatter(nullfmt)
    ysamples, log_dens = abkde.kde_scikitlearn(ydata, lims=(ymin, ymax), N=200)
    dens_kde = np.exp(log_dens)
    ax_marginal_y.fill_betweenx(
        ysamples[:, 0],
        0,
        dens_kde,
        alpha=0.5,
        facecolor=line_colours[0],
        edgecolor="none",
    )
    ax_marginal_y.plot(dens_kde, ysamples[:, 0], "-")
    ax_marginal_y.set_xlabel("$P($" + yname + "$)$")
    ax_marginal_y.set_ylim(ymin, ymax)
    ax_marginal_y.set_xlim(xmin=0.0)
    ax_marginal_y.grid()
    ax_marginal_y.spines["left"].set_visible(False)
    ax_marginal_y.spines["bottom"].set_position(("outward", 5))
    ax_marginal_y.spines["right"].set_visible(False)
    ax_marginal_y.spines["top"].set_visible(False)
    ax_marginal_y.yaxis.set_ticks_position("none")
    ax_marginal_y.xaxis.set_ticks_position("bottom")
    for xticklab in ax_marginal_y.get_xticklabels():
        xticklab.set_rotation(-90)

    return fig


def plot_joint_kde(
    xdata,
    ydata,
    xname=None,
    yname=None,
    xunit=None,
    yunit=None,
    xlims=None,
    ylims=None,
    nx=100,
    ny=100,
    lnpmin=-5,
    contour_only=False,
    show_data=False,
):
    """
    Plot the joint distribution of two variables, X and Y, for which the data set {(x_i, y_i)} is
    available.

    The joint distribution are estimated using Kernel Density Estimation (KDE). The plot is
    produced using the currently active Axes object.

    Parameters
    ----------
    xdata : array_like
        1D array of values of x_i
    ydata : array_like
        1D array of values of y_i
    xname : str
        Name of the X variable
    yname : str
        Name of Y variable
    xunit :str
        Units for the X variable
    yunit : str
        Units for the Y variable
    xlims : tuple
        Tuple with limits in x to use (min, max)
    ylims : tuple
        Tuple with limits in y to use (min, max)
    nx : int
        Number of KDE samples in X
    ny : int
        Number of KDE samples in Y
    lnpmin : float
        minimum value of the log of the KDE density (relative to maximum) to include in colour image
    contour_only : bool
        If true plot only the contours enclosing constant levels of cumulative probability
    show_data : bool
        If true plot the data points

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axes object with the plot.
    """

    if xname is None:
        xname = r"$X$"
    if yname is None:
        yname = r"$Y$"
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
    log_dens = abkde.kde2d_scikitlearn(
        xdata, ydata, xlims=(xmin, xmax), ylims=(ymin, ymax)
    )
    log_dens = log_dens - log_dens.max()

    ax_joint = plt.gca()

    if np.logical_not(contour_only):
        kde_image = ax_joint.imshow(
            log_dens,
            aspect="auto",
            cmap=mpl.cm.Blues,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
        )
        kde_image.set_clim(lnpmin, 0)
    ax_joint.contour(
        xsamples,
        ysamples,
        infr.convert_to_stdev_nan(log_dens),
        levels=(0.01, 0.683, 0.955, 0.997),
        colors="k",
    )  # line_colours[0])
    if show_data:
        ax_joint.plot(xdata, ydata, "+k")
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
    ax_joint.spines["left"].set_position(("outward", 5))
    ax_joint.spines["bottom"].set_position(("outward", 5))
    # Hide the right and top spines
    ax_joint.spines["right"].set_visible(False)
    ax_joint.spines["top"].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax_joint.yaxis.set_ticks_position("left")
    ax_joint.xaxis.set_ticks_position("bottom")

    return ax_joint

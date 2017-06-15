"""
Provides styled axes (plot boxes) inspired on the R default plotting style.

Anthony Brown Aug 2015 - May 2017
"""

from __future__ import print_function

#from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib import rc, cm
from cycler import cycler
import copy

from agabpylib.plotting.distinct_colours import get_distinct

line_colours = get_distinct(12)

# Configure matplotlib
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')
rc('font', family='serif', size=18)
rc('xtick.major', size='6')
rc('xtick.minor', size='4')
rc('ytick.major', size='6')
rc('ytick.minor', size='4')
rc('lines', linewidth=1.5)
rc('axes', linewidth=1)
#rc('axes', facecolor='f9f9f9')
rc('axes', prop_cycle=(cycler('color',line_colours)))
rc('xtick', direction='out')
rc('ytick', direction='out')
rc('grid', color='cbcbcb')
rc('grid', linestyle='-')
rc('grid', linewidth=0.5)
rc('grid', alpha=1.0)
rc('figure', facecolor='ffffff')
rc('figure', dpi=80)
rc('figure.subplot', bottom=0.125)

def get_basic_xy_axis(withgrid=False, minorticks=False):
    """
    Obtain an Axes object for basic XY plots.

    NOTE: code is not robust to empty DISPLAY environment variable. This would require hooking up
    matplotlib backends by hand instead of through the pyplot interface and then directly creating the
    Figure instance in order to use its gca() function.

    Parameters
    ----------

    None

    Keywords
    --------

    withgrid - When true a grid is displayed in the plot background
    minorticks - When true minor tickmarks are drawn.

    Returns
    -------

    Styled Axes object which can be used for further plotting instructions.
    """

    ax = plt.gca()
    configure_basic_xy_axis(ax, withgrid=withgrid, minorticks=minorticks)
    return ax

def configure_basic_xy_axis(ax, withgrid=False, minorticks=False):
    """
    Apply a basic configuration to the input axis object.

    Parameters
    ----------

    ax - The axis object to configure.

    Keywords
    --------

    withgrid - When true a grid is displayed in the plot background
    minorticks - When true minor tickmarks are drawn.

    Returns
    -------

    Nothing.
    """

    # Move left and bottom spines outward by 5 points
    ax.spines['left'].set_position(('outward', 5))
    ax.spines['bottom'].set_position(('outward', 5))
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    ax.tick_params('both', width=1.5, which='both')
    if withgrid:
        ax.grid(True)
    if minorticks:
        ax.minorticks_on()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    ax.tick_params('both', width=1.5, which='major')
    ax.set_facecolor('w')

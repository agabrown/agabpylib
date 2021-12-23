"""
Provides plotting styles.

Anthony Brown Aug 2015 - Dec 2021
"""

import matplotlib.pyplot as plt
from matplotlib import rc
from cycler import cycler
from agabpylib.plotting.distinct_colours import get_distinct


def useagab(usetex=False, fontfam='sans-serif', fontsize=18, sroncolours=False, ncolors=4, axislinewidths=1,
            linewidths=2, lenticks=6, return_colours=False):
    """
    Configure the plotting style to my liking.

    Parameters
    ----------
    usetex : boolean, optional
        Whether or not to use LaTeX text (default True).
    fontfam : str, optional
        Font family to use (default 'serif')
    fontsize : int, optional
        Font size (default 18)
    sroncolours : boolean, optional
        If true use colour-blind proof distinct colours (https://personal.sron.nl/~pault/).
    ncolors : int, optional
        Number of distinct colours to use (applies to SRON colours only, default 4)
    axislinewidths : float, optional
        Width of lines used to draw axes (default 1)
    linewidths : float, optional
        Width of lines used to draw plot elements (default 2)
    lenticks : float, optional
        Length of major tickmarks in points (default 6, minor tick marks adjusted automatically)

    Returns
    -------
    The list of colours used by the colour cycler.
    """
    if usetex:
        rc('text', usetex=True)
        rc('text.latex', preamble=r'\usepackage{amsmath}')
    else:
        rc('text', usetex=False)
    rc('font', family=fontfam, size=fontsize)
    rc('xtick.major', size=lenticks)
    rc('xtick.minor', size=lenticks * 2 / 3)
    rc('ytick.major', size=lenticks)
    rc('ytick.minor', size=lenticks * 2 / 3)
    rc('lines', linewidth=linewidths)
    rc('axes', linewidth=axislinewidths)
    rc('axes', facecolor='white')
    if sroncolours:
        line_colours = get_distinct(ncolors)
    else:
        line_colours = plt.cm.get_cmap('tab10').colors[0:ncolors]
    rc('axes', prop_cycle=(cycler('color', line_colours)))
    rc('xtick', direction='out')
    rc('ytick', direction='out')
    rc('grid', color='cbcbcb')
    rc('grid', linestyle='-')
    rc('grid', linewidth=0.5)
    rc('grid', alpha=1.0)
    rc('figure', dpi=80)
    rc('figure.subplot', bottom=0.125)

    if return_colours:
        return line_colours


def apply_tufte(ax, withgrid=False, minorticks=False, gridboth=False, yspine='left'):
    """
    Apply the "Tufte" style to the plot axes contained in the input axis object.

    This mimics the sparse style advocated by Tufte in his book "The Visual Display of Quantitative Information".

    Parameters
    ----------
    ax : matplotlib.axes instance
     The axis object to configure.
    withgrid : boolean, optional
        When true a grid is displayed in the plot background
    minorticks : boolean, optional
        When true minor tickmarks are drawn.
    gridboth : boolean, optional
        When true minor tickmarks are also used for the grid
    yspine : string, optional
        'left' set the vertical axis on the left, 'right' set vertical axis on right (default 'left')

    Returns
    -------
    Nothing.
    """

    if yspine == 'right':
        # Move right and bottom spines outward by 5 points
        ax.spines['right'].set_position(('outward', 5))
        ax.spines['bottom'].set_position(('outward', 5))
        # Hide the left and top spines
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('right')
        ax.xaxis.set_ticks_position('bottom')
    else:
        # Move left and bottom spines outward by 5 points
        ax.spines['left'].set_position(('outward', 5))
        ax.spines['bottom'].set_position(('outward', 5))
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(ax.spines[axis].get_linewidth())
    ax.tick_params('both', width=ax.spines['bottom'].get_linewidth(), which='both')
    if withgrid:
        if gridboth:
            ax.grid(which='both')
        else:
            ax.grid()
    if minorticks:
        ax.minorticks_on()
    # ax.set_facecolor('w')

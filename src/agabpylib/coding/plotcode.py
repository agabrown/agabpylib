"""Provides functions to support coding Python programmes that produce plots.

Anthony Brown Aug 2026 - Aug 2026
"""

import argparse
from matplotlib.figure import Figure
from matplotlib.pyplot import show, figure

_plot_file_formats = ("pdf", "png", "ps", "eps", "jpg", "svg")


def plotcode_parser(desc):
    """
    Create a command line argument parser that is initialized with the descprition of the programme and the options for exporting plots or showing them interactively.

    desc : str
        Description for the argument parser.

    Returns
    -------
    parser : argparse.ArgumentParser
        Initialized instance of the ArgumentParser
    """

    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        choices=_plot_file_formats,
        help="""File format for saved figure.""",
    )
    return parser


def save_or_show_plot(fig: Figure, basename: str = None, sfmt: str = None, **kwargs):
    """
    Save the plot or show it on screen.

    Parameters
    ----------
    fig: matplotlib.figure.Figure
        Figure instance with the plot to be saved or shown.
    basename : str
        Base name of file in which to save the figure (so without extension, can include path).
    sfmt : str
        Format of the file in which to save the figure. If None, show the figure on screen.
    kwargs : keywords
        Any additional keywords are passed to Figure.savefig().
    """
    if sfmt == None:
        figure(fig)
        show()
    else:
        fig.savefig(f"{basename}.{sfmt}", format=sfmt, **kwargs)

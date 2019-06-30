"""
Template for a simple script for making plots.

Anthony Brown Jan 2018 - Jun 2019
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse

from agabpylib.plotting.plotstyles import useagab, apply_tufte
from agabpylib.plotting.distinct_colours import get_distinct


def make_plot(args):
    """
    Code to make the plot.

    Parameters
    ----------

    args : dictionary
        Command line arguments

    Returns
    -------

    Nothing
    """

    basename = 'a-descriptive-name'
    if args['pdfOutput']:
        plt.savefig(basename+'.pdf')
    elif args['pngOutput']:
        plt.savefig(basename+'.png')
    else:
        plt.show()


def parseCommandLineArguments():
    """
    Set up command line parsing.
    """
    parser = argparse.ArgumentParser(description="""Describe the plot.""")
    parser.add_argument("-p", action="store_true", dest="pdfOutput", help="Make PDF plot")
    parser.add_argument("-b", action="store_true", dest="pngOutput", help="Make PNG plot")
    args = vars(parser.parse_args())
    return args


if __name__ in ('__main__'):
    args=parseCommandLineArguments()
    make_plot(args)

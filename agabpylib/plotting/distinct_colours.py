# -*- coding: iso-8859-1 -*-

"""
Colour-blind proof distinct colours module, based on work by Paul Tol
Pieter van der Meer, 2011
SRON - Netherlands Institute for Space Research

.. note::
    This code is for an older version of Tol's color scheme. It corresponds to the 2021
    "muted" color set (see `<https://personal.sron.nl/~pault/>`_).
"""

__all__ = ["get_distinct"]

# colour table in HTML hex format
hexcols = [
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#882255",
    "#AA4499",
    "#661100",
    "#6699CC",
    "#AA4466",
    "#4477AA",
]

greysafecols = ["#809BC8", "#FF6666", "#FFCC66", "#64C204"]

xarr = [
    [12],
    [12, 6],
    [12, 5, 6],
    [12, 3, 5, 6],
    [0, 1, 3, 5, 6],
    [0, 1, 3, 5, 6, 8],
    [0, 1, 2, 3, 5, 6, 8],
    [0, 1, 2, 3, 4, 5, 6, 8],
    [0, 1, 2, 3, 4, 5, 6, 7, 8],
    [0, 1, 2, 3, 4, 5, 9, 6, 7, 8],
    [0, 10, 1, 2, 3, 4, 5, 9, 6, 7, 8],
    [0, 10, 1, 2, 3, 4, 5, 9, 6, 11, 7, 8],
]


def get_distinct(nr):
    """
    Get specified number of distinct colours in HTML hex format.

    Parameters
    ----------
    nr : int
        number of colours [1..12]

    Returns
    -------
    hex_colours : list of str
        List of distinct colours in HTML hex format.
    """
    if nr < 1 or nr > 12:
        print("wrong nr of distinct colours!")
        return

    lst = xarr[nr - 1]

    #
    # generate colour list by stepping through indices and looking them up
    # in the colour table
    #

    i_col = 0
    col = [0] * nr
    for idx in lst:
        col[i_col] = hexcols[idx]
        i_col += 1
    return col


# displays usage information and produces example plot.
if __name__ == "__main__":
    import numpy as np
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt

    print(__doc__)
    print("usage examples: ")
    print("print distinct_colours.get_distinct(2)")
    print(get_distinct(2))
    print("print distinct_colours.greysafecols")
    print(greysafecols)

    print("\ngenerating example plot: distinct_colours_example.png")
    plt.close()
    t = np.arange(0.0, 2.0, 0.01)
    s = np.sin(2 * np.pi * t)
    c = np.cos(2 * np.pi * t)
    cols = get_distinct(2)
    plt.plot(t, s, linewidth=1.0, c=cols[0])
    plt.plot(t, c, linewidth=1.0, c=cols[1])

    plt.xlabel("time (s)")
    plt.ylabel("voltage (mV)")
    plt.title("Distinct colours example")
    plt.grid(True)
    plt.show()
    plt.savefig("distinct_colours_example.png")

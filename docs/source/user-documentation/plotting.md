# Plotting utilities

Package: `agabpylib.plotting`

## Introduction

Classes and functions for setting plotting styles, creating colour maps,  and making specific types of plots, all
based on [matplotlib](https://matplotlib.org).

## Plot styles

Detailed API: {py:mod}`agabpylib.plotting.plotstyles`

The plotting style can be set by invoking `useagab` which results in larger fonts, thicker lines,
specific tick lengths, and a choice for the number colours to use for the colour cycler. A colour-blind
friendly alternatice to matplotlib's tab10 colour scheme can be used, namely an older scheme by
[Paul Tol](https://personal.sron.nl/~pault/) (pre-2021).

The `apply_tufte` function mimics the sparse style advocated by Tufte in his book
"The Visual Display of Quantitative Information".

Example plots are include here to show the difference with the default matplotlib style.

```{eval-rst}
.. plot::

    import matplotlib.pyplot as plot
    import numpy as np
    from agabpylib.plotting.plotstyles import useagab, apply_tufte

    x=np.linspace(-np.pi,np.pi,1000)
    figA, axA  = plt.subplots(1, 1, figsize=(6,4.5))
    axA.plot(x, np.sin(x)*np.cos(x), label=r"$\sin(x)\cos(x)$")
    axA.set_title("default matplotlib")
    axA.legend()

    useagab()
    figB, axB  = plt.subplots(1, 1, figsize=(6,4.5))
    axB.plot(x, np.sin(x)*np.cos(x), label=r"$\sin(x)\cos(x)$")
    axB.set_title("useagab()")
    axB.legend()

    figC, axC  = plt.subplots(1, 1, figsize=(6,4.5))
    apply_tufte(axC)
    axC.plot(x, np.sin(x)*np.cos(x), label=r"$\sin(x)\cos(x)$")
    axC.set_title("useagab() + apply_tufte()")
    axC.legend()

    useagab(sroncolours=True)
    figD, axD  = plt.subplots(1, 1, figsize=(6,4.5))
    apply_tufte(axD)
    for k in range(4):
        axD.plot(x, np.sin(x)*np.cos((k+1)*x), label=rf"$\sin(x)\cos({k+1}x)$")
    axD.set_title("sroncolours=True")
    axD.legend(fontsize=12)

    plt.show()
```

## Colour maps

Detailed API: {py:mod}`agabpylib.plotting.agabcolormaps`

The `agabpylib.plotting.agabcolormaps` module adds colour maps which were collected from a variety
of sources. Most of these should not be used. The code is mostly retained for illustration of how to
create custom colour maps for use with matplotlib.

Here is a visualization of the colour maps.

```{eval-rst}
.. plot::

    from agabpylib.plotting.agabcolormaps import show_color_maps
    show_color_maps()
```

## Distinct colour-blind friendly colours

Detailed API: {py:mod}`agabpylib.plotting.distinct_colours`

The `useagab()` style offers the option to use the colour set developed by Paul Tol (SRON). The colour
set is from the 2011 code by Tol, and corresponds to the 2021 "muted" colour set (see [https://personal.sron.nl/~pault/](https://personal.sron.nl/~pault/)).

## Plots of distributions

Detailed API: {py:mod}`agabpylib.plotting.distributions`

The module `agabpylib.plotting.distributions` provides functions for making plots of 1D or 2D distributions
of data, such as samples from an MCMC run. This module is obsolete as the functionality is covered much better
by other packages such as, for example, [corner](https://github.com/dfm/corner.py)  or
[arViz](https://www.arviz.org/en/latest/).

## Plotting tools for inference

Detailed API: {py:mod}`agabpylib.plotting.inference`

The module `agabpylib.plotting.inference` provides functions that are useful when making plots in the
context of inference problems.

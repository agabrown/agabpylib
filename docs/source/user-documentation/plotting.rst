Plotting utilities (`agabpylib.plotting`)
=========================================

Introduction
------------

Classes and functions for setting plotting styles, creating colour maps,  and making specific types of plots, all
based on `matplotlib <https://matplotlib.org>`_.

Plot styles
-----------

The plotting style can be set by invoking ``useagab`` which results in larger fonts, thicker lines, 
specific tick lengths, and a choice for the number colours to use for the colour cycler. A colour-blind
friendly alternatice to matplotlib's tab10 colour scheme can be used, namely an older scheme by 
`Paul Tol <https://personal.sron.nl/~pault/>`_ (pre-2021).

The ``apply_tufte`` function mimics the sparse style advocated by Tufte in his book 
"The Visual Display of Quantitative Information".

Example plots are include here to show the difference with the default matplotlib style.

.. plot::

    import matplotlib.pyplot as plot
    import numpy as np
    from agabpylib.plotting.plotstyles import useagab, apply_tufte

    x=np.linspace(-np.pi,np.pi,1000)
    fig, axA  = plt.subplots(1, 1, figsize=(6,4.5))
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

    plt.show()

Colour maps
-----------

The ``agabpylib.plotting.agabcolormaps`` module adds colour maps which were collected from a variety
of sources. Most of these should not be used. The code is mostly retained for illustration of how to
create custom colour maps for use with matplotlib.

The colour maps are show below.

.. plot::

    from agabpylib.plotting.agabcolormaps import show_color_maps
    show_color_maps()

Reference/API
-------------

.. automodapi:: agabpylib.plotting.plotstyles
    :no-inheritance-diagram:

.. automodapi:: agabpylib.plotting.agabcolormaps
    :no-inheritance-diagram:

.. .. autosummary::
    :toctree: generated

    agabpylib.plotting.plotstyles
    agabpylib.plotting.agabcolormaps

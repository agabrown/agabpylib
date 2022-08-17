Tools for working with Gaia data
================================

Package: ``agabpylib.gaia``

Introduction
------------

This package provides tools for working with data from the `Gaia mission <https://www.cosmos.esa.int/web/gaia>`_.

.. note:: The ``gaia.ruwetools`` and ``gaia.edrthree``  modules are obsolete as of Gaia EDR3 
    (``ruwetools``) and Gaia DR3 (``edrthree``). The RUWE values have been included in the Gaia data releases
    as of Gaia EDR3 and the G-band photometry corrections for Gaia EDR3 are included already in the
    catalogue values for Gaia DR3.

Gaia EDR3 tools
---------------

Detailed API: :py:mod:`agabpylib.gaia.edrthree`

RUWE tools
----------

Detailed API: :py:mod:`agabpylib.gaia.ruwetools`

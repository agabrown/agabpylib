Statistical tools
=================

Package: `agabpylib.stats`

Provides statistical tools, including distributions not available from SciPy.

Distributions
-------------

Detailed API: :py:mod:`agabpylib.stats.distributions`

Provides statistical distributions not available from SciPy. They are implemented as subclasses of SciPy's 
`rv_continuous <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous>`_ class. 
The distributions can be imported similar to imports from `scipy.stats` as:

.. code-block:: python

    from agabpylib.stats import hoyt

Moments
-------

Detailed API: :py:mod:`agabpylib.stats.moments`

Provides functions not available from Numpy or Scipy for the calculation of
moments of distrubutions.

Robust statistics
-----------------

Detailed API: :py:mod:`agabpylib.stats.robuststats`

Functions for the calculation of robust statistical measures, such as the median
robust scatter estimate (RSE).

Robust rolling statistics
-------------------------

Detailed API: :py:mod:`agabpylib.stats.robustrollingstats`

Functions for the calculation of rolling robust statistical measures on series of data points.
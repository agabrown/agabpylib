"""
Provides rolling robust statistics for time series data, and data that can otherwise be split and ordered
according to some attribute.

Code originally writting in 2014 during Gaia commissioning.

Anthony Brown Jan 2014 - Apr 2022
"""

import numpy as np
from scipy.special import erfinv

from agabpylib.stats.robuststats import rse

__all__ = ["robust_rolling_stats", "cleanup_data"]

_rse_constant = 1.0 / (np.sqrt(2) * 2 * erfinv(0.8))


def robust_rolling_stats(series, window=5):
    """
    Calculate the rolling median and RSE for the input pandas series.

    Parameters
    ----------
    series : Pandas series
        Pandas series with the data. ASSUMED TO HAVE BEEN SORTED IN THE PROPER ORDER.
    Window : int
        Number of data points to include in rolling stats. Default 5

    Returns
    -------
    median, rse: Pandas series
        The rolling median and RSE.
    """
    rmedian = series.rolling(window).median()
    lowerq = series.rolling(window).quantile(0.1)
    upperq = series.rolling(window).quantile(0.9)
    rolling_rse = _rse_constant * (upperq - lowerq)
    #
    # treat first window data points
    #
    m = np.median(series[0:window])
    rrr = rse(series[0:window])
    rmedian[0:window] = m
    rolling_rse[0:window] = rrr

    return rmedian, rolling_rse


def cleanup_data(dframe, colname, window=50):
    """
    Remove outliers from the data frame by appling a filtering based on the rolling median and RSE. Any
    points further than 3*RSE from the local median in the series are removed.

    Parameters
    ----------
    dframe : Pandas DataFrame
        Pandas data frame with the data. ASSUMED TO HAVE BEEN SORTED IN THE PROPER ORDER.
    colname : str
        Name of column for which the data is to be cleaned.
    Window : int
        Number of data points from which to calculate the local median.

    Return
    ------
    clean_dframe : Pandas DataFrame
    rmedian : Pandas Series
    rolling_rse : Pandas Series
    cleanset : Pandas Series
        The cleaned up data frame, the rolling median series, and the rolling
        rse series. The boolean vector `cleanset` is the series indicating which
        points were selected for the clean data set.
    """
    series = dframe[colname]
    rmedian, rolling_rse = robust_rolling_stats(series, window=window)

    cleanset = abs(series - rmedian) <= 3 * rolling_rse
    return dframe[cleanset], rmedian[cleanset], rolling_rse[cleanset], cleanset

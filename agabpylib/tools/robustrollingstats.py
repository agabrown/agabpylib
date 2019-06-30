"""
Provides rolling robust statistics for time series data, and data that can otherwise be split and ordered
according to some attribute.

Code originally writting in 2014 during Gaia commissioning.

Anthony Brown Jan 2014 - June 2019
"""

import numpy as np
from agabpylib.tools.robuststats import rse


def robust_rolling_stats(series, window=5):
    """
    Calculate the rolling median and RSE for the input pandas series.

    Parameters
    ----------
    series - Pandas series with the data. ASSUMED TO HAVE BEEN SORTED IN THE PROPER ORDER.
  
    Keywords
    --------
  
    Window - Number of data points to include in rolling stats. Default 5
  
    Returns
    -------
  
    median, rse: both as pandas series
    """
    rmedian = series.rolling(window).median()
    lowerq = series.rolling(window).quantile(0.1)
    upperq = series.rolling(window).quantile(0.9)
    rolling_rse = 0.390152*(upperq-lowerq)
    #
    # treat first window data points
    #
    m=np.median(series[0:window])
    rrr=rse(series[0:window])
    rmedian[0:window]=m
    rolling_rse[0:window]=rrr
  
    return rmedian, rolling_rse


def cleanup_data(dframe, colname, window=50):
    """
    Remove outliers from the data frame by appling a filtering based on the rolling median and RSE. Any
    points further than 3*RSE from the local median in the series are removed.
  
    Parameters
    ----------
    dframe - Pandas data frame with the data. ASSUMED TO HAVE BEEN SORTED IN THE PROPER ORDER.
    colname - Name of column for which the data is to be cleaned.
  
    Keywords
    --------
  
    Window - Number of data points from which to calculate the local median. 
  
    Return
    ------
    clean_dframe, rmedian, rolling_rse, cleanset; the cleaned up data frame, the rolling median series, and
    the rolling rse series. The boolean vector cleanset is the series inidcating which points were selected
    for the clean data set.
    """
    series = dframe[colname]
    rmedian, rolling_rse = robust_rolling_stats(series, window=window)
  
    cleanset = abs(series-rmedian)<=3*rolling_rse
    return dframe[cleanset], rmedian[cleanset], rolling_rse[cleanset], cleanset

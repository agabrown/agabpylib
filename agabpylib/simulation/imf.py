"""
Provide classes and methods for the simulation of initial mass functions.

Anthony Brown Sep 2017 - Jul 2019
"""

import numpy as np
from scipy.stats import uniform
from abc import ABC, abstractmethod


class IMF(ABC):
    """
    Abstract base class for implementations of initial mass function simulations.
    """

    @abstractmethod
    def lnpdf(self, mass, min_mass, max_mass):
        """
        Evaluate the IMF (the probability density as a function of mass) for the input set of masses.

        Parameters
        ----------
        mass : float or float array
            The mass value or array of mass values for which to evaluate the IMF.
        min_mass : float
            Minimum mass of interval over which to evaluate the IMF.
        max_mass : float
            Maximum mass of interval over which to evaluate the IMF.

        Returns
        -------
        float array
            Value of the natural logarithm of the IMF for each of the input masses.
        """
        pass

    @abstractmethod
    def cdf(self, mass, min_mass, max_mass):
        """
        Evaluate the IMF in its cumulative form for the input set of masses.

        Parameters
        ----------
        mass : float or float array
            The mass value or array of mass values for which to evaluate the cumulative mass distribution
            function.
        min_mass : float
            Minimum mass of interval over which to evaluate the IMF.
        max_mass : float
            Maximum mass of interval over which to evaluate the IMF.

        Returns
        -------
        float array
            Value of the cumulative mass distribution function for each of the input masses
        """
        pass

    @abstractmethod
    def rvs(self, n, min_mass, max_mass):
        """
        Generate random masses from this IMF.

        Parameters
        ----------
        n : Number of random mass values to generate.
        min_mass : float
            Minimum mass of interval over which to evaluate the IMF.
        max_mass : float
            Maximum mass of interval over which to evaluate the IMF.

        Returns
        -------
        float array
            Array of random mass values.
        """
        pass

    @abstractmethod
    def getinfo(self):
        """
        Returns
        -------
        str :
            String with information about the IMF.
        """
        pass

    @abstractmethod
    def getmeta(self):
        """
        Returns
        -------
        dict :
            Metadata about the IMF.
        """


class MultiPartPowerLaw(IMF):
    """
    Multi-part power-law IMF.

    This class represents a multi-part power-law IMF, such as the one defined by Kroupa
    (https://ui.adsabs.harvard.edu/#abs/2001MNRAS.322..231K/abstract).
    """

    def __init__(self, slopes, break_points):
        """
        Class constructor/initializer.

        Parameters
        ----------
        slopes : float array
            The slopes alpha for each part of the power law, where the IMF is proportional to
            mass^(-alpha).
        break_points : float array
            The masses where the slopes change. Must contain one element less than the slopes array, the
            array should thus be empty for a single slope. The values should form an increasing sequence.
        """
        if np.isscalar(break_points):
            self.break_points = np.array([break_points])
        else:
            self.break_points = break_points
        if np.isscalar(slopes):
            self.slopes = np.array([slopes])
        else:
            self.slopes = slopes
        if self.slopes.size != self.break_points.size + 1:
            raise ValueError("The break_points array should contain 1 element less than the slopes array.")

    def _initialize_constants(self, min_mass, max_mass):
        """
        Initialize the constants needed for calculations with the IMF.

        Parameters
        ----------
        min_mass : float
            Minimum mass of interval over which to evaluate the IMF.
        max_mass : float
            Maximum mass of interval over which to evaluate the IMF.
        """
        active_breakpoints = np.where((self.break_points > min_mass) & (self.break_points < max_mass))
        mass_limits = self.break_points[active_breakpoints]
        mass_limits = np.concatenate(([min_mass], mass_limits, [max_mass]))
        if mass_limits.size == 2:
            active_slopes = np.array([self.slopes[np.searchsorted(self.break_points, max_mass)]])
        else:
            active_slope_indices = np.searchsorted(self.break_points, [min_mass, max_mass], side='right')
            active_slope_indices[(active_slope_indices > self.slopes.size - 1)] = self.slopes.size - 1
            active_slopes = self.slopes[active_slope_indices[0]:active_slope_indices[1] + 1]

        b = np.zeros(active_slopes.size)
        xlimits = np.zeros(mass_limits.size)
        gamma = 1.0 - active_slopes
        a = np.zeros(active_slopes.size)
        b[0] = 1.0
        a[0] = (mass_limits[1] ** gamma[0] - mass_limits[0] ** gamma[0]) / gamma[0]
        for i in range(1, active_slopes.size):
            b[i] = b[i - 1] * mass_limits[i] ** (active_slopes[i] - active_slopes[i - 1])
            a[i] = (mass_limits[i + 1] ** gamma[i] - mass_limits[i] ** gamma[i]) / gamma[i]
        normalization = np.sum(a * b)
        for i in range(xlimits.size):
            xlimits[i] = np.sum(a[0:i] * b[0:i]) / normalization

        return mass_limits, active_slopes, a, b, gamma, xlimits, normalization

    def lnpdf(self, mass, min_mass, max_mass):
        mass_limits, active_slopes, a, b, gamma, xlimits, normalization = self._initialize_constants(min_mass, max_mass)
        masses = np.array(mass)
        if masses.min() < min_mass or masses.max() > max_mass:
            raise ValueError("Mass array contains values outside interval [min_mass, max_mass]")
        lnimf = np.empty(masses.size)
        lnimf.fill(-np.inf)
        for i in range(active_slopes.size):
            indices = (mass_limits[i] <= masses) & (masses < mass_limits[i + 1])
            lnimf[indices] = np.log(b[i]) - np.log(normalization) - active_slopes[i] * np.log(masses[indices])
        indices = (masses == mass_limits[-1])
        lnimf[indices] = np.log(b[-1]) - np.log(normalization) - active_slopes[-1] * np.log(masses[indices])

        return lnimf

    def cdf(self, mass, min_mass, max_mass):
        mass_limits, active_slopes, a, b, gamma, xlimits, normalization = self._initialize_constants(min_mass, max_mass)
        masses = np.array(mass)
        if masses.min() < min_mass or masses.max() > max_mass:
            raise ValueError("Mass array contains values outside interval [min_mass, max_mass]")
        cimf = np.empty(masses.size)
        cimf.fill(-np.inf)
        for i in range(active_slopes.size):
            indices = (mass_limits[i] <= masses) & (masses < mass_limits[i + 1])
            cimf[indices] = (np.sum(a[0:i] * b[0:i]) + b[i] *
                             (masses[indices] ** gamma[i]
                              - mass_limits[i] ** gamma[i]) / gamma[i]) / normalization
        indices = (masses == mass_limits[-1])
        cimf[indices] = (np.sum(a[0:-1] * b[0:-1]) + b[-1] *
                         (masses[indices] ** gamma[-1] - mass_limits[-2] ** gamma[-1])
                         / gamma[-1]) / normalization

        return cimf

    def rvs(self, n, min_mass, max_mass):
        mass_limits, active_slopes, a, b, gamma, xlimits, normalization = self._initialize_constants(min_mass, max_mass)
        x = uniform.rvs(size=n)
        masses = np.zeros(x.size)
        for i in range(active_slopes.size):
            indices = (xlimits[i] <= x) & (x < xlimits[i + 1])
            masses[indices] = ((normalization * x[indices] - np.sum(a[0:i] * b[0:i])) /
                               b[i] * gamma[i] + mass_limits[i] ** gamma[i]) ** (1.0 / gamma[i])
        indices = (x == xlimits[-1])
        masses[indices] = ((normalization * x[indices] - np.sum(a[0:-1] * b[0:-1])) /
                           b[-1] * gamma[-1] + mass_limits[-1] ** gamma[-1]) ** (1.0 / gamma[-1])

        return masses

    def getinfo(self):
        return "Initial Mass Function\n" + \
               "---------------------\n" + \
               "Multi-part powerlaw: slopes {0}; masses of break-points {1}".format(self.slopes, self.break_points)

    def getmeta(self):
        return {'IMF': 'Multi-part powerlaw', 'IMF_slopes': self.slopes, 'IMF_break_points': self.break_points}

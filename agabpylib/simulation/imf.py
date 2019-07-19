"""
Provide classes and methods for the simulation of initial mass functions.

Anthony Brown Sep 2017 - Jun 2019
"""

import numpy as np
from scipy.stats import uniform

class IMF:
    """
    Base class for implementations of initial mass function simulations.
    """

    def pdf(self, mass, min_mass, max_mass):
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

        Value of the natural logarithm of the IMF for each of the input masses.
        """
        return self._pdf(mass, min_mass, max_mass)

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

        Value of the cumulative mass distribution function for each of the input masses
        """
        return self._cdf(mass, min_mass, max_mass)

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

        Array of random mass values.
        """
        return self._rvs(n, min_mass, max_mass)

    def showinfo(self):
        """
        Provide a string with information about the IMF.
        """
        return self._showinfo()

class MultiPartPowerLaw(IMF):
    """
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
        if (self.slopes.size != self.break_points.size+1):
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
        active_breakpoints = np.where((self.break_points>min_mass) & (self.break_points<max_mass))
        self.mass_limits = self.break_points[active_break_points]
        self.mass_limits = np.concatenate(([min_mass], self.mass_limits, [max_mass]))
        if self.mass_limits.size==2 and min_mass>self.break_points.max():
            self.active_slopes = self.slopes[-1]
        elif self.mass_limits.size==2 and max_mass<self.break_points.min():
            self.active_slopes = self.slopes[0]
        else

        self.B = np.zeros(self.slopes.size)
        self.xlimits = np.zeros(self.mass_limits.size)
        self.gamma = 1.0-self.slopes
        self.A = np.zeros(self.slopes.size)
        self.B[0] = 1.0
        self.A[0] = (self.mass_limits[1]**self.gamma[0] - self.mass_limits[0]**self.gamma[0]) / self.gamma[0]
        for i in range(1, self.slopes.size):
            self.B[i] = self.B[i-1] * self.mass_limits[i]**(self.slopes[i] - self.slopes[i-1])
            self.A[i] = (self.mass_limits[i+1]**self.gamma[i] - self.mass_limits[i]**self.gamma[i]) / self.gamma[i]
        self.normalization = np.sum(self.A*self.B)
        for i in range(self.xlimits.size):
            self.xlimits[i] = np.sum(self.A[0:i]*self.B[0:i])/self.normalization

    def _pdf(self, mass, min_mass, max_mass):
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

        Value of the natural logarithm of the IMF for each of the input masses.
        """
        self._initialize_constants(min_mass, max_mass)
        masses = np.array(mass)
        if masses.min()<min_mass or masses.max()>max_mass:
            raise ValueError("Mass array contains values outside interval [min_mass, max_mass]")
        lnimf = np.empty(masses.size)
        lnimf.fill(-np.inf)
        for i in range(self.slopes.size):
            indices = (self.mass_limits[i] <= masses) & (masses < self.mass_limits[i+1])
            lnimf[indices] = np.log(self.B[i]) - np.log(self.normalization) - self.slopes[i]*np.log(masses[indices])
        indices = (masses == self.mass_limits[-1])
        lnimf[indices] = np.log(self.B[-1]) - np.log(self.normalization) - self.slopes[-1]*np.log(masses[indices])

        return lnimf

    def _cdf(self, mass, min_mass, max_mass):
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

        Value of the cumulative mass distribution function for each of the input masses
        """
        self._initialize_constants(min_mass, max_mass)
        masses = np.array(mass)
        if masses.min()<min_mass or masses.max()>max_mass:
            raise ValueError("Mass array contains values outside interval [min_mass, max_mass]")
        cimf = np.empty(masses.size)
        cimf.fill(-np.inf)
        for i in range(self.slopes.size):
            indices = (self.mass_limits[i] <= masses) & (masses < self.mass_limits[i+1])
            cimf[indices] = (np.sum(self.A[0:i]*self.B[0:i]) + self.B[i] *
                    (masses[indices]**self.gamma[i] \
                            - self.mass_limits[i]**self.gamma[i]) / self.gamma[i]) / self.normalization
        indices = (masses == self.mass_limits[-1])
        cimf[indices] =  (np.sum(self.A[0:-1]*self.B[0:-1]) + self.B[-1] * \
                (masses[indices]**self.gamma[-1] - self.mass_limits[-2]**self.gamma[-1]) \
                / self.gamma[-1]) / self.normalization

        return cimf

    def _rvs(self, n, min_mass, max_mass):
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

        Array of random mass values.
        """
        self._initialize_constants(min_mass, max_mass)
        x = uniform.rvs(size=n)
        masses = np.zeros(x.size)
        for i in range(self.slopes.size):
            indices = (self.xlimits[i] <= x) & (x < self.xlimits[i+1])
            masses[indices] = ( (self.normalization*x[indices] - np.sum(self.A[0:i]*self.B[0:i])) / \
                    self.B[i] * self.gamma[i] + self.mass_limits[i]**self.gamma[i] ) ** (1.0/self.gamma[i])
        indices = (x == self.xlimits[-1])
        masses[indices] = ( (self.normalization*x[indices] - np.sum(self.A[0:-1]*self.B[0:-1])) / \
                    self.B[-1] * self.gamma[-1] + self.mass_limits[-1]**self.gamma[-1] ) ** (1.0/self.gamma[-1])

        return masses

    def _showinfo(self):
        """
        Provide a string with information about the IMF.
        """
        return "Multi-part power-law, slopes {0}, mass limits {1}".format(self.slopes, self.mass_limits)

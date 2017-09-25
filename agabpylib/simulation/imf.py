"""
Provide methods for the simulation of initial mass functions.

Anthony Brown Sep 2017
"""

import numpy as np

class MultiPartPowerLaw:
    """
    This class represents a multi-part power-law IMF, such as the one defined by Kroupa
    (https://ui.adsabs.harvard.edu/#abs/2001MNRAS.322..231K/abstract).
    """

    def __init__(self, slopes, mass_limits):
        """
        Class constructor/initializer.

        Parameters
        ----------

        slopes : Array of values of the slopes alpha for each part of the power law, where the IMF is
        proportional to mass^(-alpha).
        mass_limits : The limits of the mass intervals over which the slopes hold. Must contain one element
        more than the slopes array.
        """
        if np.isscalar(mass_limits):
            raise TypeError("The mass_limits parameters should be an array with at least two values.")    
        if np.isscalar(slopes):
            self.slopes = np.array([slopes])
        else:
            self.slopes = slopes
        if (self.slopes.size != mass_limits.size-1):
            raise ValueError("The mass_limits array should contain 1 element more than the slopes array.")
        self.mass_limits = mass_limits
        self.B = np.zeros(self.slopes.size)
        self.xlimits = np.zeros(self.slopes.size+2)
        gamma = 1.0-self.slopes
        A = np.zeros(self.slopes.size)
        self.B[0] = 1.0
        A[0] = (self.mass_limits[1]**gamma[0] - self.mass_limits[0]**gamma[0]) / gamma[0]
        for i in range(1, self.slopes.size):
            self.B[i] = self.B[i-1] * self.mass_limits[i]**(self.slopes[i] - self.slopes[i-1])
            A[i] = (self.mass_limits[i+1]**gamma[i] - self.mass_limits[i]**gamma[i]) / gamma[i]
        self.normalization = np.sum(A*self.B)
        for i in range(1, self.xlimits.size):
            self.xlimits[i] = np.sum(A[0:i+1]*self.B[0:i+1])
        self.xlimits[-1] = 1.0

    def evaluate(self, mass):
        """
        Evaluate the IMF (the probability density as a function of mass) for the input set of masses.

        Parameters
        ----------

        mass : The mass value or array of mass values for which to evaluate the IMF.

        Returns
        -------

        Value of the natural logarithm of the IMF for each of the input masses.
        """
        masses = np.array(mass)
        lnimf = np.empty(masses.size)
        lnimf.fill(-np.inf)
        for i in range(self.slopes.size):
            indices = (self.mass_limits[i] <= masses) & (masses < self.mass_limits[i+1])
            lnimf[indices] = np.log(self.B[i]) - np.log(self.normalization) - self.slopes[i]*np.log(masses[indices])
        indices = (masses == self.mass_limits[-1])
        lnimf[indices] = np.log(self.B[-1]) - np.log(self.normalization) - self.slopes[-1]*np.log(masses[indices])

        return lnimf
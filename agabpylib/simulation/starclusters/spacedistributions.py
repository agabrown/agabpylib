"""
Provides classes and methods to simulate the space distributions of stars in clusters

Anthony Brown Jul 2019 - Jul 2019
"""

import numpy as np
from scipy.stats import uniform
from abc import ABC, abstractmethod


class SpaceDistribution(ABC):
    """
    Abstract base class for classes representing the space distribution of stars in a cluster.
    """

    @abstractmethod
    def generate_positions(self, n):
        """
        Generate the positions of the stars randomly according to the prescribed space distribution.

        Parameters
        ----------
        n : int
            Number of star positions to generate.

        Returns
        -------
        x, y, z : astropy.units.Quantity arrays
            The x, y, z positions of the stars in units of pc.
        """
        pass

    def getinfo(self):
        """
        Returns
        -------
        str:
            String with information about the space distribution.
        """
        return "Space distribution\n" + \
               "------------------\n" + self.addinfo()

    @abstractmethod
    def addinfo(self):
        """
        Returns
        -------
        str:
            String with specific information about the space distribution.
        """
        pass

    @abstractmethod
    def getmeta(self):
        """
        Returns
        -------
        dict :
            Metadata on the space distribution of the cluster stars.
        """
        pass


class ConstantDensitySphere(SpaceDistribution):
    """
    Constant space density distribution.

    The cluster is assumed to be spherical with a given radius.

    Attributes
    ----------
    radius : astropy.units.Quantity
        Radius of the cluster in pc.
    """

    def __init__(self, r):
        """
        Class constructor/initializer.

        Parameters
        ----------
        r : astropy.units.Quantity
            Radius of the cluster in pc
        """
        self.radius = r

    def generate_positions(self, n):
        phi = uniform.rvs(loc=0, scale=2 * np.pi, size=n)
        theta = np.arcsin(uniform.rvs(loc=-1, scale=2, size=n))
        r = self.radius * uniform.rvs(loc=0, scale=1, size=n) ** (1. / 3.)
        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.cos(theta)
        z = r * np.sin(theta)
        return x, y, z

    def addinfo(self):
        return "Uniform space density distribution over sphere of radius {0}\n".format(self.radius)

    def getmeta(self):
        return {'space_distribution': 'uniform density sphere', 'space_distribution_radius': self.radius}


class PlummerSphere(SpaceDistribution):
    """
    Plummer density distribution.

    Implements a spherical Plummer distribution:
        density = C * (1+(r/a)^2)^(-5/2)
    Note that the mass of the cluster is ignored in this implementation.

    The density of stars in this model as a function of distance from the cluster centre (at (0,0,0)),
    expressed as a probability density, is given by
        rho(r) = (3/a^3) * (1+(r/a)^2)^(-5/2)
    while the number of stars per distance interval, expressed as a probability density, is:
        n(r) = (1/a)*(1+(r/a)^2)^(-3/2)

    Attributes
    ----------
    core_radius : astropy.units.Quantity
        Core radius in pc.
    """

    def __init__(self, a):
        """
        Class constructor/initializer.

        Parameters
        ----------
        a : astropy.units.Quantity
            Core radius of the cluster in pc
        """
        self.core_radius = a

    def generate_positions(self, n):
        phi = uniform.rvs(loc=0, scale=2 * np.pi, size=n)
        theta = np.arcsin(uniform.rvs(loc=-1, scale=2, size=n))
        h = uniform.rvs(loc=0, scale=1, size=n)
        r = self.core_radius / np.sqrt(h ** (-2 / 3) - 1)
        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.cos(theta)
        z = r * np.sin(theta)
        return x, y, z

    def addinfo(self):
        return "Plummer density distribution with core radius {0}".format(self.core_radius)

    def getmeta(self):
        return {'space_distribution': 'Plummer sphere', 'plummer_core_radius': self.core_radius}

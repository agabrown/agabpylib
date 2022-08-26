"""
Provides classes and methods to simulate the space distributions of stars in clusters.

.. note::
    The classes below only generate star positions according to a number density
    distribution. Only for equal mass stars will this result in a
    mass-distribution consistent with the number density distribution.

Anthony Brown Jul 2019 - Sep 2019
"""

import numpy as np
from abc import ABC, abstractmethod

__all__ = [
    "SpaceDistribution",
    "ConstantDensitySphere",
    "SphericalShell",
    "PlummerSphere",
    "TruncatedPlummerSphere",
]


class SpaceDistribution(ABC):
    """
    Abstract base class for classes representing the space distribution of stars in a cluster.
    """

    @abstractmethod
    def generate_positions(self, n, rng):
        """
        Generate the positions of the stars randomly according to the prescribed space distribution.

        Parameters
        ----------
        n : int
            Number of star positions to generate.
        rng : numpy.random.Generator
            Random number generator. This is provided separately to enable user
            control over the random number sequence.

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
        info : str
            String with information about the space distribution.
        """
        return "Space distribution\n" + "------------------\n" + self.addinfo()

    @abstractmethod
    def addinfo(self):
        """
        Returns
        -------
        info : str
            String with specific information about the space distribution.
        """
        pass

    @abstractmethod
    def getmeta(self):
        """
        Returns
        -------
        meta : dict
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

    def generate_positions(self, n, rng):
        phi = rng.uniform(low=0, high=2 * np.pi, size=n)
        theta = np.arcsin(rng.uniform(low=-1, high=1, size=n))
        r = self.radius * rng.uniform(low=0, high=1, size=n) ** (1.0 / 3.0)
        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.cos(theta)
        z = r * np.sin(theta)
        return x, y, z

    def addinfo(self):
        return "Uniform space density distribution over sphere of radius {0}\n".format(
            self.radius
        )

    def getmeta(self):
        return {
            "space_distribution": "uniform density sphere",
            "space_distribution_radius": self.radius,
        }


class SphericalShell(SpaceDistribution):
    """
    Stars are distributed in a spherical shell around the cluster centre (of
    mass) the shell has zero thickness.

    This space distribution is useful for investigating the correct simulation
    of kinematics (for example).

    Attributes
    ----------
    radius : astropy.units.Quantity
        Radius of the shell in pc.
    """

    def __init__(self, r):
        """
        Class constructor/initializer.

        Parameters
        ----------
        r : astropy.units.Quantity
            Radius of the shell in pc
        """
        self.radius = r

    def generate_positions(self, n, rng):
        phi = rng.uniform(low=0, high=2 * np.pi, size=n)
        theta = np.arcsin(rng.uniform(low=-1, high=1, size=n))
        r = self.radius
        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.cos(theta)
        z = r * np.sin(theta)
        return x, y, z

    def addinfo(self):
        return "Spherical shell of radius {0}\n".format(self.radius)

    def getmeta(self):
        return {
            "space_distribution": "spherical shell",
            "space_distribution_radius": self.radius,
        }


class PlummerSphere(SpaceDistribution):
    r"""
    Plummer density distribution.

    Implements a spherical Plummer distribution with core radius :math:`a`:

    .. math::

        \rho(r) = \frac{C}{(1+(r/a)^2)^{5/2}}

    Note that the mass of the cluster is ignored in this implementation.

    The density of stars in this model as a function of distance from the cluster centre (at (0,0,0)),
    expressed as a probability density, is given by:

    .. math::

        \rho(r) = \frac{3}{4\pi a^3} \frac{1}{(1+(r/a)^2)^{5/2}}

    The number of stars per distance interval, expressed as a probability density, is:

    .. math::

        n(r) = 4\pi r^2\rho(r)

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

    def generate_positions(self, n, rng):
        phi = rng.uniform(low=0, high=2 * np.pi, size=n)
        theta = np.arcsin(rng.uniform(low=-1, high=1, size=n))
        h = rng.uniform(low=0, high=1, size=n)
        r = self.core_radius / np.sqrt(h ** (-2 / 3) - 1)
        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.cos(theta)
        z = r * np.sin(theta)
        return x, y, z

    def addinfo(self):
        return "Plummer density distribution with core radius {0}".format(
            self.core_radius
        )

    def getmeta(self):
        return {
            "space_distribution": "Plummer sphere",
            "plummer_core_radius": self.core_radius,
        }


class TruncatedPlummerSphere(SpaceDistribution):
    """
    Truncated Plummer density distribution. A Plummer sphere truncated at a certain radius.

    Attributes
    ----------
    core_radius : astropy.units.Quantity
        Core radius in pc.
    truncation_radius : astropy.units.Quantity
        Truncation radius in pc.
    """

    def __init__(self, a, t):
        """
        Class constructor/initializer.

        Parameters
        ----------
        a : astropy.units.Quantity
            Core radius of the cluster in pc
        t : astropy.units.Quantity
            Truncation radius of the cluster in pc
        """
        self.core_radius = a
        self.truncation_radius = t

    def generate_positions(self, n, rng):
        c = (1 + (self.core_radius / self.truncation_radius) ** 2) ** (1.5)
        phi = rng.uniform(low=0, high=2 * np.pi, size=n)
        theta = np.arcsin(rng.uniform(low=-1, high=1, size=n))
        h = rng.uniform(low=0, high=1, size=n)
        r = self.core_radius / np.sqrt((c / h) ** (2 / 3) - 1)
        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.cos(theta)
        z = r * np.sin(theta)
        return x, y, z

    def addinfo(self):
        return "Truncated Plummer density distribution: core radius {0}, truncation radius {1}".format(
            self.core_radius, self.truncation_radius
        )

    def getmeta(self):
        return {
            "space_distribution": "Truncated Plummer sphere",
            "plummer_core_radius": self.core_radius,
            "plummer_truncation_radius": self.truncation_radius,
        }

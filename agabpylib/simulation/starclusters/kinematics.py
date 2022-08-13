"""
Provides classes and methods to simulate the kinematics of stars in clusters.

Anthony Brown Aug 2019 - Sep 2019
"""

import numpy as np
from scipy.stats import multivariate_normal
from abc import ABC, abstractmethod
import astropy.units as u


class Kinematics(ABC):
    """
    Abstract base class for classes representing the kinematics of stars in a cluster.
    """

    @abstractmethod
    def generate_kinematics(self, x, y, z):
        """
        Generate the velocity vectors of the stars randomly according to the prescribed kinematics of the cluster.

        Parameters
        ----------
        x, y, z : astropy.units.Quantity arrays
            The x, y, z positions of the stars with respect to the barycentre, in units of pc, where the barycentre of
            the cluster is assumed to be at (0, 0, 0) pc!

        Returns
        -------
        v_x, v_y, v_z : astropy.units.Quantity arrays
            The v_x, v_y, v_z velocity components of the stars in units of km/s.
        """
        pass

    def getinfo(self):
        """
        Returns
        -------
        str:
            String with information about the cluster kinematics.
        """
        return "Kinematics\n" + "------------------\n" + self.addinfo()

    @abstractmethod
    def addinfo(self):
        """
        Returns
        -------
        str:
            String with specific information about the cluster kinematics.
        """
        pass

    @abstractmethod
    def getmeta(self):
        """
        Returns
        -------
        dict :
            Metadata on the kinematics of the cluster stars.
        """
        pass


class LinearVelocityField(Kinematics):
    """
    General linear velocity field for the cluster stars which besides the mean cluster velocity can include a
    rotation and isotropic expansion/contraction term, both defined with respect to the cluster centre. The velocities
    are generated according to equations (3) and (4) in Lindegren et al (2000,
    https://ui.adsabs.harvard.edu/abs/2000A%26A...356.1119L/abstract), where the "dilation" terms are not used.

    Specifically the dilation terms :math:`w_1, w_2, w_3` are set to zero, while the diagonal terms of the matrix
    :math:`T` are set equal to the expansion rate :math:`\kappa` (i.e., :math:`w_4=w_5=T_{zz}=\kappa`).

    Attributes
    ----------
    v : astropy.units.Quantity 3-vector
        Mean cluster velocity [v_x, v_y, v_z] km/s.
    s : astropy.units.Quantity 3-vector
        Cluster velocity dispersion [s_x. s_y, s_z] in  km/s.
    omega : astropy.units.Quantity 3-vector
        Angular speed of the cluster [omega_x, omega_y, omega_z] in km/s/pc
    kappa : astropy.units.Quantity
        Cluster expansion/contraction rate in km/s/pc
    tmat : astropy.units.Quantity 3x3 matrix
        The matrix describing the linear velocity field of the cluster with respect to the cluster barycentre.
    """

    def __init__(self, v, s, omega, kappa):
        """
        Class constructor/initializer

        Parameters
        ----------
        v : astropy.units.Quantity 3-vector
            Mean cluster velocity [v_x, v_y, v_z] km/s.
        s : astropy.units.Quantity 3-vector
            Cluster velocity dispersion [s_x. s_y, s_z] in  km/s.
        omega : astropy.units.Quantity 3-vector
            Angular speed of the cluster [omega_x, omega_y, omega_z] in km/s/pc
        kappa : astropy.units.Quantity
            Cluster expansion/contraction rate in km/s/pc
        """
        self.v = v
        self.s = s
        self.omega = omega
        self.kappa = kappa
        self.tmat = np.empty((3, 3)) * u.km / u.s / u.pc
        self.tmat[0, 0] = kappa
        self.tmat[1, 1] = self.tmat[0, 0]
        self.tmat[2, 2] = self.tmat[0, 0]
        self.tmat[0, 1] = -self.omega[2]
        self.tmat[1, 0] = -self.tmat[0, 1]
        self.tmat[0, 2] = self.omega[1]
        self.tmat[2, 0] = -self.tmat[0, 2]
        self.tmat[1, 2] = -self.omega[0]
        self.tmat[2, 1] = -self.tmat[1, 2]

    def generate_kinematics(self, x, y, z):
        positions = np.array([x, y, z]) * x.unit
        covmat = np.zeros((3, 3))
        covmat[np.diag_indices(3)] = self.s * self.s
        v_x, v_y, v_z = multivariate_normal.rvs(
            cov=covmat, size=x.size
        ).T * u.km / u.s + np.matmul(self.tmat, positions)
        return v_x + self.v[0], v_y + self.v[1], v_z + self.v[2]

    def addinfo(self):
        return "Linear velocity field:\n v = {0}\n s = {1}\n omega = {2}\n kappa = {3}\n".format(
            self.v, self.s, self.omega, self.kappa
        )

    def getmeta(self):
        return {
            "kinematics": "Linear velocity field",
            "mean_velocity": self.v,
            "velocity_dispersion": self.s,
            "rotation_rate": self.omega,
            "expansion_rate": self.kappa,
        }

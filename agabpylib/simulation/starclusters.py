"""
Provide classes and methods for basic simulations of star clusters, where the main aim is to support
kinematic modelling studies.

Anthony Brown Jul 2019 - Jul 2019
"""

import numpy as np
from sys import stderr
from scipy.stats import norm, uniform
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.table import Table
from os import path

from agabpylib.stellarmodels.io.readisocmd import MIST, PARSEC

class StarAPs:
    """
    Class that generates the astrophysical parameters (mass, Teff, luminosity, colour, etc) for the stars
    in the cluster.
    """

    def __init__(self, age, metallicity, alphafeh, isofiles, imf, iso="parsec"):
        """
        Class constructor/initializer

        Parameters
        ----------

        age : float
            Cluster age.
        metallicity : float
            Cluster metallicity ([Fe/H] parameter in MIST/PARSEC isochrones)
        alphafeh : float
            Cluster alpha-element enhancement (afe parameter for MIST isochrones, not relevant for
            PARSEC)
        isofiles : string
            Path to folder with isochrone files. Should be compatible with isochrone set choice.
        imf : agabpylib.simulations.imf.IMF
            Instance of the agabpylib.simulations.imf.IMF class.

        Keywords
        --------

        iso : string
            Which isochrone set to use: "mist" or "parsec", default "parsec".
        """
        self.age = age
        self.logage = np.log10(age.to(u.yr).value)
        self.metallicity = metallicity
        self.afeh = alphafeh
        if path.exists(isofiles):
            self.isofiles = path.abspath(isofiles)
        else:
            raise ValueError("Path to isochrone data does not appear to exist!")
        self.imf = imf

        if self.metallicity < 0.0:
            mehsign = "m"
        else:
            mehsign = "p"
        if self.afeh < 0.0:
            afehsign = "m"
        else:
            afehsign = "p"
        if iso=="mist":
            self.isofilename = \
            "MIST_v1.2_feh_{0}{1:4.2f}_afe_{2}{3:3.1f}_vvcrit0.4_UBVRIplus.iso.cmd".format(mehsign,
                    np.abs(self.metallicity), afehsign, np.abs(self.afeh))
            isoreader = MIST
        else:
            self.isofilename = "PARSEC"
            isoreader = PARSEC

        self.isocmd = isoreader(path.join(self.isofiles,self.isofilename))

    def generate_aps(self, n):
        """
        Generate the simulated APs from the IMF and the isochrone data.

        Parameters
        ----------

        n : int
            Number of stars for which to generate the APs

        Returns
        -------

        Table with a list of stars and their properties.
        """

        ids = np.arange(n)
        ini_masses = self.imf.rvs(n)
        age_index = self.isocmd.age_index(self.logage)
        iso_ini_masses = self.isocmd.isocmds[age_index]['initial_mass']
        max_iso_ini_mass = iso_ini_masses.max()

        iso_masses = self.isocmd.isocmds[age_index]['star_mass']
        f = interp1d(iso_ini_masses, iso_masses)
        masses = f(ini_masses)*u.Msun
        f = interp1d(iso_ini_masses, self.isocmd.isocmds[age_index]['log_L'])
        logL = f(ini_masses)
        f = interp1d(iso_ini_masses, self.isocmd.isocmds[age_index]['log_Teff'])
        logTeff = f(ini_masses)
        f = interp1d(iso_ini_masses, self.isocmd.isocmds[age_index]['log_g'])
        logg = f(ini_masses)

        return Table([ids, masses, logL, logTeff, logg], names=('ID', 'mass', 'log_L', 'log_Teff', 'log_g'))

    def showinfo(self):
        """
        Print out some information on the simulation of the astrophysical parameters.
        """

        print("Astrophysical parameters")
        print("------------------------")
        print()
        print("Age: {0}".format(self.age))
        print("[M/H]: {0}".format(self.metallicity))
        print("[alpha/Fe]: {0}".format(self.afeh))
        print("IMF: {0}".format(self.imf.showinfo()))
        print("Isochrone file: {0}".format(path.join(self.isofiles,self.isofilename)))

class StarCluster:
    """
    Base class for simulation of a star cluster. The cluster stars are assumed to be single (no binaries
    or multiples) and drawn from a single isochrone, implying the same age and chemical composition for
    all cluster members. The PARSEC or MIST isochrone sets can be used to generate the simulated stars.
    The focus is on simulating Gaia observations of the clusters.
    """

    def __init__(self, n_stars, staraps):
        """
        Class constructor/initializer

        Parameters
        ----------

        n_stars : int
            Number of stars in the cluster.
        staraps : agabpylib.simulation.starclusters.StarAPs
            Class that generates the astrophysical parameters for the cluster stars.

        Keywords
        --------

        """
        self.n_stars = n_stars
        self.staraps = staraps
        self.star_table = self.staraps.generate_aps(self.n_stars)

    def showinfo(self):
        """
        Print out some information on the simulated cluster.
        """

        print("Simulated cluster paramaters")
        print("----------------------------")
        print()
        print("Number of stars: {0}".format(self.n_stars))
        print()
        self.staraps.showinfo()

    def write_star_table(self, filename):
        """
        Write the star table to file in VOTable format.

        Parameters
        ----------

        filename : string
            Name of the file to write to.
        """
        self.star_table.write(filename, format="votable")

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
from astropy.table import Table, Column
from os import path

from agabpylib.stellarmodels.io.readisocmd import MIST, PARSEC


class StarAPs:
    """
    Class that generates the astrophysical parameters (mass, Teff, luminosity, colour, etc) for the stars
    in the cluster.
    """

    def __init__(self, age, metallicity, alphafeh, vvcrit, isofiles, imf, iso="mist"):
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
        vvcrit : float
            v/vcrit parameter for the MIST isochrone set. This is ignored for the PARSEC isochrones.
        isofiles : string
            Path to folder with isochrone files. The folder is expected to contain the following sub-folders:
                MIST_v1.2_vvcrit0.0_UBVRIplus
                MIST_v1.2_vvcrit0.4_UBVRIplus
                PARSEC_v1.2S_GaiaMAW_UBVRIJHK
            The latter is for PARSEC files that where joined between the Gaia and UBVRIJHK photometric systems.
        imf : agabpylib.simulations.imf.IMF
            Instance of the agabpylib.simulations.imf.IMF class.
        iso : string, optional
            Which isochrone set to use: "mist" or "parsec", default "mist".
        """
        self.age = age
        self.logage = np.log10(age.to(u.yr).value)
        self.logageloaded = self.logage
        self.metallicity = metallicity
        self.afeh = alphafeh
        self.vvcrit = vvcrit
        self.modelset = iso
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
        if self.modelset == "mist":
            subfolder = "MIST_v1.2_vvcrit{0:3.1f}_UBVRIplus".format(self.vvcrit)
            fstring = "MIST_v1.2_feh_{0}{1:4.2f}_afe_{2}{3:3.1f}_vvcrit{4:3.1f}_UBVRIplus.iso.cmd"
            self.isofilename = fstring.format(mehsign, np.abs(self.metallicity), afehsign, np.abs(self.afeh),
                                              self.vvcrit)
            isoreader = MIST
            self.tabledict = {'initial_mass': 'initial_mass', 'mass': 'star_mass', 'log_L': 'log_L',
                              'log_Teff': 'log_Teff', 'log_g': 'log_g', 'G': 'Gaia_G_MAW', 'G_BPb': 'Gaia_BP_MAWb',
                              'G_BPf': 'Gaia_BP_MAWf', 'G_RP': 'Gaia_RP_MAW', 'V': 'Bessell_V', 'I': 'Bessell_I'}
            self.isofullpath = path.join(self.isofiles, subfolder, self.isofilename)
        else:
            subfolder = "PARSEC_v1.2S_GaiaMAW_UBVRIJHK"
            fstring = "PARSEC_v1.2S_feh_{0}{1:4.2f}_afe_{2}{3:3.1f}_GaiaMAW_UBVRIJHK.iso.cmd"
            self.isofilename = fstring.format(mehsign, np.abs(self.metallicity), afehsign, np.abs(self.afeh))
            isoreader = PARSEC
            self.tabledict = {'initial_mass': 'Mini', 'mass': 'Mass', 'log_L': 'logL',
                              'log_Teff': 'logTe', 'log_g': 'logg', 'G': 'Gmag', 'G_BPb': 'G_BPbrmag',
                              'G_BPf': 'G_BPftmag', 'G_RP': 'G_RPmag', 'V': 'Vmag', 'I': 'Imag'}
            self.isofullpath = path.join(self.isofiles, subfolder, self.isofilename)

        self.isocmd = isoreader(path.join(self.isofullpath))

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
        age_index = self.isocmd.age_index(self.logage)
        self.logageloaded = self.isocmd.ages[age_index]
        iso_ini_masses = self.isocmd.isocmds[age_index][self.tabledict['initial_mass']]

        aptable = Table()
        aptable.add_column(Column(np.arange(n)), name='ID')
        ini_masses = self.imf.rvs(n, iso_ini_masses.min(), iso_ini_masses.max())
        aptable.add_column(Column(ini_masses), name='initial_mass')

        for item in list(self.tabledict.keys())[1:]:
            y = self.isocmd.isocmds[age_index][self.tabledict[item]]
            f = interp1d(iso_ini_masses, y)
            if item == 'mass':
                aptable.add_column(Column(f(ini_masses) * u.M_sun), name=item)
            else:
                aptable.add_column(Column(f(ini_masses)), name=item)

        return aptable

    def showinfo(self):
        """
        Print out some information on the simulation of the astrophysical parameters.
        """
        print("Astrophysical parameters")
        print("------------------------")
        print()
        print("Isochrone models: {0}".format(self.modelset))
        print("Age, log(Age) specified: {0}, {1}".format(self.age, self.logage))
        print("log(Age) loaded: {0}".format(self.logageloaded))
        print("[M/H]: {0}".format(self.metallicity))
        print("[alpha/Fe]: {0}".format(self.afeh))
        if self.modelset == 'mist':
            print("[v/vcrit]: {0}".format(self.afeh))
        print("IMF: {0}".format(self.imf.showinfo()))
        print("Isochrone file: {0}".format(self.isofullpath))


class StarCluster:
    """
    Base class for simulation of a star cluster.

    The cluster stars are assumed to be single (no binaries or multiples) and drawn from a single isochrone,
    implying the same age and chemical composition for all cluster members. The PARSEC or MIST isochrone sets can be
    used to generate the simulated stars.The focus is on simulating Gaia observations of the clusters.
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

"""
Provide classes and methods for basic simulations of star clusters, where the main aim is to support
kinematic modelling studies.

Anthony Brown Jul 2019 - Aug 2019
"""

import numpy as np
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.table import QTable, Column
from os import path
from datetime import datetime
from agabpylib.stellarmodels.io.readisocmd import MIST, PARSEC


class StarAPs:
    """
    Class that generates the astrophysical parameters (mass, Teff, luminosity, colour, etc) for the stars
    in the cluster.

    Attributes
    ----------
    age : astropy.units.Quantity
        Cluster age.
    logage : float
        Log10(age).
    logageloaded : float
        Value of logage of the actual isochrone used to simulate the cluster (which may be slightly different from
        requested logage).
    metallicity : float
        Cluster metallicity ([Fe/H] parameter in MIST/PARSEC isochrones)
    alphafeh : float
        Cluster alpha-element enhancement (afe parameter for MIST isochrones, not relevant for PARSEC)
    vvcrit : float
        v/vcrit parameter for the MIST isochrone set. This is ignored for the PARSEC isochrones.
    isofiles : str
        Path to folder with isochrone files.
    imf : agabpylib.simulations.imf.IMF
        Instance of the agabpylib.simulations.imf.IMF class.
    modelset : str
        Which isochrone set is used.
    meta : dict
        Metadata describing the astrophysical modelling.
    """

    def __init__(self, age, metallicity, alphafeh, vvcrit, isofiles, imf, iso="mist"):
        """
        Class constructor/initializer

        Parameters
        ----------
        age : astropy.units.Quantity
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
                              'log_Teff': 'log_Teff', 'log_g': 'log_g', 'Gabs': 'Gaia_G_MAW',
                              'Gabs_BPb': 'Gaia_BP_MAWb', 'Gabs_BPf': 'Gaia_BP_MAWf', 'Gabs_RP': 'Gaia_RP_MAW',
                              'Vabs': 'Bessell_V', 'Iabs': 'Bessell_I'}
            self.isofullpath = path.join(self.isofiles, subfolder, self.isofilename)
        else:
            subfolder = "PARSEC_v1.2S_GaiaMAW_UBVRIJHK"
            fstring = "PARSEC_v1.2S_feh_{0}{1:4.2f}_afe_{2}{3:3.1f}_GaiaMAW_UBVRIJHK.iso.cmd"
            self.isofilename = fstring.format(mehsign, np.abs(self.metallicity), afehsign, np.abs(self.afeh))
            isoreader = PARSEC
            self.tabledict = {'initial_mass': 'Mini', 'mass': 'Mass', 'log_L': 'logL',
                              'log_Teff': 'logTe', 'log_g': 'logg', 'Gabs': 'Gmag', 'Gabs_BPb': 'G_BPbrmag',
                              'Gabs_BPf': 'G_BPftmag', 'Gabs_RP': 'G_RPmag', 'Vabs': 'Vmag', 'Iabs': 'Imag'}
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

        aptable = QTable()
        aptable.add_column(Column(np.arange(n)), name='source_id')
        ini_masses = self.imf.rvs(n, iso_ini_masses.min(), iso_ini_masses.max())
        aptable.add_column(Column(ini_masses) * u.M_sun, name='initial_mass')

        for item in list(self.tabledict.keys())[1:]:
            y = self.isocmd.isocmds[age_index][self.tabledict[item]]
            f = interp1d(iso_ini_masses, y)
            if item == 'mass':
                aptable.add_column(Column(f(ini_masses) * u.M_sun), name=item)
            else:
                aptable.add_column(Column(f(ini_masses)), name=item)

        return aptable

    def getmeta(self):
        """
        Returns
        -------
        dict :
            Dictionary containing metadata describing the astrophysical parameter model.
        """
        meta = {'ID': 'Simulated_cluster', 'age': self.age, 'logage': self.logage, 'logageloaded': self.logageloaded,
                'metallicity': self.metallicity, 'alpha_over_fe': self.afeh, 'vvcrit': self.vvcrit,
                'stellarmodels': self.modelset, 'isochronefile': self.isofullpath}
        meta.update(self.imf.getmeta())
        return meta

    def getinfo(self):
        """
        Returns
        -------
        str :
            String with information on the simulation of the astrophysical parameters.
        """
        info = ""
        info = info + "Astrophysical parameters\n" + \
               "------------------------\n" + \
               "Isochrone models: {0}\n".format(self.modelset) + \
               "Age, log(Age) specified: {0}, {1}\n".format(self.age, self.logage) + \
               "log(Age) loaded: {0}\n".format(self.logageloaded) + \
               "[M/H]: {0}\n".format(self.metallicity) + \
               "[alpha/Fe]: {0}\n".format(self.afeh)
        if self.modelset == 'mist':
            info = info + "[v/vcrit]: {0}\n".format(self.afeh)
        info = info + "Isochrone file: {0}\n\n".format(self.isofullpath)
        info = info + self.imf.getinfo()
        return info


class StarCluster:
    """
    Base class for simulation of a star cluster.

    The cluster stars are assumed to be single (no binaries or multiples) and drawn from a single isochrone,
    implying the same age and chemical composition for all cluster members. The PARSEC or MIST isochrone sets can be
    used to generate the simulated stars.The focus is on simulating Gaia observations of the clusters.
    """

    def __init__(self, n_stars, staraps, starpos, starkin):
        """
        Class constructor/initializer

        Parameters
        ----------
        n_stars : int
            Number of stars in the cluster.
        staraps : agabpylib.simulation.starclusters.StarAPs
            Class that generates the astrophysical parameters for the cluster stars.
        starpos: agabpylib.simulation.starclusters.SpaceDistribution
            The class that will generate the space positions of the stars with respect to the cluster (bary)centre.
        starkin : agabpylib.simulation.starclusters.Kinematics
            The instance of the class that will generate the cluster kinematics.
        """
        self.n_stars = n_stars
        self.staraps = staraps
        self.starpos = starpos
        self.starkin = starkin
        self.star_table = self.staraps.generate_aps(self.n_stars)
        x, y, z = starpos.generate_positions(self.n_stars)
        self.star_table.add_columns([x, y, z], names=['x', 'y', 'z'])
        vx, vy, vz = starkin.generate_kinematics(x, y, z)
        self.star_table.add_columns([vx, vy, vz], names=['v_x', 'v_y', 'v_z'])

        self.star_table.meta = {}
        self.star_table.meta.update(
            {'timestamp': datetime.now().strftime('%Y-%m-%d-%H:%M:%S'), 'n_stars': self.n_stars})
        self.star_table.meta.update(staraps.getmeta())
        self.star_table.meta.update(starpos.getmeta())
        self.star_table.meta.update(starkin.getmeta())

    def getinfo(self):
        """
        Returns
        -------
        str:
            Information on the simulated cluster.
        """
        return "Simulated cluster parameters\n" + \
               "============================\n" + \
               "Number of stars: {0}\n\n".format(self.n_stars) + \
               self.staraps.getinfo() + "\n\n" + \
               self.starpos.getinfo() + "\n\n" + \
               self.starkin.getinfo()

    def getmeta(self):
        """
        Returns
        -------
        dict :
            Metadata on the simulated star cluster.
        """
        return self.star_table.meta

    def write_star_table(self, filename, ffmt, **kwargs):
        """
        Write the star table to file in the requested format.

        Parameters
        ----------
        filename : string
            Name of the file to write to.
        ffmt : str
            File format. See astropy.Table.write.
        **kwargs : optional
            Further keyword arguments for astropy.table.Table.write()
        """
        self.star_table.write(filename, format=ffmt, **kwargs)

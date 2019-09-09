"""
Generate the Gaia observables for a given simulated cluster.

Anthony Brown Sep 2019 - Sep 2019
"""

import numpy as np
from scipy.stats import norm, multivariate_normal
from abc import ABC, abstractmethod
import astropy.units as u
from pygaia.astrometry.vectorastrometry import phaseSpaceToAstrometry, sphericalToCartesian
from pygaia.errors.astrometric import parallaxErrorSkyAvg, positionErrorSkyAvg, properMotionErrorSkyAvg, \
    errorScalingMissionLength
from pygaia.errors.photometric import gMagnitudeErrorEoM, bpMagnitudeErrorEoM, rpMagnitudeErrorEoM
from pygaia.errors.spectroscopic import vradErrorSkyAvg, _vradCalibrationFloor


class Observables(ABC):
    """
    Abstract base class for classes representing the observations made of stars in a smulated cluster.
    """

    @abstractmethod
    def generate_observations(self, cluster, distance_c, ra_c, dec_c):
        """
        Generate simulated observations of the star cluster.

        Parameters
        ----------
        cluster : astropy.table.QTable
            Table with the simulated astrophysical properties of the cluster stars.
        distance_c : astropy.units.Quantity
            Distance to cluster centre (of mass) in pc.
        ra_c : astropy.units.Quantity
            Right ascension of cluster centre (of mass) in degrees.
        dec_c : astropy.units.Quantity
            Declination of cluster centre (of mass) in degrees.

        Returns
        -------

        Nothing: the simulated observations of the cluster are appended to the input table.
        """
        pass

    def getinfo(self):
        """
        Returns
        -------
        str:
            String with information about the simulated observations.
        """
        return "Simulated observations\n" + \
               "----------------------\n" + self.addinfo()

    @abstractmethod
    def addinfo(self):
        """
        Returns
        -------
        str:
            String with specific information about the simulated observations.
        """
        pass

    @abstractmethod
    def getmeta(self):
        """
        Returns
        -------
        dict :
            Metadata on the simulated observations.
        """
        pass


class GaiaSurvey(Observables):
    """
    Gaia observations.

    Attributes
    ----------
    observation_interval : int
        Number of months of data collected by Gaia.
    cluster_distance : astropy.units.Quantity
        Distance to cluster centre (of mass) in pc
    cluster_ra : astropy.units.Quantity
        Right ascension of cluster centre (of mass) in degrees.
    cluster_dec : astropy.units.Quantity
        Declination of cluster centre (of mass) in degrees.
    """

    def __init__(self, o, distance_c, ra_c, dec_c, ):
        """
        Class constructor/initializer.

        Parameters
        ----------
        o : int
             Number of months of data collected by Gaia.
        distance_c : astropy.units.Quantity
            Distance to cluster centre (of mass) in pc
        ra_c : astropy.units.Quantity
            Right ascension of cluster centre (of mass) in degrees.
        dec_c : astropy.units.Quantity
            Declination of cluster centre (of mass) in degrees.
        """
        self.observation_interval = o
        self.cluster_distance = distance_c
        self.cluster_ra = ra_c
        self.cluster_dec = dec_c
        self.bright_faint_sep = 10.87

    def generate_observations(self, cluster):
        x_c, y_c, z_c = sphericalToCartesian(self.cluster_distance, self.cluster_ra.to(u.deg),
                                             self.cluster_dec.to(u.deg))
        star_dist = np.sqrt((cluster['x'] - x_c) ** 2 + (cluster['y'] - y_c) ** 2 + (cluster['z'] - z_c))
        dmod = 5 * np.log10(star_dist) - 5
        gmag = cluster['Gabs'] + dmod
        vmag = cluster['Vabs'] + dmod
        bpmag = np.where(gmag < self.bright_faint_sep, cluster['Gabs_BPb'], cluster['Gabs_BPf']) + dmod
        rpmag = cluster['Gabs_RP'] + dmod
        vmini = cluster['Vabs'] - cluster['Iabs']
        ra, dec, plx, pmra, pmdec, vrad = phaseSpaceToAstrometry(cluster['x'], cluster['y'], cluster['z'],
                                                                 cluster['v_x'], cluster['v_y'], cluster['v_z'])

        mission_extension = (self.observation_interval - 60.0) / 12.0
        nobs_scaled = np.round(self.observation_interval / 60.0 * 70)
        ra_error, dec_error = positionErrorSkyAvg(gmag, vmini, extension=mission_extension)
        plx_error = parallaxErrorSkyAvg(gmag, vmini, extension=mission_extension)
        pmra_error, pmdec_error = properMotionErrorSkyAvg(gmag, vmini, extension=mission_extension)
        # TODO more sophisticated radial velocity uncertainty modelling
        errscaling = errorScalingMissionLength(mission_extension, -0.5)
        vrad_error = vradErrorSkyAvg(vmag, 'F0V')
        # TODO Fix PyGaia to allow for shorter or longer mission lifetimes for vrad errors
        vrad_error = (vrad_error - _vradCalibrationFloor) * errscaling + _vradCalibrationFloor

        # TODO Fix PyGaia to allow for shorter or longer mission lifetimes for photometric errors.
        gmag_error = gMagnitudeErrorEoM(gmag, nobs=nobs_scaled)
        bpmag_error = bpMagnitudeErrorEoM(gmag, vmini, nobs=nobs_scaled)
        rpmag_error = rpMagnitudeErrorEoM(gmag, vmini, nobs=nobs_scaled)

        # convert astrometric uncertainties to milliarcsec(/yr)
        ra_error = ra_error / 1000.0
        dec_error = dec_error / 1000.0
        plx_error = plx_error / 1000.0
        pmra_error = pmra_error / 1000.0
        pmdec_error = pmdec_error / 1000.0

        gmag_obs = norm.rvs(loc=gmag, scale=gmag_error)
        bpmag_obs = norm.rvs(loc=bpmag, scale=bpmag_error)
        rpmag_obs = norm.rvs(loc=rpmag, scale=rpmag_error)
        vrad_obs = norm.rvs(loc=vrad, scale=vrad_error)

        mastorad = (1 * u.mas).to(u.rad).value
        ra_obs = norm.rvs(loc=ra, scale=ra_error * mastorad)
        dec_obs = norm.rvs(loc=dec, scale=dec_error * mastorad)
        plx_obs = norm.rvs(loc=plx, scale=plx_error)
        pmra_obs = norm.rvs(loc=pmra, scale=pmra_error)
        pmdec_obs = norm.rvs(loc=pmdec, scale=pmdec_error)

        cluster.add_columns([gmag, bpmag, rpmag, (ra * u.rad).to(u.deg), (dec * u.rad).to(u.deg), plx * u.mas,
                             pmra * u.mas / u.yr, pmdec * u.mas / u.yr, vrad * u.km / u.s],
                            names=['G', 'GBP', 'GRP', 'ra', 'dec', 'parallax', 'pmra', 'pmdec', 'radial_velocity'])
        cluster.add_columns([gmag_obs, gmag_error, bpmag_obs, bpmag_error, rpmag_obs, rpmag_error],
                            names=['G_obs', 'G_error', 'GBP_obs', 'GBP_error', 'GRP_obs', 'GRP_error'])
        cluster.add_columns([ra_obs, ra_error, dec_obs, dec_error, plx_obs, plx_error, pmra_obs, pmra_error,
                             pmdec_obs, pmdec_error, vrad_obs, vrad_error],
                            names=['ra_obs', 'ra_error', 'dec_obs', 'dec_error', 'parallax_obs', 'parallax_error',
                                   'pmra_obs', 'pmra_error', 'pmdec_obs', 'pmdec_error', 'radial_velocity_obs',
                                   'radial_velocity_error'])

    def addinfo(self):
        return "Gaia data for {0} months of data collection\n" + " Cluster distance: {1} pc\n" + \
               " Cluster position: ({2}, {3})".format(self.observation_interval, self.cluster_distance,
                                                      self.cluster_ra, self.cluster_dec)

    def getmeta(self):
        return {'simulated_survey': 'Gaia', 'data_collection_interval': self.observation_interval,
                'cluster_distance': self.cluster_distance, 'cluster_ra': self.cluster_ra,
                'cluster_dec': self.cluster_dec}

"""
Generate the Gaia observables for a given simulated cluster.

Anthony Brown Sep 2019 - Aug 2022
"""

from abc import ABC, abstractmethod

import astropy.units as u
import numpy as np
from pygaia.astrometry.vectorastrometry import (
    phase_space_to_astrometry,
    spherical_to_cartesian,
)
from pygaia.errors.astrometric import (
    parallax_uncertainty,
    position_uncertainty,
    proper_motion_uncertainty,
)
from pygaia.errors.photometric import (
    g_magnitude_uncertainty_eom,
    bp_magnitude_uncertainty_eom,
    rp_magnitude_uncertainty_eom,
)
from pygaia.errors.spectroscopic import vrad_error_sky_avg

__all__ = ["Observables", "GaiaSurvey"]


class Observables(ABC):
    """
    Abstract base class for classes representing the observations made of stars in a simulated cluster.
    """

    @abstractmethod
    def generate_observations(self, cluster):
        """
        Generate simulated observations of the star cluster.

        Parameters
        ----------
        cluster : astropy.table.QTable
            Table with the simulated astrophysical properties of the cluster stars.

        Returns
        -------
        Nothing: the simulated observations of the cluster are appended to the input table.
        """
        pass

    def getinfo(self):
        """
        Returns
        -------
        infoi : str
            String with information about the simulated observations.
        """
        return "Simulated observations\n" + "----------------------\n" + self.addinfo()

    @abstractmethod
    def addinfo(self):
        """
        Returns
        -------
        info : str
            String with specific information about the simulated observations.
        """
        pass

    @abstractmethod
    def getmeta(self):
        """
        Returns
        -------
        meta : dict
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
    rvslim : float
        RVS survey limit (default 16.0)
    """

    def __init__(self, release, distance_c, ra_c, dec_c, rvslim=16.0):
        """
        Class constructor/initializer.

        Parameters
        ----------
        release : str
            Specify the Gaia data release for which the performance is to be simulated. 'dr3' -> Gaia (E)DR3,
            'dr4' -> Gaia DR4, 'dr5' -> Gaia DR5.
        distance_c : astropy.units.Quantity
            Distance to cluster centre (of mass) in pc
        ra_c : astropy.units.Quantity
            Right ascension of cluster centre (of mass) in degrees.
        dec_c : astropy.units.Quantity
            Declination of cluster centre (of mass) in degrees.
        rvslim : float
            RVS survey limit (default 16.0)
        """
        self.gaia_release = release
        self.cluster_distance = distance_c
        self.cluster_ra = ra_c
        self.cluster_dec = dec_c
        self.rvslim = rvslim
        self.survey_limit = 20.7
        self.bright_faint_sep = 10.87
        self.n_rvs = 0
        self.n_astrophoto = 0
        self.n_plxpos = 0
        if release == "dr3":
            self.mission_extension = -26.0 / 12.0
        elif release == "dr5":
            self.mission_extension = 5.0
        else:
            self.mission_extension = 0.0
        self.rng = np.random.default_rng()

    def generate_observations(self, cluster):
        x_c, y_c, z_c = spherical_to_cartesian(
            self.cluster_distance, self.cluster_ra.to(u.rad), self.cluster_dec.to(u.rad)
        )
        star_dist = np.sqrt(
            (cluster["x"] + x_c) ** 2
            + (cluster["y"] + y_c) ** 2
            + (cluster["z"] + z_c) ** 2
        )
        dmod = 5 * np.log10(star_dist.value) - 5
        gmag = cluster["Gabs"] + dmod
        vmag = cluster["Vabs"] + dmod
        imag = cluster["Iabs"] + dmod
        bpmag = (
            np.where(
                gmag < self.bright_faint_sep, cluster["Gabs_BPb"], cluster["Gabs_BPf"]
            )
            + dmod
        )
        rpmag = cluster["Gabs_RP"] + dmod
        gminv = gmag - vmag
        vmini = vmag - imag
        grvs = gmag - (
            -0.0138 + 1.1168 * vmini - 0.1811 * vmini**2 + 0.0085 * vmini**3
        )
        ra, dec, plx, pmra, pmdec, vrad = phase_space_to_astrometry(
            (cluster["x"] + x_c).value,
            (cluster["y"] + y_c).value,
            (cluster["z"] + z_c).value,
            cluster["v_x"].value,
            cluster["v_y"].value,
            cluster["v_z"].value,
        )

        ra_error, dec_error = position_uncertainty(gmag, release=self.gaia_release)
        plx_error = parallax_uncertainty(gmag)
        pmra_error, pmdec_error = proper_motion_uncertainty(gmag)

        teff = 10 ** cluster["log_Teff"]
        logg = cluster["log_g"]
        ms = (3.5 <= logg) & (logg <= 6.0)
        condlist = [
            logg < 3.5,
            logg > 6.0,
            ms & (teff >= 30000),
            ms & ((teff < 30000) & (teff >= 15000)),
            ms & ((teff < 15000) & (teff >= 10000)),
            ms & ((teff < 10000) & (teff >= 8000)),
            ms & ((teff < 8000) & (teff >= 7000)),
            ms & ((teff < 7000) & (teff >= 6000)),
            ms & ((teff < 6000) & (teff >= 5500)),
            ms & ((teff < 5500) & (teff >= 5000)),
            ms & (teff < 5000),
        ]
        choicelist = [
            "K1III",
            "B0V",
            "B0V",
            "B5V",
            "A0V",
            "A5V",
            "F0V",
            "G0V",
            "G5V",
            "K0V",
            "K4V",
        ]
        spt = np.select(condlist, choicelist)
        vrad_error = vrad_error_sky_avg(vmag, spt, extension=self.mission_extension)

        gmag_error = g_magnitude_uncertainty_eom(gmag, extension=self.mission_extension)
        bpmag_error = bp_magnitude_uncertainty_eom(
            gmag, vmini, extension=self.mission_extension
        )
        rpmag_error = rp_magnitude_uncertainty_eom(
            gmag, vmini, extension=self.mission_extension
        )

        # convert astrometric uncertainties to milliarcsec(/yr)
        ra_error = ra_error / 1000.0
        dec_error = dec_error / 1000.0
        plx_error = plx_error / 1000.0
        pmra_error = pmra_error / 1000.0
        pmdec_error = pmdec_error / 1000.0

        gmag_obs = self.rng.normal(loc=gmag, scale=gmag_error)
        bpmag_obs = self.rng.normal(loc=bpmag, scale=bpmag_error)
        rpmag_obs = self.rng.normal(loc=rpmag, scale=rpmag_error)
        vrad_obs = self.rng.normal(loc=vrad, scale=vrad_error)

        mastorad = (1 * u.mas).to(u.rad).value
        ra_obs = self.rng.normal(loc=ra, scale=ra_error * mastorad)
        dec_obs = self.rng.normal(loc=dec, scale=dec_error * mastorad)
        plx_obs = self.rng.normal(loc=plx, scale=plx_error)
        pmra_obs = self.rng.normal(loc=pmra, scale=pmra_error)
        pmdec_obs = self.rng.normal(loc=pmdec, scale=pmdec_error)

        cluster.add_columns(
            [
                gmag * u.dimensionless_unscaled,
                bpmag * u.dimensionless_unscaled,
                rpmag * u.dimensionless_unscaled,
                grvs * u.dimensionless_unscaled,
                gminv * u.dimensionless_unscaled,
                vmag * u.dimensionless_unscaled,
                imag * u.dimensionless_unscaled,
                vmini * u.dimensionless_unscaled,
            ],
            names=["G", "GBP", "GRP", "GRVS", "GminV", "V", "I", "VminI"],
        )
        cluster.add_columns(
            [
                (ra * u.rad).to(u.deg),
                (dec * u.rad).to(u.deg),
                plx * u.mas,
                pmra * u.mas / u.yr,
                pmdec * u.mas / u.yr,
                vrad * u.km / u.s,
            ],
            names=["ra", "dec", "parallax", "pmra", "pmdec", "radial_velocity"],
        )

        phot_observables = [
            "G_obs",
            "G_obs_error",
            "GBP_obs",
            "GBP_obs_error",
            "GRP_obs",
            "GRP_obs_error",
        ]
        cluster.add_columns(
            [
                gmag_obs * u.dimensionless_unscaled,
                gmag_error * u.dimensionless_unscaled,
                bpmag_obs * u.dimensionless_unscaled,
                bpmag_error * u.dimensionless_unscaled,
                rpmag_obs * u.dimensionless_unscaled,
                rpmag_error * u.dimensionless_unscaled,
            ],
            names=phot_observables,
        )
        astrom_observables = [
            "ra_obs",
            "ra_error",
            "dec_obs",
            "dec_error",
            "parallax_obs",
            "parallax_error",
            "pmra_obs",
            "pmra_error",
            "pmdec_obs",
            "pmdec_error",
        ]
        rvs_observables = ["radial_velocity_obs", "radial_velocity_error"]
        cluster.add_columns(
            [
                (ra_obs * u.rad).to(u.deg),
                ra_error * u.mas,
                (dec_obs * u.rad).to(u.deg),
                dec_error * u.mas,
                plx_obs * u.mas,
                plx_error * u.mas,
                pmra_obs * u.mas / u.yr,
                pmra_error * u.mas / u.yr,
                pmdec_obs * u.mas / u.yr,
                pmdec_error * u.mas / u.yr,
                vrad_obs * u.km / u.s,
                vrad_error * u.km / u.s,
            ],
            names=astrom_observables + rvs_observables,
        )

        # Apply survey limits to the observables.
        for field in astrom_observables + phot_observables:
            cluster[field][gmag_obs > self.survey_limit] = np.NAN
        for field in rvs_observables:
            cluster[field][(grvs > self.rvslim)] = np.NAN
        self.n_astrophoto = plx_obs[gmag_obs <= self.survey_limit].size
        self.n_rvs = vrad_obs[grvs <= self.rvslim].size
        self.n_plxpos = plx_obs[(gmag_obs <= self.survey_limit) & (plx_obs > 0)].size

    def addinfo(self):
        return (
            "Gaia data release {0}\n"
            + " Cluster distance: {1}\n"
            + " Cluster position: ({2}, {3})\n"
            + " RVS Survey limit: Grvs={4}\n"
            + " Number of stars with astrometry and photometry: {5}\n"
            + " Number of stars with positive observed parallax: {6}\n"
            + " Number of stars with radial velocity: {7}\n"
        ).format(
            self.gaia_release,
            self.cluster_distance,
            self.cluster_ra,
            self.cluster_dec,
            self.rvslim,
            self.n_astrophoto,
            self.n_plxpos,
            self.n_rvs,
        )

    def getmeta(self):
        return {
            "simulated_survey": "Gaia",
            "data_release": self.gaia_release,
            "cluster_distance": self.cluster_distance,
            "cluster_ra": self.cluster_ra,
            "cluster_dec": self.cluster_dec,
            "rvs_survey_limit": self.rvslim,
            "num_astrophoto": self.n_astrophoto,
        }

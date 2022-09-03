"""
Classes and Functions for handling the tables listing the normalization factor u0 needed to calculate the
renormalized unit weight error (RUWE). The tables are provided on the Gaia DR2 Known Issues pages
(https://www.cosmos.esa.int/web/gaia/dr2-known-issues).

.. note::
    This module is obsolete as the RUWE values are available as a separate table in the Gaia DR2
    archive and as a field in ``gaia_source`` since Gaia EDR3.

Anthony Brown Oct 2018 - Aug 2022
"""

import numpy as np
import os
import scipy.interpolate as spint

_ROOT = os.path.abspath(os.path.dirname(__file__))

__all__ = ["U0Interpolator"]


def _get_data(path_to_file):
    """
    Obtain the path to a file located in 'data' or a subfolder thereof. Intended for package internal use only.

    Parameters
    ----------
    path_to_file : str
         Name of file or of path to the file.

    Returns
    -------
    full_path_to_file : str
        The full path to the input file.
    """
    return os.path.join(_ROOT, "data", path_to_file)


class U0Interpolator:
    """
    Class which holds functions that can used to calculate u0 values given the G-band magnitude only, or
    the G-band magnitude and the BP-RP colour of a source. The class initialization takes care of reading
    the necessary data and setting up the interpolators.
    """

    def __init__(self):
        """
        Initialize the class.
        """

        ngmagbins = 1741
        ncolorbins = 111
        self.ming = 3.6
        self.maxg = 21.0
        self.minbprp = -1.0
        self.maxbprp = 10.0

        u0data = np.genfromtxt(
            _get_data("table_u0_g_col.txt"),
            names=["g_mag", "bp_rp", "u0"],
            skip_header=1,
            delimiter=",",
        )
        gmagmesh = np.reshape(u0data["g_mag"], (ngmagbins, ncolorbins))
        bprpmesh = np.reshape(u0data["bp_rp"], (ngmagbins, ncolorbins))
        u0mesh = np.reshape(u0data["u0"], (ngmagbins, ncolorbins))

        gmag = gmagmesh[:, 0]
        bprp = bprpmesh[0, :]

        self.gbprpinterpolator = spint.RectBivariateSpline(
            gmag, bprp, u0mesh, kx=1, ky=1, s=0
        )

        u0data = np.genfromtxt(
            _get_data("table_u0_g.txt"),
            names=["g_mag", "u0"],
            skip_header=1,
            delimiter=",",
        )
        self.ginterpolator = spint.interp1d(
            u0data["g_mag"], u0data["u0"], bounds_error=True
        )

    def get_u0_g_col(self, gmag, bprp, asgrid=False):
        """
        Calculate the u0 value for the input G-band magnitude(s) and BP-RP colour(s).

        Parameters
        ----------
        gmag : float array
            Values of G (size n)
        bprp : float array
            Values of BP-RP (size m)
        asgrid : boolean
            If True treat the input gmag and bprp arrays as the coordinates for a 2D grid. If set to True
            the sizes of the input arrays n and m may be different. If set to False n=m is required.

        Returns
        -------
        u0 : array-like
            The values of u0 evaluated at the input (G, BP-RP) combinations. The returned list consists
            of n values for ``asgrid=False`` and n*m otherwise.

        Raises
        ------
        ValueError
            For input values outside the interpolation range.

        """
        return self.gbprpinterpolator(gmag, bprp, grid=asgrid)

    def get_u0_g(self, gmag):
        """
        Calculate the normalization factor u0 for the RUWE for the input G-band magnitude(s).

        Parameters
        ----------
        gmag : float array
            Values of G

        Returns
        -------
        u0 : float array
            Array of u0 values evaluated at the input G-band magnitudes.

        Raises
        ------
        ValueError
            For input values outside the interpolation range.
        """
        return self.ginterpolator(gmag)

    def get_u0(self, gmag, bprp):
        """
        Calculate the RUWE normalization factor u0 for the input G-band magnitude(s) and BP-RP colour(s).

        .. attention::
            This is a version of `get_u0_g_col` that is robust against missing (BP-RP) values and against magnitudes
            and colours outside the interpolation range. Input values outside the interpolation range will be clamped
            to the corresponding edge of that range.

        Parameters
        ----------
        gmag : float array
            Values of G (size n)
        bprp : float array
            Values of BP-RP (size m)

        Returns
        -------
        u0 : float array
            The values of u0 evaluated at the input (G, BP-RP) combinations.
        """

        if np.isscalar(gmag):
            if np.isnan(bprp):
                return self.get_u0_g(gmag)
            else:
                return self.get_u0_g_col(gmag, bprp)
        else:
            u0 = np.zeros(gmag.size)
            g = np.clip(gmag, self.ming, self.maxg)
            col = np.clip(bprp, self.minbprp, self.maxbprp)

            indnocol = np.isnan(col)
            indcol = np.logical_not(indnocol)

            u0[indcol] = self.get_u0_g_col(g[indcol], col[indcol], asgrid=False)
            u0[indnocol] = self.get_u0_g(g[indnocol])

            return u0

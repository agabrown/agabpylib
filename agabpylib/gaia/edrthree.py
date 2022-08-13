"""
Functions for the use and analysis of Gaia EDR3 data.

Anthony Brown Dec 2020 - Dec 2020
"""
import numpy as np


def correct_gband(bp_rp, astrometric_params_solved, phot_g_mean_mag, phot_g_mean_flux):
    """
    Correct the G-band fluxes and magnitudes for the input list of Gaia EDR3 data.

    Parameters
    ----------

    bp_rp: float, numpy.ndarray
        The (BP-RP) colour listed in the Gaia EDR3 archive.
    astrometric_params_solved: int, numpy.ndarray
        The astrometric solution type listed in the Gaia EDR3 archive.
    phot_g_mean_mag: float, numpy.ndarray
        The G-band magnitude as listed in the Gaia EDR3 archive.
    phot_g_mean_flux: float, numpy.ndarray
        The G-band flux as listed in the Gaia EDR3 archive.

    Returns
    -------

    The corrected G-band magnitudes and fluxes. The corrections are only applied to
    sources with a 2-parameter or 6-parameter astrometric solution fainter than G=13, for which a
    (BP-RP) colour is available.

    Example
    -------

    gmag_corr, gflux_corr = correct_gband(bp_rp, astrometric_params_solved, phot_g_mean_mag, phot_g_mean_flux)
    """

    if (
        np.isscalar(bp_rp)
        or np.isscalar(astrometric_params_solved)
        or np.isscalar(phot_g_mean_mag)
        or np.isscalar(phot_g_mean_flux)
    ):
        bp_rp = np.float64(bp_rp)
        astrometric_params_solved = np.int64(astrometric_params_solved)
        phot_g_mean_mag = np.float64(phot_g_mean_mag)
        phot_g_mean_flux = np.float64(phot_g_mean_flux)

    if not (
        bp_rp.shape
        == astrometric_params_solved.shape
        == phot_g_mean_mag.shape
        == phot_g_mean_flux.shape
    ):
        raise ValueError("Function parameters must be of the same shape!")

    do_not_correct = (
        np.isnan(bp_rp) | (phot_g_mean_mag <= 13) | (astrometric_params_solved == 31)
    )
    bright_correct = (
        np.logical_not(do_not_correct)
        & (phot_g_mean_mag > 13)
        & (phot_g_mean_mag <= 16)
    )
    faint_correct = np.logical_not(do_not_correct) & (phot_g_mean_mag > 16)
    bp_rp_c = np.clip(bp_rp, 0.25, 3.0)

    correction_factor = np.ones_like(phot_g_mean_mag)
    correction_factor[faint_correct] = (
        1.00525
        - 0.02323 * bp_rp_c[faint_correct]
        + 0.01740 * np.power(bp_rp_c[faint_correct], 2)
        - 0.00253 * np.power(bp_rp_c[faint_correct], 3)
    )
    correction_factor[bright_correct] = (
        1.00876
        - 0.02540 * bp_rp_c[bright_correct]
        + 0.01747 * np.power(bp_rp_c[bright_correct], 2)
        - 0.00277 * np.power(bp_rp_c[bright_correct], 3)
    )

    gmag_corrected = phot_g_mean_mag - 2.5 * np.log10(correction_factor)
    gflux_corrected = phot_g_mean_flux * correction_factor

    return gmag_corrected, gflux_corrected


def correct_flux_excess_factor(bp_rp, phot_bp_rp_excess_factor):
    """
    Calculate the corrected flux excess factor for the input Gaia EDR3 data.

    Parameters
    ----------

    bp_rp: float, numpy.ndarray
        The (BP-RP) colour listed in the Gaia EDR3 archive.
    phot_bp_rp_excess_factor: float, numpy.ndarray
        The flux excess factor listed in the Gaia EDR3 archive.

    Returns
    -------

    The corrected value for the flux excess factor, which is zero for "normal" stars.

    Example
    -------

    phot_bp_rp_excess_factor_corr = correct_flux_excess_factor(bp_rp, phot_bp_rp_flux_excess_factor)
    """

    if np.isscalar(bp_rp) or np.isscalar(phot_bp_rp_excess_factor):
        bp_rp = np.float64(bp_rp)
        phot_bp_rp_excess_factor = np.float64(phot_bp_rp_excess_factor)

    if bp_rp.shape != phot_bp_rp_excess_factor.shape:
        raise ValueError("Function parameters must be of the same shape!")

    do_not_correct = np.isnan(bp_rp)
    bluerange = np.logical_not(do_not_correct) & (bp_rp < 0.5)
    greenrange = np.logical_not(do_not_correct) & (bp_rp >= 0.5) & (bp_rp < 4.0)
    redrange = np.logical_not(do_not_correct) & (bp_rp > 4.0)

    correction = np.zeros_like(bp_rp)
    correction[bluerange] = (
        1.154360
        + 0.033772 * bp_rp[bluerange]
        + 0.032277 * np.power(bp_rp[bluerange], 2)
    )
    correction[greenrange] = (
        1.162004
        + 0.011464 * bp_rp[greenrange]
        + 0.049255 * np.power(bp_rp[greenrange], 2)
        - 0.005879 * np.power(bp_rp[greenrange], 3)
    )
    correction[redrange] = 1.057572 + 0.140537 * bp_rp[redrange]

    return phot_bp_rp_excess_factor - correction

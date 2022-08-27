"""
Provides useful functions for basic celestial mechanics applications.

Anthony Brown May 2020 - Aug 2022
"""

import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation as R
import astropy.constants as c
import astropy.units as u

__all__ = ["kepler_equation_solver", "orbital_elements_to_xyz"]

_base_period = (2 * np.pi / np.sqrt(c.G * c.M_sun) * c.au**1.5).to(u.yr).value


def _kepler_equation(EE, ee, MM):
    return EE - ee * np.sin(EE) - MM


def kepler_equation_solver(e, M):
    """
    Solve the Kepler equation for the input values of eccentricity and mean anomaly.

    Use a root-finding method. The input can consist of two scalar values, a combination
    of a scalar and a 1D array, or two equally sized 1D arrays.

    Parameters
    ----------
    e : float scalar or 1D array
        Value(s) of the eccentricity
    M : float scalar or 1D array
        Value(s) of the mean anomaly

    Returns
    -------
    E : float scalar or 1D array
        Array of eccentric anomaly values.
    """
    if np.ndim(e) > 1 or np.ndim(M) > 1:
        raise RuntimeError("Only scalars or 1D arrays allowed as input.")

    if np.ndim(e) == 0 and np.ndim(M) == 0:
        if e == 0 or np.mod(M, np.pi) == 0:
            return np.mod(M, 2 * np.pi)
        else:
            return sp.optimize.toms748(
                _kepler_equation, 0, 2 * np.pi, args=(e, np.mod(M, 2 * np.pi))
            )

    if np.ndim(e) == 0 and np.ndim(M) == 1:
        ecc = np.repeat(e, M.size)
        meanano = M
    elif np.ndim(e) == 1 and np.ndim(M) == 0:
        meanano = np.repeat(M, e.size)
        ecc = e
    else:
        if e.size != M.size:
            raise RuntimeError("The two input arrays must be of the same size.")
        ecc = e
        meanano = M

    E = []
    for el, Ml in zip(ecc, meanano):
        if el == 0 or np.mod(Ml, np.pi) == 0:
            E.append(np.mod(Ml, 2 * np.pi))
        else:
            E.append(
                sp.optimize.toms748(
                    _kepler_equation, 0, 2 * np.pi, args=(el, np.mod(Ml, 2 * np.pi))
                )
            )

    return np.squeeze(np.asarray(E))


def orbital_elements_to_xyz(a, e, incl, lon_nodes, argperi, tau, t):
    """
    For the given Keplerian orbital elements and the epoch t, calculate the position in (x,y,z) in the BCRS.

    The inputs can consist of all scalars or a mix of scalars an 1D arrays.

    .. note::
        For now orbits with e<1 are assumed. The orbit is treated as that of a test-particle and thus
        the period is given in multiples of 2*PI/(GM_sun)*(au)^1.5.

    Parameters
    ----------
    a : float scalar or 1D-array
        Semi-major axis in au
    e : float scalar or 1D-array
        Eccentricity
    incl : float scalar or 1D-array
        inclination in degrees (0-180)
    lon_nodes : float scalar or 1D-array
        Longitude of the line of nodes in degrees (0-360)
    argperi : float scalar or 1D-array
        Argument of perihelion in degrees (0-360)
    tau : float scalar or 1D-array
        Epoch of perihelium passage in yr
    t : float scalar or 1D-array
        Epoch at which to calculate (x,y,z) in yr

    Returns
    -------
    x, y, z : float array of shape (N,3)
        Array of values (x,y,z) of shape (N,3)
    """
    if (
        np.ndim(a) > 1
        or np.ndim(e) > 1
        or np.ndim(incl) > 1
        or np.ndim(lon_nodes) > 1
        or np.ndim(argperi) > 1
        or np.ndim(tau) > 1
        or np.ndim(t) > 1
    ):
        raise RuntimeError("Only scalars or 1D arrays allowed as input.")

    x = np.array([1, 0, 0])
    z = np.array([0, 0, 1])

    if (
        np.ndim(a)
        + np.ndim(e)
        + np.ndim(incl)
        + np.ndim(lon_nodes)
        + np.ndim(argperi)
        + np.ndim(tau)
        + np.ndim(t)
        < 1
    ):
        rx = R.from_rotvec(np.deg2rad(incl) * x)
        rz = R.from_rotvec(np.deg2rad(lon_nodes) * z)
        normal = rz.apply(rx.apply(z))
        rotation = R.from_rotvec(np.deg2rad(argperi) * normal) * rz * rx
        period = _base_period * a**1.5
        mean_anomaly = 2 * np.pi / period * (t - tau)
        ecc_anomaly = kepler_equation_solver(e, mean_anomaly)
        rvec_lmn = np.array(
            [
                a * (np.cos(ecc_anomaly) - e),
                a * np.sqrt(1 - e * e) * np.sin(ecc_anomaly),
                0,
            ]
        )
        return rotation.apply(rvec_lmn)

    numbodies = 0
    for arg in [a, e, incl, lon_nodes, argperi, tau, t]:
        if np.ndim(arg) == 1:
            numbodies = arg.size
            break

    if np.ndim(a) == 0:
        semi_major = np.repeat(a, numbodies)
    else:
        semi_major = a
    if np.ndim(e) == 0:
        eccentricity = np.repeat(e, numbodies)
    else:
        eccentricity = e
    if np.ndim(incl) == 0:
        inclination = np.repeat(np.deg2rad(incl), numbodies)
    else:
        inclination = np.deg2rad(incl)
    if np.ndim(lon_nodes) == 0:
        longitude_nodes = np.repeat(np.deg2rad(lon_nodes), numbodies)
    else:
        longitude_nodes = np.deg2rad(lon_nodes)
    if np.ndim(argperi) == 0:
        arg_perihelion = np.repeat(np.deg2rad(argperi), numbodies)
    else:
        arg_perihelion = np.deg2rad(argperi)
    if np.ndim(tau) == 0:
        tau_peri = np.repeat(tau, numbodies)
    else:
        tau_peri = tau
    if np.ndim(t) == 0:
        time = np.repeat(t, numbodies)
    else:
        time = t

    period = _base_period * semi_major**1.5
    mean_anomaly = 2 * np.pi / period * (time - tau_peri)
    rvec_xyz = np.empty((numbodies, 3))
    for index, aa, ee, ii, Omega, omega, mean_ano in zip(
        range(numbodies),
        semi_major,
        eccentricity,
        inclination,
        longitude_nodes,
        arg_perihelion,
        mean_anomaly,
    ):
        ecc_anomaly = kepler_equation_solver(ee, mean_ano)
        rx = R.from_rotvec(ii * x)
        rz = R.from_rotvec(Omega * z)
        normal = rz.apply(rx.apply(z))
        rotation = R.from_rotvec(omega * normal) * rz * rx
        rvec_lmn = np.array(
            [
                aa * (np.cos(ecc_anomaly) - ee),
                aa * np.sqrt(1 - ee * ee) * np.sin(ecc_anomaly),
                0,
            ]
        )
        rvec_xyz[index] = rotation.apply(rvec_lmn)

    return rvec_xyz

"""
Implement the evaluation of the generalized Fermi-Dirac integral according to the algorithm presented by
Aparicio (1998, https://ui.adsabs.harvard.edu/#abs/1998ApJS..117..627A/abstract). The code is based in
part on the fortran 90 implementation by F.X. Timmes
(http://cococubed.asu.edu/code_pages/fermi_dirac.shtml, as included in the MESA equation of state
libraries).

Anthony Brown Aug 2017 - Aug 2022
"""

import numpy as np
import scipy as sp

from agabpylib.numerical.integrate import fixed_quad_laguerre

__all__ = ["fd_evaluate"]

# Constants for defining the integration intervals (Table 2 in Aparicio's paper)
_D = 3.3609
_sigma = 9.1186e-2
_a1 = 6.7774
_b1 = 1.1418
_c1 = 2.9826
_a2 = 3.7601
_b2 = 9.3719e-2
_c2 = 2.1063e-2
_d2 = 3.1084e1
_e2 = 1.0056
_a3 = 7.5669
_b3 = 1.1695
_c3 = 7.5416e-1
_d3 = 6.6558
_e3 = -1.2819e-1


def _calculate_breakpoints(eta):
    """
    Calculate the integration interval breakpoints (equations 6 and 7 in Aparicio (1998)) and return
    their values.

    Parameters
    ----------
    eta : float
        Value of the parameter eta of the FD integral.

    Returns
    -------
    s1, s2, s3 : float
        The integration interval boundaries. Where the intervals are [-inf, s1], [s1,s2],
        [s2,s3], [s3,+inf]
    """
    zz = np.array(eta)
    smallexp = _sigma * (zz - _D) < 100.0
    largeexp = np.logical_not(smallexp)
    xi = np.zeros_like(zz)
    xi[smallexp] = np.log(1.0 + np.exp(_sigma * (zz[smallexp] - _D))) / _sigma
    xi[largeexp] = (_sigma * (zz[largeexp] - _D)) / _sigma

    xisqr = xi * xi
    xa = (_a1 + _b1 * xi + _c1 * xisqr) / (1 + _c1 * xi)
    xb = (_a2 + _b2 * xi + _c2 * _d2 * xisqr) / (1 + _e2 * xi + _c2 * xisqr)
    xc = (_a3 + _b3 * xi + _c3 * _d3 * xisqr) / (1 + _e3 * xi + _c3 * xisqr)

    s1 = xa - xb
    s2 = xa
    s3 = xa + xc

    return s1, s2, s3


def _fd_integrand_nearzero(z, nu, eta, theta):
    """
    Evaluate the Fermi-Dirac integrand after applying the x=z**2 transformation (see Apricio's paper,
    equation 5).

    Parameters
    ----------
    z : float or float array
        Abscissa(e) where the value of the integrand is desired.
    nu : float
        Parameter nu (index) of Fermi-Dirac integral.
    eta : float
        Parameter eta of Fermi-Dirac integral.
    theta : float
        Parameter theta of Fermi-Dirac integral.

    Returns
    -------
    val : float
        Value(s) of the integrand.
    """
    zz = np.array(z)
    zsqr = zz * zz
    smallexp = (zsqr - eta) < 100.0
    largeexp = np.logical_not(smallexp)
    result = np.zeros_like(zz)

    result[smallexp] = (
        2
        * np.power(zz[smallexp], 2 * nu + 1)
        * np.sqrt(1.0 + (zsqr[smallexp] * theta / 2.0))
    ) / (np.exp(zsqr[smallexp] - eta) + 1)
    result[largeexp] = (
        2
        * np.power(zz[largeexp], 2 * nu + 1)
        * np.sqrt(1.0 + (zsqr[largeexp] * theta / 2.0))
        * np.exp(eta - zsqr[largeexp])
    )

    return result


def _fd_integrand(x, nu, eta, theta):
    """
    Evaluate the Fermi-Dirac integrand.

    Parameters
    ----------
    x : float or float array
        Abscissa(e) where the value of the integrand is desired.
    nu : float
        Parameter nu (index) of Fermi-Dirac integral.
    eta : float
        Parameter eta of Fermi-Dirac integral.
    theta : float
        Parameter theta of Fermi-Dirac integral.

    Returns
    -------
    val : float or float array
        Value(s) of the integrand.
    """
    xx = np.array(x)
    smallexp = (xx - eta) < 100.0
    largeexp = np.logical_not(smallexp)
    result = np.zeros_like(xx)

    result[smallexp] = (
        np.power(xx[smallexp], nu) * np.sqrt(1.0 + (xx[smallexp] * theta / 2.0))
    ) / (np.exp(xx[smallexp] - eta) + 1)
    result[largeexp] = (
        np.power(xx[largeexp], nu)
        * np.sqrt(1.0 + (xx[largeexp] * theta / 2.0))
        * np.exp(eta - xx[largeexp])
    )

    return result


def fd_evaluate(nu, eta, theta):
    r"""
    Evaluate the generalized Fermi-Dirac integral.

    .. math::

        F_\nu(\eta,\theta)=\int_0^\infty \frac{x^\nu(1+\frac{1}{2}\theta x)^{1/2}}{e^{x-\eta}+1}\,dx

    Parameters
    ----------
    nu : float
        Index :math:`\nu` of Fermi-Dirac integral, must be a scalar.
    eta : float
        Value(s) of the parameter :math:`\eta`, can be an array.
    theta : float
        Value of the parameter :math:`\theta`, must be a scalar.

    Returns
    -------
    val : float
        Value of the Fermi-Dirac integral.
    """
    if not np.isscalar(nu) or not np.isscalar(theta):
        raise TypeError("fd_evaluate() is only vectorized for eta")

    order = 20
    s1, s2, s3 = _calculate_breakpoints(eta)

    if np.isscalar(eta):
        i1, dummy = sp.integrate.fixed_quad(
            _fd_integrand_nearzero, 0, np.sqrt(s1), args=(nu, eta, theta), n=order
        )
        i2, dummy = sp.integrate.fixed_quad(
            _fd_integrand, s1, s2, args=(nu, eta, theta), n=order
        )
        i3, dummy = sp.integrate.fixed_quad(
            _fd_integrand, s2, s3, args=(nu, eta, theta), n=order
        )
        i4, dummy = fixed_quad_laguerre(
            _fd_integrand, s3, args=(nu, eta, theta), n=order
        )
    else:
        i1 = np.zeros_like(eta)
        i2 = np.zeros_like(eta)
        i3 = np.zeros_like(eta)
        i4 = np.zeros_like(eta)
        for i in range(eta.size):
            i1[i], dummy = sp.integrate.fixed_quad(
                _fd_integrand_nearzero,
                0,
                np.sqrt(s1[i]),
                args=(nu, eta[i], theta),
                n=order,
            )
            i2[i], dummy = sp.integrate.fixed_quad(
                _fd_integrand, s1[i], s2[i], args=(nu, eta[i], theta), n=order
            )
            i3[i], dummy = sp.integrate.fixed_quad(
                _fd_integrand, s2[i], s3[i], args=(nu, eta[i], theta), n=order
            )
            i4[i], dummy = fixed_quad_laguerre(
                _fd_integrand, s3[i], args=(nu, eta[i], theta), n=order
            )

    return i1 + i2 + i3 + i4

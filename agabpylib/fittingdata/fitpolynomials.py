"""
Provides methods for the fitting of polynomials to (x,y) data, where both coordinates have (possibly
correlated) Gaussian errors. Based on the ideas in the paper "Data analysis recipes: Fitting a model to
data", by Hogg et al., arXiv:1008.4686v1.

Anthony Brown 2015-02-28
"""

from numpy import pi, log, isreal, atleast_2d
from numpy.polynomial.polynomial import Polynomial, polyder, polyroots, polyval

def lnP2DGaussian(zObs, z, S):
    """
    Calculate the ln(probability) (or ln(Likelihood)) for the 'observed' point zObs=(xObs,yObs), given
    the 'true' point z=(x,y) and the covariance matrix S of the Gaussian observational errors.

    Parameters
    ----------

    zObs - (x,y)_observed: passed in a form convenient for calculation (numpy array, tuple of numpy arrays)
    z    - (x,y): passed in a form convenient for calculation (numpy array, tuple of numpy arrays)
    S    - Covariance matrix for (x,y)_observed as (sigma_x^2, sigma_y^2, sigma_xy): passed in a form
           convenient for calculation: shape (3) or (N,3)

    Returns
    -------

    The value of ln(likelihood) = -0.5*[(zObs-z)^T S^{-1} (zObs-z)] - 0.5*log(detS) - log(2.0*pi)
    """

    covmat = atleast_2d(S)
    detS = covmat[:,0]*covmat[:,1]-covmat[:,2]*covmat[:,2]
    delta_x = zObs[0]-z[0]
    delta_y = zObs[1]-z[1]
    innerzObsz = covmat[:,1]*delta_x**2 - 2*covmat[:,2]*delta_x*delta_y + covmat[:,0]*delta_y**2
    return -0.5*innerzObsz/detS - 0.5*log(detS) - log(2.0*pi)

def lnL_polynomial(zObs, p, S):
    """
    Given a polynomial y=p(x) and an observed point zObs=(x,y) construct the ln(likelihood) polynomial
    given by: lnL(x) = -0.5*[(zObs-z(p))^T S^{-1} (zObs-z(p))], where z(p) = (x,p(x)).

    Parameters
    ----------

    zObs - 2-vector (x,y)_observed
    p    - Array with coefficients of polynomial y=p(x) ([1,2,3] means 1+2*x+3*x**2)
    S    - Covariance matrix for (x,y)_observed passed in as 3-vector (sigma_x^2, sigma_y^2, sigma_xy)

    Returns
    -------

    The array of ln(likelihood) polynomial coefficients.
    """
    detS = S[0]*S[1]-S[2]*S[2]
    poly_delta_x = Polynomial([zObs[0],-1])
    poly_delta_y = Polynomial([zObs[1]]) - Polynomial(p)
    poly_lnL = S[1]*poly_delta_x**2 - 2*S[2]*poly_delta_x*poly_delta_y +S[0]*poly_delta_y**2
    poly_lnL = -0.5*poly_lnL/detS
    return poly_lnL.coef

def maximize_lnL_polynomial(zObs, p, S):
    """
    Given a polynomial y=p(x) and an observed point zObs=(xObs,yObs), maximize the likelihood of the data
    given the model p(x). That is find the point (x,y) along the polynomial that maximizes the likelihood
    of zObs.

    Parameters
    ----------

    zObs - 2-vector or tuple (x,y)_observed
    p    - Array with coefficients of polynomial y=p(x) ([1,2,3] means 1+2*x+3*x**2)
    S    - Covariance matrix for (x,y)_observed passed in as 3-vector (sigma_x^2, sigma_y^2, sigma_xy)

    Returns
    -------

    Tuple (x, y, lnL) with the point (x,y) on the polynominal for which the likelihood of zObs is
    maximized, as well as the maximum likelihood value.
    """

    # Find coefficients of the polynominal describing the likelihood as a function of the model value
    # for x.
    coeffs_poly_lnL = lnL_polynomial(zObs, p, S)
    # Differentiate the polynomial, find the roots of the derivative, and look for the real root where
    # the likelihood is maximal.
    coeffs_der = polyder(coeffs_poly_lnL)
    roots = polyroots(coeffs_der)
    realroots = roots[isreal(roots)]
    x = realroots.real
    y = polyval(x,p)
    lnL = lnP2DGaussian(zObs, (x, y), S)
    maxindex = lnL.argmax()

    return (x[maxindex], y[maxindex], lnL[maxindex])

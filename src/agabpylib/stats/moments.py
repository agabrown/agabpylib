"""
Provides moments of data distributions beyond the standard examples such as mean, median, etc. For example,
weighted means of multidimensional data, accounting for (correlated) uncertainties, are implemented.

Anthony Brown Aug 2021 - Aug 2022
"""

import numpy as np
from numpy.linalg import inv

__all__ = ["weighted_mean_twod"]


def weighted_mean_twod(x, y, sx, sy, cxy):
    r"""
    Provide the weighted mean of the vectors x and y.

    The elements :math:`(x_i, y_i)` have covariance matrices:

    .. math::

        \begin{pmatrix}
        s_x^2 & c_{xy}s_xs_y \\
        c_{xy}s_xs_y & s_y^2
        \end{pmatrix}

    Parameters
    ----------
    x : float array (1D)
        Values of x-component of data vector
    y : float array (1D)
        Values of y-component of data vector
    sx : float array (1D)
        Uncertainties in x
    sy : float array (1D)
        Uncertainties in y
    cxy : float array (1D)
        Correlation coefficient of sx and sy

    Returns
    -------
    wx, wy : float
        Weighted means wx, wy.
    covw : float array (2,2)
        Covariance matrix of the weighted mean.
    """
    ndata = x.size
    cov = np.zeros((2 * ndata, 2 * ndata))
    mat_a = np.zeros((2 * ndata, 2))
    vec_b = np.zeros(2 * ndata)
    for j in range(0, 2 * ndata, 2):
        mat_a[j] = [1, 0]
        mat_a[j + 1] = [0, 1]
        k = int(j / 2)
        vec_b[j] = x[k]
        vec_b[j + 1] = y[k]
        cov[j, j] = sx[k] ** 2
        cov[j + 1, j + 1] = sy[k] ** 2
        cov[j + 1, j] = sx[k] * sy[k] * cxy[k]
        cov[j, j + 1] = cov[j + 1, j]
    cinv = inv(cov)
    covw = inv(np.dot(mat_a.T, np.dot(cinv, mat_a)))
    wx, wy = np.dot(covw, np.dot(mat_a.T, np.dot(cinv, vec_b)))
    return wx, wy, covw

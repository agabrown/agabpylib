#
# Anthony Brown June 2019 = June 2019
#

import numpy as np
from scipy.stats import rv_continuous
from scipy.special import iv

class hoyt_gen(rv_continuous):
    r"""
    Hoyt distribution.

    This distribution applies to the Euclidian norm of a 2-vector for which the components are
    distributed as indepenent random normal variables with variances :math:`\sigma^2_1` and
    :math:`\sigma^2_2`, with :math:`0 < \sigma^2_2 \leq \sigma^2_1`. The
    parameters :math:`a`and :math:`b` of the Hoyt distribution are then given by:

    .. math:
        a = \sqrt{\frac{\soigma^2_2}{\sigma^2_1}}\,,\quad b = \sigma^2_1+\sigma^2_2 

    See footnote (2) in https://members.noa.gr/tronto/IEEE_TR_WC_DEC08.pdf (page 5012).
    The probability density function for `hoyt` is:

    .. math:

        f(x, a, b) = \frac{(1+a^2)x}{ab} \exp\left[-\frac{(1+a^2)^2 x^2}{4a^2b}\right]
                        I_0\left(\frac{(1-a^4)x^2}{4a^2b}

    where :math:`I_0` is the modified Bessel function of order zero (`scipy.special.iv`).

    Note that for correlated random components of the vector one can use the variances
    :math:`\sigma^2_1'` and :math:`\sigma^2_2'` of the diagonalized covariance matrix.

    %(before_notes)s

    Notes
    -----

    `hoyt` takes :math:`a` and :math:`b` as shape parameters.

    %(after_notes)s

    """

    def _pdf(self, x, a, b):
        four_qsqrw = 4*a**2*b
        return (1.0+a**2)*x/(a*b) * np.exp(-(1.0+a**2)**2 * x**2/four_qsqrw) * \
                iv(0,(1.0-a**4)*x**2/four_qsqrw)

hoyt = hoyt_gen(name='hoyt')

_distn_names = ['hoyt']

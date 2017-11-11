"""
This module provides classes and functions that implement the Metropolis MCMC sampling algorithm.
"""

import numpy as np

class MetropolisSampler:
    """
    Base class for Metropolis samplers. Can deal with multi-dimensional sampling problems.

    Limitations:

    - Simplistic implementation
    - Only one chain at a time
    - No conservation of state
    """

    def __init__(self, lnprob, proposal_rvs):
        """
        Class constructor/initializer.

        Parameters
        ----------

        lnprob : callable
            Function returning the natural logarithm of the probability density distribution to be
            sampled. Should take a vector of floats as its single input parameter.
        proposal_rvs : callable
            Function returning a random variate from the proposal distribution. Should take a vector
            of floats as its single input parameter.

        Keywords
        --------

        Returns
        -------

        Nothing.
        """

        self.lnprob = lnprob
        self.proposal_rvs = proposal_rvs

    def run_mcmc(self, theta_init, n_iter, burnin=0, thin=1):
        """
        Run the sampler.

        Parameters
        ----------

        theta_init : float (scalar or vector)
            Initial values of the samples theta_k.
        n_iter : int
            Number of MCMC iterations to perform.

        Keywords
        --------

        burnin : int
            Number of "burn-in" steps to take.
        thin : int
            Only output every thin sample.
        """

        theta_samples = None
        if np.isscalar(theta_init):
            theta_samples = np.empty(n_iter)
        else:
            theta_samples = np.empty((n_iter, theta_init.size))
        theta_samples[0] = theta_init
        for i in range(1, n_iter):
            theta_prop = self.proposal_rvs(theta_samples[i-1])
            lnr = np.log(np.random.uniform())
            lnpdf_theta_k = self.lnprob(theta_samples[i-1])
            lnpdf_theta_prop = self.lnprob(theta_prop)
            if (lnpdf_theta_prop - lnpdf_theta_k > lnr):
                theta_samples[i] = theta_prop
            else:
                theta_samples[i] = theta_samples[i-1]

        return theta_samples

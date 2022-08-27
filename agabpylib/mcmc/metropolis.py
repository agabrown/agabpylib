"""
This module provides classes and functions that implement the Metropolis MCMC sampling algorithm.
"""

import numpy as np

__all__ = ["MetropolisSampler"]


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
            sampled. Should take a 1D array of floats as its single input parameter.
        proposal_rvs : callable
            Function returning a random variate from the proposal distribution. Should take a 1D array
            of floats as its single input parameter.

        Returns
        -------
        Nothing.
        """

        self.lnprob = lnprob
        self.proposal_rvs = proposal_rvs
        self.samples = None
        self.numiter = None
        self.accepted_samples = 1

    def run_mcmc(self, theta_init, n_iter, burnin=0, thin=1):
        """
        Run the sampler.

        TODO: Handle the case where the probability densities for both the current and proposed sample
        are -np.inf.

        Parameters
        ----------
        theta_init : float, or float array
            Initial values of the samples theta_k.
        n_iter : int
            Number of MCMC iterations to perform.
        burnin : int
            Number of "burn-in" steps to take.
        thin : int
            Only output every thin samples.

        Returns
        -------
        Nothing
        """

        self.numiter = n_iter
        if np.isscalar(theta_init):
            self.samples = np.empty(n_iter)
        else:
            self.samples = np.empty((n_iter, theta_init.size))
        self.samples[0] = theta_init
        for i in range(1, n_iter):
            theta_prop = self.proposal_rvs(self.samples[i - 1])
            lnr = np.log(np.random.uniform())
            lnpdf_theta_k = self.lnprob(self.samples[i - 1])
            lnpdf_theta_prop = self.lnprob(theta_prop)
            #   HANDLE THIS CASE!
            # if (np.isinf(lnpdf_theta_prop) and np.isinf(lnpdf_theta_k)):
            if lnpdf_theta_prop - lnpdf_theta_k > lnr:
                self.samples[i] = theta_prop
                self.accepted_samples += 1
            else:
                self.samples[i] = self.samples[i - 1]

    def get_samples(self):
        """
        Get the MCMC samples.

        Parameters
        ----------
        None

        Returns
        -------
        samples : float array
            The array with MCMC samples (shape (n_samples, n_parameters)).
        """

        if np.any(self.samples == None):
            raise Exception("No samples generated: Please invoke run_mcmc() first!")
        return self.samples

    def get_acceptance_fraction(self):
        """
        Get the acceptance fraction for the MCMC chain.

        Parameters
        ----------
        None

        Returns
        -------
        acceptance_frac : float
            The acceptance fraction.
        """
        if np.any(self.samples == None):
            raise Exception("No samples generated: Please invoke run_mcmc() first!")
        return self.accepted_samples / self.numiter

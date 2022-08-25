MCMC tools
==========

Package: ``agabpylib.mcmc``

Metropolis samplers
-------------------

Detailed API: :py:mod:`agabpylib.mcmc.metropolis`

This module was started as part of following the exercises on MCMC samplers in
`Hogg & Foreman-Mackey (2017) <https://arxiv.org/abs/1710.06068>`_. It is only intended
as a learning exercise. For actual MCMC applications proper samplers should
be used such as `Stan <https://mc-stan.org>`_, `emcee <https://github.com/dfm/emcee>`_, 
or `PyMC <https://github.com/pymc-devs/pymc>`_.

Stan tools
----------

`agabpylib.mcmc.stantools`

Some tools for working with the `PyStan <https://mc-stan.org/users/interfaces/pystan.html>`_ 
implementation of `Stan <https://mc-stan.org>`_.

.. note:: 
    The code in this module is incompatible with PyStan 3+ versions and I have switched
    to using `CmdStanPy <https://github.com/stan-dev/cmdstanpy>`_ + `CmdStan <https://mc-stan.org/users/interfaces/cmdstan.html>`_.
    Hence this module is not needed anymore.
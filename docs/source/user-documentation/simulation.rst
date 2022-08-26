Simulation tools
================

Package: `agabpylib.simulation`

Star cluster simulations
------------------------

Package: `agabpylib.simulation.starclusters`

This package contains several modules for the simulation of star clusters. The simulations generate
the following:

* A set of single stars according to a certain initial mass function.
* Stellar astrophysical parameters according to the stars' age and metallicity.
* The space distribution of the stars.
* Kinematics of the stars (cluster bulk velocity and internal velocity field).
* Simulated Gaia observations of the cluster stars.

The following code listing shows an example of generating a simulated cluster.

.. code-block:: python

    import nump as np
    import astropy.units as u

    from agabpylib.simulation.starclusters.cluster import StarCluster, StarAPs
    from agabpylib.simulation.imf import MultiPartPowerLaw as mppl
    from agabpylib.simulation.starclusters.spacedistributions import TruncatedPlummerSphere
    from agabpylib.simulation.starclusters.kinematics import LinearVelocityField
    from agabpylib.simulation.starclusters.observables import GaiaSurvey

    # Random number generator
    rangen = np.random.default_rng()

    # Kroupa type IMF
    imf = mppl(np.array([0.3,1.3,2.3]), np.array([0.1,0.5]))

    # Astrophysical parameters of the cluster stars
    age = 650*u.Myr
    isodir = "/home/brown/Stars/Modelgrids"
    feh = 0.0
    afeh = 0.0
    vvcrit = 0.0

    # Number of stars and core radius for Plummer distribution
    nstars = 5000
    rcore = 6*u.pc
    rtrunc = 3*rcore

    # Kinematic parameters
    v=np.array([-6.30, 45.44, 5.32])*u.km/u.s
    s=np.array([1.69, 1.95, 1.06])*u.km/u.s
    omega = np.array([0.0,0.0,0.0])*u.km/u.s/u.pc
    kappa = 0.0*u.km/u.s/u.pc

    aps = StarAPs(age, feh, afeh, vvcrit, isodir, imf, iso="mist")
    pos = TruncatedPlummerSphere(rcore, rtrunc)
    kin = LinearVelocityField(v, s, omega, kappa)

    distance_c = 45.0*u.pc
    ra_c = 60.5*u.deg
    dec_c = 15.9*u.deg
    gaiadr = "dr3"
    survey = GaiaSurvey(distance_c, ra_c, dec_c, release=gaiadr, rvslim=12.0)

    cluster = StarCluster(nstars, aps, pos, kin, survey, rangen)

Star clusters
^^^^^^^^^^^^^

Module: :py:mod:`agabpylib.simulation.starclusters.cluster`

Contains the class to simulate clusters and the class to generate a set of stars
according to an IMF together with their astrophysical parameters.

Star cluster kinematics
^^^^^^^^^^^^^^^^^^^^^^^

Module: :py:mod:`agabpylib.simulation.starclusters.kinematics`

Contains classes for simulating the velocity field of a star cluster.

Star cluster observables
^^^^^^^^^^^^^^^^^^^^^^^^

Module: :py:mod:`agabpylib.simulation.starclusters.observables`

Classes for generating observations of stars in a star cluster (for now only the Gaia survey).

Star cluster space distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Module: :py:mod:`agabpylib.simulation.starclusters.spacedistributions`

Classes for generating the space distribution of stars in a cluster.

Initial mass function
---------------------

Module :py:mod:`agabpylib.simulation.imf`

Classes for generating stellar masses according to a specific IMF.

Parallax surveys
----------------

Module :py:mod:`agabpylib.simulation.parallaxsurveys`

Simulations of simple parallax surveys. Only the parallax and apparent
magnitudes of the stars are simulated. The classes are useful for investigating
the issue around inferring distances from parallaxes. Some of the code was used
in support of the paper on using Gaia parallaxes (`Luri et al. 2018
<https://ui.adsabs.harvard.edu/abs/2018A%26A...616A...9L/abstract>`_). See the
corresponding `github repository
<https://github.com/agabrown/astrometry-inference-tutorials>`_
"""
Provide classes and methods for simulating simple parallax surveys. These consist of measurements of the
parallax and the apparent magnitude of the stars.

Anthony Brown 2011-2017
"""

#__all__ = ['simParallaxesConstantSpaceDensity', 'simGaussianAbsoluteMagnitude',
#        'UniformDistributionSingleLuminosityHip', 'showSurveyStats']

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, cm
from sys import stderr
from scipy.stats import norm, uniform
from scipy.special import erf
from cycler import cycler
from os import path

from agabpylib.plotting.plotstyles import useagab, apply_tufte
from agabpylib.plotting.distinct_colours import get_distinct
from agabpylib.densityestimation.kde import kde_scikitlearn
from agabpylib.tools.robuststats import rse

_ROOT = path.abspath(path.dirname(__file__))
def data_file_path(mypath):
    return path.join(_ROOT, 'data', mypath)

def simParallaxesConstantSpaceDensity(numStars, minParallax, maxParallax):
    """
    Simulate parallaxes for stars distributed uniformly in space around the Sun.

    Parameters
    ----------
  
    numStars - number of stars to simulate
    minParallax - lower limit of the true parallax (volume limit of survey, arbitrary units)
    maxParallax - upper limit of the true parallax (closest possible star, arbitrary units)
  
    Returns
    -------

    Vector of parallax values
    """
    x=uniform.rvs(loc=0.0, scale=1.0, size=numStars)
    minPMinThird=np.power(minParallax,-3.0)
    maxPMinThird=np.power(maxParallax,-3.0)
    return np.power(minPMinThird-1.0*x*(minPMinThird-maxPMinThird),-1.0/3.0)

def simGaussianAbsoluteMagnitude(numStars, mean, stddev):
    """
    Simulate absolute magnitudes following a Gaussian distribution.

    Parameters
    ----------

    numStars - number of stars to simulate
    mean - mean of the distribution
    stddev - standard deviation of the distribution

    Returns
    -------

    Vector of magnitudes.
    """
    return norm.rvs(loc=mean, scale=stddev, size=numStars)

class ParallaxSurvey:
    """
    Base class for simulating a parallax survey. The survey is assumed to have been conducted for a set
    of stars distributed in space between some minimum aand maximum parallax value, according to some
    spatial distribution to be defined by the sub-classes. Likewise the distribution in apparent
    magnitudes is assumed to be specified by the sub-classes.
    """

    def __init__(self, numberOfStars, minParallax, maxParallax, surveyLimit=np.Inf):
        """
        Class constructor/initializer.

        Parameters
        ----------

        numberOfStars - Number of stars to simulate
        minParallax   - Lower limit of the true parallax (volume limit of survey, mas)
        maxParallax   - Upper limit of the true parallax (closest possible star, mas)

        Keywords
        --------

        surveyLimit - Apparent magnitude limit of the survey (default: no limit)
        """
        self.numberOfStars=numberOfStars
        self.numberOfStarsInSurvey=numberOfStars
        self.minParallax=minParallax
        self.maxParallax=maxParallax
        self.apparentMagnitudeLimit=surveyLimit

    def setRandomNumberSeed(self, seed):
        """
        (Re-)Set the random number seed for the simulations. NOTE, also applies to the scipy.stats functions.
  
        Parameters
        ----------
  
        seed - Value of random number seed
        """
        np.random.seed(seed)

    def _applyApparentMagnitudeLimit(self):
        """
        Apply the apparent magnitude limit to the simulated survey.
        """
        indices=(self.observedMagnitudes <= self.apparentMagnitudeLimit)
        self.trueParallaxes=self.trueParallaxes[indices]
        self.absoluteMagnitudes=self.absoluteMagnitudes[indices]
        self.apparentMagnitudes=self.apparentMagnitudes[indices]
        self.parallaxErrors=self.parallaxErrors[indices]
        self.magnitudeErrors=self.magnitudeErrors[indices]
        self.observedParallaxes=self.observedParallaxes[indices]
        self.observedMagnitudes=self.observedMagnitudes[indices]
        self.numberOfStarsInSurvey=len(self.observedMagnitudes)

class UniformSpaceDistributionSingleLuminosity(ParallaxSurvey):
    """
    Base class for simulated parallax surveys in which the stars are distributed uniformly in space
    between a minimum and maximum distance and with all stars having an absolute magnitude draw from the
    same Normal distribution.
    """
    def __init__(self, numberOfStars, minParallax, maxParallax, meanAbsoluteMagnitude,
            stddevAbsoluteMagnitude, surveyLimit=np.Inf):
        """
        Class constructor/initializer.

        Parameters
        ----------
    
        numberOfStars             - number of stars to simulate
        minParallax               - lower limit of the true parallax (volume limit of survey, mas)
        maxParallax               - upper limit of the true parallax (closest possible star, mas)
        meanAbsoluteMagnitude     - Mean of Gaussian absolute magnitude distribution
        stddevAbsoluteMagnitude   - Standard deviation of Gaussian absolute magnitude distribution
  
        Keywords
        --------

        surveyLimit - Apparent magnitude limit of the survey (default: no limit)
        """
        super().__init__(numberOfStars, minParallax, maxParallax, surveyLimit)
        self.meanAbsoluteMagnitude=meanAbsoluteMagnitude
        self.stddevAbsoluteMagnitude=stddevAbsoluteMagnitude

    def generateObservations(self):
        """
        Generate the simulated observations.
        """
        self.trueParallaxes = simParallaxesConstantSpaceDensity(self.numberOfStars, self.minParallax,
                self.maxParallax)
        self.absoluteMagnitudes = simGaussianAbsoluteMagnitude(self.numberOfStars,
                self.meanAbsoluteMagnitude, self.stddevAbsoluteMagnitude)
        self.apparentMagnitudes = self.absoluteMagnitudes-5.0*np.log10(self.trueParallaxes)+10.0
        self.parallaxErrors = self._generateParallaxErrors()
        self.magnitudeErrors = self._generateApparentMagnitudeErrors()
        self.observedParallaxes = norm.rvs(loc=self.trueParallaxes, scale=self.parallaxErrors)
        self.observedMagnitudes = norm.rvs(loc=self.apparentMagnitudes, scale=self.magnitudeErrors)
        self._applyApparentMagnitudeLimit()

    def apparentMagnitude_lpdf(self, m):
        """
        Calculate the natural logarithm of the analytical probability density function of the apparent
        magnitudes in the simulated survey. This can be derived from the know parallax and absolute
        magnitude PDFs.

        Parameters
        ----------

        m - The apparent magnitude(s) for which to calculate the PDF.
        """
        c = 0.6*np.log(10.0)
        a = np.power(self.minParallax, -3.0) - np.power(self.maxParallax, -3.0)
        dmH = -5*np.log10(self.minParallax)+10
        dmL = -5*np.log10(self.maxParallax)+10
        zH = (dmH - m + self.meanAbsoluteMagnitude - c*self.stddevAbsoluteMagnitude**2) / \
                (np.sqrt(2.0)*self.stddevAbsoluteMagnitude)
        zL = (dmL - m + self.meanAbsoluteMagnitude - c*self.stddevAbsoluteMagnitude**2) / \
                (np.sqrt(2.0)*self.stddevAbsoluteMagnitude)
        lpdf = np.log(c) - np.log(a) + c*(m - self.meanAbsoluteMagnitude + \
                0.5*c*self.stddevAbsoluteMagnitude**2 - 10.0) + \
                -np.log(2.0) + np.log(erf(zH)-erf(zL))

        return lpdf

class UniformDistributionSingleLuminosityHip(UniformSpaceDistributionSingleLuminosity):
    """
    Simulate a parallax survey for stars distributed uniformly in space around the sun. The stars all
    have the same luminosity drawn from a Gaussian distribution. The errors on the observed parallaxes
    and apparent magnitudes roughly follow the characteristics of the Hipparcos Catalogue.
    """

    def __init__(self, numberOfStars, minParallax, maxParallax, meanAbsoluteMagnitude,
            stddevAbsoluteMagnitude, surveyLimit=np.Inf):
        """
        Class constructor/initializer.

        Parameters
        ----------
    
        numberOfStars             - number of stars to simulate
        minParallax               - lower limit of the true parallax (volume limit of survey, mas)
        maxParallax               - upper limit of the true parallax (closest possible star, mas)
        meanAbsoluteMagnitude     - Mean of Gaussian absolute magnitude distribution
        stddevAbsoluteMagnitude   - Standard deviation of Gaussian absolute magnitude distribution
  
        Keywords
        --------

        surveyLimit - Apparent magnitude limit of the survey (default: no limit)
        """
        super().__init__(numberOfStars, minParallax, maxParallax, meanAbsoluteMagnitude,
                stddevAbsoluteMagnitude, surveyLimit)
        self.parallaxErrorNormalizationMagnitude=5.0
        self.parallaxErrorSlope=0.2
        self.parallaxErrorCalibrationFloor=0.2
        self.magnitudeErrorNormalizationMagnitude=5.0
        self.magnitudeErrorSlope=0.006
        self.magnitudeErrorCalibrationFloor=0.001

    def _generateParallaxErrors(self):
        """
        Generate the parallax errors according to an ad-hoc function of parallax error as a function of
        magnitude.
        """
        errors = (np.power(10.0,0.2*(self.apparentMagnitudes -
            self.parallaxErrorNormalizationMagnitude))*self.parallaxErrorSlope)
        indices = (errors < self.parallaxErrorCalibrationFloor)
        errors[indices] = self.parallaxErrorCalibrationFloor
        return errors

    def _generateApparentMagnitudeErrors(self):
        """
        Generate the apparent magnitude errors to an ad-hoc function of magnitude error as a function of
        magnitude.
        """
        errors = (np.power(10.0,0.2*(self.apparentMagnitudes -
            self.magnitudeErrorNormalizationMagnitude))*self.magnitudeErrorSlope)
        indices = (errors < self.magnitudeErrorCalibrationFloor)
        errors[indices] = self.magnitudeErrorCalibrationFloor
        return errors

class UniformDistributionSingleLuminosityTGAS(UniformSpaceDistributionSingleLuminosity):
    """
    Simulate a parallax survey for stars distributed uniformly in space around the sun. The stars all
    have the same luminosity drawn from a Gaussian distribution. The errors on the observed parallaxes
    and apparent magnitudes roughly follow the characteristics of the TGAS Catalogue.
    """

    def __init__(self, numberOfStars, minParallax, maxParallax, meanAbsoluteMagnitude,
            stddevAbsoluteMagnitude, surveyLimit=np.Inf):
        """
        Class constructor/initializer.

        Parameters
        ----------
    
        numberOfStars             - number of stars to simulate
        minParallax               - lower limit of the true parallax (volume limit of survey, mas)
        maxParallax               - upper limit of the true parallax (closest possible star, mas)
        meanAbsoluteMagnitude     - Mean of Gaussian absolute magnitude distribution
        stddevAbsoluteMagnitude   - Standard deviation of Gaussian absolute magnitude distribution
  
        Keywords
        --------

        surveyLimit - Apparent magnitude limit of the survey (default: no limit)
        """
        super().__init__(numberOfStars, minParallax, maxParallax, meanAbsoluteMagnitude,
                stddevAbsoluteMagnitude, surveyLimit)
        self.magnitudeErrorNormalizationMagnitude=13.0
        self.magnitudeErrorLogarithmicSlope=0.21
        self.magnitudeErrorLogCalibrationFloor=-3.4
        self.tgasErrorsPdfFile = data_file_path('TGAS-parallax-errors-pdf.csv')
        self.tgasErrPdf = None
        if path.isfile(self.tgasErrorsPdfFile):
            self.tgasErrPdf = np.genfromtxt(self.tgasErrorsPdfFile, comments='#', skip_header=1,
                delimiter=',', names=['err','logdens'], dtype=None)
        else:
            print("Cannot find file {0}".format(self.tgasErrorsPdfFile))
            print("exiting")
            exit()

    def _generateParallaxErrors(self):
        """
        Generate the parallax errors according to an ad-hoc function of parallax error as a function of
        magnitude.
        """
        probs = np.exp(self.tgasErrPdf['logdens'])/np.sum(np.exp(self.tgasErrPdf['logdens']))
        errors = np.random.choice(self.tgasErrPdf['err'], size=self.apparentMagnitudes.size, p=probs)
        return errors

    def _generateApparentMagnitudeErrors(self):
        """
        Generate the apparent magnitude errors to an ad-hoc function of magnitude error as a function of
        magnitude.
        """
        errors = self.magnitudeErrorLogCalibrationFloor + self.magnitudeErrorLogarithmicSlope * \
                (self.apparentMagnitudes - self.magnitudeErrorNormalizationMagnitude)
        indices = (errors < self.magnitudeErrorLogCalibrationFloor)
        errors[indices] = self.magnitudeErrorLogCalibrationFloor
        return np.power(10.0,errors)

def showSurveyStatistics(simulatedSurvey, pdfFile=None, pngFile=None):
    """
    Produce a plot with the survey statistics.

    Parameters
    ----------

    simulatedSurvey - Object containing the simulated survey.

    Keywords
    --------

    pdfFile - Name of optional PDF file in which to save the plot.
    pngFile - Name of optional PNG file in which to save the plot.
    """
    try:
        _ = simulatedSurvey.observedParallaxes.shape
    except AttributeError:
        stderr.write("You have not generated the observations yet!\n")
        return

    parLimitPlot=50.0

    positiveParallaxes = (simulatedSurvey.observedParallaxes > 0.0)
    goodParallaxes = (simulatedSurvey.observedParallaxes/simulatedSurvey.parallaxErrors >= 5.0)
    estimatedAbsMags = (simulatedSurvey.observedMagnitudes[positiveParallaxes] +
            5.0*np.log10(simulatedSurvey.observedParallaxes[positiveParallaxes])-10.0)
    relParErr = (simulatedSurvey.parallaxErrors[positiveParallaxes] /
            simulatedSurvey.observedParallaxes[positiveParallaxes])
    deltaAbsMag = estimatedAbsMags - simulatedSurvey.absoluteMagnitudes[positiveParallaxes]

    useagab()
    fig = plt.figure(figsize=(16,10))
  
    axA = fig.add_subplot(2,2,1)
    apply_tufte(axA, withgrid=False)
    axA.set_prop_cycle(cycler('color', get_distinct(3)))

    minPMinThird=np.power(simulatedSurvey.minParallax,-3.0)
    maxPMinThird=np.power(parLimitPlot,-3.0)
    x=np.linspace(simulatedSurvey.minParallax,np.min([parLimitPlot,simulatedSurvey.maxParallax]),1001)
    axA.plot(x,3.0*np.power(x,-4.0)/(minPMinThird-maxPMinThird),'--', label='model', lw=3)

    scatter = rse(simulatedSurvey.trueParallaxes)
    bw = 1.06*scatter*simulatedSurvey.numberOfStarsInSurvey**(-0.2)
    samples, logdens = kde_scikitlearn(simulatedSurvey.trueParallaxes, N=200,
            lims=(simulatedSurvey.observedParallaxes.min(),
                np.min([parLimitPlot,simulatedSurvey.observedParallaxes.max()])), kde_bandwidth=bw)
    axA.plot(samples, np.exp(logdens), '-', lw=3, label='true')

    scatter = rse(simulatedSurvey.observedParallaxes)
    bw = 1.06*scatter*simulatedSurvey.numberOfStarsInSurvey**(-0.2)
    samples, logdens = kde_scikitlearn(simulatedSurvey.observedParallaxes, N=200,
            lims=(simulatedSurvey.observedParallaxes.min(),
                np.min([parLimitPlot,simulatedSurvey.observedParallaxes.max()])), kde_bandwidth=bw)
    axA.plot(samples, np.exp(logdens), '-', lw=3, label='observed')

    axA.set_xlabel(r'$\varpi$ [mas]')
    axA.set_ylabel(r'$p(\varpi)$')
    #axA.set_ylim(0,0.15)
    leg=axA.legend(loc='best', handlelength=1.0)
    for t in leg.get_texts():
        t.set_fontsize(14)

    axB = fig.add_subplot(2,2,2)
    apply_tufte(axB, withgrid=False)
    axB.set_prop_cycle(cycler('color', get_distinct(3)))

    scatter = rse(simulatedSurvey.absoluteMagnitudes)
    bw = 1.06*scatter*simulatedSurvey.numberOfStarsInSurvey**(-0.2)
    samples, logdens = kde_scikitlearn(simulatedSurvey.absoluteMagnitudes, N=200, kde_bandwidth=bw)
    x=np.linspace(samples.min(),samples.max(),300)
    axB.plot(x, norm.pdf(x,loc=simulatedSurvey.meanAbsoluteMagnitude,
        scale=simulatedSurvey.stddevAbsoluteMagnitude), '--', lw=3, label='model')
    axB.plot(samples, np.exp(logdens), '-', label='true', lw=3)
    scatter = rse(simulatedSurvey.absoluteMagnitudes[goodParallaxes])
    bw = 1.06*scatter*simulatedSurvey.absoluteMagnitudes[goodParallaxes].size**(-0.2)
    samples, logdens = kde_scikitlearn(simulatedSurvey.absoluteMagnitudes[goodParallaxes], lims=(x.min(),
        x.max()), N=200, kde_bandwidth=bw)
    axB.plot(samples, np.exp(logdens), '-', label=r'$\varpi/\sigma_\varpi\geq5$', lw=3)
    axB.set_xlabel("$M$")
    axB.set_ylabel("$p(M)$")
    leg=axB.legend(loc='upper left', handlelength=1.0)
    for t in leg.get_texts():
        t.set_fontsize(14)

    axC = fig.add_subplot(2,2,3)
    apply_tufte(axC, withgrid=False)
    axC.set_prop_cycle(cycler('color', get_distinct(3)))

    m = np.linspace(simulatedSurvey.observedMagnitudes.min(), simulatedSurvey.observedMagnitudes.max(), 1000)
    axC.plot(m, np.exp(simulatedSurvey.apparentMagnitude_lpdf(m)), '--', lw=3, label='model')

    scatter = rse(simulatedSurvey.apparentMagnitudes)
    bw = 1.06*scatter*simulatedSurvey.numberOfStarsInSurvey**(-0.2)
    samples, logdens = kde_scikitlearn(simulatedSurvey.apparentMagnitudes, N=200, kde_bandwidth=bw)
    axC.plot(samples, np.exp(logdens), '-', label='true', lw=3)

    scatter = rse(simulatedSurvey.observedMagnitudes)
    bw = 1.06*scatter*simulatedSurvey.numberOfStarsInSurvey**(-0.2)
    samples, logdens = kde_scikitlearn(simulatedSurvey.observedMagnitudes, N=200, kde_bandwidth=bw)
    axC.plot(samples, np.exp(logdens), '-', label='observed', lw=3)

    axC.set_xlabel("$m$")
    axC.set_ylabel("$p(m)$")
    leg=axC.legend(loc='upper left', handlelength=1.0)
    for t in leg.get_texts():
        t.set_fontsize(14)

    axD = fig.add_subplot(2,2,4)
    apply_tufte(axD, withgrid=False)
    axD.set_prop_cycle(cycler('color', get_distinct(2)))
    if len(relParErr) < 1000:
        axD.semilogx(relParErr,deltaAbsMag,'.')
        axD.set_xlabel("$\\sigma_\\varpi/\\varpi_\\mathrm{o}$")
        axD.set_xlim(1.0e-3,100)
    else:
        axD.hexbin(np.log10(relParErr),deltaAbsMag,C=None, bins='log', cmap=cm.Blues_r, mincnt=1)
        axD.set_xlabel("$\\log\\sigma_\\varpi/\\varpi_\\mathrm{o}]$")
        axD.set_xlim(-3,2)
    axD.set_ylabel("$\\widetilde{M}-M_\\mathrm{true}$")
    axD.set_ylim(-10,6)

    plt.suptitle("Simulated survey statistics: $N_\\mathrm{{stars}}={0}$, ".format(simulatedSurvey.numberOfStars) +
            "$m_\\mathrm{{lim}}={0}$, ".format(simulatedSurvey.apparentMagnitudeLimit) +
            "$N_\\mathrm{{survey}}={0}$, ".format(simulatedSurvey.numberOfStarsInSurvey) +
            "${0}\\leq\\varpi\\leq{1}$, ".format(simulatedSurvey.minParallax, simulatedSurvey.maxParallax)+
            "$\\mu_M={0}$, ".format(simulatedSurvey.meanAbsoluteMagnitude) + 
            "$\\sigma_M={0:.2f}$".format(simulatedSurvey.stddevAbsoluteMagnitude))
  
    if pdfFile is not None:
        plt.savefig(pdfFile)
    if pngFile is not None:
        plt.savefig(pngFile)
    if (pdfFile is None and pngFile is None):
        plt.show()

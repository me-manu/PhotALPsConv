"""
Class to calculate turbulent B field

History:
--------
- 04/08/2014: version 0.01 - created
"""

__author__ = "Manuel Meyer // manuel.meyer@fysik.su.se"
__version__ = 0.01

# --- Imports ------------ #
import numpy as np
from numpy.random import rand
from numpy import log,log10,pi,meshgrid,cos,sum,sqrt
from math import ceil

class Bgaussian(object):
    """
    Class to calculate a magnetic field with gaussian turbulence and power-law spectrum
    """
    def __init__(self,**kwargs):
	"""
	Initializa gaussian turbulence B field spectrum. Defaults assume values typical for a galaxy cluster

	kwargs
	------
	kH:		float, upper wave number cutoff, should be at at least > 1. / osc. wavelength (default = 1 / (1 kpc))
	kL:		float, lower wave number cutoff, should be of same size as the system (default = 1 / (100 kpc))
	B:  		float, B field strength, energy is B^2 / 4pi (default = 1 muG)
	q:  		float, power-law turbulence spectrum (default: q = 11/3 is Kolmogorov type spectrum)
	dkType:		string, either linear, log, or random. Determine the spacing of the dk intervals 	
	dkSteps: 	int, number of dkSteps. For log spacing, number of steps per decade / number of decades ~ 10
			should be chosen.
	"""
# --- Set the defaults 
	kwargs.setdefault('kL',1. / 100.)
	kwargs.setdefault('kH',1. / 1.)
	kwargs.setdefault('B',1.)
	kwargs.setdefault('q',-11. / 3.)
	kwargs.setdefault('dkType','log')
	kwargs.setdefault('dkSteps',0)

	self.__dict__.update(kwargs)

	if not self.dkSteps:
	    self.dkSteps = int(ceil(10. * (log10(self.kH) - log10(self.kL)) ** 2.))

# initialize the k values and intervalls.
	if self.dkType == 'linear':
	    self.kn = np.linspace(self.kL, self.kH, self.dkSteps)
	    self.dk = self.kn[1:] - self.kn[:-1]
	    self.kn = self.kn[:-1]
	elif self.dkType == 'log':
	    self.kn = 10.**np.linspace(log10(self.kL), log10(self.kH), self.dkSteps)
	    self.dk = self.kn[1:] - self.kn[:-1]
	    self.kn = self.kn[:-1]
	elif self.dkType == 'random':
	    self.dk = rand(self.dkSteps)
	    self.dk *= (self.kH - self.kL) / sum(self.dk)
	    self.kn = np.array([self.kL + sum(self.dk[:n]) for n in range(self.dk.shape[0])])
	else:
	    raise ValueError("dkType has to either 'linear', 'log', or 'random', not {0:s}".format(self.dkType))


	self.Un = rand(self.kn.shape[0])
	self.Vn = rand(self.kn.shape[0])

	return

    def new_random_numbers(self):
	"""Generate new random numbers for Un,Vn, and kn if knType == random"""
	if self.dkType == 'random':
	    self.dk = rand(self.dkSteps)
	    self.dk *= (self.kH - self.kL) / sum(self.dk)
	    self.kn = np.array([self.kL + sum(self.dk[:n]) for n in range(self.dk.shape[0])])

	self.Un = rand(self.kn.shape[0])
	self.Vn = rand(self.kn.shape[0])
	return

    def _Fq(self,x):
	"""
	Calculate the F_q function for given x,kL, and kH

	Arguments
	---------
	x:	n-dim array, Ratio between k and kH

	Returns
	-------
	n-dim array with Fq values
	"""
	if self.q == 0.:
	    return 3. * self.kH **2. / (self.kH ** 3. - self.kL ** 3.) * ( 0.5 * (1. - x*x) - x * x * log(x) )
	elif self.q == -2.:
	    return ( 0.5 * (1. - x*x) - log(x) ) / (self.kH  - self.kL )
	elif self.q == -3.:
	    return 1. / log(self.kH / self.kL) / self.kH / x  / 3. * (-x*x*x - 3. * x + 4.)
	else:
	    return self.kH ** (self.q + 2.) / (self.kH ** (self.q + 3.) - self.kL ** (self.q + 3.)) * \
		    (self.q + 3.) / (self.q * ( self.q + 2.)) * \
		    (self.q + x * x * ( 2. + self.q - 2. * (1. + self.q) * x ** self.q))

    def _corrTrans(self,k):
	"""
	Calculate the transversal correlation function for wave number k

	Arguments
	---------
	k:	n-dim array, wave number

	Returns
	-------
	n-dim array with values of the correlation function
	"""
	return pi / 4. * self.B * self.B * self._Fq(k / self.kH)

    def Bgaus(self, z):
	"""
	Calculate the magnetic field for a gaussian turbulence field along the line of sight direction, denoted by z.

	Arguments
	---------
	z:	m-dim array, distance traversed in magnetic field

	Returns
	-------
	m-dim array with values of transversal field
	"""
	zz, kk = meshgrid(z,self.kn)
	zz, dd = meshgrid(z,self.dk)
	zz, uu = meshgrid(z,self.Un)
	zz, vv = meshgrid(z,self.Vn)

	B = sum(sqrt(self._corrTrans(kk) / pi * dd * log(1. / uu)) * cos(kk * zz + 2. * pi * vv), axis = 0)
	return B

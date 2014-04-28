"""
Class for the calculation of photon-ALPs conversion in galaxy clusters

History:
- 06/01/12: created
- 07/18/13: cleaned up
"""
__version__=0.02
__author__="M. Meyer // manuel.meyer@physik.uni-hamburg.de"


import numpy as np
from math import ceil
import eblstud.ebl.tau_from_model as Tau
from eblstud.misc.constants import *
import logging
import warnings
from numpy.random import rand, seed
from PhotALPsConv.Bturb import Bgaussian as Bgaus

# --- Conversion without absorption, designed to match values in Clusters -------------------------------------------#
from deltas import *
class PhotALPs_ICM(object):
    """
    Class for photon ALP conversion in galaxy clusters and the intra cluster medium (ICM) 

    Attributes
    ----------
    Lcoh:	coherence length / domain size of turbulent B-field in the cluster in kpc
    B:		field strength of transverse component of the cluster B-field, in muG
    r_abell:	size of cluster filled with the constant B-field in kpc
    g:		Photon ALP coupling in 10^{-11} GeV^-1
    m:		ALP mass in neV
    n:		thermal electron density in the cluster, in 10^{-3} cm^-3
    Nd:		number of domains, Lcoh/r_abell
    Psin:	random angle in domain n between transverse B field and propagation direction
    T1:		Transfer matrix 1 (3x3xNd)-matrix
    T2:		Transfer matrix 2 (3x3xNd)-matrix		
    T3:		Transfer matrix 3 (3x3xNd)-matrix
    Un:		Total transfer matrix in all domains (3x3xNd)-matrix
    Dperp:	Mixing matrix parameter Delta_perpedicular in n-th domain
    Dpar:	Mixing matrix parameter Delta_{||} in n-th domain
    Dag:	Mixing matrix parameter Delta_{a\gamma} in n-th domain
    Da:		Mixing matrix parameter Delta_{a} in n-th domain
    alph:	Mixing angle
    Dosc:	Oscillation Delta

    EW1: 	Eigenvalue 1 of mixing matrix
    EW2:	Eigenvalue 2 of mixing matrix
    EW3:	Eigenvalue 3 of mixing matrix

    Notes
    -----
    For Photon - ALP mixing theory see e.g. De Angelis et al. 2011 and also Horns et al. 2012
    http://adsabs.harvard.edu/abs/2011PhRvD..84j5030D
    http://adsabs.harvard.edu/abs/2012PhRvD..86g5024H
    """
    def __init__(self, **kwargs):
	"""
	init photon axion conversion in intracluster medium

	Parameters
	----------
	Lcoh:		coherence length / domain size of turbulent B-field in the cluster in kpc, default: 10 kpc
	B:		field strength of transverse component of the cluster B-field, in muG, default: 1 muG
	r_abell:	size of cluster filled with the constant B-field in kpc. default: 1500 * h
	g:		Photon ALP coupling in 10^{-11} GeV^-1, default: 1.
	m:		ALP mass in neV, default: 1.
	n:		thermal electron density in the cluster, in 10^{-3} cm^-3, default: 1.

	Bn_const:	boolean, if True n and B are constant all over the cluster
			if False than B and n are modeled, see notes
	Bgauss:		boolean, if True, B field calculated from gaussian turbulence spectrum,
			if False then domain-like structure is assumed.

	kH:		float, upper wave number cutoff, should be at at least > 1. / osc. wavelength (default = 1 / (1 kpc))
	kL:		float, lower wave number cutoff, should be of same size as the system (default = 1 / (r_abell kpc))
	q:  		float, power-law turbulence spectrum (default: q = 11/3 is Kolmogorov type spectrum)
	dkType:		string, either linear, log, or random. Determine the spacing of the dk intervals 	
	dkSteps: 	int, number of dkSteps. For log spacing, number of steps per decade / number of decades ~ 10
			should be chosen.

	r_core:		Core radius for n and B modeling in kpc, default: 200 kpc
	beta:		power of n dependence, default: 2/3
	eta:		power with what B follows n, see Notes. Typical values: 0.5 <= eta <= 1. default: 1.

	Returns
	-------
	Nothing.

	Notes
	-----

	If Bn_const = False then electron density is modeled according to Carilli & Taylor (2002) Eq. 2:
	    n_e(r)  = n * (1 - (r/r_core)**2.)**(-3/2*beta)
	with typical values of r_core = 200 kpc and beta = 2/3.

	The magnetic field is supposed to follow n_e(r) with (Feretti et al. 2012, p. 41, section 7.1)
	    B(r) = B * (n_e(r)/n) ** eta
	with typical values 1 muG <= B <= 15muG and 0.5 <= eta <= 1
	"""
# --- Set the defaults 
	kwargs.setdefault('g',1.)
	kwargs.setdefault('m',1.)
	kwargs.setdefault('B',1.)
	kwargs.setdefault('n',1.)
	kwargs.setdefault('Lcoh',1.)
	kwargs.setdefault('r_abell',100.)
	kwargs.setdefault('r_core',200.)
	kwargs.setdefault('E_GeV',1.)

	kwargs.setdefault('B_gauss',False)
	kwargs.setdefault('kL',1. / kwargs['r_abell'])
	kwargs.setdefault('kH',200.)
	kwargs.setdefault('q',-11. / 3.)
	kwargs.setdefault('dkType','log')
	kwargs.setdefault('dkSteps',0)

	kwargs.setdefault('Bn_const',True)
	kwargs.setdefault('beta',2. / 3.)
	kwargs.setdefault('eta',1.)
# --------------------
	self.update_params(**kwargs)

	super(PhotALPs_ICM,self).__init__()

    def update_params(self, new_Bn = True, **kwargs):
	"""Update all parameters with new values and initialize all matrices
	
	kwargs
	------
	new_B_n:	boolean, if True, recalculate B field and electron density
	
	"""

	self.__dict__.update(kwargs)

	self.Nd	= int(self.r_abell / self.Lcoh)	# number of domains, no expansion assumed
	self.r	= np.linspace(self.Lcoh, self.r_abell + self.Lcoh, int(self.Nd))

	if self.B_gauss:
	    self.bfield	= Bgaus(**kwargs)		# init gaussian turbulent field

	if new_Bn:
	    self.new_B_n()

	self.T1		= np.zeros((3,3,self.Nd),np.complex)	# Transfer matrices
	self.T2		= np.zeros((3,3,self.Nd),np.complex)
	self.T3		= np.zeros((3,3,self.Nd),np.complex)
	self.Un		= np.zeros((3,3,self.Nd),np.complex)

	return

    def new_B_n(self):
	"""
	Recalculate Bfield and density, if Kolmogorov turbulence is set to true, new random values for B and Psi are calculated.
	"""

	if self.B_gauss:
	    Bt		= self.bfield.Bgaus(self.r)	# calculate first transverse component
	    self.bfield.new_random_numbers()		# new random numbers
	    Bu		= self.bfield.Bgaus(self.r)	# calculate second transverse component
	    self.B	= np.sqrt(Bt ** 2. + Bu ** 2.)	# calculate total transverse component 
	    self.Psin	= np.arctan2(Bt , Bu)		# and angle to x2 (t) axis -- use atan2 to get the quadrants right

	if self.Bn_const:
	    self.n		= self.n * np.ones(int(self.Nd))	# assuming a constant electron density over all domains
	    if not self.B_gauss:
		self.B		= self.B * np.ones(int(self.Nd))	# assuming a constant B-field over all domains
	else:
	    if np.isscalar(self.n):
		n0 = self.n
	    else:
		n0 = self.n[0]
	    self.n =  n0 * (np.ones(int(self.Nd)) + self.r**2./self.r_core**2.)**(-1.5 * self.beta)
	    self.B = self.B * (self.n / n0 )**self.eta
	return

    def new_random_psi(self):
	"""
	Calculate new random psi values

	Parameters:
	-----------
	None

	Returns:
	--------
	Nothing
	"""
	self.Psin	= 2. * np.pi * rand(1,int(self.Nd))[0]	# angle between photon propagation on B-field in i-th domain 
	return

    def __setDeltas(self):
	"""
	Set Deltas of mixing matrix for each domain
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""

	self.Dperp	= Delta_pl_kpc(self.n,self.E) + 2.*Delta_QED_kpc(self.B,self.E)		# np.arrays , self.Nd-dim
	self.Dpar	= Delta_pl_kpc(self.n,self.E) + 3.5*Delta_QED_kpc(self.B,self.E)	# np.arrays , self.Nd-dim
	self.Dag	= Delta_ag_kpc(self.g,self.B)						# np.array, self.Nd-dim
	self.Da		= Delta_a_kpc(self.m,self.E) * np.ones(int(self.Nd))			# np.ones, so that it is np.array, self.Nd-dim
	self.alph	= 0.5 * np.arctan(2. * self.Dag / (self.Dpar - self.Da)) 
	self.Dosc	= np.sqrt((self.Dpar - self.Da)**2. + 4.*self.Dag**2.)

	return

    def __setEW(self):
	"""
	Set Eigenvalues
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	# Eigen values are all self.Nd-dimensional
	self.__setDeltas()
	self.EW1 = self.Dperp
	self.EW2 = 0.5 * (self.Dpar + self.Da - self.Dosc)
	self.EW3 = 0.5 * (self.Dpar + self.Da + self.Dosc)
	return
	

    def __setT1n(self):
	"""
	Set T1 in all domains
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	c = np.cos(self.Psin)
	s = np.sin(self.Psin)
	self.T1[0,0,:]	= c*c
	self.T1[0,1,:]	= -1. * c*s
	self.T1[1,0,:]	= self.T1[0,1]
	self.T1[1,1,:]	= s*s
	return

    def __setT2n(self):
	"""
	Set T2 in all domains
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	c = np.cos(self.Psin)
	s = np.sin(self.Psin)
	ca = np.cos(self.alph)
	sa = np.sin(self.alph)
	self.T2[0,0,:] = s*s*sa*sa
	self.T2[0,1,:] = s*c*sa*sa
	self.T2[0,2,:] = -1. * s * sa *ca

	self.T2[1,0,:] = self.T2[0,1]
	self.T2[1,1,:] = c*c*sa*sa
	self.T2[1,2,:] = -1. * c *ca * sa

	self.T2[2,0,:] = self.T2[0,2]
	self.T2[2,1,:] = self.T2[1,2]
	self.T2[2,2,:] = ca * ca
	return

    def __setT3n(self):
	"""
	Set T3 in all domains
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	c = np.cos(self.Psin)
	s = np.sin(self.Psin)
	ca = np.cos(self.alph)
	sa = np.sin(self.alph)
	self.T3[0,0,:] = s*s*ca*ca
	self.T3[0,1,:] = s*c*ca*ca
	self.T3[0,2,:] = s*sa*ca

	self.T3[1,0,:] = self.T3[0,1]
	self.T3[1,1,:] = c*c*ca*ca
	self.T3[1,2,:] = c * sa *ca

	self.T3[2,0,:] = self.T3[0,2]
	self.T3[2,1,:] = self.T3[1,2]
	self.T3[2,2,:] = sa*sa
	return

    def __setUn(self):
	"""
	Set Transfer Matrix Un in n-th domain
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	self.Un = np.exp(1.j * self.EW1 * self.Lcoh) * self.T1 + \
	np.exp(1.j * self.EW2 * self.Lcoh) * self.T2 + \
	np.exp(1.j * self.EW3 * self.Lcoh) * self.T3
	return

    def SetDomainN(self):
	"""
	Set Transfer matrix in all domains and multiply it

	Parameters
	----------
	None (self only)

	Returns
	-------
	Transfer matrix as 3x3 complex numpy array
	"""
	if not self.Nd == self.Psin.shape[0]:
	    raise TypeError("Number of domains (={0:n}) is not equal to number of angles (={1:n})!".format(self.Nd,self.Psin.shape[0]))
	self.__setEW()
	self.__setT1n()
	self.__setT2n()
	self.__setT3n()
	self.__setUn()	# self.Un contains now all 3x3 matrices in all self.Nd domains
	# do the martix multiplication
	for i in range(self.Un.shape[2]):
	    if not i:
		U = self.Un[:,:,i]
	    else:
		U = np.dot(U,self.Un[:,:,i])	# first matrix on the left
	return U

"""
Class for the calculation of photon-ALPs conversion in AGN jets

History:
- 11/13/13: created
"""
__version__=0.01
__author__="M. Meyer // manuel.meyer@fysik.su.se"


import numpy as np
from math import ceil,floor
from eblstud.misc.constants import *
import logging
import warnings
from numpy.random import rand, seed

# --- Conversion without absorption, designed to match values in Clusters -------------------------------------------#
from deltas import *
class PhotALPs_Jet(object):
    """
    Class for photon ALP conversion in AGN jets 

    Attributes
    ----------
    B:		field strength in jet at r = R_BLR in G
    R_BLR:	Radius of broad line region in pc
    r:		distance where mixing is evaluated in pc
    s:		Exponent for electron density (between 1 and 3)
    p:		Exponent for magnetic field (1=toroidal, 2=poloidal)
    g:		Photon ALP coupling in 10^{-11} GeV^-1
    m:		ALP mass in neV
    E:		energy in GeV
    n:		electron density in the jet at r = R_BLR, in cm^-3
    T1:		Transfer matrix 1 (3x3xNd)-matrix
    T2:		Transfer matrix 2 (3x3xNd)-matrix		
    T3:		Transfer matrix 3 (3x3xNd)-matrix
    Un:		Total transfer matrix in all domains (3x3xNd)-matrix
    Dperp:	Mixing matrix parameter Delta_perpedicular 
    Dpar:	Mixing matrix parameter Delta_{||} 
    Dag:	Mixing matrix parameter Delta_{a\gamma} 
    Da:		Mixing matrix parameter Delta_{a} 
    alph:	Mixing angle
    Dosc:	Oscillation Delta

    EW1: 	Eigenvalue 1 of mixing matrix
    EW2:	Eigenvalue 2 of mixing matrix
    EW3:	Eigenvalue 3 of mixing matrix

    Notes
    -----
    For Photon - ALP mixing in AGN jet see e.g. Tavecchio et al. 2012 and Mena & Razzaque 2013:
    http://adsabs.harvard.edu/abs/2011PhRvD..84j5030D
    http://adsabs.harvard.edu/abs/2012PhRvD..86g5024H
    """
    def __init__(self, **kwargs):
	"""
	init photon axion conversion in intracluster medium

	Parameters
	----------
	None

	kwargs	
	------
	Rmax:		distance up to which jet extends, in pc.
	B:		field strength at r = R_BLR in G, default: 0.1 G
	E:		Energy in GeV for mixing is evaluated, default: 1 TeV.
	R_BLR:		Distance of broad line region (BLR) to centrum in pc, default: 0.3 pc
	g:		Photon ALP coupling in 10^{-11} GeV^-1, default: 1.
	m:		ALP mass in neV, default: 1.
	n:		electron density in the jet at r = R_BLR, in cm^-3, default: 1e9
	s:		exponent for scaling of electron density, default: 2.
	p:		exponent for scaling of magneitc field, default: 1.
	sens:		scalar < 1., sets the number of domains, for the B field in the n-th domain, it will have changed by B_n = sens * B_{n-1}
	Psi:		scalar, angle between B field and transversal photon polarization, default: 0.

	Returns
	-------
	Nothing.

	Notes
	-----

	The magnetic field is modeled according to 
	    B(r) = B * (r / R_BLR) ** -p
	p = 1: toroidal field
	p = 2: poloidal field
	See e.g. Rees (1987).
	The electron density is modeled according to 
	    n(r) = n * (r / R_BLR) ** -s
	"""

# --- Set the defaults
	kwargs.setdefault('R_BLR',0.3)
	kwargs.setdefault('E',1.)
	kwargs.setdefault('g',1.)
	kwargs.setdefault('m',1.)
	kwargs.setdefault('njet',1e3)
	kwargs.setdefault('Rmax',1000.)
	kwargs.setdefault('Bjet',0.01)
	kwargs.setdefault('sjet',2.)
	kwargs.setdefault('pjet',1.)
	kwargs.setdefault('sens',0.99)
	kwargs.setdefault('Psi',0.)
# --------------------
	self.update_params_Jet(**kwargs)

	super(PhotALPs_Jet,self).__init__()

	return

    def update_params_Jet(self, **kwargs):
	"""Update all parameters with new values and initialize all matrices"""

	self.__dict__.update(kwargs)

	self.Bf		= lambda r: self.Bjet * (r / self.R_BLR) ** -self.pjet
	self.nf		= lambda r: self.njet * (r / self.R_BLR) ** -self.sjet

	self.Nd_jet	= ceil( -1. * self.pjet * np.log(self.Rmax/self.R_BLR) / np.log(self.sens) )	
	self.Lcoh_jet	= self.R_BLR *  self.sens ** ( - np.linspace(1.,self.Nd_jet,self.Nd_jet) / self.pjet ) * (1. - self.sens) # domain length
	self.r		= self.R_BLR *  self.sens ** ( - np.linspace(0.,self.Nd_jet,self.Nd_jet) / self.pjet ) 		# distance from BLR



	self.Br = self.Bf(self.r)
	self.nr = self.nf(self.r)

	self.T1jet	= np.zeros((3,3,self.Nd_jet),np.complex)	# Transfer matrices
	self.T2jet	= np.zeros((3,3,self.Nd_jet),np.complex)
	self.T3jet	= np.zeros((3,3,self.Nd_jet),np.complex)
	self.Unjet	= np.zeros((3,3,self.Nd_jet),np.complex)

	self.Psin_jet	= np.ones(self.Nd_jet) * self.Psi
	return

    def __setDeltas_Jet(self):
	"""
	Set Deltas of mixing matrix for each domain in units of 1/pc
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""

	self.Dperp	= 1e-3 * (Delta_pl_kpc(self.nr * 1e3,self.E) + 2.*Delta_QED_kpc(self.Br * 1e6,self.E))
	self.Dpar	= 1e-3 * (Delta_pl_kpc(self.nr * 1e3,self.E) + 3.5*Delta_QED_kpc(self.Br * 1e6,self.E))	
	self.Dag	= 1e-3 * (Delta_ag_kpc(self.g,self.Br * 1e6))
	self.Da		= 1e-3 * (Delta_a_kpc(self.m,self.E) * np.ones(int(self.Nd_jet )))
	self.alph	= 0.5 * np.arctan(2. * self.Dag / (self.Dpar - self.Da)) 
	self.Dosc	= np.sqrt((self.Dpar - self.Da)**2. + 4.*self.Dag**2.)

	return

    def __setEW_Jet(self):
	"""
	Set Eigenvalues
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	# Eigen values are all self.Nd_jet-dimensional
	self.__setDeltas_Jet()
	self.EW1jet = self.Dperp
	self.EW2jet = 0.5 * (self.Dpar + self.Da - self.Dosc)
	self.EW3jet = 0.5 * (self.Dpar + self.Da + self.Dosc)
	return
	

    def __setT1n_Jet(self):
	"""
	Set T1 in all domains
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	c = np.cos(self.Psin_jet)
	s = np.sin(self.Psin_jet)
	self.T1jet[0,0,:]	= c*c
	#self.T1jet[0,1,:]	= -1. * c*s
	self.T1jet[1,0,:]	= self.T1jet[0,1]
	self.T1jet[1,1,:]	= s*s
	return

    def __setT2n_Jet(self):
	"""
	Set T2 in all domains
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	c = np.cos(self.Psin_jet)
	s = np.sin(self.Psin_jet)
	ca = np.cos(self.alph)
	sa = np.sin(self.alph)
	#self.T2jet[0,0,:] = s*s*sa*sa
	#self.T2jet[0,1,:] = s*c*sa*sa
	#self.T2jet[0,2,:] = -1. * s * sa *ca

	self.T2jet[1,0,:] = self.T2jet[0,1]
	self.T2jet[1,1,:] = c*c*sa*sa
	self.T2jet[1,2,:] = -1. * c *ca * sa

	self.T2jet[2,0,:] = self.T2jet[0,2]
	self.T2jet[2,1,:] = self.T2jet[1,2]
	self.T2jet[2,2,:] = ca * ca
	return

    def __setT3n_Jet(self):
	"""
	Set T3 in all domains
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	c = np.cos(self.Psin_jet)
	s = np.sin(self.Psin_jet)
	ca = np.cos(self.alph)
	sa = np.sin(self.alph)
	#self.T3jet[0,0,:] = s*s*ca*ca
	#self.T3jet[0,1,:] = s*c*ca*ca
	#self.T3jet[0,2,:] = s*sa*ca

	self.T3jet[1,0,:] = self.T3jet[0,1]
	self.T3jet[1,1,:] = c*c*ca*ca
	self.T3jet[1,2,:] = c * sa *ca

	self.T3jet[2,0,:] = self.T3jet[0,2]
	self.T3jet[2,1,:] = self.T3jet[1,2]
	self.T3jet[2,2,:] = sa*sa
	return

    def __setUn_Jet(self):
	"""
	Set Transfer Matrix Un in n-th domain
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	self.Unjet = np.exp(1.j * self.EW1jet * self.Lcoh_jet) * self.T1jet + \
	np.exp(1.j * self.EW2jet * self.Lcoh_jet) * self.T2jet + \
	np.exp(1.j * self.EW3jet * self.Lcoh_jet) * self.T3jet
	return

    def SetDomainN_Jet(self):
	"""
	Set Transfer matrix in all domains and multiply it

	Parameters
	---------- None (self only) 
	Returns
	-------
	Transfer matrix as 3x3 complex numpy array
	"""
	self.__setEW_Jet()
	self.__setT1n_Jet()
	self.__setT2n_Jet()
	self.__setT3n_Jet()
	self.__setUn_Jet()	# self.Un contains now all 3x3 matrices in all self.Nd_jet domains
	# do the martix multiplication
	for i in range(self.Unjet.shape[2]):
	    if not i:
		U = self.Unjet[:,:,i]
	    else:
		U = np.dot(U,self.Unjet[:,:,i])	# first matrix on the left
	return U

    def analytical_U(self):
	"""
	Calculate transfer matrix with analytical formula of Eq. (60) in Tavecchio (2012)

	Parameters
	---------- 
	None (self only) 

	Returns
	-------
	Transfer matrix as 3x3 complex numpy array
	"""
	U	= np.zeros((3,3),np.complex)
	U[0,0]	= 1.
	x	= (1e-3 * (Delta_ag_kpc(self.g,self.Bf(np.array([self.Rmax])) * 1e6)) * (self.Rmax / self.R_BLR) ** self.pjet * self.R_BLR * np.log(self.Rmax / self.R_BLR))[0]
	#x	= self.Dag * (self.r / self.R_BLR) ** self.pjet * self.R_BLR * np.log(self.r / self.R_BLR)
	U[1,1]  = np.cos(x)
	U[1,2]  = 1.j * np.sin(x)
	U[2,1]  = U[1,2]
	U[2,2]	= U[1,1]
	return U

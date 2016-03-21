
"""
Class for the calculation of photon-ALPs conversion in the BLR

History:
- 04/24/15: created
"""
__version__=0.01
__author__="M. Meyer // manuel.meyer@fysik.su.se"

# --- Imports --------------------------------------------#
import numpy as np
import logging
import warnings
from eblstud.blr.absorption import OptDepth_BLR
from eblstud.misc.constants import *
from math import ceil
from numpy.random import rand, seed
from deltas import *
# --------------------------------------------------------#

class PhotALPs_BLR(object):
    """
    Class for photon ALP conversion in galaxy clusters and the intra cluster medium (ICM) 

    Attributes
    ----------
    B_BLR:	field strength of transverse component of the cluster B-field, in G
    g:		Photon ALP coupling in 10^{-11} GeV^-1
    m:		ALP mass in neV
    n_BLR:	thermal electron density in the cluster, in cm^-3
    Nd_BLR:		number of domains, R_BLR/L_BLR
    Psin_BLR:	random angle in domain n between the transverse B field and the z-axis
    T1_BLR:		Transfer matrix 1 (3x3xNd)-matrix
    T2_BLR:		Transfer matrix 2 (3x3xNd)-matrix		
    T3_BLR:		Transfer matrix 3 (3x3xNd)-matrix
    Un_BLR:		Total transfer matrix in all domains (3x3xNd)-matrix
    Dperp_BLR:	Mixing matrix parameter Delta_perpedicular in n-th domain
    Dpar_BLR:	Mixing matrix parameter Delta_{||} in n-th domain
    Dag_BLR:	Mixing matrix parameter Delta_{a\gamma} in n-th domain
    Da_BLR:		Mixing matrix parameter Delta_{a} in n-th domain
    Dosc_BLR:	Oscillation Delta

    EW1_BLR: 	Eigenvalue 1 of mixing matrix
    EW2_BLR:	Eigenvalue 2 of mixing matrix
    EW3_BLR:	Eigenvalue 3 of mixing matrix

    Notes
    -----
    For Photon - ALP mixing theory see e.g. De Angelis et al. 2011 and also Horns et al. 2012
    http://adsabs.harvard.edu/abs/2011PhRvD..84j5030D
    http://adsabs.harvard.edu/abs/2012PhRvD..86g5024H
    """
    def __init__(self, **kwargs):
	"""
	init photon axion conversion in intracluster medium

	kwargs	
	------
	L_BLR:		float, coherence length / domain size of turbulent B-field in pc, 
			default: same as R_BLR, i.e. one domain only
	B_BLR:		float, Magnetic field in BLR in G, default: 0.2
	R_BLR:		float, distance of BLR to SMBH in pc, default: 0.3
	n_BLR:		float, thermal electron density in the cluster, in cm^-3, default: 1e5
	g:		float, Photon ALP coupling in 10^{-11} GeV^-1, default: 1.
	m:		float, ALP mass in neV, default: 1.
	z:		float, Redshift of the source, default: 0.1
	Elines: 	list, line energies in eV, used to calculate absorption, 
			default: [13.6,10.2,8.0,24.6,21.2]
	Nlines: 	list, column densities of each line in log10 (cm^2), 
			used to calculate absorption, 
			default: [24.,24.,24.,24.,24.]
	A:		bool, toggles absorption on or off, default: 1
	Returns
	-------
	Nothing.
	"""
# --- Set the defaults
        kwargs.setdefault('g',1)
	kwargs.setdefault('m',1.)
	kwargs.setdefault('z',0.1)
	kwargs.setdefault('B_BLR',0.2)
	kwargs.setdefault('n_BLR',1e5)
	kwargs.setdefault('R_BLR',0.3)
	kwargs.setdefault('L_BLR',kwargs['R_BLR'])
	kwargs.setdefault('A',1)
	kwargs.setdefault('Elines',np.array([13.6,10.2,8.0,24.6,21.2]))
	kwargs.setdefault('Nlines',np.array([24.,24.,24.,24.,24.]))
# --------------------
	self.update_params_BLR(**kwargs)
		
	super(PhotALPs_BLR,self).__init__()

	return
	

    def update_params_BLR(self, new_Bn_BLR = True, **kwargs):
	"""
	Update all parameters with new values and initialize all matrices

	kwargs
	------
	new_B_n:	boolean, if True, recalculate B field and electron density
	
	"""
	self.__dict__.update(kwargs)

	# number of domains, no expansion assumed
	self.Nd_BLR	= int(self.R_BLR / self.L_BLR)	
	if self.Nd_BLR == 0:
	    raise TypeError("Error: Coherence length greater than total distance.")

	self.r_BLR	= np.linspace(self.L_BLR, self.R_BLR + self.L_BLR, int(self.Nd_BLR))  

	if new_Bn_BLR:
	    self.new_B_n_BLR()

	self.T1_BLR	= np.zeros((3,3,self.Nd_BLR),np.complex)	# Transfer matrices
	self.T2_BLR	= np.zeros((3,3,self.Nd_BLR),np.complex)    
	self.T3_BLR	= np.zeros((3,3,self.Nd_BLR),np.complex)
	self.Un_BLR	= np.zeros((3,3,self.Nd_BLR),np.complex)

	# Optical depth class
	self.tt = OptDepth_BLR(Elines = self.Elines, Nlines = self.Nlines, z = self.z)  

	return

    def new_B_n_BLR(self): 
	"""
	Recalculate Bfield and density, 
	if Kolmogorov turbulence is set to true, 
	new random values for B and Psi are calculated.
	"""
	# assuming a constant electron density over all domains
	self.n_BLR		= self.n_BLR * np.ones(int(self.Nd_BLR))	
	# assuming a constant B-field over all domains
	self.B_BLR		= self.B_BLR * np.ones(int(self.Nd_BLR))	
	self.new_random_psi_BLR()

	return

    def new_random_psi_BLR(self):
	"""
	Calculate new random psi values

	Parameters:
	-----------
	None

	Returns:
	--------
	Nothing
	""" 							
        self.Psin_BLR	= 2. * np.pi * rand(1,int(self.Nd_BLR))[0]
        return

    def __setDeltas_BLR(self):
	"""
	Set Deltas of mixing matrix for each domain
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	
	# Changing units to agree with deltas.py
	self.n_BLR = self.n_BLR*1e3
	self.B_BLR = self.B_BLR*1e6
	self.R_BLR = self.R_BLR*1e-3 
	self.L_BLR = self.L_BLR*1e-3

	# np.arrays , self.Nd-dim
	self.Dperp_BLR	= Delta_pl_kpc(self.n_BLR,self.E) +\
	    2.*Delta_QED_kpc(self.B_BLR,self.E) +\
	    bool(self.A)*(self.tt(self.E).sum(axis = 0) / (2. * self.R_BLR)) * 1.j	

	# np.arrays , self.Nd-dim
	self.Dpar_BLR	= Delta_pl_kpc(self.n_BLR,self.E) + \
	    3.5*Delta_QED_kpc(self.B_BLR,self.E) +\
	    bool(self.A)*(self.tt(self.E).sum(axis = 0)/ (2. * self.R_BLR)) * 1.j 	

	# np.array, self.Nd-dim
	self.Dag_BLR	= Delta_ag_kpc(self.g,self.B_BLR)				
	# np.ones, to get np.array, self.Nd-dim
	self.Da_BLR	= Delta_a_kpc(self.m,self.E) * np.ones(int(self.Nd_BLR))											
	self.Dosc_BLR	= np.sqrt((self.Dpar_BLR - self.Da_BLR)**2. + 4.*self.Dag_BLR**2.)
	self.Ecrit_BLR	= Ecrit_GeV(self.m,self.n_BLR,self.B_BLR,self.g)
	self.Emax_BLR 	= Emax_GeV(self.B_BLR,self.g)

	# Changing back
	self.n_BLR = self.n_BLR*1e-3
	self.B_BLR = self.B_BLR*1e-6
	self.R_BLR = self.R_BLR*1e3 
	self.L_BLR = self.L_BLR*1e3

	return

    def __setEW_BLR(self):
	"""
	Set Eigenvalues
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	# Eigenvalues are all self.Nd-dimensional
	self.__setDeltas_BLR()
	self.EW1_BLR = self.Dperp_BLR
	self.EW2_BLR = 0.5 * (self.Dpar_BLR + self.Da_BLR - self.Dosc_BLR)
	self.EW3_BLR = 0.5 * (self.Dpar_BLR + self.Da_BLR + self.Dosc_BLR)
	return
	

    def __setT1n_BLR(self): 
	"""
	Set T1 in all domains
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	c = np.cos(self.Psin_BLR) 
	s = np.sin(self.Psin_BLR) 
	self.T1_BLR[0,0,:]	= c*c
	self.T1_BLR[0,1,:]	= -1. * c*s
	self.T1_BLR[1,0,:]	= self.T1_BLR[0,1,:]
	self.T1_BLR[1,1,:]	= s*s
	return

    def __setT2n_BLR(self):
	"""
	Set T2 in all domains
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	c = np.cos(self.Psin_BLR)
	s = np.sin(self.Psin_BLR)

	A = 0.5*(self.Da_BLR - self.Dpar_BLR + self.Dosc_BLR)/self.Dosc_BLR
	B = 0.5*(-self.Da_BLR + self.Dpar_BLR + self.Dosc_BLR)/self.Dosc_BLR
	C = self.Dag_BLR/self.Dosc_BLR

	self.T2_BLR[0,0,:] = s*s*A
	self.T2_BLR[0,1,:] = s*c*A
	self.T2_BLR[0,2,:] = -1.*s*C

	self.T2_BLR[1,0,:] = self.T2_BLR[0,1,:]
	self.T2_BLR[1,1,:] = c*c*A
	self.T2_BLR[1,2,:] = -1.*c*C

	self.T2_BLR[2,0,:] = self.T2_BLR[0,2,:]
	self.T2_BLR[2,1,:] = self.T2_BLR[1,2,:]
	self.T2_BLR[2,2,:] = B 
	return

    def __setT3n_BLR(self):
	"""
	Set T3 in all domains
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	c = np.cos(self.Psin_BLR)
	s = np.sin(self.Psin_BLR)

	A = 0.5*(self.Da_BLR - self.Dpar_BLR + self.Dosc_BLR)/self.Dosc_BLR
	B = 0.5*(-self.Da_BLR + self.Dpar_BLR + self.Dosc_BLR)/self.Dosc_BLR
	C = self.Dag_BLR/self.Dosc_BLR

	self.T3_BLR[0,0,:] = s*s*B
	self.T3_BLR[0,1,:] = s*c*B
	self.T3_BLR[0,2,:] = s*C

	self.T3_BLR[1,0,:] = self.T3_BLR[0,1,:]
	self.T3_BLR[1,1,:] = c*c*B
	self.T3_BLR[1,2,:] = c * C

	self.T3_BLR[2,0,:] = self.T3_BLR[0,2,:]
	self.T3_BLR[2,1,:] = self.T3_BLR[1,2,:]
	self.T3_BLR[2,2,:] = A
	return

    def __setUn_BLR(self):
	"""
	Set Transfer Matrix Un in n-th domain
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	self.L_BLR = self.L_BLR*1e-3 # Changing units to kpc 

	self.Un_BLR = np.exp(1.j * self.EW1_BLR * self.L_BLR) * self.T1_BLR + \
	np.exp(1.j * self.EW2_BLR * self.L_BLR) * self.T2_BLR + \
	np.exp(1.j * self.EW3_BLR * self.L_BLR) * self.T3_BLR
 
	self.L_BLR = self.L_BLR*1e3 # Changing back to pc

	return

    def SetDomainN_BLR(self):
	"""
	Set Transfer matrix in all domains and multiply it

	Parameters
	----------
	None (self only)

	Returns
	-------
	Transfer matrix as 3x3 complex numpy array
	"""
	if not self.Nd_BLR == self.Psin_BLR.shape[0]:
	    raise TypeError("Number of domains (={0:n}) is not equal to number of angles (={1:n})!".format(
	    self.Nd_BLR,self.Psin_BLR.shape[0])
	    )
	self.__setEW_BLR()
	self.__setT1n_BLR()
	self.__setT2n_BLR()
	self.__setT3n_BLR()
	self.__setUn_BLR()	# self.Un contains now all 3x3 matrices in all self.Nd domains
	# do the matrix multiplication
	for i in range(self.Un_BLR.shape[2]): 
	    if not i:                     
		U = self.Un_BLR[:,:,i]        
	    else:                         
		U = np.dot(U,self.Un_BLR[:,:,i])	# first matrix on the left
	return U

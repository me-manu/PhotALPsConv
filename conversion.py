"""
Class for the calculation of photon-ALPs conversion
in intergalactic magnetic field

History:
- 11/15/11: created
- 07/18/13: updated and cleaned up
"""
__version__=0.03
__author__="M. Meyer // manuel.meyer@physik.uni-hamburg.de"


import numpy as np
from math import ceil
import eblstud.ebl.tau_from_model as Tau
from eblstud.misc.constants import *
from numpy.random import rand, seed
import logging
import warnings
from deltas import *

def Tau_Fit(z,E):
    """
    Tau calculation by Giorgio Galanti
    Fit to Franceschini Model
    E in eV

    Parameters
    ----------
    z: float, redshift
    E: float, energy in eV

    Returns
    -------
    Optical depth from Franceschini et al. (2008)
    """
    a00=29072.8078002930
    a01=-12189.9033508301
    a02=2032.30382537842
    a03=-168.504407882690
    a04=6.95066644996405
    a05=-0.114138037664816

    E *= 1e12

    c_a=[1003.34072943900,1744.79443325556,-3950.79983395431,3095.04470168520]

    anomal=c_a[0]+c_a[1]*z+c_a[2]*z**2.+c_a[3]*z**3.

    power = np.log10(E/(0.999+z)**-0.6)
    return anomal*z*10.**(a00+a01*power+a02*power**2. \
	    +a03*power**3.+a04*power**4.+a05*power**5.)

class PhotALPs(object):
    """
    Class for photon ALP conversion in the ingtergalactic magnetic field (IGMF)

    Attributes
    ----------
    L0 = L0
    dz = redshift step

    tau = optical depth class
    E0 = initial energy
    Nd = number of domains
    z = maximum redshift

    Psin	= angle between IGMF and propagation direction in n-th domain
    dn		= delta parameter, see Notes
    EW1n	= Eigenvalue 1 of mixing matrix in n-th domain
    EW2n	= Eigenvalue 2 of mixing matrix in n-th domain
    EW3n	= Eigenvalue 3 of mixing matrix in n-th domain
    E0		= Energy in GeV at z = 0
    B0		= intergalactic magnetic field in nG at z = 0
    Dn		= sqrt(1 - 4* dn**2.), see notes
    T1		= Transfer matrix 1
    T2		= Transfer matrix 2
    T3		= Transfer matrix 3
    Un		= Total transfermatrix in n-th domain
    ebl_norm	= normalization of optical depth

    Notes
    -----
    For Photon - ALP mixing theory see e.g. De Angelis et al. 2011 
    http://adsabs.harvard.edu/abs/2011PhRvD..84j5030D
    """

    def __init__(self, **kwargs):
	"""
	Init photon-ALPs conversion class with 

	kwargs	
	------
	z: redshift of source
	L0: domain size at z=0 in Mpc, default: 5.
	B0: intergalactic magnetic field at z=0 in nG, default: 1
	g : photon-ALPs coupling strength in 10^-11 GeV^-1, default: 1
	model: EBL model to be used, default: 'kneiske'
	ebl_norm: additional normalization of optical depth, default: 1
	m : ALP mass in neV, default is 1. (only for energy dependent calculation)
	n0: electron density at z=0 in 10^7, default is 1. (only for energy dependent calculation)

	Returns
	-------
	Nothing
	"""
# --- Set the defaults 
	kwargs.setdefault('z',None)
	kwargs.setdefault('B0',1.)
	kwargs.setdefault('L0',5.)
	kwargs.setdefault('g',1.)
	kwargs.setdefault('m',1.)
	kwargs.setdefault('n0',1.)
	kwargs.setdefault('ebl','gilmore')
	kwargs.setdefault('ebl_norm',1.)
	kwargs.setdefault('filename','None')
# --------------------
	self.update_params_IGM(**kwargs) 

	super(PhotALPs,self).__init__()

	return

    def update_params_IGM(self, **kwargs):
	"""Update all parameters with new values and initialize all matrices"""

	for k in kwargs.keys():
	    if kwargs[k] == None:
		logging.error("kwarg {0:s} cannot be None type.".format(k))

	self.__dict__.update(kwargs)

	self.dz = 1.17e-3 * self.L0 / 5.

	self.tau = Tau.OptDepth()

	if self.filename == 'None':
	    self.tau.readfile(model = self.ebl)
	else:
	    self.tau.readfile(model = self.ebl, file_name = self.filename)

	self.E0 = 0.	# Energy in GeV

	self.Nd_IGM = int(ceil(0.85e3*5./self.L0*self.z))	# Number of domains 
								# see De Angelis et al. Eq 114
	self.dn		= 0.
	self.EW1n_IGM	= 0.
	self.EW2n_IGM	= 0.
	self.EW3n_IGM	= 0.
	self.Dn		= 0.
								# random realizations
	self.T1_IGM		= np.zeros((3,3,self.Nd_IGM),np.complex)
	self.T2_IGM		= np.zeros((3,3,self.Nd_IGM),np.complex)
	self.T3_IGM		= np.zeros((3,3,self.Nd_IGM),np.complex)
	self.Un_IGM		= np.zeros((3,3,self.Nd_IGM),np.complex)

	return 

    def new_random_psi_IGM(self):
	"""
	Calculate new random psi values

	Parameters:
	-----------
	None

	Returns:
	--------
	Nothing
	"""
	self.Psin_IGM	= 2. * np.pi * rand(1,int(self.Nd_IGM))[0]	# angle between photon propagation on B-field in i-th domain 
	return

#--- Energy dependent calculations -------------------------------------------------#
    def __SetT1n_IGM(self):
	"""
	Set T1 in all domains for energy dependent calculation (stron mixing regime not required)
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	s = np.sin(self.Psin_IGM)
	c = np.cos(self.Psin_IGM)
	self.T1_IGM[0,0,:] = c*c 
	self.T1_IGM[0,1,:] = -1.*s*c
	self.T1_IGM[1,0,:] = self.T1_IGM[0,1]
	self.T1_IGM[1,1,:] = s*s
	return

    def __SetT2n_IGM(self):
	"""
	Set T2 in all domains for energy dependent calculation (stron mixing regime not required)
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	s = np.sin(self.Psin_IGM)
	c = np.cos(self.Psin_IGM)
	self.T2_IGM[0,0,:] = 0.5* (self.delta_aa_n - self.delta_par_n - self.delta_abs_n + self.Dn) / self.Dn * s*s
	self.T2_IGM[0,1,:] = 0.5* (self.delta_aa_n - self.delta_par_n - self.delta_abs_n + self.Dn) / self.Dn * s*c
	self.T2_IGM[0,2,:] = -1. *  self.delta_ag_n / self.Dn * s
	self.T2_IGM[1,0,:] = self.T2_IGM[0,1]
	self.T2_IGM[2,0,:] = self.T2_IGM[0,2]

	self.T2_IGM[1,1,:] = 0.5* (self.delta_aa_n - self.delta_par_n - self.delta_abs_n + self.Dn) / self.Dn * c*c
	self.T2_IGM[1,2,:] = -1. * self.delta_ag_n / self.Dn * c
	self.T2_IGM[2,1,:] = self.T2_IGM[1,2]

	self.T2_IGM[2,2,:] = 0.5* (-1. * self.delta_aa_n + self.delta_par_n + self.delta_abs_n +self.Dn) / self.Dn
	return 

    def __SetT3n_IGM(self):
	"""
	Set T3 in all domains for energy dependent calculation (stron mixing regime not required)
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	s = np.sin(self.Psin_IGM)
	c = np.cos(self.Psin_IGM)

	self.T3_IGM[0,0,:] = 0.5* (-1. * self.delta_aa_n + self.delta_par_n + self.delta_abs_n +self.Dn) / self.Dn * s*s
	self.T3_IGM[0,1,:] = 0.5* (-1. * self.delta_aa_n + self.delta_par_n + self.delta_abs_n +self.Dn) / self.Dn * s*c
	self.T3_IGM[0,2,:] = self.delta_ag_n / self.Dn * s
	self.T3_IGM[1,0,:] = self.T3_IGM[0,1]
	self.T3_IGM[2,0,:] = self.T3_IGM[0,2]

	self.T3_IGM[1,1,:] = 0.5* (-1. * self.delta_aa_n + self.delta_par_n + self.delta_abs_n +self.Dn) / self.Dn * c*c
	self.T3_IGM[1,2,:] = self.delta_ag_n / self.Dn * c
	self.T3_IGM[2,1,:] = self.T3_IGM[1,2]

	self.T3_IGM[2,2,:] = 0.5 * (self.delta_aa_n - self.delta_par_n - self.delta_abs_n +self.Dn) / self.Dn

	return 

    def __SetUn_IGM(self):
	"""
	Set Transfer Matrix Un in all domains for energy dependent calculation (no stron mixing regime required)
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	self.Un_IGM = np.exp(1.j*self.EW1n_IGM* self.Ln) * self.T1_IGM \
	    + np.exp(1.j*self.EW2n_IGM* self.Ln) * self.T2_IGM \
	    + np.exp(1.j*self.EW3n_IGM* self.Ln) * self.T3_IGM 
	return

    def SetDomainN_IGM(self):
	"""
	Set domain length, energy, magnetic field, mean free path and delta in all domains 
	and calculate total transfer matrix, with energy dependence included (strong mixing regime not required)

	Parameters
	----------
	n:	Number of domain, 0 < n <= self.Nd_IGM

	Returns:
	--------
	U:	3x3 complex numpy array with total transfer matrix 
	"""

	ones	= np.ones(self.Nd_IGM)
	n	= np.array(range(1,self.Nd_IGM+1))
	En	= self.E0*(ones + (n-ones)*self.dz)		# Energy in all domains in GeV
	Bn	= self.B0*(ones + (n-ones)*self.dz)**2.		# B-field in all domains in nG
	neln	= self.n0*(ones + (n-ones)*self.dz)**3.		# electron density in all domains in 1e-7 cm^-3

	# calculate mean free path according to De Angelis et al. (2011) Eq. 131
	difftau	= (self.tau.opt_depth_array(n*self.dz , self.E0 / 1e3) - self.tau.opt_depth_array((n-ones)*self.dz , self.E0 / 1e3)).transpose()[0]
	difftau[difftau < 1e-20] = np.ones(difftau.shape[0])[difftau < 1e-20] * 1e-20	# set to 1e-20 if difference is smaller

	self.Ln	= 4.29e3*self.dz / (ones + 1.45*(n - ones)*self.dz)

	mfn	= self.Ln / difftau / self.ebl_norm 	# mean free path

	delta_pl_n		= Delta_pl_Mpc(neln,En / 1e3)
	delta_QED_n		= Delta_QED_Mpc(Bn,En / 1e3)
	self.delta_par_n	= delta_pl_n + 3.5 * delta_QED_n
	self.delta_perp_n	= delta_pl_n + 2.* delta_QED_n
	self.delta_aa_n		= Delta_a_Mpc(self.m * 10.,En / 1e3)
	self.delta_ag_n		= Delta_ag_Mpc(self.g,Bn)
	self.delta_abs_n	= 0.5j/mfn

### DEBUG
	#self.delta_par_n	= 0. * ones
	#self.delta_perp_n	= 0. * ones
	#self.delta_aa_n	= 0. * ones
###
	self.Dn	= np.sqrt((self.delta_aa_n - self.delta_par_n - self.delta_abs_n) ** 2. + 4.*self.delta_ag_n**2.)

	self.EW1n_IGM = self.delta_perp_n + self.delta_abs_n
	self.EW2n_IGM = 0.5 * (self.delta_aa_n + self.delta_par_n + self.delta_abs_n - self.Dn)
	self.EW3n_IGM = 0.5 * (self.delta_aa_n + self.delta_par_n + self.delta_abs_n + self.Dn)

	self.__SetT1n_IGM()
	self.__SetT2n_IGM()
	self.__SetT3n_IGM()

	self.__SetUn_IGM()
	for i in range(self.Nd_IGM):
	    if not i:
		U = self.Un_IGM[:,:,i]
	    else:
		U = np.dot(self.Un_IGM[:,:,i],U)
	return U

"""
Class for the calculation of photon-ALPs conversion

History:
- 11/15/11: created
- 05/08/12: adding reconversion in galactic magnetic field (GMF)
"""
__version__=0.02
__author__="M. Meyer // manuel.meyer@physik.uni-hamburg.de"


import numpy as np
from math import ceil
import eblstud.ebl.tau_from_model as Tau
from eblstud.ebl import mfn_model as MFN
from eblstud.misc.constants import *
from numpy.random import rand, seed
import logging
import warnings

def Tau_Giorgio(z,E):
    """
    Tau calculation by Giorgio Galanti
    Fit to Franceschini Model
    E in eV
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
    Ld = Ldom1
    xi = g * B
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
    E0		= Energy in TeV at z = 0
    B0		= intergalactic magnetic field in nG at z = 0
    Dn		= sqrt(1 - 4* dn**2.), see notes
    T1		= Transfer matrix 1
    T2		= Transfer matrix 2
    T3		= Transfer matrix 3
    Un		= Total transfermatrix in n-th domain

    Notes
    -----
    For Photon - ALP mixing theory see e.g. De Angelis et al. 2011 
    http://adsabs.harvard.edu/abs/2011PhRvD..84j5030D
    """

    def __init__(self, z, B0 = 1., Ldom1=5., g = 1., model='kneiske',filename='None'):
	"""
	Init photon-ALPs conversion class with 

	Parameters
	----------
	z: redshift of source
	Ldom1: domain size at z=0 in Mpc, default: 5.
	B0: intergalactic magnetic field at z=0 in nG, default: 1
	g : photon-ALPs coupling strength in 10^-11 GeV^-1, default: 1
	model: EBL model to be used, default: 'kneiske'

	Returns
	-------
	Nothing
	"""
	self.Ld = Ldom1
	self.B0 = B0
	self.xi = B0 * g
	self.dz = 1.17e-3 * self.Ld / 5.

	self.tau = Tau.OptDepth()
	#self.mfn = MFN.MFNModel(file_name='/home/manuel/projects/blazars/EBLmodelFiles/mfn_kneiske.dat.gz', model = 'kneiske')

	if filename == 'None':
	    self.tau.readfile(model = model)
	else:
	    self.tau.readfile(model = model, file_name = filename)

	self.E0 = 0.	# Energy in TeV

	self.z = z
	self.Nd = int(ceil(0.85e3*5./self.Ld*z))	# Number of domains 
							# see De Angelis et al. Eq 114

	self.dn		= 0.
	self.EW1n	= 0.
	self.EW2n	= 0.
	self.EW3n	= 0.
	self.Dn		= 0.
	self.Psin	= 2. * np.pi * rand(1,int(self.Nd))[0]	# angle between photon propagation on B-field in i-th domain 
								# random realizations
	self.T1		= np.zeros((3,3,self.Nd),np.complex)
	self.T2		= np.zeros((3,3,self.Nd),np.complex)
	self.T3		= np.zeros((3,3,self.Nd),np.complex)
	self.Un		= np.zeros((3,3,self.Nd),np.complex)

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

    def __SetT1n(self):
	"""
	Set T1 in all domains
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	s = np.sin(self.Psin)
	c = np.cos(self.Psin)
	self.T1[0,0,:] = c*c 
	self.T1[0,1,:] = -1.*s*c
	self.T1[1,0,:] = self.T1[0,1]
	self.T1[1,1,:] = s*s
	return

    def __SetT2n(self):
	"""
	Set T2 in all domains
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	s = np.sin(self.Psin)
	c = np.cos(self.Psin)
	ones = np.ones(self.Nd)
	self.T2[0,0,:] = 0.5* (ones + self.Dn) / self.Dn * s*s
	self.T2[0,1,:] = 0.5* (ones + self.Dn) / self.Dn * s*c
	self.T2[0,2,:] = -1.j * self.dn/self.Dn * s
	self.T2[1,0,:] = self.T2[0,1]
	self.T2[2,0,:] = self.T2[0,2]

	self.T2[1,1,:] = 0.5* (ones + self.Dn) / self.Dn * c*c
	self.T2[1,2,:] = -1.j * self.dn/self.Dn * c
	self.T2[2,1,:] = self.T2[1,2]

	self.T2[2,2,:] = 0.5* ( -1.*ones + self.Dn ) / self.Dn
	return 

    def __SetT3n(self):
	"""
	Set T3 in all domains
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	s = np.sin(self.Psin)
	c = np.cos(self.Psin)
	ones = np.ones(self.Nd)

	self.T3[0,0,:] = 0.5* (-1.*ones + self.Dn) / self.Dn * s*s
	self.T3[0,1,:] = 0.5* (-1.*ones + self.Dn) / self.Dn * s*c
	self.T3[0,2,:] = 1.j * self.dn/self.Dn * s
	self.T3[1,0,:] = self.T3[0,1]
	self.T3[2,0,:] = self.T3[0,2]

	self.T3[1,1,:] = 0.5* (-1.*ones + self.Dn) / self.Dn * c*c
	self.T3[1,2,:] = 1.j * self.dn/self.Dn * c
	self.T3[2,1,:] = self.T3[1,2]

	self.T3[2,2,:] = 0.5* ( 1.*ones + self.Dn ) / self.Dn
	return 

    def __SetUn(self):
	"""
	Set Transfer Matrix Un in all domains
	
	Parameters
	----------
	None (self only)

	Returns
	-------
	Nothing
	"""
	self.Un = np.exp(self.EW1n* self.Ln) * self.T1 \
	    + np.exp(self.EW2n* self.Ln) * self.T2 \
	    + np.exp(self.EW3n* self.Ln) * self.T3 
	return

    def SetDomainN(self):
	"""
	Set domain length, energy, magnetic field, mean free path and delta in all domains 
	and calculate total transfer matrix

	Parameters
	----------
	n:	Number of domain, 0 < n <= self.Nd

	Returns:
	--------
	U:	3x3 complex numpy array with total transfer matrix 
	"""

	ones	= np.ones(self.Nd)
	n	= np.array(range(1,self.Nd+1))
	En	= self.E0*(ones + (n-ones)*self.dz)
	Bn	= self.B0*(ones + (n-ones)*self.dz)**2.

	# calculate mean free path according to De Angelis et al. Eq 131
	difftau	= (self.tau.opt_depth_array(n*self.dz , self.E0) - self.tau.opt_depth_array((n-ones)*self.dz , self.E0)).transpose()[0]
	difftau[difftau < 1e-20] = np.ones(len(difftau < 1e-20)) * 1e-20	# set to 1e-20 if difference is smaller

	self.Ln	= 4.29e3*self.dz / (ones + 1.45*(n - ones)*self.dz)

	mfn	= self.Ln / difftau 

	#mfn = self.mfn.get_mfn(n*self.dz , self.E0)*100./Mpc2cm
	# What's right? Alessandro (1) or Cssaki (2) et al.? 
	# The (1 + z)**2 factor comes from the B-field scaling
	#self.dn = 3.04e-2*self.xi*mfn*(1. + (n-1.)*self.dz)**2.

	self.dn	= 0.11*self.xi*mfn*(ones + (n-ones)*self.dz)**2.
	self.Dn	= ones - 4.*self.dn**2. + 0.j*ones
	self.Dn	= np.sqrt(self.Dn)
	self.EW1n = -0.5 / mfn 
	self.EW2n = -0.25 / mfn * ( ones + self.Dn)
	self.EW3n = -0.25/ mfn * ( ones - self.Dn)

	self.__SetT1n()
	self.__SetT2n()
	self.__SetT3n()

	self.__SetUn()
	for i in range(self.Nd):
	    if not i:
		U = self.Un[:,:,i]
	    else:
		U = np.dot(self.Un[:,:,i],U)
	return U

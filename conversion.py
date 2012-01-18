"""
Class for the calculation of photon-ALPs conversion

History:
- 11/15/11: created
"""
__version__=0.01
__author__="M. Meyer // manuel.meyer@physik.uni-hamburg.de"


import numpy as np
from math import ceil
import eblstud.ebl.tau_from_model as Tau
from eblstud.ebl import mfn_model as MFN
from eblstud.misc.constants import *

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

class PhotALPs:
    def __init__(self, Ldom1=5., xi= 1., model='kneiske',filename='None'):
	"""
	Init photon-ALPs conversion class with 
	Ldom1: domain size at z=0 in Mpc (default: 5)
	xi: intergalactic magnetic field at z=0 in nG times photon-ALPs coupling strength in 10^-11 GeV^-1 (defautl: 1)
	model: EBL model to be used
	"""
	self.Ld = Ldom1
	self.xi = xi
	self.dz = 1.17e-3 * self.Ld / 5.

	self.tau = Tau.OptDepth()
	self.mfn = MFN.MFNModel(file_name='/home/manuel/projects/blazars/EBLmodelFiles/mfn_kneiske.dat.gz', model = 'kneiske')

	if filename == 'None':
	    self.tau.readfile(model = model)
	else:
	    self.tau.readfile(model = model, file_name = filename)

	self.E0 = 0.
	self.z = 0.
	self.Nd = 0

	self.mfn_print	= 0.
	self.dtau	= 0.
	self.Psin	= 0.
	self.dn		= 0.
	self.EW1n	= 0.
	self.EW2n	= 0.
	self.EW3n	= 0.
	self.Dn		= 0.
	self.T1		= np.zeros((3,3),np.complex)
	self.T2		= np.zeros((3,3),np.complex)
	self.T3		= np.zeros((3,3),np.complex)
	self.Un		= np.zeros((3,3),np.complex)

    def readz(self,z):
	self.z = z
	self.Nd = int(ceil(0.85e3*5./self.Ld*z))
	return 

    def SetT1n(self):
	s = np.sin(self.Psin)
	c = np.cos(self.Psin)
	self.T1[0,0] = c*c 
	self.T1[0,1] = -1.*s*c
	self.T1[1,0] = self.T1[0,1]
	self.T1[1,1] = s*s
	return

    def SetT2n(self):
	s = np.sin(self.Psin)
	c = np.cos(self.Psin)
	self.T2[0,0] = 0.5* (1. + self.Dn) / self.Dn * s*s
	self.T2[0,1] = 0.5* (1. + self.Dn) / self.Dn * s*c
	#self.T2[0,2] = -1. * self.dn/self.Dn * s
	self.T2[0,2] = -1.j * self.dn/self.Dn * s
	self.T2[1,0] = self.T2[0,1]
	#self.T2[2,0] = -1. * self.T2[0,2]
	self.T2[2,0] = self.T2[0,2]

	self.T2[1,1] = 0.5* (1. + self.Dn) / self.Dn * c*c
	#self.T2[1,2] = -1. * self.dn/self.Dn * c
	self.T2[1,2] = -1.j * self.dn/self.Dn * c
	#self.T2[2,1] = -1. * self.T2[1,2]
	self.T2[2,1] = self.T2[1,2]

	self.T2[2,2] = 0.5* ( -1. + self.Dn ) / self.Dn
	return 

    def SetT3n(self):
	s = np.sin(self.Psin)
	c = np.cos(self.Psin)
	self.T3[0,0] = 0.5* (-1. + self.Dn) / self.Dn * s*s
	self.T3[0,1] = 0.5* (-1. + self.Dn) / self.Dn * s*c
	#self.T3[0,2] = 1. * self.dn/self.Dn * s
	self.T3[0,2] = 1.j * self.dn/self.Dn * s
	self.T3[1,0] = self.T3[0,1]
	#self.T3[2,0] = -1. * self.T3[0,2]
	self.T3[2,0] = self.T3[0,2]

	self.T3[1,1] = 0.5* (-1. + self.Dn) / self.Dn * c*c
	#self.T3[1,2] = 1. * self.dn/self.Dn * c
	self.T3[1,2] = 1.j * self.dn/self.Dn * c
	#self.T3[2,1] = -1. * self.T3[1,2]
	self.T3[2,1] = self.T3[1,2]

	self.T3[2,2] = 0.5* ( 1. + self.Dn ) / self.Dn
	return 

    def SetUn(self):
	self.Un = np.exp(self.EW1n* self.Ln) * self.T1 \
	    + np.exp(self.EW2n* self.Ln) * self.T2 \
	    + np.exp(self.EW3n* self.Ln) * self.T3 
	return

    def SetDomainN(self,n):
	"""
	Set domain length, energy, magnetic field, mean free path and delta to n-th domain
	"""

	En	= self.E0*(1. + (n-1.)*self.dz)
	#Bn	= self.B0*(1. + (n-1.)*self.dz)**2.
	difftau	=self.tau.opt_depth(n*self.dz , self.E0) - self.tau.opt_depth((n-1.)*self.dz , self.E0)
	#difftau	=Tau_Giorgio(n*self.dz , self.E0) - Tau_Giorgio((n-1.)*self.dz , self.E0)
	self.Ln	= 4.29e3*self.dz / (1. + 1.45*(n - 1.)*self.dz)
	if difftau:
	    mfn	= self.Ln / difftau 
	else:
	    raise ValueError("difftau is zero!")

	#mfn = self.mfn.get_mfn(n*self.dz , self.E0)*100./Mpc2cm
	self.mfn_print = mfn
	self.dtau= difftau
	# What's right? Alessandro (1) or Cssaki (2) et al.? 
	# The (1 + z)**2 factor comes from the B-field scaling
	#self.dn = 3.04e-2*self.xi*mfn*(1. + (n-1.)*self.dz)**2.
	self.dn = 0.11*self.xi*mfn*(1. + (n-1.)*self.dz)**2.
	self.Dn	= 1. - 4.*self.dn**2. + 0.j
	self.Dn = np.sqrt(self.Dn)
	self.EW1n = -0.5 / mfn 
	self.EW2n = -0.25 / mfn * ( 1. + self.Dn)
	self.EW3n = -0.25/ mfn * ( 1. - self.Dn)

	self.SetT1n()
	self.SetT2n()
	self.SetT3n()

	self.SetUn()
	return self.Un

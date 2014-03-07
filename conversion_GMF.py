"""
Class for the calculation of photon-ALPs conversion in the galactic magnetic field (GMF)

History:
- 06/01/12: created
- 11/06/13: added the Pshirkov model
"""
__version__=0.02
__author__="M. Meyer // manuel.meyer@fysik.su.se"


import numpy as np
from math import ceil
import eblstud.ebl.tau_from_model as Tau
from eblstud.misc.constants import *
from PhotALPsConv.conversion_ICM import PhotALPs_ICM
import logging
import warnings
import pickle
import subprocess,os,glob
# --- Conversion in the galactic magnetic field ---------------------------------------------------------------------#
from gmf import gmf
from gmf.trafo import *
from kapteyn import wcs
from scipy.integrate import simps
from gmf.ne2001 import density_2001_los as dl

class PhotALPs_GMF(PhotALPs_ICM):
    """
    Class for conversion of ALP into photons in the regular component of the galactic magnetic field (GMF)

    Attributes
    ----------
    l: galactic longitude of source
    b: galactic latitude of source
    B: B GMF field instance (B-field in muG)
    g: ALP-photon coupling in 10^-11 GeV^-1
    m: ALP mass in 10^-9 eV
    n: electron density in 10^-3 cm^-3
    E: Energy in GeV
    d: postition of origin along x axis in GC coordinates
    NE2001: flag if NE2001 code is used to compute thermal enectron density
    """

    #def __init__(self, pol_t = 1./np.sqrt(2.) , pol_u = 1./np.sqrt(2.), g = 1., m = 1., n = 1.e4, 
		#galactic = -1., rho_max = 20., zmax = 50., d = -8.5,Lcoh = 0.01, NE2001 = False, model = 'jansson', model_sym = 'ASS'):
    def __init__(self, **kwargs):
	"""
	init GMF class

	kwargs	
	------
	pol_t: float (optional)
	    polarization of photon beam in one transverse direction
	    default: 0.5
	pol_u: float (optional)
	    polarization of photon beam in the other transverse direction
	    default: 0.5
	galactic: float (optional)
	    if -1: source is considered extragalactic. Otherwise provide distance from the sun in kpc
	    default: -1
	rho_max: float (optional)
	    maximal rho of GMF in kpc
	    default: 20 kpc
	zmax: float (optional)
	    maximal z of GMF in kpc
	    default: 50 kpc
	d : float (optional)
	    position of origin along x axis in GC coordinates in kpc
	    default is postion of the sun, i.e. d = -8.5kpc
	Lcoh : float (optional)
	    coherence length or step size for integration
	NE2001: bool (optional, default = False)
	    if True, NE2001 code is used to compute electron density instead of constant value
	NE2001file: string (optional, default = None)
	    file with electron density along line of sight. If None, it will be created.
	model: string (default = jansson)
	    GMF model that is used. Currently the model by Jansson & Farrar (2012) and Pshirkov et al. (2011) are implemented. 
	    Usage: model=[jansson, phsirkov]
	model_sym: string (default = ASS)
	    Only applies if pshirkov model is chosen: you can choose between the axis- and bisymmetric version by setting model_sym to ASS or BSS.

	Returns
	-------
	Nothing.
	"""


# --- set defaults 
	kwargs.setdefault('pol_t',0.5)
	kwargs.setdefault('pol_u',0.5)
	kwargs.setdefault('pol_a',0.)
	kwargs.setdefault('g',1.)
	kwargs.setdefault('m', 1.)
	kwargs.setdefault('nGMF',1.e4)
	kwargs.setdefault('galactic',-1.)
	kwargs.setdefault('rho_max',20.)
	kwargs.setdefault('zmax',50.)
	kwargs.setdefault('d',-8.5)
	kwargs.setdefault('Lcoh',0.01)
	kwargs.setdefault('NE2001', False)
	kwargs.setdefault('NE2001file',None)
	kwargs.setdefault('model','jansson')
	kwargs.setdefault('model_sym','ASS')

	self.update_params_GMF(**kwargs)

	# ALP parameters g,m,n: already provided in super call
	#super(PhotALPs_GMF,self).__init__(g = self.g, m = self.m, n = self.n, Lcoh = self.Lcoh)	#Inherit everything from PhotALPs_ICM
	super(PhotALPs_GMF,self).__init__(**kwargs)	#Inherit everything from PhotALPs_ICM

	return

    def update_params_GMF(self, **kwargs):
	"""Update all parameters with new values and initialize GMF model"""

	self.__dict__.update(**kwargs)			# form instance of kwargs

	if not (self.model == 'jansson' or self.model == 'pshirkov'):
	    raise ValueError('Unknown GMF model: {0}! Use pshirkov or jansson.'.format(self.model))
	if not (self.model_sym == 'ASS' or self.model_sym == 'BSS'):
	    raise ValueError('Unknown symmetry for GMF model: {0}! Use ASS or BSS.'.format(self.model))

	self.l		= 0.
	self.b		= 0.
	self.smax	= 0.
	self.E		= 0.	# Energy in GeV

	if self.model == 'jansson':
	    self.Bgmf = gmf.GMF()	# Initialize the Bfield class
	elif self.model == 'pshirkov':
	    self.Bgmf = gmf.GMF_Pshirkov(mode = self.model_sym)	

	return

    def __set_coordinates(self, ra, dec):
	"""
	set the coordinates l,b and the the maximum distance smax where |GMF| > 0

	Parameters
	----------
	ra: float
	    right ascension of source in degrees
	dec: float
	    declination of source in degrees

	Returns
	-------
	l: float
	    galactic longitude
	b: float
	    galactic latitude
	smax: float
	    maximum distance in kpc from sun considered here where |GMF| > 0
	"""

	# Transformation RA, DEC -> L,B
	tran = wcs.Transformation("EQ,fk5,J2000.0", "GAL")
	self.l,self.b = tran.transform((ra,dec)) # return galactic coordinates in degrees
	self.l *= np.pi / 180.	# transform to radian
	self.b *= np.pi / 180.	# transform to radian
	d = self.d

	if self.galactic < 0.:	# if source is extragalactic, calculate maximum distance that beam traverses GMF to Earth
	    cl = np.cos(self.l)
	    cb = np.cos(self.b)
	    sb = np.sin(self.b)
	    self.smax = np.amin([self.zmax/np.abs(sb),1./np.abs(cb) * (-d*cl + np.sqrt(d**2 + cl**2 - d**2*cb + self.rho_max**2))])
	else:
	    self.smax = self.galactic

	return self.l, self.b, self.smax

    def Bgmf_calc(self,s,l=0.,b=0.):
	"""
	compute GMF at (s,l,b) position where origin at self.d along x-axis in GC coordinates is assumed

	Parameters
	-----------
	s: N-dim np.array, distance from sun in kpc for all domains
	l (optional): galactic longitude, scalar or N-dim np.array
	b (optional): galactic latitude, scalar or N-dim np.array

	Returns
	-------
	2-dim tuple containing:
	    (3,N)-dim np.parray containing GMF for all domains in galactocentric cylindrical coordinates (rho, phi, z) 
	    N-dim np.array, field strength for all domains
	"""
	if np.isscalar(l):
	    if not l:
		l = self.l
	if np.isscalar(b):
	    if not b:
		b = self.b

	rho	= rho_HC2GC(s,l,b,self.d)	# compute rho in GC coordinates for s,l,b
	phi	= phi_HC2GC(s,l,b,self.d)	# compute phi in GC coordinates for s,l,b
	z	= z_HC2GC(s,l,b,self.d)		# compute z in GC coordinates for s,l,b

	B = self.Bgmf.Bdisk(rho,phi,z)[0] 	# add all field components
	B += self.Bgmf.Bhalo(rho,z)[0] 
	if self.model == 'jansson':
	    B += self.Bgmf.BX(rho,z)[0] 

	### Single components for debugging ###
	#B = self.Bgmf.Bdisk(rho,phi,z)[0] 	
	#B = self.Bgmf.Bhalo(rho,z)[0] 
	#B = self.Bgmf.BX(rho,z)[0] 

	Babs = np.sqrt(np.sum(B**2., axis = 0))	# compute overall field strength

	return B,Babs

    def Pag_TM(self, E, ra, dec, pol, pol_final = None):
	"""
	Compute the conversion probability using the Transfer matrix formalism

	Parameters
	----------
	E:		 	 float, Energy in GeV
	ra, dec:  		 float, float, coordinates of the source in degrees
	pol:			 np.array((3,3)): 3x3 matrix of the initial polarization
	Lcoh (optional):	 float, cell size
	pol_final (optional):	 np.array((3,3)): 3x3 matrix of the final polarization
				 if none, results for final polarization 
				 in t,u and ALPs direction are returned

	Returns
	-------
	Pag: float, photon ALPs conversion probability
	"""

	self.__set_coordinates(ra,dec)
	self.E	= E

	# first domain is that one farthest away from us, i.e. that on the edge of the milky way
	if int(self.smax/self.Lcoh) < 100:				# at least 100 domains
	    sa	= np.linspace(self.smax,0., 100,endpoint = False)	# divide distance into smax / Lcoh large cells
	    self.Lcoh = self.smax / 100.
	else:
	    sa	= np.linspace(self.smax,0., int(self.smax/self.Lcoh),endpoint = False)	# divide distance into smax / Lcoh large cells

	self.s = sa
	# --- Calculate B-field in all domains ---------------------------- #
	B,Babs	= self.Bgmf_calc(sa)
	Bs, Bt, Bu	= GC2HCproj(B, sa, self.l, self.b,self.d)	# Compute Bgmf and the projection to HC coordinates (s,b,l)
	
	self.B	= np.sqrt(Bt**2. + Bu**2.)	# Abs value of transverse component in all domains
	# Debug:
	#self.B	= np.sqrt(Bt**2.)		# Abs value of transverse component in all domains
	#self.B	= np.sqrt(Bu**2.)		# Abs value of transverse component in all domains
	# ----------------------------------------------------------------- #

	# --- Calculate Angle between B in prop direction in all domains -- #
	self.Nd		= sa.shape[0]
	self.Psin	= np.zeros(self.Nd)
	m		= self.B > 0.


	# Psi = arctan( B_transversal / B_along prop. direction)
	# Debug:
	#self.Psin[m]	= np.arctan2(self.B[m],Bs[m])	# arctan2 selects the right quadrant
	self.Psin[m]	= np.ones((self.Psin[m]).shape[0]) * np.pi / 4.

	self.T1		= np.zeros((3,3,self.Nd),np.complex)
	self.T2		= np.zeros((3,3,self.Nd),np.complex)
	self.T3		= np.zeros((3,3,self.Nd),np.complex)
	self.Un		= np.zeros((3,3,self.Nd),np.complex)
	# ----------------------------------------------------------------- #

	# --- Calculate density in all domains: ----------------------------#
	if self.NE2001:
	    if self.NE2001file == None:
		self.NE2001file = os.path.join(os.environ['NE2001_PATH'],'data/smax{0:.1f}_l{1:.2f}_b{2:.2f}_Lcoh{3}.pickle'.format(self.smax,self.l,self.b,self.Lcoh))
	    try:
		f = open(self.NE2001file)	# check if n has already been calculated for this l,b, smax and Lcoh
						# returns n in cm^-3
		self.n = pickle.load(f) *1e3	# convert into 1e-3 cm^-3
		f.close()
	    except IOError:			# if not already calculated, do it now and save to file with function dl
		self.n = dl(sa,self.l,self.b,self.NE2001file,d=self.d) * 1e3		# convert into 1e-3 cm^-3
	else:
	    self.n = self.nGMF * np.ones(self.Nd)
	# ----------------------------------------------------------------- #

	self.n[self.n == 0.] = 1e-4 * np.ones(np.sum(self.n == 0.))

#	for i,Bi in enumerate(self.B):
#	    logging.debug("B,Bt,Bu,Psi: {0:20.2f},{1:20.2f},{2:20.2f},{3:20.2f}".format(Bi,Bt[i],Bu[i],self.Psin[i]))


	U = super(PhotALPs_GMF,self).SetDomainN()		# calculate product of all transfer matrices

	if pol_final == None:
	    pol_t = np.zeros((3,3),np.complex)
	    pol_t[0,0] += 1.
	    pol_u = np.zeros((3,3),np.complex)
	    pol_u[1,1] += 1.
	    pol_unpol = 0.5*(pol_t + pol_u)
	    pol_a = np.zeros((3,3),np.complex)
	    pol_a[2,2] += 1.
	    Pt = np.sum(np.diag(np.dot(pol_t,np.dot(U,np.dot(pol,U.transpose().conjugate())))))	#Pt = Tr( pol_t U pol U^\dagger )
	    Pu = np.sum(np.diag(np.dot(pol_u,np.dot(U,np.dot(pol,U.transpose().conjugate())))))	#Pu = Tr( pol_u U pol U^\dagger )
	    Pa = np.sum(np.diag(np.dot(pol_a,np.dot(U,np.dot(pol,U.transpose().conjugate())))))	#Pa = Tr( pol_a U pol U^\dagger )
	    return Pt,Pu,Pa
	else:
	    return np.sum(np.diag(np.dot(pol_final,np.dot(U,np.dot(pol,U.transpose().conjugate())))))

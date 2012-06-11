"""
Class for the calculation of photon-ALPs conversion in the galactic magnetic field (GMF)

History:
- 06/01/12: created
"""
__version__=0.01
__author__="M. Meyer // manuel.meyer@physik.uni-hamburg.de"


import numpy as np
from math import ceil
import eblstud.ebl.tau_from_model as Tau
from eblstud.ebl import mfn_model as MFN
from eblstud.misc.constants import *
from PhotALPsConv.conversion_ICM import PhotALPs_ICM
import logging
import warnings
import pickle
# --- Conversion in the galactic magnetic field ---------------------------------------------------------------------#
from gmf import gmf
from gmf.trafo import *
from kapteyn import wcs
from scipy.integrate import simps
from gmf.ne2001 import density_2001_los as dl

pertubation = 0.10	# Limit for pertubation theory to be valid

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

    def __init__(self, pol_t = 1./np.sqrt(2.) , pol_u = 1./np.sqrt(2.), g = 1., m = 1., n = 1.e4, galactic = -1., rho_max = 20., zmax = 50., d = -8.5,Lcoh = 0.01, NE2001 = False):
	"""
	init GMF class

	Parameters
	----------
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
	    position of origin along x axis in GC coordinates
	    default is postion of the sun, i.e. d = -8.5kpc
	Lcoh : float (optional)
	    coherence length or step size for integration
	NE2001: bool (optional, default = False)
	    if True, NE2001 code is used to compute electron density instead of constant value

	Returns
	-------
	None.
	"""
	super(PhotALPs_GMF,self).__init__(g = g, m = m, n = n, Lcoh = Lcoh)	#Inherit everything from PhotALPs_ICM

	self.l		= 0.
	self.b		= 0.
	self.smax	= 0.
	self.pol_t	= pol_t
	self.pol_u	= pol_u
	self.rho_max	= rho_max
	self.zmax	= zmax
	self.galactic	= galactic
	self.d		= d
	self.NE2001	= NE2001


	self.E		= 0.	# Energy an GeV
	# ALP parameters: already provided in super call
	#self.g		= g
	#self.m		= m
	#self.n		= n
	# Initialize the Bfield the class
	self.Bgmf = gmf.GMF()

	return

    def set_coordinates(self, ra, dec):
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
	    maximum distance from sun considered here where |GMF| > 0
	"""

	# Transformation RA, DEC -> L,B
	tran = wcs.Transformation("EQ,fk5,J2000.0", "GAL")
	self.l,self.b = tran.transform((ra,dec))
	d = self.d

	if self.galactic < 0.:
	    cl = np.cos(self.l)
	    cb = np.cos(self.b)
	    sb = np.sin(self.b)
	    self.smax = np.amin([self.zmax/np.abs(sb),1./np.abs(cb) * (-d*cl + np.sqrt(d**2 + cl**2 - d**2*cb + self.rho_max**2))])
	    #logging.debug("l,b,cl,cb,sb,smax: {0:.3f},{1:.3f},{2:.3f},{3:.3f},{4:.3f},{5:.3f}".format(self.l,self.b,cl,cb,sb,self.smax))
	else:
	    self.smax = self.galactic

	return self.l, self.b, self.smax

    def Bgmf_calc(self,s,l = 0.,b = 0.):
	"""
	compute GMF at (s,l,b) position where origin at self.d along x-axis in GC coordinates is assumed

	Parameters
	-----------
	s: float, distance from sun in kpc
	l: galactic longitude
	b: galactic latitude

	Returns
	-------
	GMF at this positon in galactocentric cylindrical coordinates (rho, phi, z) and field strength
	"""
	if not l:
	    l = self.l
	if not b:
	    b = self.b

	rho	= rho_HC2GC(s,l,b,self.d)	# compute rho in GC coordinates for s,l,b
	phi	= phi_HC2GC(s,l,b,self.d)	# compute phi in GC coordinates for s,l,b
	z	= z_HC2GC(s,l,b,self.d)		# compute z in GC coordinates for s,l,b

	B = self.Bgmf.Bdisk(rho,phi,z)[0] 	# add all field components
	B += self.Bgmf.Bhalo(rho,z)[0] 
	B += self.Bgmf.BX(rho,z)[0] 

	# Single components for debugging
	#B = self.Bgmf.Bdisk(rho,phi,z)[0] 	# add all field components
	#B = self.Bgmf.Bhalo(rho,z)[0] 
	#B = self.Bgmf.BX(rho,z)[0] 

	Babs = np.sqrt(np.sum(B**2.))	# compute overall field strength

	#logging.debug('rho,phi,z,Babs, Bfield = {0:.2f},{1:.2f},{2:.2f},{3},{4}'.format(rho,phi,z,Babs,B))

	return B,Babs

    def integrate_los(self,ra = 0.,dec = 0.):
	"""
	compute the line of sight integral of ALPs - Conversion probability,
	I_{t/u} = | \int_0^{s_\mathrm{max}} \mathrm{d}s \Delta_{a\gamma}^{t/u}(s) \exp(i(\Delta_a(s) - \Delta_{|| / \perp}(s))s) |^2

	Parameters
	-----------
	None
	(l,b,smax and energy need to be set!)

	Returns
	-------
	Values of I_t and I_u

	Notes
	-----
	See Mirizzi, Raffelt & Serpico (2007) Eq. A23
	and Simet, Hooper and Serpico (2008) Eq. 1
	and my theory lab book p. 149f
	"""

	#sa	= np.linspace(0.,self.smax,100)
	sa	= np.linspace(0.,self.smax,int(self.smax/self.Lcoh))
	kernel_t = np.zeros(sa.shape[0],np.complex)
	kernel_u = np.zeros(sa.shape[0],np.complex)
	#logging.debug("smax,l,b = {0},{1},{2}".format(self.smax,self.l,self.b))
	for i,s in enumerate(sa):

	    if self.NE2001:
		n	= density_2001(x_HC2GC(s,self.l,self.b,self.d),y_HC2GC(s,self.l,self.b,self.d),z_HC2GC(s,self.l,self.b,self.d))
		if n >= 0:	# computation succeeded
		    self.n = n
	    B,Babs	= self.Bgmf_calc(s)
	    Bs, Bt, Bu	= GC2HCproj(B, s, self.l, self.b,self.d)	# Compute Bgmf and the projection to HC coordinates (s,b,l)
	    sb,tb,ub	= HC_base(self.l,self.b)
	    Btrans	= Bt * tb + Bu * ub				# transverse Component
	    Btrans_abs	= np.sqrt(Bt**2. + Bu**2.)			# absolute value of transverse Component, needed for Delta_QED

#	    if Btrans_abs:
#		self.Psin	= np.arccos(Bt/Btrans_abs)				# angle between B and propagation direction, cos(Psi) = < B,s > / |B||s|
#		if Bu < 0.:
#		    self.Psin = 2.*np.pi - self.Psin
#	    else:
#		self.Psin = 0.
	    #logging.debug("Integrate Psi: {0:.5f}, Bu: {1:.5f}".format(self.Psin, Bu))
	    #logging.debug("Integrate: B cos(phi),B sin(phi), Bt, Bu: {0:.3f},{1:.3f},{2:.3f},{3:.3f}".format(Btrans_abs * np.cos(self.Psin), Btrans_abs * np.sin(self.Psin), Bt, Bu))

#	    Bu = np.sin(self.Psin) * Btrans_abs	# doesn't matter results are unchanged

	    #logging.debug("s,Bt,Bu,Btrans: {0:.3f},{1:.3f},{2:.3f},{3:.3f}".format(s,Bt,Bu,Btrans_abs))

	    # Compute the Delta factors
	    #Delta_ag_t	= Delta_ag_kpc(self.g,np.abs(Bt))
	    #Delta_ag_u	= Delta_ag_kpc(self.g,np.abs(Bu))
	    Delta_ag_t	= Delta_ag_kpc(self.g,Bt)
	    Delta_ag_u	= Delta_ag_kpc(self.g,Bu)


	    if Delta_ag_t > pertubation:
		warnings.warn("Warning: large Delta_ag_t = {0:.5f} detected in Integration at s = {1}".format(Delta_ag_t,s),RuntimeWarning)
	    if Delta_ag_u > pertubation:
		warnings.warn("Warning: large Delta_ag_u = {0:.5f} detected in Integration at s = {1}".format(Delta_ag_u,s),RuntimeWarning)


	    Delta_a	= Delta_a_kpc(self.m,self.E)
	    Delta_perp	= Delta_pl_kpc(self.n,self.E) + 2.*Delta_QED_kpc(Btrans_abs,self.E)	# perp goes together with t component
	    Delta_par 	= Delta_pl_kpc(self.n,self.E) + 3.5*Delta_QED_kpc(Btrans_abs,self.E)	# par goes together with u component

	    #logging.debug("Delta_a, Delta_perp, Delta_par: {0:.5f},{1:.5f},{2:.5f}".format(Delta_a, Delta_perp, Delta_par))
	    #logging.debug("Delta t,u : {0:.5f},{1:.5f}".format(Delta_ag_t, Delta_ag_u))

	    kernel_t[i] = Delta_ag_t*np.exp(1.j*s*(Delta_a - Delta_perp))	# Compute the kernel for the t polarization 
	    kernel_u[i] = Delta_ag_u*np.exp(1.j*s*(Delta_a - Delta_par))	# Compute the kernel for the u polarization

	    #logging.debug("s, kernet_t , kernel_u : {0:.3f} {1:.3f} {2:.3f}".format(s,kernel_t[i],kernel_u[i]))
	    #logging.debug("s, Delta_a - Delta_perp : {0} {1}".format(s,Delta_a - Delta_perp))
	    #logging.debug("exp t : {0}".format(np.exp(1.j*s*(Delta_a - Delta_perp))))

	m_t = kernel_t*np.conjugate(kernel_t) > 1e-20
	m_u = kernel_u*np.conjugate(kernel_u) > 1e-20
	
	#logging.debug('kernel t,u: {0}, {1}'.format(kernel_t[m_t], kernel_u[m_u]))

	if np.sum(m_t):
	    I_t = simps(kernel_t[m_t].real,sa[m_t]) + 1.j * simps(kernel_t[m_t].imag,sa[m_t])
	else:
	    I_t = 0.
	if np.sum(m_u):
	    I_u = simps(kernel_u[m_u].real,sa[m_u]) + 1.j * simps(kernel_u[m_u].imag,sa[m_u])
	else:
	    I_u = 0.

	#assert (I_t * np.conjugate(I_t)) == I_t.real**2. + I_t.imag**2.
	#logging.debug("I t , |I t|^2,u : {0}, {1}".format(I_t,I_t.real**2. + I_t.imag**2.))

	return (I_t * np.conjugate(I_t)).real, (I_u * np.conjugate(I_u)).real, (np.conjugate(I_u) * I_t).real

    def Pag(self, E, ra, dec, pol_t = -1 , pol_u = -1):
	"""
	compute the line of sight integral of ALPs - Conversion probability,
	I_{t/u} = | \int_0^{s_\mathrm{max}} \mathrm{d}s \Delta_{a\gamma}^{t/u}(s) \exp(i(\Delta_a(s) - \Delta_{|| / \perp}(s))s) |^2

	Parameters
	-----------
	E: float, Energy in GeV
	ra, dec: float, float, coordinates of the source in degrees
	pol_t: float, polarization of t direction (optional)
	pol_u: float, polarization of u direction (optional)

	Returns
	-------
	Pag: float, photon ALPs conversion probability

	Notes
	-----
	(1) See Mirizzi, Raffelt & Serpico (2007)
	(2) and Simet, Hooper and Serpico (2008)
	(3) and my theory lab book p. 149f
	"""
	self.set_coordinates(ra,dec)
	self.E		= E
	if pol_t > 0.:
	    self.pol_t	= pol_t
	if pol_u > 0.:
	    self.pol_u	= pol_u

	It, Iu, ItIu = self.integrate_los()

	# This is a hack - what to do if results largen than one? Pertubation theory not applicable?
	# I think my Integration
#	if It > 1:
#	    It = 1.
#	if Iu > 1:
#	    Iu = 1.
#	if ItIu > 1:
#	    ItIu = 1.
	    
	return self.pol_t ** 2. * It + self.pol_u ** 2. * Iu		# (1) - Eq. A23
	#return 2. * (pol_t ** 2. * It + pol_u ** 2. * Iu)	# (2) - Eq. 1
	#return (pol_t ** 2. + pol_u ** 2.) * (pol_t ** 2. * It + pol_u ** 2. * Iu + 2.*pol_u*pol_t*ItIu)	# (3) - p.149f
    def Pag_TM(self, E, ra, dec, pol, pol_final = None):
	"""
	Compute the conversion probability using the Transfer matrix formalism

	Parameters
	----------
	E: float, Energy in GeV
	ra, dec: float, float, coordinates of the source in degrees
	pol: np.array((3,3)): 3x3 matrix of the initial polarization
	Lcoh (optional): float, cell size
	pol_final (optional): np.array((3,3)): 3x3 matrix of the final polarization
	if none, results for final polarization in t,u and ALPs direction are returned

	Returns
	-------
	Pag: float, photon ALPs conversion probability

	Notes
	-----
	"""
	self.set_coordinates(ra,dec)
	self.E	= E
	# first domain is that one farthest away from us, i.e. that on the edge of the milky way
	#logging.debug("smax {0:.5f}".format(self.smax))
	if int(self.smax/self.Lcoh) < 100:	# at least 100 domains
	    sa	= np.linspace(self.smax,0., 100,endpoint = False)	# divide distance into smax / Lcoh large cells
	    self.Lcoh = self.smax / 100.
	else:
	    sa	= np.linspace(self.smax,0., int(self.smax/self.Lcoh),endpoint = False)	# divide distance into smax / Lcoh large cells

	U 	= np.diag(np.diag(np.ones((3,3), np.complex)))

	pol_t = np.zeros((3,3),np.complex)
	pol_t[0,0] += 1.
	pol_u = np.zeros((3,3),np.complex)
	pol_u[1,1] += 1.
	pol_unpol = 0.5*(pol_t + pol_u)
	pol_a = np.zeros((3,3),np.complex)
	pol_a[2,2] += 1.

	if self.NE2001:
	    filename = '/nfs/astrop/d6/meyerm/axion_cluster/data/NE2001/smax{0:.1f}_l{1:.1f}_b{2:.1f}_Lcoh{3}.pickle'.format(self.smax,self.l,self.b,self.Lcoh)
	    try:
		f = open(filename)
		# returns n in cm^-3
		n = pickle.load(f) *1e3		# convert into 1e-3 cm^-3
		f.close()
	    except IOError:
		# returns n in cm^-3
		n = dl(sa,self.l,self.b,filename,d=self.d) * 1e3		# convert into 1e-3 cm^-3

	for i,s in enumerate(sa):
	    if self.NE2001:
		self.n = n[i]
		    
	    B,Babs	= self.Bgmf_calc(s)
	    Bs, Bt, Bu	= GC2HCproj(B, s, self.l, self.b,self.d)	# Compute Bgmf and the projection to HC coordinates (s,b,l)
	    sb,tb,ub	= HC_base(self.l,self.b)
	    Btrans	= Bt * tb + Bu * ub				# transverse Component
	    self.B	= np.sqrt(Bt**2. + Bu**2.)
	    if self.B:
		self.Psin	= np.arccos(Bt/self.B)			# angle between B and propagation direction, cos(Psi) = < B,s > / |B||s|
		if Bu < 0.:
		    self.Psin = 2.*np.pi - self.Psin
	    else:
		self.Psin = 0.

	    #logging.debug("Psi: {0:.5f}, Bu: {1:.5f}".format(self.Psin, Bu))
	    #logging.debug("s,Bt,Bu,Btrans: {0:.3f},{1:.3f},{2:.3f},{3:.3f}".format(s,Bt,Bu,self.B))
	    #logging.debug("B cos(phi),B sin(phi): {0:.3f},{1:.3f}".format(self.B * np.cos(self.Psin), self.B * np.sin(self.Psin)))

	    #assert np.round(self.B* np.cos(self.Psin),5) == np.round(Bt,5)
	    #assert np.round(self.B* np.sin(self.Psin),5) == np.round(Bu,5)

	    U = np.dot(U,super(PhotALPs_GMF,self).SetDomainN())							# calculate product of all transfer matrices

	    #logging.debug("s,U : {0}--------------\n{1}\n{2}\n{3}".format(s,Un[0,:],Un[1,:],Un[2,:]))
	    

	if pol_final == None:
	    Pt = np.sum(np.diag(np.dot(pol_t,np.dot(U,np.dot(pol,U.transpose().conjugate())))))	#Pt = Tr( pol_t U pol U^\dagger )
	    Pu = np.sum(np.diag(np.dot(pol_u,np.dot(U,np.dot(pol,U.transpose().conjugate())))))	#Pu = Tr( pol_u U pol U^\dagger )
	    Pa = np.sum(np.diag(np.dot(pol_a,np.dot(U,np.dot(pol,U.transpose().conjugate())))))	#Pa = Tr( pol_a U pol U^\dagger )
	    return Pt,Pu,Pa
	else:
	    return np.sum(np.diag(np.dot(pol_final,np.dot(U,np.dot(pol,U.transpose().conjugate())))))

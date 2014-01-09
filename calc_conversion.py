"""
Module to wrap calculate conversion from photon to ALPs for different B field environments


History:
--------
- 01/05/2014: version 0.01 - created
"""

__author__ = "Manuel Meyer // manuel.meyer@fysik.su.se"
__version__ = 0.01

# --- Imports ------------ #
import numpy as np
import iminuit as minuit
import sys
import yaml
from math import floor
from eblstud.tools.lsq_fit import *
from scipy.integrate import simps
from scipy.interpolate import interp1d
import logging
# --- ALP imports 
import PhotALPsConv.conversion_Jet as JET
import PhotALPsConv.conversion as IGM 
import PhotALPsConv.conversion_ICM as ICM 
import PhotALPsConv.conversion_GMF as GMF 
# --- EBL imports
import eblstud.ebl.tau_from_model as TAU
from eblstud.misc.bin_energies import calc_bin_bounds
from eblstud.tools.iminuit_fit import pl,lp
# ------------------------ #

class Calc_Conv(IGM.PhotALPs,JET.PhotALPs_Jet,GMF.PhotALPs_GMF):
    """
    Class to wrap the calculation for photons to ALPs.
    """
    def __init__(self, **kwargs):
	"""
	Init class to calculate conversion from photons to ALPs

	kwargs
	------
	scenario:	string or list of strings with the Bfield environments that are used. Possibilities are:
			Jet:	Mixing in the AGN jet
			ICM:	Mixing in a galaxy cluster
			IGM:	Mixing in the intergalactic magnetic field
			GMF:	Mixing in the Galactic magnetic field
			default: ['ICM','GMF']

	config:		string, path to config yaml file. If none, use kwargs.
	z:		float, redshift of the source
	ra:		float, right ascension of the source, in degree
	dec:		float, declination of the source, in degree

	g:		Photon ALP coupling in 10^{-11} GeV^-1, default: 1.
	m:		ALP mass in neV, default: 1.

	Rmax:		distance up to which jet extends, in pc.
	Bjet:		field strength at r = R_BLR in G, default: 0.1 G
	R_BLR:		Distance of broad line region (BLR) to centrum in pc, default: 0.3 pc
	njet:		electron density in the jet at r = R_BLR, in cm^-3, default: 1e3
	s:		exponent for scaling of electron density, default: 2.
	p:		exponent for scaling of magneitc field, default: 1.
	sens:		scalar < 1., sets the number of domains, for the B field in the n-th domain, 
			it will have changed by B_n = sens * B_{n-1}
	Psi:		scalar, angle between B field and transversal photon polarization in the jet, default: 0.

	model:		GMF model that is used. Currently available: pshirkov (ASS), jansson (default)
	ebl:		EBL model that is used. defaut: gilmore
	NE2001:		bool, if true, use ne2001 code to calculate electron density in the Milky Way
			default: True

	pol_t: 		float, initial photon polarization
	pol_u: 		float, initial photon polarization
	pol_a: 		float, initial ALP polarization

	Notes
	-----
	pol_t + pol_u + pol_a != 1

	"""
	# --- set defaults
	kwargs.setdefault('config','None')
	kwargs.setdefault('scenario',['ICM','GMF'])
	# -----------------

	if not kwargs['config'] == 'None':
	    kwargs = yaml.load(open(kwargs['config']))

	super(Calc_Conv,self).__init__(**kwargs)	# init the jet mixing and gmf mixing, see e.g.
							# for a small example
	self.update_params_all(**kwargs)

	# --- init random angles
	try:
	    self.scenario.index('IGM')
	    self.new_random_psi_IGM()
	except ValueError:
	    pass
	try:
	    self.scenario.index('ICM')
	    self.new_random_psi()
	except ValueError:
	    pass

	return

    def update_params_all(self,new_init=True,**kwargs):
	"""
	Update all parameters and initial all matrices.

	kwargs
	------
	new_init:	bool, if True, re-init all polarization matrices
	"""

	self.__dict__.update(kwargs)			# form instance of kwargs

	# --- init params
	try:
	    self.scenario.index('IGM')
	    self.update_params_IGM(**kwargs)
	except ValueError:
	    pass
	try:
	    self.scenario.index('ICM')
	    self.update_params(**kwargs)
	except ValueError:
	    pass
	try:
	    self.scenario.index('Jet')
	    self.update_params_Jet(**kwargs)
	except ValueError:
	    pass
	try:
	    self.scenario.index('GMF')
	    self.update_params_GMF(**kwargs)
	except ValueError:
	    pass

	# --- init initial and final polarization states ------ #
	if new_init:
	    self.pol	= np.zeros((3,3))	# initial polarization state
	    self.pol[0,0]	= self.pol_t
	    self.pol[1,1]	= self.pol_u
	    self.pol[2,2]	= self.pol_a

	    self.polt	= np.zeros((3,3))	
	    self.polu	= np.zeros((3,3))
	    self.pola	= np.zeros((3,3))

	    self.polt[0,0]	= 1.
	    self.polu[1,1]	= 1.
	    self.pola[2,2]	= 1.

	self.kwargs = kwargs	# save kwargs

	return

    def calc_conversion(self,EGeV,new_angles = True):
	"""
	Calculate conversion probailities for energies EGeV

	Paramaters
	----------
	EGeV:	n-dim array, energies in GeV

	kwargs
	------
	new_angles:	bool, if True, calculate new random angles. Default: True

	Returns
	-------
	tuple with conversion probabilities in t,u, and a polarization
	"""

	Pt,Pu,Pa	= np.ones(EGeV.shape[0]) * 1e-40,np.ones(EGeV.shape[0]) * 1e-40,np.ones(EGeV.shape[0]) * 1e-40
	# --- calculate new random angles
	try:
	    self.scenario.index('IGM')
	    if new_angles:
		self.new_random_psi_IGM()
	except ValueError:
	    pass
	try:
	    self.scenario.index('ICM')
	    if new_angles:
		self.new_random_psi()
	    Psin	= self.Psin		# save values of Psi and Nd, altered by GMF calculation
	except ValueError:
	    pass

	# --- calculate transfer matrix for every energy
	for i,E in enumerate(EGeV):
	    pol		= self.pol
	    self.E	= E
	    try:
		self.scenario.index('Jet')
		T	= self.SetDomainN_Jet()
		pol	= np.dot(T,np.dot(pol,T.transpose().conjugate()))	# new polarization matrix
	    except ValueError:
		pass
	    try:
		self.scenario.index('ICM')
		self.update_params(**(self.kwargs))
		T	= self.SetDomainN()
		pol	= np.dot(T,np.dot(pol,T.transpose().conjugate()))	# new polarization matrix
	    except ValueError:
		pass
	    try:
		self.scenario.index('IGM')
		self.E0 = E
		T	= self.SetDomainN_IGM()	
		pol	= np.dot(T,np.dot(pol,T.transpose().conjugate()))	# new polarization matrix
		atten	= 1.
	    except ValueError:
		atten		= np.exp(-1. * self.ebl_norm * self.tau.opt_depth(self.z,E / 1e3))

	    Pt[i]	= np.real(np.sum(np.diag(np.dot(self.polt,pol))))
	    Pu[i]	= np.real(np.sum(np.diag(np.dot(self.polu,pol))))
	    Pa[i]	= np.real(np.sum(np.diag(np.dot(self.pola,pol))))

	    pol		= np.diag([Pt[i] * atten,Pu[i] * atten, Pa[i]])
	    try:
		self.scenario.index('GMF')
		Pt[i],Pu[i],Pa[i]= np.real(self.Pag_TM(self.E,self.ra,self.dec,pol))	# mixing in GMF
		try:
		    self.scenario.index('ICM')
		    self.Psin	= Psin		# restore values of Psin 
		except ValueError:
		    pass
	    except ValueError:
		pass
	return Pt,Pu,Pa

    def calc_pggave_conversion(self, bins, func, pfunc, new_angles = True, Esteps = 50):
	"""
	Calculate average photon transfer matrix from an interpolation

	Parameters
	----------
	bins:	n+1 -dim array with bin boundaries in GeV
	func:	function used for averaging, has to be called with func(pfunc,E)
	pfunc:	parameters for function

	kwargs
	------
	Esteps: int, number of energies to interpolate photon survival probability, default: 50
	new_angles:	bool, if True, calculate new random angles. Default: True

	Returns
	-------
	n+1-dim array with average photon survival probability for each bin
	"""

	logEGeV = np.linspace(np.log(bins[0] * 0.9), np.log(bins[-1] * 1.1), Esteps)

	# --- calculate the photon survival probability
	Pt,Pu,Pa = self.calc_conversion(np.exp(logEGeV), new_angles = new_angles)

	# --- calculate the average with interpolation
	self.pgg	= interp1d(logEGeV,np.log(Pt + Pu))

	# --- calculate average correction for each bin
	for i,E in enumerate(bins):
	    if not i:
		logE_array	= np.linspace(np.log(E),np.log(bins[i+1]),self.Esteps / 3)
		pgg_array	= np.exp(self.pgg(logE_array))
	    elif i == len(bins) - 1:
		break
	    else:
		logE		= np.linspace(np.log(E),np.log(bins[i+1]),self.Esteps / 3)
		logE_array	= np.vstack((logE_array,logE))
		pgg_array	= np.vstack((pgg_array,np.exp(self.pgg(logE))))
	# average transfer matrix over the bins
	return	simps(self.func(self.pobs,np.exp(logE_array)) * pgg_array * logE_array, logE_array, axis = 1) / \
		simps(self.func(self.pobs,np.exp(logE_array)) * logE_array, logE_array, axis = 1)

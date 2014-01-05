"""
Module to fit a spectrum with ALP modifications to data points. 

Fitting is done with iminuit routine

An observed spectrum will be corrected for EBL absorption including ALPs and it will be fitted 
with different functions.

History:
--------
- 12/16/2013: version 0.01 - created
"""

__author__ = "Manuel Meyer // manuel.meyer@fysik.su.se"
__version__ = 0.01

# --- Imports ------------ #
import numpy as np
import iminuit as minuit
import sys
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

# - Chi functions --------------------------------------------------------------------------- #
errfunc = lambda func, p, x, y, s: (func(p, x)-y) / s

# - Power Law Fit to data corrected for abs. w/ ALP within Jet and GMF B-fields ------------- #
class Fit_JetGMF(JET.PhotALPs_Jet,GMF.PhotALPs_GMF):
    def __init__(self,x,y,s,bins = None, **kwargs):
	"""
	Class to fit Powerlaw to data using minuit.migrad
	Data is corrected for absorption using the B-field environments of an AGN Jet and the GMF

	y(x) = p['Prefactor'] * ( x / p['Scale'] ) ** p['Index']
	y_abs = < P gg > * y_obs
	< P gg > is bin averaged transfer function including ALPs. 
	Additional fit parameters are B,g,m,Rmax,R_BLR,n

	Parameters
	----------
	x:		n-dim array containing the measured x values
	y:		n-dim array containing the measured y values, i.e. y = y(x)
	s: 		n-dim array with (symmetric) measurment uncertainties on y

	kwargs
	------
	bins:		n+1 dim array with boundaries. if none, comupted from x data.

	z:		float, redshift of the source
	ra:		float, right ascension of the source, in degree
	dec:		float, declination of the source, in degree
	func:		Function that describes the observed spectrum and takes x and pobs as parameters, y = func(x,pobs)
	pobs:		parameters for function

	Rmax:		distance up to which jet extends, in pc.
	Bjet:		field strength at r = R_BLR in G, default: 0.1 G
	R_BLR:		Distance of broad line region (BLR) to centrum in pc, default: 0.3 pc
	g:		Photon ALP coupling in 10^{-11} GeV^-1, default: 1.
	m:		ALP mass in neV, default: 1.
	njet:		electron density in the jet at r = R_BLR, in cm^-3, default: 1e3
	s:		exponent for scaling of electron density, default: 2.
	p:		exponent for scaling of magneitc field, default: 1.
	sens:		scalar < 1., sets the number of domains, for the B field in the n-th domain, 
			it will have changed by B_n = sens * B_{n-1}
	Psi:		scalar, angle between B field and transversal photon polarization, default: 0.
	model:		GMF model that is used. Currently available: pshirkov (ASS), jansson (default)
	ebl:		EBL model that is used. defaut: gilmore
	nE:		int, number of energy points used to calculate average Pgg in each energy bin, default: 30
	Esteps:		int, number of energy points used for interpolation of Pgg, default: 50
	NE2001:		bool, if true, use ne2001 code to calculate electron density in the Milky Way
			default: True

	pol_t: 		float, initial photon polarization
	pol_u: 		float, initial photon polarization
	pol_a: 		float, initial ALP polarization
	t + u + a != 1

	"""
# --- Set the defaults 
	kwargs.setdefault('z',None)
	kwargs.setdefault('ra',None)
	kwargs.setdefault('dec',None)
	kwargs.setdefault('func',None)
	kwargs.setdefault('pobs',None)
# --------------------
	kwargs.setdefault('nE',30)
	kwargs.setdefault('Esteps',50)
	kwargs.setdefault('model','jansson')
	kwargs.setdefault('ebl','gilmore')
# --------------------

	for k in kwargs.keys():
	    if kwargs[k] == None:
		raise TypeError("kwarg {0:s} cannot be None type.".format(k))
	if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	    raise TypeError("Lists must have same length!")

	super(Fit_JetGMF,self).__init__(**kwargs)	# init the jet mixing and gmf mixing, see e.g.
							# http://stackoverflow.com/questions/3277367/how-does-pythons-super-work-with-multiple-inheritance
							# for a small example
	self.__dict__.update(kwargs)			# form instance of kwargs

	self.x = np.array(x)
	self.y = np.array(y)
	self.yerr = np.array(s)

	self.exp = floor(np.log10(y[0]))
	self.y /= 10.**self.exp
	self.yerr /= 10.**self.exp

	if bins == None or not len(bins) == x.shape[0] + 1:
	    self.bins = calc_bin_bounds(x)

# --- init ALP mixing 
	self.pol		= np.zeros((3,3))
	self.polt	= np.zeros((3,3))
	self.polu	= np.zeros((3,3))
	self.pola	= np.zeros((3,3))

	self.polt[0,0]	= 1.
	self.polu[1,1]	= 1.
	self.pola[2,2]	= 1.
	self.pol[0,0]	= self.pol_t
	self.pol[1,1]	= self.pol_u
	self.pol[2,2]	= self.pol_a

	self.tau = TAU.OptDepth(model = kwargs['ebl'])	# init opt depth

# --- create 2-dim energy and tau arrays to compute average mixing in each energy bin
	for i,E in enumerate(self.bins):
	    if not i:
		self.logE_array	= np.linspace(np.log(E),np.log(self.bins[i+1]),self.nE)
		self.t_array	= self.tau.opt_depth_array(self.z,np.exp(self.logE_array))[0]

	    elif i == len(self.bins) - 1:
		break

	    else:
		logE	= np.linspace(np.log(E),np.log(self.bins[i+1]),self.nE)
		self.t_array	= np.vstack((self.t_array,self.tau.opt_depth_array(self.z,np.exp(logE))[0]))
		self.logE_array	= np.vstack((self.logE_array,logE))

	return
	
    def calc_PggAve(self, Esteps = 50):
	"""
	Calculate average transfer matrix for mixing in Jet and GMF and EBL absorption
	from an interpolation

	kwargs
	------
	Esteps: int, number of energies to interpolate photon survival probability, default: 50

	Returns
	-------
	n-dim array with average photon survival probability for each bin
	"""

	logETeV = np.linspace(np.log(self.bins[0] * 0.9), np.log(self.bins[-1] * 1.1), Esteps)
	atten	= np.exp(-self.tau.opt_depth_array(self.z,np.exp(logETeV))[0])
	Pt,Pu,Pa = np.zeros(logETeV.shape[0]),np.zeros(logETeV.shape[0]),np.zeros(logETeV.shape[0])

	# --- calculate the photon survival probability
	for i,E in enumerate(logETeV):
	    self.E	= np.exp(logETeV[i]) * 1e3
	    Tjet	= self.SetDomainN_Jet()	# mixing in jet
	    pol_jet	= np.dot(Tjet,np.dot(self.pol,Tjet.transpose().conjugate()))
	    polt	= np.real(np.sum(np.diag(np.dot(self.polt,pol_jet))))
	    polu	= np.real(np.sum(np.diag(np.dot(self.polu,pol_jet))))
	    pola	= np.real(np.sum(np.diag(np.dot(self.pola,pol_jet))))
	    pol_new	= np.diag([polt * atten[i],polu * atten[i], pola])
	    Pt[i], Pu[i], Pa[i] = np.real(self.Pag_TM(self.E,self.ra,self.dec,pol_new))	# mixing in GMF
	    #Pt[i], Pu[i], Pa[i] = polu,polt,pola

	# --- calculate the average with interpolation
	self.pgg	= interp1d(logETeV,np.log(Pt + Pu))

	# --- calculate average correction for each bin
	for i,E in enumerate(self.bins):
	    if not i:
		logE_array	= np.linspace(np.log(E),np.log(self.bins[i+1]),self.Esteps / 3)
		pgg_array	= np.exp(self.pgg(logE_array))
	    elif i == len(self.bins) - 1:
		break
	    else:
		logE		= np.linspace(np.log(E),np.log(self.bins[i+1]),self.Esteps / 3)
		logE_array	= np.vstack((logE_array,logE))
		pgg_array	= np.vstack((pgg_array,np.exp(self.pgg(logE))))
	# average transfer matrix over the bins
	return	simps(self.func(self.pobs,np.exp(logE_array)) * pgg_array * logE_array, logE_array, axis = 1) / \
		simps(self.func(self.pobs,np.exp(logE_array)) * logE_array, logE_array, axis = 1)


    def FillChiSq(self,Prefactor,Index,Scale,Rmax,Bjet,g,m,njet):
	"""
	Calculate the chi^2 value

	Parameters
	----------
	Prefactor:	float, power-law normalization
	Index:		float, power-law index
	Scale:		float, power-law pivo energy
	Rmax:		float, maximum radius of Bfield region, in pc
	B:		float, magnetic field at r = R_BLR, in pc
	g:		float, photon-ALP coupling constant, in 10^-11 GeV^-1
	m:		float, ALP mass in neV
	n:		float, ambient electron density at r = R_BLR, in cm^-3

	Returns
	-------
	float, chi^2 value
	"""

	params = {'Prefactor': Prefactor, 'Index': Index, 'Scale': Scale}

	# if any ALP parameters have changed, re-calculate the average correction
	if self.init or not g == self.g or not m == self.m or not Bjet == self.Bjet or not njet == self.njet or not Rmax == self.Rmax:
	    alppar = {'Rmax': Rmax, 'Bjet': Bjet, 'g': g, 'm': m, 'njet': njet, 'R_BLR': self.R_BLR}
	    self.update_params(**alppar)		# get the new params.
	    self.g = g
	    self.m = m
	    self.Bjet = Bjet
	    self.njet = njet
	    self.Rmax = Rmax
	# --- calculate the new deabsorbed data points
	    self.PggAve = self.calc_PggAve(Esteps = self.Esteps)
	    if self.init:
		self.init = False

	# calculate chi^2
	return np.sum(errfunc(pl,params,self.x,self.y / self.PggAve, self.yerr / self.PggAve)**2.)

    def fit(self, **kwargs):
	"""
	Fit a power law to the intrinsic spectrum, deabsorbed with ALPs

	kwargs
	-------
	full_output:	bool, if True, errors will be estimated additionally with minos, covariance matrix will also be returned
	minos_conf:	float, confidence level for minos error estimation, default: 1.
	print_level:	0,1, level of verbosity, defualt = 0 means nothing is printed
	int_steps:	float, initial step width, multiply with initial values of errors, default = 0.1
	strategy:	0 = fast, 1 = default (default), 2 = thorough
	tol:		float, required tolerance of fit = 0.001*tol*UP, default = 1.
	up		float, errordef, 1 (default) for chi^2, 0.5 for log-likelihood
	ncall:		int, number of maximum calls, default = 1000
	pedantic:	bool, if true (default), give all warnings
	limits:		dictionary containing 2-tuple for all fit parameters
	pinit:		dictionary with initial fit for all fit parameters
	fix:		dictionary with booleans if parameter is frozen for all fit parameters

	Returns
	-------
	tuple containing
	    0. list of Fit Stats: ChiSq, Dof, P-value
	    1. dictionary with final fit parameters
	    2. dictionary with 1 Sigma errors of final fit parameters
	if full_output = True:
	    3. dictionary with +/- 1 Sigma Minos errors
	    4. dictionary with covariance matrix

	Notes
	-----
	iminuit documentation: http://iminuit.github.io/iminuit/index.html
	"""
# --------------------
	kwargs.setdefault('full_output',True)
	kwargs.setdefault('minos_conf',1.)
	kwargs.setdefault('print_level',0)		# no output
	kwargs.setdefault('int_steps',0.1)		# Initial step width, multiply with initial values in m.errors
	kwargs.setdefault('strategy',1)		# 0 = fast, 1 = default, 2 = thorough
	kwargs.setdefault('tol',1.)			# Tolerance of fit = 0.001*tol*UP
	kwargs.setdefault('up',1.)			# 1 for chi^2, 0.5 for log-likelihood
	kwargs.setdefault('ncall',1000.)		# number of maximum calls
	kwargs.setdefault('pedantic',True)		# Give all warnings
	kwargs.setdefault('limits',{})
	kwargs.setdefault('pinit',{})
	kwargs.setdefault('fix',{'Prefactor': False,'Scale': True,'Index': False,'g': False,'m': True,'n':True ,'B': True,'Rmax': True})	
# --------------------
	self.init = True	# first function call to FillChiSq

	if not len(kwargs['pinit']):
	    kwargs['pinit']['Scale']	= self.x[np.argmax(self.y/self.yerr)]
	    kwargs['pinit']['Prefactor']= prior_norm(self.x / kwargs['pinit']['Scale'],self.y)
	    kwargs['pinit']['Index']	= prior_pl_ind(self.x / kwargs['pinit']['Scale'],self.y)
	else:
	    kwargs['pinit']['Prefactor'] /= 10.**self.exp
	if not len(kwargs['limits']):
	    kwargs['limits']['Prefactor'] = (kwargs['pinit']['Prefactor'] / 1e2, kwargs['pinit']['Prefactor'] * 1e2)
	    kwargs['limits']['Index'] = (-10.,2.)
	    kwargs['limits']['Scale'] = (kwargs['pinit']['Scale'] / 1e2, kwargs['pinit']['Scale'] * 1e2)
	    kwargs['limits']['g'] = (0.5,8.)
	    kwargs['limits']['m'] = (0.01,50.)
	    kwargs['limits']['njet']	= (self.njet / 10. ,self.njet * 10. )
	    kwargs['limits']['Bjet']	= (self.Bjet / 10. ,self.Bjet * 10. )
	    kwargs['limits']['Rmax']	= (self.Rmax / 10. ,self.Rmax * 10. )

	m = minuit.Minuit(self.FillChiSq, print_level = kwargs['print_level'],
			    # initial values
			    Prefactor	= kwargs['pinit']["Prefactor"],
			    Index = kwargs['pinit']["Index"],
			    Scale = kwargs['pinit']["Scale"],
			    g = self.g,
			    m = self.m,
			    Bjet = self.Bjet,
			    Rmax = self.Rmax,
			    njet = self.njet,
			    # errors
			    error_Prefactor	= kwargs['pinit']['Prefactor'] * kwargs['int_steps'],
			    error_Index		= kwargs['pinit']['Index'] * kwargs['int_steps'],
			    error_Scale		= 0.,
			    error_g		= self.g* kwargs['int_steps'],
			    error_m		= self.m* kwargs['int_steps'],
			    error_Bjet		= self.Bjet * kwargs['int_steps'],
			    error_Rmax		= self.Rmax * kwargs['int_steps'],
			    error_njet		= self.njet * kwargs['int_steps'],
			    # limits
			    limit_Prefactor = kwargs['limits']['Prefactor'],
			    limit_Index	= kwargs['limits']['Index'],
			    limit_Scale	= kwargs['limits']['Scale'],
			    limit_g	= kwargs['limits']["g"],
			    limit_m	= kwargs['limits']["m"],
			    limit_Bjet	= kwargs['limits']["Bjet"],
			    limit_njet	= kwargs['limits']["njet"],
			    limit_Rmax	= kwargs['limits']["Rmax"],
			    # freeze parametrs 
			    fix_Prefactor	= kwargs['fix']['Prefactor'],
			    fix_Index	= kwargs['fix']['Index'],
			    fix_Scale	= kwargs['fix']['Scale'],
			    fix_g		= kwargs['fix']["g"],
			    fix_m		= kwargs['fix']["m"],
			    fix_Bjet		= kwargs['fix']["Bjet"],
			    fix_njet		= kwargs['fix']["njet"],
			    fix_Rmax	= kwargs['fix']["Rmax"],
			    # setup
			    pedantic	= kwargs['pedantic'],
			    errordef	= kwargs['up'],
			    )

	npar = 0
	for k in kwargs['fix']:
	    if not kwargs['fix'][k]:
		npar += 1

	# Set initial fit control variables
	m.tol	= kwargs['tol']
	m.strategy	= kwargs['strategy']

	m.migrad(ncall = kwargs['ncall'])
	# second fit
	#m = minuit.Minuit(FillChiSq, print_level = kwargs['print_level'],errordef = kwargs['up'], **m.fitarg)
	#m.migrad(ncall = kwargs['ncall'])
	logging.info("PL Jet GMF: Migrad minimization finished")

	m.hesse()
	logging.info("PL Jet GMF: Hesse matrix calculation finished")

	if kwargs['full_output']:
	    for k in kwargs['fix'].keys():
		if kwargs['fix'][k]:
		    continue
		logging.info("PL Jet GMF: Running Minos for error estimation for parameter {0:s} at confidence level {1:.1f}".format(k,kwargs['minos_conf']))
		m.minos(k,kwargs['minos_conf'])
	    logging.info("PL_JetGMF: Minos finished")

	fit_stat = m.fval, float(len(self.x) - npar), pvalue(float(len(self.x) - npar), m.fval)

	m.values['Prefactor'] *= 10.**self.exp
	m.errors['Prefactor'] *= 10.**self.exp

	for k in kwargs['limits'].keys():
	    if kwargs['fix'][k]:
		continue
	    m.covariance[k,'Prefactor'] *= 10.**self.exp 
	    m.covariance['Prefactor',k] *= 10.**self.exp 


	if kwargs['full_output']:
	    return fit_stat,m.values, m.errors,m.merrors, m.covariance
	else:	
	    return fit_stat,m.values, m.errors

"""
Module to fit a spectrum with ALP modifications to data points. 

Fitting is done with iminuit routine

An observed spectrum will be corrected for EBL absorption including ALPs and it will be fitted 
with different functions.

History:
--------
- 12/16/2013: version 0.01 - created
- 01/08/2014: version 0.02 - added fit for ICM environment and included calc_conversion class
"""

__author__ = "Manuel Meyer // manuel.meyer@fysik.su.se"
__version__ = 0.02

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
import PhotALPsConv.calc_conversion as CC
# --- EBL imports
import eblstud.ebl.tau_from_model as TAU
from eblstud.misc.bin_energies import calc_bin_bounds
from eblstud.tools.iminuit_fit import pl,lp
# ------------------------ #

logging.basicConfig(level = logging.DEBUG)
# - Chi functions --------------------------------------------------------------------------- #
errfunc = lambda func, p, x, y, s: (func(p, x)-y) / s

# ------------------------------------------------------------------------------------------- #
# - Init the class -------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------- #
class Fit_JetICMGMF(CC.Calc_Conv):
    def __init__(self,x,y,s,bins = None, **kwargs):
	"""
	Class to fit Powerlaw to data using minuit.migrad
	Data is corrected for absorption using the B-field environments of either an AGN Jet or intracluster medium  and the GMF

	y(x) = p['Prefactor'] * ( x / p['Scale'] ) ** p['Index']
	y_abs = < P gg > * y_obs
	< P gg > is bin averaged transfer function including ALPs. 
	Additional fit parameters are B,g,m,Rmax,R_BLR,n

	Parameters
	----------
	x:		n-dim array containing the measured x values in TeV
	y:		n-dim array containing the measured y values, i.e. y = y(x)
	s: 		n-dim array with (symmetric) measurment uncertainties on y

	kwargs
	------
	bins:		n+1 dim array with boundaries. if none, comupted from x data.

	func:		Function that describes the observed spectrum and takes x and pobs as parameters, y = func(x,pobs)
	pobs:		parameters for function

	and all kwargs from PhotALPsConv.calc_conversion.Calc_Conv.

	"""
# --- Set the defaults 
	kwargs.setdefault('func',None)
	kwargs.setdefault('pobs',None)
# --------------------
	kwargs.setdefault('nE',30)
	kwargs.setdefault('Esteps',50)
# --------------------

	for k in kwargs.keys():
	    if kwargs[k] == None:
		raise TypeError("kwarg {0:s} cannot be None type.".format(k))
	if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	    raise TypeError("Lists must have same length!")

	super(Fit_JetICMGMF,self).__init__(**kwargs)	# init the jet mixing and gmf mixing, see e.g.
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

# ----------------------------------------------------------------------------- #
# ---- Chi square functions --------------------------------------------------- #
# ----------------------------------------------------------------------------- #
# --- Jet + GMF scenario ------------------------------------------------------ #
# ----------------------------------------------------------------------------- #
    def __FillChiSq_JetGMF(self,Prefactor,Index,Scale,Rmax,Bjet,g,m,njet):
	"""
	Calculate the chi^2 value for ALP conversion in Jet and GMF

	Parameters
	----------
	Prefactor:	float, power-law normalization
	Index:		float, power-law index
	Scale:		float, power-law pivo energy
	Rmax:		float, maximum radius of Bfield region, in pc
	Bjet:		float, magnetic field at r = R_BLR, in pc
	g:		float, photon-ALP coupling constant, in 10^-11 GeV^-1
	m:		float, ALP mass in neV
	njet:		float, ambient electron density at r = R_BLR, in cm^-3

	Returns
	-------
	float, chi^2 value
	"""

	params = {'Prefactor': Prefactor, 'Index': Index, 'Scale': Scale}

	# if any ALP parameters have changed, re-calculate the average correction
	if self.init or not g == self.g or not m == self.m or not Bjet == self.Bjet or not njet == self.njet or not Rmax == self.Rmax:
	    alppar = {'Rmax': Rmax, 'Bjet': Bjet, 'g': g, 'm': m, 'njet': njet, 'R_BLR': self.R_BLR}
	    self.update_params_Jet(**alppar)		# get the new params.
	    self.g = g
	    self.m = m
	    self.Bjet = Bjet
	    self.njet = njet
	    self.Rmax = Rmax
	# --- calculate the new deabsorbed data points
	    #self.PggAve = self.calc_PggAve(Esteps = self.Esteps)
	    self.PggAve = self.calc_pggave_conversion(self.bins *1e3, self.func, self.pobs, Esteps = self.Esteps)
	    if self.init:
		self.init = False

	# calculate chi^2
	logging.debug("{0} {1}".format(self.g, g))
	logging.debug("{0}".format(self.PggAve))
	return np.sum(errfunc(pl,params,self.x,self.y / self.PggAve, self.yerr / self.PggAve)**2.)

# ----------------------------------------------------------------------------- #
# --- ICM + GMF scenario ------------------------------------------------------ #
# ----------------------------------------------------------------------------- #

    def __FillChiSq_ICMGMF(self,Prefactor,Index,Scale,B,r_abell,Lcoh,g,m,n):
	"""
	Calculate the chi^2 value for ALP conversion in ICM and GMF

	Parameters
	----------
	Prefactor:	float, power-law normalization
	Index:		float, power-law index
	Scale:		float, power-law pivo energy
	r_abell:	float, radius of cluster, in kpc
	B:		float, magnetic field 
	g:		float, logarithm (naturalis) of photon-ALP coupling constant, in 10^-11 GeV^-1
	m:		float, ALP mass in neV
	n:		float, ambient electron density 
	Lcoh:		float, coherence length in kpc

	Returns
	-------
	float, chi^2 value
	"""
	params = {'Prefactor': Prefactor, 'Index': Index, 'Scale': Scale}

	# if any ALP parameters have changed, re-calculate the average correction
	if np.isscalar(self.n):
	    self.n = np.ones(self.Nd) * self.n
	if np.isscalar(self.B):
	    self.B = np.ones(self.Nd) * self.B

	#if self.init or not g == self.g or not m == self.m or not B == self.B[0] or not n == self.n[0] \
	if self.init or not np.exp(g) == self.g or not m == self.m or not B == self.B[0] or not n == self.n[0] \
	    or not r_abell == self.r_abell or not Lcoh == self.Lcoh:
	    alppar = {'r_abell': r_abell, 'B': B, 'g': np.exp(g), 'm': m, 'n': n, 'Lcoh': Lcoh}
	    #alppar = {'r_abell': r_abell, 'B': B, 'g': g, 'm': m, 'n': n, 'Lcoh': self.Lcoh}
	    self.update_params(**alppar)		# get the new params.
	    self.kwargs.update(alppar)
# --- calculate the new deabsorbed data points
	    self.PggAve = self.calc_pggave_conversion(self.bins *1e3, self.func, self.pobs, Esteps = self.Esteps, new_angles = False)
	    self.update_params(**alppar)		# B,n,L changed to GMF calc. With this call, they are restored to the ICM values.
	    if self.init:
		self.init = False

	# calculate chi^2
	logging.debug("{0} {1}".format(self.g, g))
	logging.debug("{0}".format(self.PggAve))
	return np.sum(errfunc(pl,params,self.x,self.y / self.PggAve, self.yerr / self.PggAve)**2.)

# ----------------------------------------------------------------------------- #
# ---- The fit function ------------------------------------------------------- #
# ----------------------------------------------------------------------------- #

    def fit(self, **kwargs):
	"""
	Fit a power law to the intrinsic spectrum, deabsorbed with ALPs in B-field of either AGN Jet or ICM and GMF

	kwargs
	-------
	full_output:	bool, if True, errors will be estimated additionally with minos, covariance matrix will also be returned
	minos_conf:	float or list, confidence level for minos error estimation, default: 1.
	minos_only:	string, if None, use minos for all parameters, else use minos only for specified parameter
	sample_chi2:	int, if > 0 and minos_only and full_output are set, this sets the number of bins to 
			profile over Chi2 (using Minos) for parameter specified with minos_only within 3 sigma of best fit.
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
	kwargs.setdefault('minos_only','None')
	kwargs.setdefault('sample_chi2',0)
	kwargs.setdefault('print_level',0)		# no output
	kwargs.setdefault('int_steps',0.1)		# Initial step width, multiply with initial values in m.errors
	kwargs.setdefault('strategy',1)		# 0 = fast, 1 = default, 2 = thorough
	kwargs.setdefault('tol',0.1)			# Tolerance of fit = 0.001*tol*UP
	kwargs.setdefault('up',1.)			# 1 for chi^2, 0.5 for log-likelihood
	kwargs.setdefault('ncall',1000.)		# number of maximum calls
	kwargs.setdefault('pedantic',True)		# Give all warnings
	kwargs.setdefault('limits',{})
	kwargs.setdefault('pinit',{})
	try:
	    self.scenario.index('Jet')
	    kwargs.setdefault('fix',{'Prefactor': False,'Scale': True,'Index': False,'g': False,'m': True,'njet':True ,'Bjet': True,'Rmax': True})	
	except ValueError:
	    pass
	try:
	    self.scenario.index('ICM')
	    kwargs.setdefault('fix',{'Prefactor': False,'Scale': True,'Index': False,'g': False,'m': True,'n':True ,'B': True,'r_abell': True, 'Lcoh':True})
	except ValueError:
	    pass
# --------------------
	self.init = True	# first function call to FillChiSq


	if not len(kwargs['limits']):
	    kwargs['limits']['Index'] = (-10.,2.)
	    kwargs['limits']['g'] = (0.1,8.)
	    kwargs['limits']['m'] = (0.01,50.)
	    try:
		self.scenario.index('Jet')
		kwargs['limits']['njet']	= (self.njet / 10. ,self.njet * 10. )
		kwargs['limits']['Bjet']	= (self.Bjet / 10. ,self.Bjet * 10. )
		kwargs['limits']['Rmax']	= (self.Rmax / 10. ,self.Rmax * 10. )
	    except ValueError:
		pass
	    try:
		self.scenario.index('ICM')
		kwargs['limits']['n']	= (self.n / 10. ,self.n * 10. )
		kwargs['limits']['B']	= (self.B / 10. ,self.B * 10. )
		kwargs['limits']['r_abell']	= (self.r_abell/ 10. ,self.r_abell* 10. )
		kwargs['limits']['Lcoh']	= (self.Lcoh / 10. ,self.Lcoh * 10. )
	    except ValueError:
		pass


	if not len(kwargs['pinit']):
	    cc = CC.Calc_Conv(**self.kwargs)		# has to be used here instead of self in order not to change self.kwargs
	    Pgg = cc.calc_pggave_conversion(self.bins *1e3, self.func, self.pobs, Esteps = self.Esteps, new_angles = False)
	    del cc
	    kwargs['pinit']['Scale']	= self.x[np.argmax(self.y/self.yerr)]
	    kwargs['pinit']['Prefactor']= prior_norm(self.x / kwargs['pinit']['Scale'],self.y / Pgg)
	    kwargs['pinit']['Index']	= prior_pl_ind(self.x / kwargs['pinit']['Scale'],self.y / Pgg)
	else:
	    kwargs['pinit']['Prefactor'] /= 10.**self.exp

	try:
	    kwargs['limits'].keys().index('Prefactor')
	except ValueError:
	    kwargs['limits']['Prefactor'] = (kwargs['pinit']['Prefactor'] / 1e2, kwargs['pinit']['Prefactor'] * 1e2)
	try:
	    kwargs['limits'].keys().index('Scale')
	except ValueError:
	    kwargs['limits']['Scale'] = (kwargs['pinit']['Scale'] / 1e2, kwargs['pinit']['Scale'] * 1e2)

	try:
	    self.scenario.index('Jet')
	    m = minuit.Minuit(self.__FillChiSq_JetGMF, print_level = kwargs['print_level'],
			    # --- initial values
			    Prefactor	= kwargs['pinit']["Prefactor"],
			    Index = kwargs['pinit']["Index"],
			    Scale = kwargs['pinit']["Scale"],
			    g = self.g,
			    m = self.m,
			    Bjet = self.Bjet,
			    Rmax = self.Rmax,
			    njet = self.njet,
			    # --- errors
			    error_Prefactor	= kwargs['pinit']['Prefactor'] * kwargs['int_steps'],
			    error_Index		= kwargs['pinit']['Index'] * kwargs['int_steps'],
			    error_Scale		= 0.,
			    error_g		= self.g* kwargs['int_steps'],
			    error_m		= self.m* kwargs['int_steps'],
			    error_Bjet		= self.Bjet * kwargs['int_steps'],
			    error_Rmax		= self.Rmax * kwargs['int_steps'],
			    error_njet		= self.njet * kwargs['int_steps'],
			    # --- limits
			    limit_Prefactor = kwargs['limits']['Prefactor'],
			    limit_Index	= kwargs['limits']['Index'],
			    limit_Scale	= kwargs['limits']['Scale'],
			    limit_g	= kwargs['limits']["g"],
			    limit_m	= kwargs['limits']["m"],
			    limit_Bjet	= kwargs['limits']["Bjet"],
			    limit_njet	= kwargs['limits']["njet"],
			    limit_Rmax	= kwargs['limits']["Rmax"],
			    # --- freeze parametrs 
			    fix_Prefactor	= kwargs['fix']['Prefactor'],
			    fix_Index	= kwargs['fix']['Index'],
			    fix_Scale	= kwargs['fix']['Scale'],
			    fix_g		= kwargs['fix']["g"],
			    fix_m		= kwargs['fix']["m"],
			    fix_Bjet		= kwargs['fix']["Bjet"],
			    fix_njet		= kwargs['fix']["njet"],
			    fix_Rmax	= kwargs['fix']["Rmax"],
			    # --- setup
			    pedantic	= kwargs['pedantic'],
			    errordef	= kwargs['up'],
			    )
	except ValueError:
	    pass
	try:
	    self.scenario.index('ICM')
	    m = minuit.Minuit(self.__FillChiSq_ICMGMF, print_level = kwargs['print_level'],
			    # --- initial values
			    Prefactor	= kwargs['pinit']["Prefactor"],
			    Index = kwargs['pinit']["Index"],
			    Scale = kwargs['pinit']["Scale"],
			    g = np.log(self.g),
			    #g = self.g,
			    m = self.m,
			    B = self.B,
			    r_abell = self.r_abell,
			    Lcoh = self.Lcoh,
			    n = self.n,
			    # --- errors
			    error_Prefactor	= kwargs['pinit']['Prefactor'] * kwargs['int_steps'],
			    error_Index		= kwargs['pinit']['Index'] * kwargs['int_steps'],
			    error_Scale		= 0.,
			    error_g		= np.log(self.g * kwargs['int_steps']),
			    #error_g		= self.g * kwargs['int_steps'],
			    error_m		= self.m * kwargs['int_steps'],
			    error_B 		= self.B * kwargs['int_steps'],
			    error_r_abell	= self.r_abell * kwargs['int_steps'],
			    error_Lcoh		= self.Lcoh * kwargs['int_steps'],
			    error_n		= self.n * kwargs['int_steps'],
			    # --- limits
			    limit_Prefactor = kwargs['limits']['Prefactor'],
			    limit_Index	= kwargs['limits']['Index'],
			    limit_Scale	= kwargs['limits']['Scale'],
			    limit_g	= np.log(kwargs['limits']["g"]),
			    #limit_g	= kwargs['limits']["g"],
			    limit_m	= kwargs['limits']["m"],
			    limit_B	= kwargs['limits']["B"],
			    limit_n	= kwargs['limits']["n"],
			    limit_r_abell= kwargs['limits']["r_abell"],
			    limit_Lcoh	= kwargs['limits']["Lcoh"],
			    # --- freeze parametrs 
			    fix_Prefactor	= kwargs['fix']['Prefactor'],
			    fix_Index	= kwargs['fix']['Index'],
			    fix_Scale	= kwargs['fix']['Scale'],
			    fix_g	= kwargs['fix']["g"],
			    fix_m	= kwargs['fix']["m"],
			    fix_B	= kwargs['fix']["B"],
			    fix_n	= kwargs['fix']["n"],
			    fix_r_abell	= kwargs['fix']["r_abell"],
			    fix_Lcoh	= kwargs['fix']["Lcoh"],
			    # --- setup
			    pedantic	= kwargs['pedantic'],
			    errordef	= kwargs['up'],
			    )
	except ValueError:
	    pass

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

#Prefactor,Index,Scale,B,r_abell,Lcoh,g,m,n

	m.hesse()
	logging.info("PL Jet GMF: Hesse matrix calculation finished")

	if kwargs['full_output']:
	    merr = {}
	    for k in kwargs['fix'].keys():
		if kwargs['fix'][k]:
		    continue
		if not kwargs['minos_only'] == 'None':
		    if not k == kwargs['minos_only']:
			continue
		if np.isscalar(kwargs['minos_conf']):
		    logging.info("PL Jet GMF: Running Minos for error estimation for parameter {0:s} at confidence level {1:.1f}".format(k,kwargs['minos_conf']))
		    t = m.minos(k,kwargs['minos_conf'])
		    if t['g']['lower_valid']:
			merr[(k,-1. * kwargs['minos_conf'])] = t['g']['lower']
		    else:
			merr[(k,-1. * kwargs['minos_conf'])] = 0.
		    if t['g']['upper_valid']:
			merr[(k,kwargs['minos_conf'])] = t['g']['upper']
		    else:
			merr[(k,kwargs['minos_conf'])] = 0.
		    del t
		else:
		    for i in kwargs['minos_conf']:
			logging.info("PL Jet GMF: Running Minos for error estimation for parameter {0:s} at confidence level {1:.1f}".format(k,i))
			t = m.minos(k,i)
			if t['g']['lower_valid']:
			    merr[(k,-1. * i )] = t['g']['lower']
			else:
			    merr[(k,-1. * i )] = 0.
			if t['g']['upper_valid']:
			    merr[(k,i )] = t['g']['upper']
			else:
			    merr[(k,i )] = 0.
			del t
	    logging.info("PL_JetGMF: Minos finished")
	if kwargs['sample_chi2'] and not kwargs['minos_only'] == 'None' and kwargs['full_output']:
	    #self.kwargs.update(m.fitarg)
	    #self.update_params_all(**self.kwargs)
	    #print m.fitarg
	    profile = m.mnprofile(kwargs['minos_only'],bins = kwargs['sample_chi2'], bound = 3, subtract_min = True)

	fit_stat = m.fval, float(len(self.x) - npar), pvalue(float(len(self.x) - npar), m.fval)


	m.values['Prefactor'] *= 10.**self.exp
	m.errors['Prefactor'] *= 10.**self.exp

	if kwargs['full_output']:
	    for k in kwargs['limits'].keys():
		if kwargs['fix'][k]:
		    continue
		m.covariance[k,'Prefactor'] *= 10.**self.exp 
		m.covariance['Prefactor',k] *= 10.**self.exp 
	    if kwargs['sample_chi2']:
		return fit_stat,m.values, m.errors,merr, m.covariance, profile
	    else:
		return fit_stat,m.values, m.errors,merr, m.covariance
	else:	
	    return fit_stat,m.values, m.errors
# end.

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
from math import floor, ceil
from eblstud.tools.lsq_fit import *
from scipy.integrate import simps
from scipy.interpolate import interp1d
import logging
import copy
# --- ALP imports 
import PhotALPsConv.conversion_Jet as JET
import PhotALPsConv.conversion as IGM 
import PhotALPsConv.conversion_ICM as ICM 
import PhotALPsConv.conversion_GMF as GMF 
from PhotALPsConv.tools import median_contours
from PhotALPsConv.deltas import Ecrit_GeV,Delta_Osc_kpc_array
# --- EBL imports
import eblstud.ebl.tau_from_model as TAU
from eblstud.misc.bin_energies import calc_bin_bounds
from eblstud.tools.iminuit_fit import *
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
	theta_jet:	float, angle between jet and l.o.s. in degrees, default: 3.
	Gamma:		float, bulk lorentz factor, default: 10.

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
	    if not self.B_gauss:
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
	    self.scenario.index('GMF')
	    self.update_params_GMF(**kwargs)
	except ValueError:
	    pass

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
	# change galactic magnetic field model
	try:
	    if not self.GMF_FitVal == 0 and self.model == 'jansson':
		parGMF			= {}
		parGMF['Bn']		= self.Bgmf.Bn
		parGMF['Bs']		= self.Bgmf.Bs
		parGMF['rhon']		= self.Bgmf.rhon
		parGMF['rhos']		= self.Bgmf.rhos
		parGMF['whalo']		= self.Bgmf.whalo
		parGMF['z0']		= self.Bgmf.z0
		parGMF['BX0']		= self.Bgmf.BX0
		parGMF['ThetaX0']	= self.Bgmf.ThetaX0
		parGMF['rhoXc']		= self.Bgmf.rhoXc
		parGMF['rhoX']		= self.Bgmf.rhoX

		parGMF['Bn_unc']	= self.Bgmf.Bn_unc
		parGMF['Bs_unc']	= self.Bgmf.Bs_unc
		parGMF['rhon_unc']	= self.Bgmf.rhon_unc
		parGMF['rhos_unc']	= self.Bgmf.rhos_unc
		parGMF['whalo_unc']	= self.Bgmf.whalo_unc
		parGMF['z0_unc']	= self.Bgmf.z0_unc
		parGMF['BX0_unc']	= self.Bgmf.BX_unc
		parGMF['ThetaX0_unc']	= self.Bgmf.ThetaX0_unc
		parGMF['rhoXc_unc']	= self.Bgmf.rhoXc_unc
		parGMF['rhoX_unc']	= self.Bgmf.rhoX_unc

		for k in parGMF.keys():
		    if k.find('unc') >= 0 : continue
		    parGMF[k] = parGMF[k] + self.GMF_FitVal * parGMF[k + '_unc']
		self.Bgmf.__dict__.update(**parGMF)
	except AttributeError:
	    self.GMF_FitVal = 0
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
	if np.isscalar(EGeV):
	    EGeV = np.array([EGeV])

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
		if self.B_gauss:
		    self.new_B_n()
		else:
		    self.new_random_psi()
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
		T	= self.SetDomainN()
		pol	= np.dot(T,np.dot(pol,T.transpose().conjugate()))	# new polarization matrix
		if not i:
		# store values that are modified by GMF conversion calculation
		    B,Lcoh,Nd,Psin,n = copy.copy(self.B),copy.copy(self.Lcoh),copy.copy(self.Nd),copy.copy(self.Psin),copy.copy(self.n)
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
		    # if ICM also considered, restore these values
		    self.scenario.index('ICM')
		    self.B,self.Lcoh,self.Nd,self.Psin,self.n = copy.copy(B),copy.copy(Lcoh),copy.copy(Nd),copy.copy(Psin),copy.copy(n)
		    self.T1		= np.zeros((3,3,self.Nd),np.complex)	# Transfer matrices
		    self.T2		= np.zeros((3,3,self.Nd),np.complex)
		    self.T3		= np.zeros((3,3,self.Nd),np.complex)
		    self.Un		= np.zeros((3,3,self.Nd),np.complex)
		except ValueError:
		    pass
	    except ValueError:
		pass
	return Pt,Pu,Pa

    def calc_pggave_conversion(self, bins, func=None, pfunc=None, new_angles = True, logPgg = 'None', Esteps = 50):
	"""
	Calculate average photon transfer matrix from an interpolation

	Parameters
	----------
	bins:	n+1 -dim array with bin boundaries in GeV

	kwargs
	------
	Esteps: int, number of energies to interpolate photon survival probability, default: 50
	new_angles:	bool, if True, calculate new random angles. Default: True
	func:	function used for averaging, has to be called with func(pfunc,E)
	pfunc:	parameters for function
	logPgg: Function for photon survival log(probability) versus log(energy). If not given it will be calculated.

	Returns
	-------
	n+1-dim array with average photon survival probability for each bin
	"""
	if not func == None:
	    self.funcAve = func
	if not pfunc == None:
	    self.pobs = pfunc

	logEGeV = np.linspace(np.log(bins[0] * 0.9), np.log(bins[-1] * 1.1), Esteps)

	# --- calculate the photon survival probability
	if logPgg == 'None':
	    Pt,Pu,Pa = self.calc_conversion(np.exp(logEGeV), new_angles = new_angles)

	# --- calculate the average with interpolation
	    self.pgg	= interp1d(logEGeV,np.log(Pt + Pu))
	else:
	    self.pgg	= logPgg

	# --- calculate average correction for each bin
	for i,E in enumerate(bins):
	    if not i:
		logE_array	= np.linspace(np.log(E),np.log(bins[i+1]),Esteps / 3)
		pgg_array	= np.exp(self.pgg(logE_array))
	    elif i == len(bins) - 1:
		break
	    else:
		logE		= np.linspace(np.log(E),np.log(bins[i+1]),Esteps / 3)
		logE_array	= np.vstack((logE_array,logE))
		pgg_array	= np.vstack((pgg_array,np.exp(self.pgg(logE))))
	# average transfer matrix over the bins
	return	simps(self.funcAve(self.pobs,np.exp(logE_array)) * pgg_array * np.exp(logE_array), logE_array, axis = 1) / \
		simps(self.funcAve(self.pobs,np.exp(logE_array)) * np.exp(logE_array), logE_array, axis = 1)

# --- Convenience function to plot a spectrum together with it's absorption corrected versions ----- #
    def plot_spectrum(self, x,y,s, logPgg = 'None', xerr = 'None',filename = 'spectrum.pdf',Emin = 0., Emax = 0.):
	"""
	Plot an observed gamma-ray spectrum together with it's abosrption corrected (w/ and w/o ALPs) versions.

	Arguments
	---------
	x:  n-dim array, Energy of spectrum in TeV
	y:  n-dim array, observed Flux of spectrum in dN / dE
	s:  n-dim array, dF

	kwargs
	------
	logPgg: Function for photon survival log(probability) versus log(energy). If not given it will be calculated.
	xerr: (n+1)-dim array, bin boundaries. If not given it will be caclulated
	Emin: minimum plot energy in TeV
	Emax: maximum plot energy  in TeV
	"""

	import matplotlib.pyplot as plt

	assert x.shape[0] == y.shape[0]
	assert x.shape[0] == s.shape[0]

	m = y > 0.
	x = x[m]
	y = y[m]
	s = s[m]

	stat,p,err,merr,cov,func,dfunc = {},{},{},{},{},{},{}
	f,df = {'obs': y}, {'obs': s}

	# calculate the bin bounds
	if xerr == 'None':
	    xerr = calc_bin_bounds(x)

	if not Emin:
	    Emin  = xerr[0]
	if not Emax:
	    Emin  = xerr[-1]
	xplot = 10.**np.linspace(np.log10(Emin),np.log10(Emax),200)

	# pl fit to observed spectrum
	stat['obs'],p['obs'], err['obs'],merr['obs'], cov['obs'] = MinuitFitPL(x,y,s , full_output = True)
	func['obs']	= pl
	dfunc['obs']	= butterfly_pl
 
	# calculate the average absorption correction
	pggAve	= self.calc_pggave_conversion(xerr * 1e3, func=func['obs'], pfunc=p['obs'], logPgg = logPgg)
	tauAve	= self.tau.opt_depth_Ebin(self.z,xerr,func['obs'],p['obs'])
	atten	= np.exp(-tauAve)
	f['alp']	= y / pggAve
	df['alp']	= s / pggAve
	f['tau']	= y / atten
	df['tau']	= s / atten

	# pl fit to deabs spectra
	stat['alp'],p['alp'], err['alp'],merr['alp'], cov['alp'] = MinuitFitPL(x,y / pggAve,s / pggAve , full_output = True)
	func['alp']	= pl
	dfunc['alp']	= butterfly_pl

	stat['tau'],p['tau'], err['tau'],merr['tau'], cov['tau'] = MinuitFitPL(x,y / atten ,s / atten, full_output = True)
	func['tau']	= pl
	dfunc['tau']	= butterfly_pl

	for k in stat.keys():
	    print '{0:s}:'.format(k)
	    for l in p[k].keys():
		print "{0:s} = {1:.2e} +/- {2:.2e}".format(l,p[k][l],err[k][l])

	# plot everything
	fig	= plt.figure()
	ax	= plt.subplot(111)
	ax.set_xscale('log')
	ax.set_yscale('log')
	cp	= plt.cm.Dark2
	marker = {'obs': 's', 'tau': 'v', 'alp':'o'}
	label  = {'obs': 'observed', 'tau': 'deabs. w/o ALPs', 'alp':'deabs. w/ ALPs'}
	for i,k in enumerate(f.keys()):
	    plt.errorbar(x,f[k], yerr = df[k], xerr = [x - xerr[:-1], xerr[1:] - x], 
		ls	= 'None',
		marker	= marker[k],
		color	= cp(i / (len(f.keys()) + 1.)),
		label	= label[k],
		)
	    plt.plot(xplot, func[k](p[k],xplot), 
		ls	= '-',
		lw	= 1.,
		color	= cp(i / (len(f.keys()) + 1.))
		)
	    plt.fill_between(xplot, func[k](p[k],xplot)*(1. - dfunc[k](p[k],xplot,err[k],cov[k])), 
		y2	= func[k](p[k],xplot)*(1. + dfunc[k](p[k],xplot,err[k],cov[k])),
		lw	= 1.,
		facecolor = 'None',
		edgecolor = cp(i / (len(f.keys()) + 1.))
		)
	plt.legend(loc = 0)
	plt.xlabel('Energy')
	plt.ylabel('$\mathrm{d}N / \mathrm{d} E$')
	v = plt.axis()
	plt.axis([Emin * 0.8, Emax / 0.8, v[2], v[3]])

	ax2 = ax.twinx()
	ax2.set_xscale('log')
	ax2.set_yscale('log')

	plt.plot(xplot, np.exp(logPgg(np.log(xplot * 1e3))),
	    lw		= 2.,
	    color	= 'red'
	    )
	plt.plot(xplot, np.exp(-self.tau.opt_depth_array(self.z,xplot))[0],
	    lw		= 2.,
	    color	= '0.',
	    ls = '--'
	    )

	v = plt.axis()
	plt.axis([Emin * 0.8, Emax / 0.8, v[2], v[3]])

	plt.savefig(filename, format = filename.split('.')[-1])
	plt.show()
	return

    def plot_counts(self, x,S,B, logPgg, xerr = 'None',alpha = 0.2,filename = 'counts.pdf'):
	"""
	Plot the expected counts spectra of a gamma-ray observation.

	Arguments
	---------
	S:  dictionary with n-dim arrays, containing the expected signal counts
	B:  dictionary with n-dim arrays, containing the expected bkg counts
	logPgg: Function for photon survival log(probability) versus log(energy)

	kwargs
	------
	xerr: (n+1)-dim array, bin boundaries. If not given it will be caclulated
	alpha: float, ratio of exposure between ON and OFF
	"""

	import matplotlib.pyplot as plt

	# calculate the bin bounds
	if xerr == 'None':
	    xerr = calc_bin_bounds(x)

	xplot = 10.**np.linspace(np.log10(xerr[0]),np.log10(xerr[-1]),200)

	# plot everything
	fig	= plt.figure(1)
	ax	= plt.subplot(111)
	ax.set_xscale('log')
	ax.set_yscale('log')
	cp	= plt.cm.Dark2
	marker = ['s','o']
	for i,k in enumerate(S.keys()):
	    plt.errorbar(x,S[k], xerr = [x - xerr[:-1], xerr[1:] - x], 
		ls	= 'None',
		marker	= 'o',
		lw	= 1.,
		color	= cp(i / (len(S.keys()) + 1.)),
		mec	= cp(i / (len(S.keys()) + 1.)),
		mfc	= 'None',
		label	= 'exp. signal {0:s}'.format(k),
		)
	    plt.errorbar(x,B[k], xerr = [x - xerr[:-1], xerr[1:] - x], 
		ls	= 'None',
		lw	= 1.,
		marker	= 's',
		color	= cp(i / (len(S.keys()) + 1.)),
		mec	= cp(i / (len(S.keys()) + 1.)),
		mfc	= 'None',
		label	= 'exp. bkg {0:s}'.format(k),
		)
	plt.legend(loc = 0)
	plt.xlabel('Energy')
	plt.ylabel('$\mathrm{d}N / \mathrm{d} E$')
	v = plt.axis()
	plt.axis([xerr[0] * 0.8, xerr[-1] / 0.8, v[2], v[3]])

	ax2 = ax.twinx()
	ax2.set_xscale('log')
	ax2.set_yscale('log')

	plt.plot(xplot, np.exp(-self.tau.opt_depth_array(self.z,xplot))[0],
	    lw		= 2.,
	    color	= '0.'
	    )
	plt.plot(xplot, np.exp(logPgg(np.log(xplot * 1e3))),
	    lw		= 2.,
	    color	= 'red'
	    )

	v = plt.axis()
	plt.axis([xerr[0] * 0.8, xerr[-1] / 0.8, v[2], v[3]])
	plt.savefig(filename + '.spec', format = filename.split('.')[-1])

# histogramm with NON values
	from scipy.stats import poisson
	fig	= plt.figure(2)
	for i,k in enumerate(S.keys()):
	    NonExp	= (S[k] + B[k])[0] * np.ones(5000)
	    NON		= poisson.rvs(NonExp)
	    plt.hist(NON, label = k)
	plt.legend(loc = 0)
	plt.savefig(filename + '.hist', format = filename.split('.')[-1])

	plt.show()
	return

    def EcritAve(self):
	"""
	calculate average critical energy if B and density are changing with radius

	Returns
	-------
	Average critical energy is float
	"""
	Bave = simps(self.B * self.r, np.log(self.r)) / (self.r[-1] - self.r[0])
	nave = simps(self.n * self.r, np.log(self.r)) / (self.r[-1] - self.r[0])

	return Ecrit_GeV(self.m,nave,Bave,self.g)

    def plot_Pgg(self, EGeV, P, axis = 0, plot_all = False, filename = None, plot_one = True, EcritLabel = '', y0 = 1e-3, y1 = 1.1, loc = 3):
	"""
	Plot the transfer function

	Arguments
	---------
	EGeV:	m-dim array with energies in GeV
	P:	nxm-dim array with transfer function

	kwargs
	------
	axis:	int, axis with B-field realisations (if more than one, only used if n > 1)
	plot_all: bool, if true, plot all realizations of transfer function
	plot_one: bool, if true, plot one realization of transfer function
	y0: float, min of y-axis of main plot
	y1: float, max of y-axis of main plot
	loc: int, location of legend
	"""

	import matplotlib.pyplot as plt

	fig = plt.figure()
	ax = plt.subplot(111)
	ax2 = fig.add_axes([0.2, 0.2, 0.4, 0.4])	# left bottom width height

	try:
	    self.scenario.index('ICM')
	    Ecrit = self.EcritAve()
	except ValueError:
	    Ecrit = Ecrit_GeV(self.m,self.kwargs['n'],self.kwargs['B'],self.g)

	imin = np.argmin(np.abs(EGeV - Ecrit))

	for i,a in enumerate([ax,ax2]):
	    a.set_xscale('log')
	    a.set_yscale('log')


	    if len(P.shape) > 1:
		MedCon = median_contours(P, axis = axis)
		a.fill_between(EGeV,MedCon['conf_95'][0],y2 = MedCon['conf_95'][1], color = plt.cm.Greens(0.8))
		a.fill_between(EGeV,MedCon['conf_68'][0],y2 = MedCon['conf_68'][1], color = plt.cm.Greens(0.5))
		if plot_one:
		    a.plot(EGeV,P[0], ls = '-', color = '0.', label = '$P_{\gamma\gamma}$, one realization', lw = 2)
		if plot_all:
		    for i in range(1,P.shape[axis]):
			a.plot(EGeV,P[i], ls = ':', color = '0.', lw = 1)

		ymin = ceil(np.log10(MedCon['median'][imin])) - 0.3
		ymax = ceil(np.log10(MedCon['median'][imin]))
	    else:
		a.plot(EGeV,P, ls = '-', color = '0.', label = '$P_{\gamma\gamma}$')
		ymin = ceil(np.log10(P[imin])) - 0.3
		ymax = ceil(np.log10(P[imin]))

	    a.axvline(Ecrit, 
		ls = '--', color = '0.',lw = 1., 
		label = r'$E_\mathrm{{crit}}^\mathrm{{{0:s}}}$'.format(EcritLabel)
		)

	    a.plot(EGeV,np.exp(-1. * self.ebl_norm * self.tau.opt_depth_array(self.z,EGeV / 1e3)[0]), ls = '-', color = 'red', lw = 2.,label = r'$\exp(-\tau)$')
	    if not i:
		a.legend(loc = loc, fontsize = 'small')
		v = a.axis([EGeV[0],EGeV[-1],y0,y1])
		ax.set_xlabel("Energy (GeV)")
		ax.set_ylabel("Photon survival probability")
	    else:
		xmin = np.log10(Ecrit) - 0.5
		xmax = np.log10(Ecrit) + 0.5
		a.axis([10.**xmin,10.**xmax,10.**ymin,10.**ymax])

	if not filename == None:
	    plt.savefig(filename + '.pdf', format = 'pdf')
	    plt.savefig(filename + '.png', format = 'png')
	    plt.show()


	return ax,ax2

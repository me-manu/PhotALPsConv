"""
Module Containing tools for fitting with minuit migrad routine.

History of changes:
Version 1.0
- Created 30th November 2011
	* MinuitFit1Pol
	* MinuitFit2Pol
	* MinuitFit3Pol
"""

import numpy as np
from eblstud.stats import misc
from PhotALPsConv.lsq_fit import *
import scipy.special
import minuit
import warnings

class FitMinuit:
    def __init__(self,minos = True):
# - Set Minuit Fit parameters --------------------------------------------------------------- #
	self.FitParams = {
	    'int_steps' : 0.01,		# Initial step width, multiply with initial values in m.errors
	    'strategy'  : 1,		# 0 = fast, 1 = default, 2 = thorough
	    'printMode' : 0,		# Shut Minuit up
	    'maxcalls'  : 5000,		# Maximum Number of Function calls, default is None
	    'tol'       : 0.01,		# Tolerance of fit = 0.001*tol*fit
	    'up'        : 1.,		# 1 for chi^2, 0.5 for log-likelihood
	}
	self.minos = minos
	return
# - Initial Fit Parameters ------------------------------------------------------------------ #
    def SetFitParams(self,Minuit):
	Minuit.maxcalls = self.FitParams['maxcalls']
	Minuit.strategy = self.FitParams['strategy']
	Minuit.tol	= self.FitParams['tol']
	Minuit.up	= self.FitParams['up']
	Minuit.printMode= self.FitParams['printMode']
	return

# - 1st order polynomial -------------------------------------------------------------------- #
    def MinuitFit1Pol(self,x,y,s,pinit=[],limits=(),full_output=False):
	"""Function to fit 1st order Polynomial to data using minuit.migrad
    
	pinit[0] = constant
	    pinit[1] = slope
	returns 3 lists:
	1. Fit Stats: ChiSq, Dof, P-value
	2. Final fit parameters
	3. 1 Sigma errors of Final Fit parameters
	"""
	npar = 2
	fitfunc = err_1st_pol

	if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	    raise TypeError("Lists must have same length!")
	if not len(x) > npar:
	    print "Not sufficient number of data points => Returning -1"
	    return -1

	x = np.array(x)
	y = np.array(y)
	s = np.array(s)

	def FillChiSq(p0,p1):
	    params = p0,p1
	    result = 0.
	    for i,xval in enumerate(x):
	        result += fitfunc(params,xval,y[i],s[i])**2.
	    return result
	m = minuit.Minuit(FillChiSq)

       # Set initial fit control variables
	self.SetFitParams(m)

       # Set initial Fit parameters, initial step width and limits of parameters
	if not len(pinit):
	    m.values['p0'] = prior_p0(x,y)
	    m.values['p1'] = prior_p1(x,y)
	else:
	    m.values['p0'] = pinit[0]
	    m.values['p1'] = pinit[1]
	if len(limits):
	    for i in range(len(limits)):
		if i == 0:
		    m.limits['p0'] = limits[i]
		if i == 1:
		    m.limits['p1'] = limits[i]
	for val in m.errors:
	    m.errors[val] = m.values[val] * self.FitParams['int_steps']

	pfinal = np.zeros((npar,))
	fit_err = np.zeros((npar,))
	try:
	    m.migrad()
	    m.hesse()
	    if self.minos and pvalue(float(len(x) - npar), m.fval) >= 0.05:
		m.minos('p1',1.)
	except minuit.MinuitError:
            warnings.simplefilter("always")
            warnings.warn('Minuit could not fit function!',RuntimeWarning)
	    if full_output:
		return (np.inf,float(npar),np.inf),pfinal,fit_err,m.covariance
	    else:
		return (np.inf,float(npar),np.inf),pfinal,fit_err

	fit_stat = m.fval, float(len(x) - npar), pvalue(float(len(x) - npar), m.fval)
	for i,val in enumerate(m.values):
	    pfinal[i] = m.values[val]
	    fit_err[i] = m.errors[val]
	if full_output:
	    return fit_stat,pfinal,fit_err,m.covariance
	else:	
	    return fit_stat,pfinal,fit_err
# - 2nd order polynomial -------------------------------------------------------------------- #
    def MinuitFit2Pol(self,x,y,s,pinit=[],limits=(),full_output=False):
	"""Function to fit 2nd order Polynomial to data using minuit.migrad
    
	pinit[0] = constant
	pinit[1] = slope
	pinit[2] = slope on x^2
	returns 3 lists:
	1. Fit Stats: ChiSq, Dof, P-value
	2. Final fit parameters
	3. 1 Sigma errors of Final Fit parameters
	"""
	npar = 3
	fitfunc = err_2nd_pol

	if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	    raise TypeError("Lists must have same length!")
	if not len(x) > npar:
	    print "Not sufficient number of data points => Returning -1"
	    return -1

	x = np.array(x)
	y = np.array(y)
	s = np.array(s)

	def FillChiSq(p0,p1,p2):
	    params = p0,p1,p2
	    result = 0.
	    for i,xval in enumerate(x):
	        result += fitfunc(params,xval,y[i],s[i])**2.
	    return result
	m = minuit.Minuit(FillChiSq)

       # Set initial fit control variables
	self.SetFitParams(m)

       # Set initial Fit parameters, initial step width and limits of parameters
	if not len(pinit):
	    m.values['p0'] = prior_p0(x,y)
	    m.values['p1'] = prior_p1(x,y)
	    m.values['p2'] = prior_p2(x,y)
	else:
	    m.values['p0'] = pinit[0]
	    m.values['p1'] = pinit[1]
	    m.values['p2'] = pinit[2]
	if len(limits):
	    for i in range(len(limits)):
		if i == 0:
		    m.limits['p0'] = limits[i]
		if i == 1:
		    m.limits['p1'] = limits[i]
		if i == 2:
		    m.limits['p2'] = limits[i]
	for val in m.errors:
	    m.errors[val] = m.values[val] * self.FitParams['int_steps']

	pfinal = np.zeros((npar,))
	fit_err = np.zeros((npar,))
	try:
	    m.migrad()
	    m.hesse()
	    if self.minos and pvalue(float(len(x) - npar), m.fval) >= 0.05:
		m.minos('p1',1.)
		m.minos('p2',1.)
	except minuit.MinuitError:
            warnings.simplefilter("always")
            warnings.warn('Minuit could not fit function!',RuntimeWarning)
	    if full_output:
		return (np.inf,float(npar),np.inf),pfinal,fit_err,m.covariance
	    else:
		return (np.inf,float(npar),np.inf),pfinal,fit_err

	fit_stat = m.fval, float(len(x) - npar), pvalue(float(len(x) - npar), m.fval)
	pfinal[0], fit_err[0] = m.values['p0'],m.errors['p0']
	pfinal[1], fit_err[1] = m.values['p1'],m.errors['p1']
	pfinal[2], fit_err[2] = m.values['p2'],m.errors['p2']
	if full_output:
	    return fit_stat,pfinal,fit_err,m.covariance
	else:	
	    return fit_stat,pfinal,fit_err
# - 3rd order polynomial -------------------------------------------------------------------- #
    def MinuitFit3Pol(self,x,y,s,pinit=[],limits=(),full_output=False):
	"""Function to fit 3rd order Polynomial to data using minuit.migrad
    
	pinit[0] = constant
	pinit[1] = slope
	pinit[2] = slope on x^2
	pinit[3] = slope on x^3
	returns 3 lists:
	1. Fit Stats: ChiSq, Dof, P-value
	2. Final fit parameters
	3. 1 Sigma errors of Final Fit parameters
	"""
	npar = 4
	fitfunc = err_3rd_pol

	if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	    raise TypeError("Lists must have same length!")
	if not len(x) > npar:
	    print "Not sufficient number of data points => Returning -1"
	    return -1

	x = np.array(x)
	y = np.array(y)
	s = np.array(s)

	def FillChiSq(p0,p1,p2,p3):
	    params = p0,p1,p2,p3
	    result = 0.
	    for i,xval in enumerate(x):
	        result += fitfunc(params,xval,y[i],s[i])**2.
	    return result
	m = minuit.Minuit(FillChiSq)

       # Set initial fit control variables
	self.SetFitParams(m)

       # Set initial Fit parameters, initial step width and limits of parameters
	if not len(pinit):
	    m.values['p0'] = prior_p0(x,y)
	    m.values['p1'] = prior_p1(x,y)
	    m.values['p2'] = prior_p2(x,y)
	    m.values['p3'] = prior_p3(x,y)
	else:
	    m.values['p0'] = pinit[0]
	    m.values['p1'] = pinit[1]
	    m.values['p2'] = pinit[2]
	    m.values['p3'] = pinit[3]
	if len(limits):
	    for i in range(len(limits)):
		if i == 0:
		    m.limits['p0'] = limits[i]
		if i == 1:
		    m.limits['p1'] = limits[i]
		if i == 2:
		    m.limits['p2'] = limits[i]
		if i == 3:
		    m.limits['p3'] = limits[i]
	for val in m.errors:
	    m.errors[val] = m.values[val] * self.FitParams['int_steps']

	pfinal = np.zeros((npar,))
	fit_err = np.zeros((npar,))
	try:
	    m.migrad()
	    m.hesse()
	    if self.minos and pvalue(float(len(x) - npar), m.fval) >= 0.05:
		m.minos('p1',1.)
		m.minos('p2',1.)
		m.minos('p3',1.)
	except minuit.MinuitError:
            warnings.simplefilter("always")
            warnings.warn('Minuit could not fit function!',RuntimeWarning)
	    if full_output:
		return (np.inf,float(npar),np.inf),pfinal,fit_err,m.covariance
	    else:
		return (np.inf,float(npar),np.inf),pfinal,fit_err

	fit_stat = m.fval, float(len(x) - npar), pvalue(float(len(x) - npar), m.fval)
	pfinal[0], fit_err[0] = m.values['p0'],m.errors['p0']
	pfinal[1], fit_err[1] = m.values['p1'],m.errors['p1']
	pfinal[2], fit_err[2] = m.values['p2'],m.errors['p2']
	pfinal[3], fit_err[3] = m.values['p3'],m.errors['p3']
	if full_output:
	    return fit_stat,pfinal,fit_err,m.covariance
	else:	
	    return fit_stat,pfinal,fit_err

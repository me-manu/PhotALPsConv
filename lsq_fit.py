"""
Module Containing tools for fitting with least squares scipy routine:
1st ,2nd, and 3rd order polynomial

History of changes:
Version 1.0
- Created 30th November 2011

"""

__version__ = 1.0
__author__ = "M. Meyer // manuel.meyer@physik.uni-hamburg.de"

import numpy as np
from scipy.optimize import leastsq
from scipy.integrate import quad
from eblstud.stats import misc
import scipy.special

gammainc = scipy.special.gammainc
gamma = scipy.special.gamma
pvalue = lambda dof, chisq: 1. - gammainc(.5 * dof, .5 * chisq)

# Fitting Functions #############################################
fit_1st_pol = lambda p,x: p[1] * x + p[0]
fit_2nd_pol = lambda p,x: p[2] * x**2. + p[1] * x + p[0]
fit_3rd_pol = lambda p,x: p[3]* x **3. + p[2] * x**2. + p[1] * x + p[0]

err_1st_pol = lambda p,x,y,err: (fit_1st_pol(p,x) - y )/ err
err_2nd_pol = lambda p,x,y,err: (fit_2nd_pol(p,x) - y )/ err
err_3rd_pol = lambda p,x,y,err: (fit_3rd_pol(p,x) - y )/ err

# Jacobians #####################################################
def jacobian_1st(p,x,y,err):
    J = np.zeros((2,len(x)))
    J[0:,] = 1./err
    J[1:,] = x / err
    return J

def jacobian_2nd(p,x,y,err):
    J = np.zeros((3,len(x)))
    J[0:,] = 1./err
    J[1:,] = x / err
    J[2:,] = x**2. / err
    return J

def jacobian_3rd(p,x,y,err):
    J = np.zeros((4,len(x)))
    J[0:,] = 1./err
    J[1:,] = x / err
    J[2:,] = x**2. / err
    J[3:,] = x**3. / err
    return J

# Priors (initial conditions) #####################################################
def prior_p0(x,y):
    return y[np.where(np.abs(x) == np.min(np.abs(x)))]
def prior_p1(x,y):
    if len(x) > 3:
	return (y[-2] - y[1]) / (x[-2] - x[1])
    else:
	return (y[-1] - y[0]) / (x[-1] - x[0])
def prior_p2(x,y):
    return 1.
def prior_p3(x,y):
    return 1.

#################################

def lsq_1stfit(x,y,s,pinit=[],full_output = False):
    """Function to fit 1st order polynomial with scipy.leastsquares
    
    pinit[0] = y-axes
    pinit[1] = slope
    """

    npar = 2

    if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	raise TypeError("Lists must have same length!")
    if not len(x) > npar:
	print "Not sufficient number of data points => Returning -1"
	return -1

    x = np.array(x)
    y = np.array(y)
    s = np.array(s)

    if not len(pinit):
	pinit = np.zeros((npar,))
	pinit[0] = prior_p0(x,y)
	pinit[1] = prior_p1(x,y)

    out = leastsq(err_1st_pol, pinit,
	       args=(x, y, s), Dfun=jacobian_1st, full_output=1, col_deriv=1
	       )
    pfinal = out[0]
    covar = out[1]

    if not covar==None:
	fit_err = [np.sqrt(covar[i][i]) for i in range(npar)]
    else:
	fit_err = np.zeros((npar,))

    chisq = np.sum(err_1st_pol(pfinal,x,y,s)**2)
    dof = len(y) - npar 
    pval = pvalue(dof,chisq)

    fit_stat = chisq,dof,pval
    
    if full_output:
	return fit_stat,pfinal,fit_err,covar

    return fit_stat,pfinal,fit_err

def lsq_2ndfit(x,y,s,pinit=[],full_output = False):
    """Function to fit 2nd order polynomial with scipy.leastsquares
    
    pinit[0] = y-axes
    pinit[1] = slope of x
    pinit[2] = slope of x^2
    """

    npar = 3

    if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	raise TypeError("Lists must have same length!")
    if not len(x) > npar:
	print "Not sufficient number of data points => Returning -1"
	return -1

    x = np.array(x)
    y = np.array(y)
    s = np.array(s)

    if not len(pinit):
	pinit = np.zeros((npar,))
	pinit[0] = prior_p0(x,y)
	pinit[1] = prior_p1(x,y)
	pinit[2] = prior_p2(x,y)

    out = leastsq(err_2nd_pol, pinit,
	       args=(x, y, s), Dfun=jacobian_2nd, full_output=1, col_deriv=1
	       )
    pfinal = out[0]
    covar = out[1]

    if not covar==None:
	fit_err = [np.sqrt(covar[i][i]) for i in range(npar)]
    else:
	fit_err = np.zeros((npar,))

    chisq = np.sum(err_2nd_pol(pfinal,x,y,s)**2)
    dof = len(y) - npar 
    pval = pvalue(dof,chisq)

    fit_stat = chisq,dof,pval
    
    if full_output:
	return fit_stat,pfinal,fit_err,covar

    return fit_stat,pfinal,fit_err
def lsq_3rdfit(x,y,s,pinit=[],full_output = False):
    """Function to fit 3rd order polynomial with scipy.leastsquares
    
    pinit[0] = y-axes
    pinit[1] = slope of x
    pinit[2] = slope of x^2
    pinit[2] = slope of x^3
    """

    npar = 4

    if not len(x) == len(y) or not len(x) == len(s) or not len(y) == len(s):
	raise TypeError("Lists must have same length!")
    if not len(x) > npar:
	print "Not sufficient number of data points => Returning -1"
	return -1

    x = np.array(x)
    y = np.array(y)
    s = np.array(s)

    if not len(pinit):
	pinit = np.zeros((npar,))
	pinit[0] = prior_p0(x,y)
	pinit[1] = prior_p1(x,y)
	pinit[2] = prior_p2(x,y)
	pinit[3] = prior_p2(x,y)

    out = leastsq(err_3rd_pol, pinit,
	       args=(x, y, s), Dfun=jacobian_3rd, full_output=1, col_deriv=1
	       )
    pfinal = out[0]
    covar = out[1]

    if not covar==None:
	fit_err = [np.sqrt(covar[i][i]) for i in range(npar)]
    else:
	fit_err = np.zeros((npar,))

    chisq = np.sum(err_3rd_pol(pfinal,x,y,s)**2)
    dof = len(y) - npar 
    pval = pvalue(dof,chisq)

    fit_stat = chisq,dof,pval
    
    if full_output:
	return fit_stat,pfinal,fit_err,covar

    return fit_stat,pfinal,fit_err

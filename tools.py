"""
Some handy tools for the calculation of statistical properties 
of photon - ALP survival probabilities

History:
- 11/20/13: created
"""

__version__=0.01
__author__="M. Meyer // manuel.meyer@fysik.su.se"

# --- Imports -------------- #
#from numpy import mean,nanmean,sqrt,sort,median,array
from numpy import mean,sqrt,sort,median,array
from math import floor,ceil
# -------------------------- #

# calculate index for lower confidence contour 
# for matrix P along axis axis and for confidence level conf
ind_lo = lambda P,conf,axis: int(floor((P.shape[axis]*0.5*(1. - conf))))

# calculate index for upper confidence contour 
# for matrix P along axis axis and for confidence level conf
ind_up = lambda P,conf,axis: int(ceil((P.shape[axis]*0.5*(1. + conf))))

def rms(x, axis=None):
    """calculate rms of x along axis axis"""
    return sqrt(mean(x**2, axis=axis))

#def nanrms(x, axis=None):
#    """calculate rms of x if x contains nans along axis axis"""
#    return sqrt(nanmean(x**2, axis=axis))

def median_contours(P,axis = 0, conf = [0.68,0.95]):
    """
    Calculate median and 68,95 % confidence contours of survival probability matrix P

    Parameters
    ----------
    P:	np.array with photon survival probabilities, either n or n x m dimensional

    kwargs
    ------
    axis:	int, axis along which median etc. is calculated, default: 0
    conf:	list with confidence levels, defaut: [0.68,0.95]

    Returns
    -------
    dictionary with entries
	median:	n [or m] dimensional array with median entries
	conf_{int(100 * conf)} 2 x n [or m] dimensional array with confidence contours around median
    """

    result = {}
    for c in conf:
	idx_low	= ind_lo(P,c,axis)
	idx_up	= ind_up(P,c,axis)
	if idx_up > P.shape[axis] - 1:
	        idx_up = P.shape[axis] - 1
	if axis:
	    result['conf_{0:n}'.format(int(c * 100))] = array([sort(P, axis = axis)[:,idx_low],sort(P, axis = axis)[:,idx_up]])
	else:
	    result['conf_{0:n}'.format(int(c * 100))] = array([sort(P, axis = axis)[idx_low,:],sort(P, axis = axis)[idx_up,:]])
    result['median'] = median(P,axis = axis)
    return result

"""
Functions to calculate the Delta parameters that enter into the photon-Axion Mixing matrix.

Version 1.0
- 11/15/11: created
"""

import numpy as np

__version__=0.03

#g is photon axion coupling in 10^-11 GeV^-1
#B is magnetic field in nG
#returns Delta in Mpc^-1
#taken from Mirizzi & Montanino 2009
Delta_ag_Mpc= lambda g,B: 1.52e-2*g*B

#g is photon axion coupling in 10^-11 GeV^-1
#B is magnetic field in muG
#returns Delta in kpc^-1
Delta_ag_kpc= lambda g,B: 1.52e-2*g*B

#m is photon axion mass in 10^-10 eV
#E is Energy in TeV
#returns Delta in Mpc^-1
#taken from Mirizzi & Montanino 2009
Delta_a_Mpc= lambda m,E: -7.8e-4*m**2./E

#m is photon axion mass in 10^-9 eV
#E is Energy in GeV
#returns Delta in kpc^-1
Delta_a_kpc= lambda m,E: -7.8e-2*m**2./E

#n is electron density in 10^-7 cm^-3
#E is Energy in TeV
#returns Delta in Mpc^-1
#taken from Mirizzi & Montanino 2009
Delta_pl_Mpc= lambda n,E: -1.1e-11*n/E + Delta_CMB_Mpc(E)

#n is electron density in 10^-3 cm^-3
#E is Energy in GeV
#returns Delta in kpc^-1
Delta_pl_kpc= lambda n,E: -1.1e-7*n/E + Delta_CMB_kpc(E)

#E is Energy in GeV
#returns Delta in kpc^-1
# additional term from Dobrynina+ 2014, 1412.4771
Delta_CMB_kpc= lambda E: 0.8e-7*E

#E is Energy in TeV
#returns Delta in Mpc^-1
# additional term from Dobrynina+ 2014, 1412.4771
Delta_CMB_Mpc= lambda E: 0.8e-1*E

#B is magnetic field in nG
#E is Energy in TeV
#returns Delta in Mpc^-1
#taken from Mirizzi & Montanino 2009
Delta_QED_Mpc= lambda B,E: 4.1e-9*E*B**2.

#B is magnetic field in muG
#E is Energy in GeV
#returns Delta in kpc^-1
Delta_QED_kpc= lambda B,E: 4.1e-9*E*B**2.

def Delta_Osc_kpc_array(m,n,g,B,E): 
    """
    Compute Delta Osc

    Parameters
    ----------
    m: ALP mass, scalar, in neV
    n: el. density in 10^-3 cm^-3, n-dim array
    g: photon-ALP coyupling strength, scalar
    B: magnetic field in muG, n-dim array
    E: energy in GeV, m-dim array

    Returns
    -------
    Delta_osc as mxn-dim array in kpc^-1
    """
    if np.isscalar(E):
	E = np.array([E])
    if np.isscalar(B):
	B = np.array([B])
    if np.isscalar(n):
	n = np.array([n])
    if not B.shape[0] == n.shape[0]:
	raise ValueError("B and n array have to have the same shape. B.shape: {0}, n.shape: {1}".format(B.shape[0],n.shape[0]))
    result = -7.8e-2 * m ** 2. * (np.ones((E.shape[0],B.shape[0])).transpose()/E).transpose()	# Delta_a as ExB-shaped matrix
    result -= -1.1e-7*((np.ones((E.shape[0],B.shape[0]))*n).transpose()/E).transpose()		# Delta_pl as ExB-shaped matrix
    result *= result
    result += 4. * np.ones((E.shape[0],B.shape[0]))* (1.52e-2*g*B)**2.
    return np.sqrt(result)

def Delta_Osc_Mpc_array(m,n,g,B,E): 
    """
    Compute Delta Osc

    Parameters
    ----------
    m: ALP mass, scalar, in 10^-10 eV
    n: el. density in 10^-7 cm^-3, n-dim array
    g: photon-ALP coupling strength, scalar, in 10^-11 GeV^-1
    B: magnetic field in nG, n-dim array
    E: energy in TeV, m-dim array

    Returns
    -------
    Delta_osc as mxn-dim array in Mpc^-1
    """
    if np.isscalar(E):
	E = np.array([E])
    if np.isscalar(B):
	B = np.array([B])
    if np.isscalar(n):
	n = np.array([n])
    if not B.shape[0] == n.shape[0]:
	raise ValueError("B and n array have to have the same shape. B.shape: {0}, n.shape: {1}".format(B.shape[0],n.shape[0]))
    result = -7.8e-4 * m ** 2. * (np.ones((E.shape[0],B.shape[0])).transpose()/E).transpose()	# Delta_a as ExB-shaped matrix
    result -= -1.1e-11*((np.ones((E.shape[0],B.shape[0]))*n).transpose()/E).transpose()		# Delta_pl as ExB-shaped matrix
    result *= result
    result += 4. * np.ones((E.shape[0],B.shape[0]))* (1.52e-2*g*B)**2.
    return np.sqrt(result)

#Plasma freq in 10^-10 eV
#n is electron density in 10^-7 cm^-3
w_pl_e10 = lambda n: 0.000117*np.sqrt(n)

#Plasma freq in 10^-9 eV
#n is electron density in 10^-3 cm^-3
w_pl_e9 = lambda n: 0.00117*np.sqrt(n)


#from math import abs
#Critical energy for strong mixing regime in TeV
#m is photon axion mass in 10^-10 eV
#n is electron density in 10^-7 cm^-3
#B is magnetic field in nG
#g is photon axion coupling in 10^-11 GeV^-1
Ecrit_TeV= lambda m,n,B,g: 2.5e-2*abs(m**2. - w_pl_e10(n)**2.)/B/g

#Critical energy for strong mixing regime in GeV
#m is axion mass in 10^-09 eV
#n is electron density in 10^-3 cm^-3
#B is magnetic field in muG
#g is photon axion coupling in 10^-11 GeV^-1
Ecrit_GeV= lambda m,n,B,g: 2.5e0*abs(m**2. - w_pl_e9(n)**2.)/B/g

#Maximum energy for strong mixing regime in GeV
#B is magnetic field in muG
#g is photon axion coupling in 10^-11 GeV^-1
Emax_GeV= lambda B,g: 2.12e6 * g / B

# mixing angle
#m is axion mass in 10^-09 eV
#n is electron density in 10^-3 cm^-3
#B is magnetic field in muG
#g is photon axion coupling in 10^-11 GeV^-1
#E is energy in GeV
alpha_kpc = lambda g,B,n,E,m: 0.5 * np.arctan2(2. * Delta_ag_kpc(g,B) , (Delta_pl_kpc(n,E) + 3.5*Delta_QED_kpc(B,E) - Delta_a_kpc(m,E))) 

# mixing angle
#m is axion mass in 10^-10 eV
#n is electron density in 10^-7 cm^-3
#B is magnetic field in nG
#g is photon axion coupling in 10^-11 GeV^-1
#E is energy in TeV
alpha_Mpc = lambda g,B,n,E,m: 0.5 * np.arctan2(2. * Delta_ag_Mpc(g,B) , (Delta_pl_Mpc(n,E) + 3.5*Delta_QED_Mpc(B,E) - Delta_a_Mpc(m,E))) 

# oscillation wave number
#m is axion mass in 10^-09 eV
#n is electron density in 10^-3 cm^-3
#B is magnetic field in muG
#g is photon axion coupling in 10^-11 GeV^-1
#E is energy in GeV
Delta_osc_kpc = lambda g,B,n,E,m  : np.sqrt((Delta_pl_kpc(n,E) + 3.5*Delta_QED_kpc(B,E) - Delta_a_kpc(m,E)) ** 2. + 4. * Delta_ag_kpc(g,B) ** 2.)

# oscillation wave number
#m is axion mass in 10^-10 eV
#n is electron density in 10^-7 cm^-3
#B is magnetic field in nG
#g is photon axion coupling in 10^-11 GeV^-1
#E is energy in TeV
Delta_osc_Mpc = lambda g,B,n,E,m  : np.sqrt((Delta_pl_Mpc(n,E) + 3.5*Delta_QED_Mpc(B,E) - Delta_a_Mpc(m,E)) ** 2. + 4. * Delta_ag_Mpc(g,B) ** 2.)

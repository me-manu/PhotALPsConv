"""
Functions to calculate the Delta parameters that enter into the photon-Axion Mixing matrix.

Version 1.0
- 11/15/11: created
"""

__version__=0.01

#g is photon axion coupling in 10^-11 GeV^-1
#B is magnetic field in nG
#returns Delta in Mpc^-1
#taken from Mirizzi & Montanino 2009
Delta_ag= lambda g,B: 1.52e-2*g*B

#m is photon axion mass in 10^-10 eV
#E is Energy in TeV
#returns Delta in Mpc^-1
#taken from Mirizzi & Montanino 2009
Delta_a= lambda m,E: -7.8e-4*m**2./E

#n is electron density in 10^-7 cm^-3
#E is Energy in TeV
#returns Delta in Mpc^-1
#taken from Mirizzi & Montanino 2009
Delta_pl= lambda n,E: -1.1e-11*n/E

#B is magnetic field in nG
#E is Energy in TeV
#returns Delta in Mpc^-1
#taken from Mirizzi & Montanino 2009
Delta_QED= lambda n,E: 4.1e-9*E*B

#Plasma freq in 10^-10 eV
#n is electron density in 10^-7 cm^-3
w_pl = lambda n: 1.17e-4*n


from math import abs
#Critical energy for strong mixing regime in TeV
#m is photon axion mass in 10^-10 eV
#n is electron density in 10^-7 cm^-3
#B is magnetic field in nG
#g is photon axion coupling in 10^-11 GeV^-1
Ecrit= lambda m,n,B,g: 2.5e-2*abs(m**2. - w_pl(n)**2.)/B/g

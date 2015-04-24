#Example script to show use of PhotALPsConv.calc_conversion module.

# ---- Imports ------------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
import yaml
from PhotALPsConv.conversion_BLR import PhotALPs_BLR as PggBLR
from PhotALPsConv.tools import median_contours
from optparse import OptionParser
# -------------------------------------------------- #

# read in the config file 
parser=OptionParser()
parser.add_option("-c","--config",dest="c",help="config yaml file",action="store")
(opt, args) = parser.parse_args()

par = yaml.load(open(opt.c))
# init the conversion calculation in the BLR
cc = PggBLR(**par)

# the energy array in GeV
EGeV = np.logspace(cc.log10Estart,cc.log10Estop,cc.Estep)

# init matrices that will store the conversion probabilities
# for the different photon (t,u) and ALP (a) polarization states
Pt = np.zeros(EGeV.shape[0])	
Pu = np.zeros(EGeV.shape[0])
Pa = np.zeros(EGeV.shape[0])

polInit		= np.zeros((3,3))	# initial polarization state
polInit[0,0]	= par['pol_t']
polInit[1,1]	= par['pol_u']
polInit[2,2]	= par['pol_a']

polt	= np.zeros((3,3))	
polu	= np.zeros((3,3))
pola	= np.zeros((3,3))

polt[0,0]	= 1.
polu[1,1]	= 1.
pola[2,2]	= 1.

# calculate the mixing
for i,E in enumerate(EGeV):
    pol		= polInit
    cc.E	= E
    T		= cc.SetDomainN()
    pol		= np.dot(T,np.dot(pol,T.transpose().conjugate()))	# new polarization matrix
    Pt[i]	= np.real(np.sum(np.diag(np.dot(polt,pol))))
    Pu[i]	= np.real(np.sum(np.diag(np.dot(polu,pol))))
    Pa[i]	= np.real(np.sum(np.diag(np.dot(pola,pol))))

# plot the results
plt.figure()
ax = plt.subplot(111)
ax.set_xscale("log")
ax.set_yscale("log")

# plot the photon survival probability including ALPs
plt.plot(EGeV,Pt + Pu, ls = '-', color = '0.', lw = 3.)

plt.xlabel("Energy (GeV)", size = 'x-large')
plt.ylabel("Photon survival probability", size = 'x-large')
plt.legend(loc = 3, fontsize = 'x-large')

plt.axis([EGeV[0],EGeV[-1],1e-1,1.1])
plt.show()

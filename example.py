#Example script to show use of PhotALPsConv.calc_conversion module.

# ---- Imports ------------------------------------- #
import numpy as np
import matplotlib.pyplot as plt
import PhotALPsConv.calc_conversion as CC
from PhotALPsConv.tools import median_contours
from optparse import OptionParser
# -------------------------------------------------- #

# read in the config file 
parser=OptionParser()
parser.add_option("-c","--config",dest="c",help="config yaml file",action="store")
(opt, args) = parser.parse_args()

# init the conversion calculation 
cc = CC.Calc_Conv(config = opt.c)

# the energy array in GeV
EGeV = np.logspace(cc.log10Estart,cc.log10Estop,cc.Estep)

# init matrices that will store the conversion probabilities
# for the different photon (t,u) and ALP (a) polarization states
Pt = np.zeros((cc.nsim,EGeV.shape[0]))	
Pu = np.zeros((cc.nsim,EGeV.shape[0]))
Pa = np.zeros((cc.nsim,EGeV.shape[0]))

# calculate the mixing, nsim > 0 only if ICM or IGM are included
for i in range(cc.nsim):
    try:
	cc.scenario.index('Jet')
	new_angles = False
    except ValueError:
	new_angles = True
# calculate the mixing for all energies. If new_angles = True, 
# new random angles will be generated for each random realizatioin
    Pt[i],Pu[i],Pa[i] = cc.calc_conversion(EGeV, new_angles = new_angles)

# calculate the median and 68% and 95% confidence contours
MedCon = median_contours(Pt + Pu)

# plot the results
plt.figure()
ax = plt.subplot(111)
ax.set_xscale("log")
ax.set_yscale("log")

# plot the contours if we are dealing with many realizations
if cc.nsim > 1:
    plt.fill_between(EGeV,MedCon['conf_95'][0],y2 = MedCon['conf_95'][1], color = plt.cm.Greens(0.8))
    plt.fill_between(EGeV,MedCon['conf_68'][0],y2 = MedCon['conf_68'][1], color = plt.cm.Greens(0.5))
    plt.plot(EGeV,MedCon['median'], ls = '-', color = 'gold', label = 'median')
    label = 'one realization'
else:
    label = 'w/ ALPs'

# plot the standard attenuation
plt.plot(EGeV,np.exp(-1. * cc.ebl_norm * cc.tau.opt_depth_array(cc.z,EGeV / 1e3)[0]), ls = '--', color = 'red', label = r'w/o ALPs', lw = 3.)

# plot the photon survival probability including ALPs
plt.plot(EGeV,Pt[0] + Pu[0], ls = '-', color = '0.', label = label, lw = 3.)

plt.xlabel("Energy (GeV)", size = 'x-large')
plt.ylabel("Photon survival probability", size = 'x-large')
plt.legend(loc = 3, fontsize = 'x-large')

plt.axis([EGeV[0],EGeV[-1],1e-1,1.1])
plt.show()

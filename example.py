"""
Example script to show use of PhotALPsConv.calc_conversion module.
"""

import PhotALPsConv.calc_conversion as CC
from PhotALPsConv.tools import median_contours
import numpy as np
import matplotlib.pyplot as plt
import yaml

cc = CC.Calc_Conv(config = "./EXAMPLE.yaml")

EGeV = 10.**np.linspace(-1.,4.5,50)

Pt = np.zeros((cc.nsim,EGeV.shape[0]))	# init matrices that will store the conversion probabilities
Pu = np.zeros((cc.nsim,EGeV.shape[0]))
Pa = np.zeros((cc.nsim,EGeV.shape[0]))

# calculate the mixing, nsim > 0 only if ICM or IGM are included
for i in range(cc.nsim):
    Pt[i],Pu[i],Pa[i] = cc.calc_conversion(EGeV)

# calculate the median and 68% and 95% confidence contours
MedCon = median_contours(Pt + Pu)

# plot the result
plt.figure()
ax = plt.subplot(111)
ax.set_xscale("log")
ax.set_yscale("log")

plt.fill_between(EGeV,MedCon['conf_95'][0],y2 = MedCon['conf_95'][1], color = plt.cm.Greens(0.8))
plt.fill_between(EGeV,MedCon['conf_68'][0],y2 = MedCon['conf_68'][1], color = plt.cm.Greens(0.5))

plt.plot(EGeV,np.exp(-1. * cc.ebl_norm * cc.tau.opt_depth_array(cc.z,EGeV / 1e3)[0]), ls = '--', color = 'red', label = r'$\exp(-\tau)$')

plt.plot(EGeV,MedCon['median'], ls = '-', color = '0.', label = 'median')
plt.plot(EGeV,Pt[0] + Pu[0], ls = '-', color = 'gold', label = 'one realization')

plt.xlabel("Energy (GeV)")
plt.ylabel("Photon survival probability")
plt.legend(loc = 0)

plt.axis([EGeV[0],EGeV[-1],1e-4,1.1])

plt.show()

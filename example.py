"""
Example script to show use of PhotALPsConv.calc_conversion module.
"""

import PhotALPsConv.calc_conversion as CC
from PhotALPsConv.tools import median_contours
import numpy as np
import matplotlib.pyplot as plt
import yaml
from optparse import OptionParser
from scipy.integrate import simps
from scipy.interpolate import interp1d
from PhotALPsConv.deltas import Ecrit_GeV,Delta_Osc_kpc

parser=OptionParser()
parser.add_option("-c","--config",dest="c",help="config yaml file",action="store")
(opt, args) = parser.parse_args()

cc = CC.Calc_Conv(config = opt.c)

elogmin = -1
elogmax = 5
EGeV = 10.**np.linspace(elogmin,elogmax,200)

Pt = np.zeros((cc.nsim,EGeV.shape[0]))	# init matrices that will store the conversion probabilities
Pu = np.zeros((cc.nsim,EGeV.shape[0]))
Pa = np.zeros((cc.nsim,EGeV.shape[0]))

# calculate the mixing, nsim > 0 only if ICM or IGM are included
for i in range(cc.nsim):
    Pt[i],Pu[i],Pa[i] = cc.calc_conversion(EGeV, new_angles = True)
# calculate the median and 68% and 95% confidence contours
MedCon = median_contours(Pt + Pu)


# --- calculate energy dispersion for one realizations
Esteps = 100
Edisp = lambda E,Etrue,sigE :  np.exp(-0.5 * (E - Etrue)**2. / sigE ** 2.) / np.sqrt(2 * np.pi) / sigE

interp = interp1d(np.log(EGeV),np.log(Pt + Pu)[0])
Pgg = lambda E: np.exp(interp(np.log(E)))
Pgglog = lambda logE: np.exp(interp(logE))

EGeVn = 10.**np.linspace(elogmin + 1,elogmax -1,150)

for i,E in enumerate(EGeVn):
    sE = 0.05 * E
    if not i:
	logEarray	= np.linspace(np.log(E - 5. * sE),np.log(E + 5. * sE),Esteps)
	kernel		= Edisp(np.exp(logEarray),E, sE) * Pgglog(logEarray)
    else:
	logE	= np.linspace(np.log(E - 5. * sE),np.log(E + 5. * sE),Esteps)
	logEarray = np.vstack((logEarray,logE))
	ker	= Edisp(np.exp(logE),E, sE) * Pgglog(logE)
	kernel = np.vstack((kernel,ker))

PggEdisp = simps(kernel * np.exp(logEarray), logEarray, axis = 1)

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
plt.plot(EGeVn,PggEdisp, ls = '-', color = 'blue', label = 'one realization - smeared with 6% energy dispersion')

Bave = simps(cc.B,cc.r) / (cc.r[-1] - cc.r[0])
nave = simps(cc.n,cc.r) / (cc.r[-1] - cc.r[0])
Dosc = Delta_Osc_kpc(cc.m,nave,cc.g,Bave,EGeV)
print 'maximum osc. Delta: {0:.2e}'.format(np.max(Dosc))

plt.axvline(Ecrit_GeV(cc.m,nave,Bave,cc.g), ls = '--', color = '0.', label = r'$E_\mathrm{{crit}}(\langle B \rangle = {0:.2f}\mu\mathrm{{G}})$'.format(Bave))

plt.xlabel("Energy (GeV)")
plt.ylabel("Photon survival probability")
plt.legend(loc = 0)

plt.axis([EGeV[0],EGeV[-1],1e-2,1.1])
plt.savefig('conversion.pdf',format = 'pdf')
plt.show()

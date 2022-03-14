import numpy as np
import matplotlib.pyplot as plt
import sys
import pyccl as ccl
from scipy.special import erf


from pathlib import Path
import time

# get project directory
path = Path.cwd().parent.parent


start_time = time.perf_counter()

params = {'lines.linewidth' : 3.5,
          'font.size' : 20,
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large',
          'mathtext.fontset': 'stix',
          'font.family': 'STIXGeneral',
          'figure.figsize': (10, 10)
          }
plt.rcParams.update(params)
markersize = 10

###############################################################################
###############################################################################
###############################################################################

# Create new Cosmology object with a given set of parameters. This keeps track of previously-computed cosmological functions
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, w0=-1, wa=0, h=0.67, sigma8=0.815583, n_s=0.96, m_nu=0.06, Omega_k=1-(0.27+0.05)-0.68)


# Define redshift distribution of sources kernels
ztab = np.arange(0, 2.5, 0.001)

zmin, zmax = 0.001, 2.5
zi = np.array([
    [zmin, 0.418, 0.56, 0.678, 0.789, 0.9, 1.019, 1.155, 1.324, 1.576],
    [0.418, 0.56, 0.678, 0.789, 0.9, 1.019, 1.155, 1.324, 1.576, zmax]
])
Nbins = len(zi[0])

nzEuclid = 30*(ztab/0.9*np.sqrt(2))**2*np.exp(-(ztab/0.9*np.sqrt(2))**1.5)

fout, cb, zb, sigmab, c0, z0, sigma0 = 0.1, 1, 0, 0.05, 1, 0.1, 0.05

nziEuclid = np.array([ nzEuclid
                      *1/2/c0/cb*(
                          cb*fout*(erf((ztab-z0-c0*zi[0, iz])/np.sqrt(2)/(1+ztab)/sigma0)
                                   -erf((ztab-z0-c0*zi[1,iz])/np.sqrt(2)/(1+ztab)/sigma0))
                          +c0*(1-fout)*(erf((ztab-zb-cb*zi[0, iz])/np.sqrt(2)/(1+ztab)/sigmab)
                                        -erf((ztab-zb-cb*zi[1,iz])/np.sqrt(2)/(1+ztab)/sigmab))
                      ) for iz in range(Nbins) ])

plt.xlabel('$z$')
plt.ylabel('$n_i(z)\,[\mathrm{arcmin}^{-2}]$')
[ plt.plot(ztab, nziEuclid[iz]) for iz in range(Nbins)]
plt.show()


#Import look-up tables for IAs

IAFILE = np.genfromtxt('InputFiles/scaledmeanlum-E2Sa.dat')
FIAzNoCosmoNoGrowth = -1 * 1.72 * 0.0134 * (1+IAFILE[:, 0])**(-0.41) * IAFILE[:, 1]**2.17

# Computes the WL (w/ and w/o IAs) and GCph kernels

#WCS = [ ccl.WeakLensingTracer(cosmo, dndz=(ztab, nziEuclid[iz])) for iz in range(Nbins) ]

FIAz = FIAzNoCosmoNoGrowth*(cosmo.cosmo.params.Omega_c+cosmo.cosmo.params.Omega_b)/ccl.growth_factor(cosmo,1/(1+IAFILE[:, 0]))
WL = [ ccl.WeakLensingTracer(cosmo, dndz=(ztab, nziEuclid[iz]), ia_bias=(IAFILE[:, 0], FIAz), use_A_ia=False) for iz in range(Nbins) ]



# Import fiducial P(k,z)
PkFILE = np.genfromtxt('InputFiles/pkz-Fiducial.txt')

# Populates vectors for z, k [1/Mpc], and P(k,z) [Mpc^3]
zlist = np.unique(PkFILE[:, 0])
klist = PkFILE[:int(len(PkFILE[:, 2])/len(zlist)),1]*cosmo.cosmo.params.h
Pklist = PkFILE[:, 3].reshape(len(zlist), len(klist))/cosmo.cosmo.params.h**3

# Create a Pk2D object
Pk = ccl.Pk2D(a_arr=1/(1+zlist[::-1]), lk_arr=np.log(klist), pk_arr=Pklist, is_logp=False)

ell = np.geomspace(10, 5000, 20)


CLL = np.array([ [ ccl.angular_cl(cosmo, WL[iz], WL[jz], ell, p_of_k_a=Pk) for iz in range(Nbins) ] for jz in range(Nbins) ])

A_deg = 15e3
f_sky = A_deg*(np.pi/180)**2 / (4*np.pi)
n_gal = 30*(180*60/np.pi)**2
sigma_e = 0.3




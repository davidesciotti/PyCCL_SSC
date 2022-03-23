import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
from scipy.special import erf

# get project directory
project_path = Path.cwd().parent

start_time = time.perf_counter()

params = {'lines.linewidth': 3.5,
          'font.size': 20,
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
          'mathtext.fontset': 'stix',
          'font.family': 'STIXGeneral',
          'figure.figsize': (10, 10)
          }
plt.rcParams.update(params)
markersize = 10

###############################################################################
###############################################################################
###############################################################################

# Create new Cosmology object with a given set of parameters. This keeps track of previously-computed cosmological
# functions
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, w0=-1, wa=0, h=0.67, sigma8=0.815583, n_s=0.96, m_nu=0.06,
                      Omega_k=1 - (0.27 + 0.05) - 0.68)

nbl = 20

# Define redshift distribution of sources kernels
ztab = np.arange(0, 2.5, 0.001)
zmin, zmax = 0.001, 2.5
zi = np.array([
    [zmin, 0.418, 0.56, 0.678, 0.789, 0.9, 1.019, 1.155, 1.324, 1.576],
    [0.418, 0.56, 0.678, 0.789, 0.9, 1.019, 1.155, 1.324, 1.576, zmax]
])
Nbins = len(zi[0])

nzEuclid = 30 * (ztab / 0.9 * np.sqrt(2)) ** 2 * np.exp(-(ztab / 0.9 * np.sqrt(2)) ** 1.5)

fout, cb, zb, sigmab, c0, z0, sigma0 = 0.1, 1, 0, 0.05, 1, 0.1, 0.05

nziEuclid = np.array([nzEuclid
                      * 1 / 2 / c0 / cb * (
                              cb * fout * (erf((ztab - z0 - c0 * zi[0, iz]) / np.sqrt(2) / (1 + ztab) / sigma0)
                                           - erf((ztab - z0 - c0 * zi[1, iz]) / np.sqrt(2) / (1 + ztab) / sigma0))
                              + c0 * (1 - fout) * (erf((ztab - zb - cb * zi[0, iz]) / np.sqrt(2) / (1 + ztab) / sigmab)
                                                   - erf(
                                  (ztab - zb - cb * zi[1, iz]) / np.sqrt(2) / (1 + ztab) / sigmab))
                      ) for iz in range(Nbins)])

plt.xlabel('$z$')
plt.ylabel('$n_i(z)\,[\mathrm{arcmin}^{-2}]$')
[plt.plot(ztab, nziEuclid[iz]) for iz in range(Nbins)]
# plt.show()

# Import look-up tables for IAs

IAFILE = np.genfromtxt(project_path / 'input/scaledmeanlum-E2Sa.dat')
FIAzNoCosmoNoGrowth = -1 * 1.72 * 0.0134 * (1 + IAFILE[:, 0]) ** (-0.41) * IAFILE[:, 1] ** 2.17

# Computes the WL (w/ and w/o IAs) and GCph kernels

# WCS = [ ccl.WeakLensingTracer(cosmo, dndz=(ztab, nziEuclid[iz])) for iz in range(Nbins) ]

FIAz = FIAzNoCosmoNoGrowth * (cosmo.cosmo.params.Omega_c + cosmo.cosmo.params.Omega_b) / ccl.growth_factor(cosmo, 1 / (
        1 + IAFILE[:, 0]))
WL = [ccl.WeakLensingTracer(cosmo, dndz=(ztab, nziEuclid[iz]), ia_bias=(IAFILE[:, 0], FIAz), use_A_ia=False) for iz in
      range(Nbins)]


# Import fiducial P(k,z)
PkFILE = np.genfromtxt(project_path / 'input/pkz-Fiducial.txt')

# Populates vectors for z, k [1/Mpc], and P(k,z) [Mpc^3]
zlist = np.unique(PkFILE[:, 0])
klist = PkFILE[:int(len(PkFILE[:, 2]) / len(zlist)), 1] * cosmo.cosmo.params.h
Pklist = PkFILE[:, 3].reshape(len(zlist), len(klist)) / cosmo.cosmo.params.h ** 3

# Create a Pk2D object
a_arr = 1 / (1 + zlist[::-1])
lk_arr = np.log(klist)
Pk = ccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=Pklist, is_logp=False)

ell = np.geomspace(10, 5000, nbl)


CLL = np.array(
    [[ccl.angular_cl(cosmo, WL[iz], WL[jz], ell, p_of_k_a=Pk) for iz in range(Nbins)] for jz in range(Nbins)])


A_deg = 15e3
f_sky = A_deg * (np.pi / 180) ** 2 / (4 * np.pi)
n_gal = 30 * (180 * 60 / np.pi) ** 2
sigma_e = 0.3

# TODO we have no clue about the values of Delta and rho_type
# name = 'Bhattacharya13'
# mass_def = ccl.halos.massdef.MassDef(Delta='vir', rho_type='matter', c_m_relation=name)

# from https://ccl.readthedocs.io/en/latest/api/pyccl.halos.massdef.html?highlight=.halos.massdef.MassDef#pyccl.halos.massdef.MassDef200c
mass_def = ccl.halos.massdef.MassDef200c(c_m='Duffy08')

# TODO pass mass_def object? plus, understand what the hell is mass_def_strict
# mass_def must not be None
# don't use a default MassFunction or HaloBias classes, they're probably abstract classes
# it's all default settings except for mass_def

# massfunc = ccl.halos.hmfunc.MassFunc(cosmo, mass_def=mass_def, mass_def_strict=True)
massfunc = ccl.halos.hmfunc.MassFuncTinker08(cosmo, mass_def=mass_def, mass_def_strict=True)

# hbias = ccl.halos.hbias.HaloBias(cosmo, mass_def=mass_def, mass_def_strict=True)
hbias = ccl.halos.hbias.HaloBiasTinker10(cosmo, mass_def=mass_def, mass_def_strict=True)

hmc = ccl.halos.halo_model.HMCalculator(cosmo, massfunc, hbias, mass_def=mass_def,
                                        log10M_min=8.0, log10M_max=16.0, nlog10M=128,
                                        integration_method_M='simpson', k_min=1e-05)

# TODO pick a non-random one from https://ccl.readthedocs.io/en/latest/api/pyccl.halos.profiles.html#pyccl.halos.profiles.HaloProfile
prof_GNFW = ccl.halos.profiles.HaloProfilePressureGNFW(mass_bias=0.8, P0=6.41, c500=1.81, alpha=1.33, alpha_P=0.12,
                                                       beta=4.13,
                                                       gamma=0.31, P0_hexp=-1.0, qrange=(0.001, 1000.0), nq=128,
                                                       x_out=np.inf)

# from https://ccl.readthedocs.io/en/latest/api/pyccl.halos.halo_model.html?highlight=trispectrum#pyccl.halos.halo_model.halomod_Tk3D_SSC
tkka = ccl.halos.halo_model.halomod_Tk3D_SSC(cosmo, hmc, prof1=prof_GNFW, prof2=None, prof12_2pt=None, prof3=None,
                                             prof4=None, prof34_2pt=None, normprof1=False, normprof2=False,
                                             normprof3=False, normprof4=False,
                                             p_of_k_a=Pk, lk_arr=lk_arr, a_arr=a_arr, extrap_order_lok=1,
                                             extrap_order_hik=1, use_log=False)

cov_SSC = np.zeros(())
# TODO correct fsky?
for i in range(Nbins):
    for j in range(Nbins):
        for k in range(Nbins):
            for l in range(Nbins):
                cltracer1, cltracer2 = CLL, CLL

                WL_i = ccl.WeakLensingTracer(cosmo, dndz=(ztab, nziEuclid[i]), ia_bias=(IAFILE[:, 0], FIAz), use_A_ia=False)
                WL_j = ccl.WeakLensingTracer(cosmo, dndz=(ztab, nziEuclid[j]), ia_bias=(IAFILE[:, 0], FIAz), use_A_ia=False)
                WL_k = ccl.WeakLensingTracer(cosmo, dndz=(ztab, nziEuclid[k]), ia_bias=(IAFILE[:, 0], FIAz), use_A_ia=False)
                WL_l = ccl.WeakLensingTracer(cosmo, dndz=(ztab, nziEuclid[l]), ia_bias=(IAFILE[:, 0], FIAz), use_A_ia=False)

                CLL_ij = ccl.angular_cl(cosmo, WL_i, WL_j, ell, p_of_k_a=Pk)
                CLL_kl = ccl.angular_cl(cosmo, WL_k, WL_l, ell, p_of_k_a=Pk)

                cov_SSC[:, :, i, j, k, l] = ccl.covariances.angular_cl_cov_SSC(cosmo, CLL_ij, CLL_kl, ell, tkka, sigma2_B=None, fsky=0.37,
                                                   cltracer3=None, cltracer4=None, ell2=None, integration_method='qag_quad')


# from https://ccl.readthedocs.io/en/latest/api/pyccl.core.html?highlight=trispectrum#pyccl.core.Cosmology.angular_cl_cov_SSC
# cov_SSC_wishfulthinking = ccl.angular_cl_cov_SSC(cltracer1=CLL, cltracer2=CLL, ell, tkka, sigma2_B=None, fsky=1.0, cltracer3=CLL,
#                                              cltracer4=CLL, ell2=None, integration_method='qag_quad')


print('done')

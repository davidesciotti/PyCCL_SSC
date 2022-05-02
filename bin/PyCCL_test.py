import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
from scipy.special import erf

sys.path.append(str('/Users/davide/Documents/Lavoro/Programmi/SSC_restructured/lib'))
import my_module as mm

matplotlib.use('Qt5Agg')

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

# define fsky Euclid
survey_area = 15000  # deg^2
deg2_in_sphere = 41252.96  # deg^2 in a spere
fsky_IST = survey_area / deg2_in_sphere

# Create new Cosmology object with a given set of parameters. This keeps track of previously-computed cosmological
# functions
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, w0=-1., wa=0., h=0.67, sigma8=0.815583, n_s=0.96, m_nu=0.06,
                      Omega_k=1 - (0.27 + 0.05) - 0.68)

nbl = 20

# Define redshift distribution of sources kernels
ztab = np.arange(0, 2.5, 0.001)
zmin, zmax = 0.001, 2.5
zi = np.array([
    [zmin, 0.418, 0.56, 0.678, 0.789, 0.9, 1.019, 1.155, 1.324, 1.576],
    [0.418, 0.56, 0.678, 0.789, 0.9, 1.019, 1.155, 1.324, 1.576, zmax]
])

zbins = len(zi[0])

# get number of redshift pairs
npairs_auto, npairs_asimm, npairs_tot = mm.get_pairs(zbins)

nzEuclid = 30 * (ztab / 0.9 * np.sqrt(2)) ** 2 * np.exp(-(ztab / 0.9 * np.sqrt(2)) ** 1.5)

fout, cb, zb, sigmab, c0, z0, sigma0 = 0.1, 1, 0, 0.05, 1, 0.1, 0.05

nziEuclid = np.array([nzEuclid * 1 / 2 / c0 / cb * (cb * fout * (
        erf((ztab - z0 - c0 * zi[0, iz]) / np.sqrt(2) / (1 + ztab) / sigma0)
        - erf((ztab - z0 - c0 * zi[1, iz]) / np.sqrt(2) / (1 + ztab) / sigma0))
                                                    + c0 * (1 - fout) * (
                                                                erf((ztab - zb - cb * zi[0, iz]) / np.sqrt(2) / (
                                                                            1 + ztab) / sigmab)
                                                                - erf((ztab - zb - cb * zi[1, iz]) / np.sqrt(2) / (
                                                                    1 + ztab) / sigmab))) for iz in range(zbins)])

# plt.xlabel('$z$')
# plt.ylabel('$n_i(z)\,[\mathrm{arcmin}^{-2}]$')
# [plt.plot(ztab, nziEuclid[iz]) for iz in range(Nbins)]
# plt.show()

# Import look-up tables for IAs

IAFILE = np.genfromtxt(project_path / 'input/scaledmeanlum-E2Sa.dat')
FIAzNoCosmoNoGrowth = -1 * 1.72 * 0.0134 * (1 + IAFILE[:, 0]) ** (-0.41) * IAFILE[:, 1] ** 2.17

# Computes the WL (w/ and w/o IAs) and GCph kernels

# WCS = [ ccl.WeakLensingTracer(cosmo, dndz=(ztab, nziEuclid[iz])) for iz in range(Nbins) ]

FIAz = FIAzNoCosmoNoGrowth * (cosmo.cosmo.params.Omega_c + cosmo.cosmo.params.Omega_b) / ccl.growth_factor(cosmo, 1 / (
        1 + IAFILE[:, 0]))
WL = [ccl.WeakLensingTracer(cosmo, dndz=(ztab, nziEuclid[iz]), ia_bias=(IAFILE[:, 0], FIAz), use_A_ia=False) for iz in
      range(zbins)]

# ztab = np.expand_dims(ztab, axis=0)
# ztab = np.repeat(ztab, repeats=10, axis=0)
# WL_obj = ccl.WeakLensingTracer(cosmo, dndz=(ztab, nziEuclid.T), ia_bias=(IAFILE[:, 0], FIAz), use_A_ia=False)

# Import fiducial P(k,z)
PkFILE = np.genfromtxt(project_path / 'input/pkz-Fiducial.txt')

# Populates vectors for z, k [1/Mpc], and P(k,z) [Mpc^3]
zlist = np.unique(PkFILE[:, 0])
k_points = int(len(PkFILE[:, 2]) / len(zlist))
klist = PkFILE[:k_points, 1] * cosmo.cosmo.params.h
z_points = len(zlist)
Pklist = PkFILE[:, 3].reshape(z_points, k_points) / cosmo.cosmo.params.h ** 3

# Create a Pk2D object
a_arr = 1 / (1 + zlist[::-1])
lk_arr = np.log(klist)  # it's the natural log, not log10
Pk = ccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=Pklist, is_logp=False)

# IST:F?
# ell = np.geomspace(10, 5000, nbl)

# IST:NL
ell_bins = np.linspace(np.log(10.), np.log(5000.), 21)
ell = (ell_bins[:-1] + ell_bins[1:]) / 2.
ell = np.exp(ell)
deltas = np.diff(np.exp(ell_bins))

CLL = np.array(
    [[ccl.angular_cl(cosmo, WL[iz], WL[jz], ell, p_of_k_a=Pk) for iz in range(zbins)] for jz in range(zbins)])

A_deg = 15e3
f_sky = A_deg * (np.pi / 180) ** 2 / (4 * np.pi)
n_gal = 30 * (180 * 60 / np.pi) ** 2
sigma_e = 0.3

# notebook per mass_relations: https://github.com/LSSTDESC/CCLX/blob/master/Halo-mass-function-example.ipynb
# Cl notebook: https://github.com/LSSTDESC/CCL/blob/v2.0.1/examples/3x2demo.ipynb

# TODO we're nto sure' about the values of Delta and rho_type
# mass_def = ccl.halos.massdef.MassDef(Delta='vir', rho_type='matter', c_m_relation=name)

# from https://ccl.readthedocs.io/en/latest/api/pyccl.halos.massdef.html?highlight=.halos.massdef.MassDef#pyccl.halos.massdef.MassDef200c

# ! HALO MODEL PRESCRIPTIONS:
# KiDS_1000 Methodology:
# https://www.pure.ed.ac.uk/ws/portalfiles/portal/188893969/2007.01844v2.pdf, after (E.10)

# Krause_2017: https://arxiv.org/pdf/1601.05779.pdf
# about the mass definition, the paper says:
# "Throughout this paper we define halo properties using the over density ∆ = 200 ¯ρ, with ¯ρ the mean matter density"

hm_recipe = 'KiDS_1000'

# ! mass definition
if hm_recipe == 'KiDS_1000':
    c_m = 'Duffy08'  # ! NOT SURE AT ALL!
elif hm_recipe == 'Krause_2017':
    c_m = 'Bhattacharya13'  # see paper, after Eq. 1

mass_def = ccl.halos.MassDef200m(c_m=c_m)

# TODO pass mass_def object? plus, understand what exactly is mass_def_strict
# mass_def must not be None

# ! mass function
massfunc = ccl.halos.hmfunc.MassFuncTinker10(cosmo, mass_def=mass_def, mass_def_strict=True)

# ! halo bias
hbias = ccl.halos.hbias.HaloBiasTinker10(cosmo, mass_def=mass_def, mass_def_strict=True)

# ! concentration-mass relation
if hm_recipe == 'KiDS_1000':
    c_M_relation = ccl.halos.concentration.ConcentrationDuffy08(mdef=mass_def)
elif hm_recipe == 'Krause_2017':
    c_M_relation = ccl.halos.concentration.ConcentrationBhattacharya13(mdef=mass_def)  # over Eq. 12


# TODO understand better this object. We're calling the abstract class, is this ok?
# ! HMCalculator
hmc = ccl.halos.halo_model.HMCalculator(cosmo, massfunc, hbias, mass_def=mass_def,
                                        log10M_min=8.0, log10M_max=16.0, nlog10M=128,
                                        integration_method_M='simpson', k_min=1e-05)

# ! halo profile
halo_profile = ccl.halos.profiles.HaloProfileNFW(c_M_relation=c_M_relation,
                                                 fourier_analytic=True, projected_analytic=False,
                                                 cumul2d_analytic=False, truncated=True)

# it was p_of_k_a=Pk, but it should use the LINEAR power spectrum (see documentation:
# https://ccl.readthedocs.io/en/latest/api/pyccl.halos.halo_model.html?highlight=halomod_Tk3D_SSC#pyccl.halos.halo_model.halomod_Tk3D_SSC)
# 🐛 bug solved: normprof shoud be True
tkka = ccl.halos.halo_model.halomod_Tk3D_SSC(cosmo, hmc,
                                             prof1=halo_profile, prof2=None, prof12_2pt=None,
                                             prof3=None, prof4=None, prof34_2pt=None,
                                             normprof1=True, normprof2=True, normprof3=True, normprof4=True,
                                             p_of_k_a=None, lk_arr=lk_arr, a_arr=a_arr, extrap_order_lok=1,
                                             extrap_order_hik=1, use_log=False)


zbins = 2
cov_PyCCL_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
# for ell2_idx, ell2 in enumerate(ell): # TODO is this necessary?
for i in range(zbins):
    for j in range(zbins):
        start = time.perf_counter()
        for k in range(zbins):
            for l in range(zbins):
                cov_PyCCL_6D[:, :, i, j, k, l] = ccl.covariances.angular_cl_cov_SSC(cosmo, WL[i], WL[j], ell, tkka,
                                                                                    sigma2_B=None, fsky=fsky_IST,
                                                                                    cltracer3=WL[k], cltracer4=WL[l],
                                                                                    ell2=None,
                                                                                    integration_method='spline')

        print(f'i, j redshift bins: {i}, {j}, computed in  {(time.perf_counter() - start):.2f} seconds')

mm.matshow(cov_PyCCL_6D[:, :, 0, 0, 0, 0], log=True, title='cov_PyCCL_6D')

# SAVE
# np.save(project_path / 'output/cov_SSC_p_of_k_a=None.npy', cov_SSC)


ind = np.genfromtxt(
    '/Users/davide/Documents/Lavoro/Programmi/SSC_restructured/config/common_data/ind/indici_cloe_like.dat').astype(
    'int') - 1
ind_LL = ind[:npairs_auto, :]

# reshape
cov_PyCCL_4D = mm.cov_6D_to_4D(cov_PyCCL_6D, nbl=nbl, npairs=npairs_auto, ind=ind_LL)

# import Robin and PySSC
cov_Robin_2D = np.load(
    '/Users/davide/Documents/Lavoro/Programmi/SSC_paper_jan22/PySSC_vs_CosmoLike/Robin/cov_SS_full_sky_rescaled/lmax5000_noextrap/davides_reshape/cov_R_WL_SSC_lmax5000_2D.npy')
cov_PySSC_4D = np.load(project_path / 'input/CovMat-ShearShear-SSC-20bins-NL_flag_2_4D.npy')

# reshape
cov_Robin_4D = mm.cov_2D_to_4D(cov_Robin_2D, nbl=nbl, npairs=npairs_auto, block_index='vincenzo')
cov_PySSC_6D = mm.cov_4D_to_6D(cov_PySSC_4D, nbl, zbins, probe='LL', ind=ind_LL)

# show the various versions
mm.matshow(cov_PySSC_4D[:, :, 0, 0], log=True, title='cov_PySSC_4D')
mm.matshow(cov_Robin_4D[:, :, 0, 0], log=True, title='cov_Robin_4D')
mm.matshow(cov_PyCCL_4D[:, :, 0, 0], log=True, title='cov_PyCCL_4D')

# compute and plot percent difference (not in log scale)
rob_over_PyCCL = mm.percent_diff(cov_Robin_4D, cov_PyCCL_4D)
rob_over_PySSC = mm.percent_diff(cov_Robin_4D, cov_PySSC_4D)

mm.matshow(rob_over_PyCCL[:, :, 0, 0], log=False, title='rob_over_PyCCL')
mm.matshow(rob_over_PySSC[:, :, 0, 0], log=False, title='rob_over_PySSC')

# correlation matrix
# corr_PyCCL = mm.correlation_from_covariance(cov_SSC[:, :, 0, 0, 0, 0])
# corr_PySSC = mm.correlation_from_covariance(cov_SS_WL_old_6D[1, :, :, 0, 0, 0, 0])
# mm.matshow(corr_PyCCL, log=True, title='PyCCL')
# mm.matshow(corr_PySSC, log=True, title='PySSC')


# from https://ccl.readthedocs.io/en/latest/api/pyccl.core.html?highlight=trispectrum#pyccl.core.Cosmology.angular_cl_cov_SSC
# cov_SSC_wishfulthinking = ccl.angular_cl_cov_SSC(cltracer1=CLL, cltracer2=CLL, ell, tkka, sigma2_B=None, fsky=1.0, cltracer3=CLL,
#                                              cltracer4=CLL, ell2=None, integration_method='qag_quad')


print('done')

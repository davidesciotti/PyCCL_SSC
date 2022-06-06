import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
from scipy.special import erf

# get project directory
project_path = Path.cwd().parent

sys.path.append(str(project_path.parent))
import SSC_restructured_v2.lib.my_module as mm

matplotlib.use('Qt5Agg')

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

def bias(z, zi):
    zbins = len(zi[0])
    z_minus = zi[0, :]  # lower edge of z bins
    z_plus = zi[1, :]  # upper edge of z bins
    z_mean = (z_minus + z_plus) / 2  # cener of the z bins

    for i in range(zbins):
        if z_minus[i] <= z < z_plus[i]:
            return b(i, z_mean)
        if z > z_plus[-1]:  # max redshift bin
            return b(9, z_mean)


def b(i, z_mean):
    return np.sqrt(1 + z_mean[i])


def compute_SSC_PyCCL(cosmo, kernel_A, kernel_B, kernel_C, kernel_D, ell, tkka, f_sky, integration_method='spline'):

    cov_SS_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))

    start_SSC_timer = time.perf_counter()
    for i in range(zbins):
        for j in range(zbins):
            start = time.perf_counter()
            for k in range(zbins):
                for l in range(zbins):
                    cov_SS_6D[:, :, i, j, k, l] = ccl.covariances.angular_cl_cov_SSC(cosmo, kernel_A[i], kernel_B[j],
                                                                                     ell, tkka,
                                                                                     sigma2_B=None, fsky=f_sky,
                                                                                     cltracer3=kernel_C[k],
                                                                                     cltracer4=kernel_D[l],
                                                                                     ell2=None,
                                                                                     integration_method=integration_method)

            print(f'i, j redshift bins: {i}, {j}, computed in  {(time.perf_counter() - start):.2f} seconds')
    print(f'SSC computed in  {(time.perf_counter() - start_SSC_timer):.2f} seconds')

    return cov_SS_6D


###############################################################################
###############################################################################
###############################################################################

# ! POTENTIAL ISSUES:
# 1. input files (WF, ell, a, pk...)
# 2. halo model recipe
# 3. ordering of the resulting covariance matrix


# Create new Cosmology object with a given set of parameters. This keeps track of previously-computed cosmological
# functions
cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.05, w0=-1., wa=0., h=0.67, sigma8=0.815583, n_s=0.96, m_nu=0.06,
                      Omega_k=1 - (0.27 + 0.05) - 0.68)

# Define redshift distribution of sources kernels
zmin, zmax = 0.001, 2.5
ztab = np.arange(zmin, zmax, 0.001)  # ! should it start from 0 instead?
zi = np.array([
    [zmin, 0.418, 0.56, 0.678, 0.789, 0.9, 1.019, 1.155, 1.324, 1.576],
    [0.418, 0.56, 0.678, 0.789, 0.9, 1.019, 1.155, 1.324, 1.576, zmax]
])

zbins = len(zi[0])

# get number of redshift pairs
npairs_auto, npairs_asimm, npairs_tot = mm.get_pairs(zbins)

nzEuclid = 30 * (ztab / 0.9 * np.sqrt(2)) ** 2 * np.exp(-(ztab / 0.9 * np.sqrt(2)) ** 1.5)

fout, cb, zb, sigmab, c0, z0, sigma0 = 0.1, 1, 0, 0.05, 1, 0.1, 0.05

nziEuclid = np.array([nzEuclid * 1 / 2 / c0 / cb * (cb * fout *
                                                    (erf((ztab - z0 - c0 * zi[0, iz]) / np.sqrt(2) / (
                                                            1 + ztab) / sigma0) -
                                                     erf((ztab - z0 - c0 * zi[1, iz]) / np.sqrt(2) / (
                                                             1 + ztab) / sigma0)) + c0 * (1 - fout) *
                                                    (erf((ztab - zb - cb * zi[0, iz]) / np.sqrt(2) / (
                                                            1 + ztab) / sigmab) -
                                                     erf((ztab - zb - cb * zi[1, iz]) / np.sqrt(2) / (
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
wil = [ccl.WeakLensingTracer(cosmo, dndz=(ztab, nziEuclid[iz]), ia_bias=(IAFILE[:, 0], FIAz), use_A_ia=False) for iz in
       range(zbins)]

# GALAXY KERNELS

# construct the bias array -
b_array = np.asarray([bias(z, zi) for z in ztab])
# it should be the same for all redshift bins:
# b_array = np.repeat(b_array[:, np.newaxis], zbins, axis=1)  # this is useless, I can just pass the same array each
# time in the call below

# ! the bias is not used by this function!!!
wig = [ccl.tracers.NumberCountsTracer(cosmo, has_rsd=False, dndz=(ztab, nziEuclid[iz]), bias=(ztab, b_array),
                                      mag_bias=None) for iz in range(zbins)]

# get kernels and store them into arrays

# scale factor and comoving distance
# a_arr = 1 / (1 + ztab[::-1])
# chi = ccl.background.comoving_radial_distance(cosmo, a_arr[::-1])  # in Mpc
# wig_array = np.zeros((zbins, len(ztab)))
# wil_array = np.zeros((zbins, len(ztab)))
#
# for zbin in range(zbins):
#     # wil_values = wil[zbin].get_kernel(chi=chi)
#     wig_array[zbin, :] = wig[zbin].get_kernel(chi=chi)[0, :]
#     wil_array[zbin, :] = wil[zbin].get_kernel(chi=chi)[0, :]
#     plt.plot(ztab, wil_array[zbin, :])


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

# choose the ell binning
ell_min = 10
ell_max_WL = 5000
nbl = 30

# ! settings
which_ells = 'IST-F'
compute_SS_WL = False
compute_SS_GC = True
save_SSC = True
compute_cNG = False
hm_recipe = 'KiDS_1000'
# ! settings

if which_ells == 'IST-F':
    nbl = 30
elif which_ells == 'IST-NL':
    nbl = 20
else:
    raise ValueError('which_ells should be IST-F or IST-NL')

# TODO why should nbl be equal to 20? was I drunk?
# if which_ells == 'IST-F' and nbl == 20:
if which_ells == 'IST-F':
    # IST:F recipe:
    ell_WL = np.logspace(np.log10(ell_min), np.log10(ell_max_WL), nbl + 1)  # WL
    # central values of each bin
    l_centr_WL = (ell_WL[1:] + ell_WL[:-1]) / 2
    # take the log10 of the values
    logarithm_WL = np.log10(l_centr_WL)
    # update the ell_WL, ell_GC arrays with the right values
    ell_WL = logarithm_WL
    # ell values in linear scale:
    l_lin_WL = 10 ** ell_WL
    ell = l_lin_WL

# elif which_ells == 'IST-NL' and nbl == 20:
elif which_ells == 'IST-NL':
    # this is slightly different
    ell_bins = np.linspace(np.log(ell_min), np.log(ell_max_WL), nbl + 1)
    ell = (ell_bins[:-1] + ell_bins[1:]) / 2.
    ell = np.exp(ell)

else:
    raise ValueError('Wrong choice of ell bins: which_ells must be either IST-F or IST-NL, and nbl must be 20')

# jsut a check on the settings
print(
    f'settings:\nwhich_ells = {which_ells}\nnbl = {nbl}\nhm_recipe = {hm_recipe}\ncompute_SS_WL = {compute_SS_WL}'
    f'\ncompute_SS_GC = {compute_SS_GC} \ncompute_cNG = {compute_cNG}')

CLL = np.array([[ccl.angular_cl(cosmo, wil[iz], wil[jz], ell, p_of_k_a=Pk)
                 for iz in range(zbins)]
                for jz in range(zbins)])

A_deg = 15e3
f_sky = A_deg * (np.pi / 180) ** 2 / (4 * np.pi)
n_gal = 30 * (180 * 60 / np.pi) ** 2
sigma_e = 0.3

# save wf and cl for validation
np.save(project_path / 'output/wl_and_cl_validation/ztab.npy', ztab)
# np.save(project_path / 'output/wl_and_cl_validation/wil_array.npy', wil_array)
# np.save(project_path / 'output/wl_and_cl_validation/wig_array.npy', wig_array)
np.save(project_path / 'output/wl_and_cl_validation/ell.npy', ell)
np.save(project_path / 'output/wl_and_cl_validation/C_LL.npy', CLL)
np.save(project_path / 'output/wl_and_cl_validation/nziEuclid.npy', nziEuclid)



# notebook per mass_relations: https://github.com/LSSTDESC/CCLX/blob/master/Halo-mass-function-example.ipynb
# Cl notebook: https://github.com/LSSTDESC/CCL/blob/v2.0.1/examples/3x2demo.ipynb

# TODO we're not sure about the values of Delta and rho_type
# mass_def = ccl.halos.massdef.MassDef(Delta='vir', rho_type='matter', c_m_relation=name)

# from https://ccl.readthedocs.io/en/latest/api/pyccl.halos.massdef.html?highlight=.halos.massdef.MassDef#pyccl.halos.massdef.MassDef200c

# ! HALO MODEL PRESCRIPTIONS:
# KiDS_1000 Methodology:
# https://www.pure.ed.ac.uk/ws/portalfiles/portal/188893969/2007.01844v2.pdf, after (E.10)

# Krause_2017: https://arxiv.org/pdf/1601.05779.pdf
# about the mass definition, the paper says:
# "Throughout this paper we define halo properties using the over density ∆ = 200 ¯ρ, with ¯ρ the mean matter density"

# mass definition
if hm_recipe == 'KiDS_1000':
    c_m = 'Duffy08'  # ! NOT SURE ABOUT THIS
    mass_def = ccl.halos.MassDef200m(c_m=c_m)
    c_M_relation = ccl.halos.concentration.ConcentrationDuffy08(mdef=mass_def)
elif hm_recipe == 'Krause_2017':
    c_m = 'Bhattacharya13'  # see paper, after Eq. 1
    mass_def = ccl.halos.MassDef200m(c_m=c_m)
    c_M_relation = ccl.halos.concentration.ConcentrationBhattacharya13(mdef=mass_def)  # over Eq. 12
else:
    raise ValueError('Wrong choice of hm_recipe: it must be either "KiDS_1000" or "Krause_2017".')

# TODO pass mass_def object? plus, understand what exactly is mass_def_strict

# mass function
massfunc = ccl.halos.hmfunc.MassFuncTinker10(cosmo, mass_def=mass_def, mass_def_strict=True)

# halo bias
hbias = ccl.halos.hbias.HaloBiasTinker10(cosmo, mass_def=mass_def, mass_def_strict=True)

# concentration-mass relation

# TODO understand better this object. We're calling the abstract class, is this ok?
# HMCalculator
hmc = ccl.halos.halo_model.HMCalculator(cosmo, massfunc, hbias, mass_def=mass_def,
                                        log10M_min=8.0, log10M_max=16.0, nlog10M=128,
                                        integration_method_M='simpson', k_min=1e-05)

# halo profile
halo_profile = ccl.halos.profiles.HaloProfileNFW(c_M_relation=c_M_relation,
                                                 fourier_analytic=True, projected_analytic=False,
                                                 cumul2d_analytic=False, truncated=True)

# it was p_of_k_a=Pk, but it should use the LINEAR power spectrum (see documentation:
# https://ccl.readthedocs.io/en/latest/api/pyccl.halos.halo_model.html?highlight=halomod_Tk3D_SSC#pyccl.halos.halo_model.halomod_Tk3D_SSC)
# 🐛 bug solved: normprof shoud be True
# 🐛 bug solved?: p_of_k_a=None instead of Pk
tkka = ccl.halos.halo_model.halomod_Tk3D_SSC(cosmo, hmc,
                                             prof1=halo_profile, prof2=None, prof12_2pt=None,
                                             prof3=None, prof4=None, prof34_2pt=None,
                                             normprof1=True, normprof2=True, normprof3=True, normprof4=True,
                                             p_of_k_a=None, lk_arr=lk_arr, a_arr=a_arr, extrap_order_lok=1,
                                             extrap_order_hik=1, use_log=False)

# ! note that the ordering is such that out[i2, i1] = Cov(ell2[i2], ell[i1]). Transpose 1st 2 dimensions??

# ! super-sample
if compute_SS_WL:
    cov_SS_WL_6D = compute_SSC_PyCCL(cosmo, kernel_A=wil, kernel_B=wil, kernel_C=wil, kernel_D=wil,
                                  ell=ell, tkka=tkka, f_sky=f_sky, integration_method='spline')
if compute_SS_GC:
    cov_SS_GC_6D = compute_SSC_PyCCL(cosmo, kernel_A=wig, kernel_B=wig, kernel_C=wig, kernel_D=wig,
                                  ell=ell, tkka=tkka, f_sky=f_sky, integration_method='qag_quad')

if save_SSC:
    np.save(f'{project_path}/output/cov_PyCCL_SS_WL_nbl{nbl}_ells{which_ells}_hm_recipe{hm_recipe}_6D.npy', cov_SS_WL_6D)
    np.save(f'{project_path}/output/cov_PyCCL_SS_GC_nbl{nbl}_ells{which_ells}_hm_recipe{hm_recipe}_6D.npy', cov_SS_GC_6D)

if compute_cNG:
    # ! connected non-Gaussian
    cov_cNG_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    start_cNG = time.perf_counter()
    for i in range(zbins):
        for j in range(zbins):
            start = time.perf_counter()
            for k in range(zbins):
                for l in range(zbins):
                    cov_cNG_6D[:, :, i, j, k, l] = ccl.covariances.angular_cl_cov_cNG(cosmo, wil[i], wil[j], ell,
                                                                                      tkka, fsky=f_sky,
                                                                                      cltracer3=wil[k],
                                                                                      cltracer4=wil[l], ell2=None,
                                                                                      integration_method='spline')
            print(f'i, j redshift bins: {i}, {j}, computed in  {(time.perf_counter() - start):.2f} seconds')
    print(f'connected non-Gaussian computed in {(time.perf_counter() - start_cNG):.2f} seconds')

    np.save(f'{project_path}/output/cov_PyCCL_cNG_nbl{nbl}_ells{which_ells}_hm_recipe{hm_recipe}_6D.npy', cov_cNG_6D)


assert 1 > 2, 'stop here'

path_ind = '/Users/davide/Documents/Lavoro/Programmi/SSC_restructured/config/common_data/ind'
ind = np.genfromtxt(f'{path_ind}/indici_vincenzo_like.dat').astype('int') - 1
ind_LL = ind[:npairs_auto, :]

# load CosmoLike (Robin) and PySSC
robins_cov_path = '/Users/davide/Documents/Lavoro/Programmi/SSC_paper_jan22/PySSC_vs_CosmoLike/Robin/cov_SS_full_sky_rescaled'
cov_Robin_2D = np.load(robins_cov_path + '/lmax5000_noextrap/davides_reshape/cov_R_WL_SSC_lmax5000_2D.npy')
cov_PySSC_4D = np.load(project_path / 'input/CovMat-ShearShear-SSC-20bins-NL_flag_2_4D.npy')

# reshape
cov_SS_4D = mm.cov_6D_to_4D(cov_SS_6D, nbl=nbl, npairs=npairs_auto, ind=ind_LL)
cov_SS_2D = mm.cov_4D_to_2D(cov_SS_4D, nbl=nbl, npairs_AB=npairs_auto, npairs_CD=None, block_index='vincenzo')

cov_Robin_4D = mm.cov_2D_to_4D(cov_Robin_2D, nbl=nbl, npairs=npairs_auto, block_index='vincenzo')
cov_PySSC_6D = mm.cov_4D_to_6D(cov_PySSC_4D, nbl=nbl, zbins=10, probe='LL', ind=ind_LL)

# save PyCCL 2D
np.save(f'{project_path}/output/cov_PyCCL_nbl{nbl}_ells{which_ells}_2D.npy', cov_SS_2D)

assert 1 > 2

# check if the matrics are symmetric in ell1 <-> ell2
print(np.allclose(cov_Robin_4D, cov_Robin_4D.transpose(1, 0, 2, 3), rtol=1e-10))
print(np.allclose(cov_SS_4D, cov_SS_4D.transpose(1, 0, 2, 3), rtol=1e-10))
print(np.allclose(cov_PySSC_4D, cov_PySSC_4D.transpose(1, 0, 2, 3), rtol=1e-10))

mm.matshow(cov_SS_6D[:, :, 0, 0, 0, 0], log=True, title='cov_PyCCL_6D')
mm.matshow(cov_PySSC_6D[:, :, 0, 0, 0, 0], log=True, title='cov_PySSC_6D')

# show the various versions
mm.matshow(cov_PySSC_4D[:, :, 0, 0], log=True, title='cov_PySSC_4D')
mm.matshow(cov_Robin_4D[:, :, 0, 0], log=True, title='cov_Robin_4D')
mm.matshow(cov_SS_4D[:, :, 0, 0], log=True, title='cov_PyCCL_4D')

# compute and plot percent difference (not in log scale)
PyCCL_vs_rob = mm.compare_2D_arrays(cov_SS_4D[:, :, 0, 0], cov_Robin_4D[:, :, 0, 0], 'cov_PyCCL_4D', 'cov_Robin_4D',
                                    log_arr=True)
PySSC_vs_PyCCL = mm.compare_2D_arrays(cov_SS_4D[:, :, 0, 0], cov_PySSC_4D[:, :, 0, 0], 'cov_PyCCL_4D',
                                      'cov_PySSC_4D', log_arr=True)

# mm.matshow(rob_vs_PyCCL[:, :, 0, 0], log=False, title='rob_vs_PyCCL [%]')
# mm.matshow(rob_vs_PySSC[:, :, 0, 0], log=False, title='rob_vs_PySSC [%]')

# correlation matrices
corr_PySSC_4D = mm.correlation_from_covariance(cov_PySSC_4D[:, :, 0, 0])
corr_Robin_4D = mm.correlation_from_covariance(cov_Robin_4D[:, :, 0, 0])
corr_PyCCL_4D = mm.correlation_from_covariance(cov_SS_4D[:, :, 0, 0])

corr_PyCCL_vs_rob = mm.compare_2D_arrays(corr_PyCCL_4D, corr_Robin_4D, 'corr_PyCCL_4D', 'corr_Robin_4D', log_arr=False)
corr_PySSC_vs_PyCCL = mm.compare_2D_arrays(corr_PyCCL_4D, corr_PySSC_4D, 'corr_PyCCL_4D', 'corr_PySSC_4D',
                                           log_arr=False)

print('done')

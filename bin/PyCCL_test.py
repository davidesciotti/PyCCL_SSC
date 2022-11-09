import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
from scipy.special import erf
import scipy.io as sio
import ray

ray.shutdown()
ray.init()

# get project directory adn import useful modules
project_path = Path.cwd().parent

sys.path.append(f'{project_path.parent}/common_data/common_lib')
import my_module as mm

sys.path.append(f'{project_path.parent}/SSC_restructured_v2/bin')
import ell_values_running as ell_utils

sys.path.append(f'{project_path.parent}/common_data/common_config')
import ISTF_fid_params as ISTF_fid

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
            print(f'i, j redshift bins: {i}, {j}, computed in  {(time.perf_counter() - start):.2f} s')
    print(f'SSC computed in  {(time.perf_counter() - start_SSC_timer):.2f} s')

    return cov_SS_6D


def compute_cNG_PyCCL(cosmo, kernel_A, kernel_B, kernel_C, kernel_D, ell, tkka, f_sky, integration_method='spline'):
    cov_cNG_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    start_cNG_timer = time.perf_counter()

    for i in range(zbins):
        for j in range(zbins):
            start = time.perf_counter()
            for k in range(zbins):
                for l in range(zbins):
                    cov_cNG_6D[:, :, i, j, k, l] = ccl.covariances.angular_cl_cov_cNG(cosmo, kernel_A[i], kernel_B[j],
                                                                                      ell, tkka, fsky=f_sky,
                                                                                      cltracer3=kernel_C[k],
                                                                                      cltracer4=kernel_D[l],
                                                                                      ell2=None,
                                                                                      integration_method=integration_method)
            print(f'i, j redshift bins: {i}, {j}, computed in  {(time.perf_counter() - start):.2f} s')
    print(f'cNG computed in {(time.perf_counter() - start_cNG_timer):.2f} s')

    return cov_cNG_6D


compute_SSC_PyCCL_ray = ray.remote(compute_SSC_PyCCL)
compute_cNG_PyCCL_ray = ray.remote(compute_cNG_PyCCL)
###############################################################################
###############################################################################
###############################################################################

# ! POTENTIAL ISSUES:
# 1. input files (WF, ell, a, pk...)
# 2. halo model recipe
# 3. ordering of the resulting covariance matrix

# * fanstastic collection of notebooks: https://github.com/LSSTDESC/CCLX


# Create new Cosmology object with a given set of parameters. This keeps track of previously-computed cosmological
# functions
Om_c0 = ISTF_fid.primary['Om_m0'] - ISTF_fid.primary['Om_b0']
cosmo = ccl.Cosmology(Omega_c=Om_c0, Omega_b=ISTF_fid.primary['Om_b0'], w0=ISTF_fid.primary['w0'],
                      wa=ISTF_fid.primary['w_a'], h=ISTF_fid.primary['h'], sigma8=ISTF_fid.primary['sigma8'],
                      n_s=ISTF_fid.primary['n_s'], m_nu=ISTF_fid.primary['m_nu'],
                      Omega_k=1 - (Om_c0 + ISTF_fid.primary['Om_b0']) - ISTF_fid.extensions['Om_Lambda0'])

# Define redshift distribution of sources kernels
zmin, zmax, dz = 0.001, 2.5, 0.001
ztab = np.arange(zmin, zmax, dz)  # ! should it start from 0 instead?

# these are the bin edges
zi = np.array([
    [zmin, 0.418, 0.56, 0.678, 0.789, 0.9, 1.019, 1.155, 1.324, 1.576],
    [0.418, 0.56, 0.678, 0.789, 0.9, 1.019, 1.155, 1.324, 1.576, zmax]
])

zbins = len(zi[0])

# get number of redshift pairs
npairs_auto, npairs_asimm, npairs_tot = mm.get_pairs(zbins)

z_median = ISTF_fid.photoz_bins['z_median']
n_gal = ISTF_fid.other_survey_specs['n_gal']
survey_area = ISTF_fid.other_survey_specs['survey_area']

fout = ISTF_fid.photoz_pdf['f_out']
cb, zb, sigmab = ISTF_fid.photoz_pdf['c_b'], ISTF_fid.photoz_pdf['z_b'], ISTF_fid.photoz_pdf['sigma_b']
c0, z0, sigma0 = ISTF_fid.photoz_pdf['c_o'], ISTF_fid.photoz_pdf['z_o'], ISTF_fid.photoz_pdf['sigma_o']

nzEuclid = n_gal * (ztab / z_median * np.sqrt(2)) ** 2 * np.exp(-(ztab / z_median * np.sqrt(2)) ** 1.5)

nziEuclid = np.array([nzEuclid * 1 / 2 / c0 / cb * (cb * fout *
                                                    (erf((ztab - z0 - c0 * zi[0, iz]) / np.sqrt(2) /
                                                         (1 + ztab) / sigma0) -
                                                     erf((ztab - z0 - c0 * zi[1, iz]) / np.sqrt(2) /
                                                         (1 + ztab) / sigma0)) +
                                                    c0 * (1 - fout) *
                                                    (erf((ztab - zb - cb * zi[0, iz]) / np.sqrt(2) /
                                                         (1 + ztab) / sigmab) -
                                                     erf((ztab - zb - cb * zi[1, iz]) / np.sqrt(2) /
                                                         (1 + ztab) / sigmab))) for iz in range(zbins)])

# normalize nz: this should be the denominator of Eq. (112) of IST:f
for i in range(zbins):
    norm_factor = np.sum(nziEuclid[i, :]) * dz
    nziEuclid[i, :] /= norm_factor

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

# GALAXY KERNELS

# construct the bias array -
b_array = np.asarray([bias(z, zi) for z in ztab])
# it should be the same for all redshift bins:
# b_array = np.repeat(b_array[:, np.newaxis], zbins, axis=1)  # this is useless, I can just pass the same array each
# time in the call below

wil = [ccl.WeakLensingTracer(cosmo, dndz=(ztab, nziEuclid[iz]), ia_bias=(IAFILE[:, 0], FIAz), use_A_ia=False) for iz in
       range(zbins)]
wig = [ccl.tracers.NumberCountsTracer(cosmo, has_rsd=False, dndz=(ztab, nziEuclid[iz]), bias=(ztab, b_array),
                                      mag_bias=None) for iz in range(zbins)]

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
ell_max = 5000
nbl = 30

# ! settings
ell_recipe = 'ISTF'
probe = 'WL'
compute_cNG = False
save_covs = True
hm_recipe = 'Krause2017'
GL_or_LG = 'GL'
# ! settings

# for which_ells in ['ISTF', 'ISTNL']:
# for hm_recipe in ['KiDS1000', 'Krause2017']:
# for probe in ['WL', 'GC']:

path_ind = '/Users/davide/Documents/Lavoro/Programmi/common_data/ind_files'
ind = np.genfromtxt(f'{path_ind}/indici_vincenzo_like_int.dat', dtype=int)
ind_LL = ind[:npairs_auto, :]

if ell_recipe == 'ISTF':
    nbl = 30
elif ell_recipe == 'ISTNL':
    nbl = 20
else:
    raise ValueError('ell_recipe must be "ISTF" or "ISTNL"')

if probe == 'WL':
    ell_max = 5000
elif probe == 'GC' or '3x2pt':
    ell_max = 3000
else:
    raise ValueError('probe must be "WL", "GC" or "3x2pt"')

# 3x2pt stuff
probe_wf_dict = {
    'L': wil,
    'G': wig
}

probe_ordering = ('LL', f'{GL_or_LG}', 'GG')
probe_idx_dict = {'L': 0, 'G': 1}
# upper diagonal of blocks of the covariance matrix
probe_combinations_3x2pt = ((probe_ordering[0][0], probe_ordering[0][1], probe_ordering[0][0], probe_ordering[0][1]),
                            (probe_ordering[0][0], probe_ordering[0][1], probe_ordering[1][0], probe_ordering[1][1]),
                            (probe_ordering[0][0], probe_ordering[0][1], probe_ordering[2][0], probe_ordering[2][1]),
                            (probe_ordering[1][0], probe_ordering[1][1], probe_ordering[1][0], probe_ordering[1][1]),
                            (probe_ordering[1][0], probe_ordering[1][1], probe_ordering[2][0], probe_ordering[2][1]),
                            (probe_ordering[2][0], probe_ordering[2][1], probe_ordering[2][0], probe_ordering[2][1]))

ell, _ = ell_utils.compute_ells(nbl, ell_min, ell_max, ell_recipe)

# just a check on the settings
print(
    f'settings:\nwhich_ells = {ell_recipe}\nnbl = {nbl}\nhm_recipe = {hm_recipe}\nprobe = {probe}'
    f'\ncompute_cNG = {compute_cNG}')

# CLL = np.array([[ccl.angular_cl(cosmo, wil[iz], wil[jz], ell, p_of_k_a=Pk)
#                  for iz in range(zbins)]
#                 for jz in range(zbins)])

# ! this has to go in a cfg file!!
A_deg = 15e3
f_sky = A_deg * (np.pi / 180) ** 2 / (4 * np.pi)
n_gal = 30 * (180 * 60 / np.pi) ** 2
sigma_e = 0.3

# save wf and cl for validation
# np.save(project_path / 'output/wl_and_cl_validation/ztab.npy', ztab)
# np.save(project_path / 'output/wl_and_cl_validation/wil_array.npy', wil_array)
# np.save(project_path / 'output/wl_and_cl_validation/wig_array.npy', wig_array)
# np.save(project_path / 'output/wl_and_cl_validation/ell.npy', ell)
# np.save(project_path / 'output/wl_and_cl_validation/C_LL.npy', CLL)
# np.save(project_path / 'output/wl_and_cl_validation/nziEuclid.npy', nziEuclid)

# notebook for mass_relations: https://github.com/LSSTDESC/CCLX/blob/master/Halo-mass-function-example.ipynb
# Cl notebook: https://github.com/LSSTDESC/CCL/blob/v2.0.1/examples/3x2demo.ipynb

# TODO we're not sure about the values of Delta and rho_type
# mass_def = ccl.halos.massdef.MassDef(Delta='vir', rho_type='matter', c_m_relation=name)

# from https://ccl.readthedocs.io/en/latest/api/pyccl.halos.massdef.html?highlight=.halos.massdef.MassDef#pyccl.halos.massdef.MassDef200c

# ! HALO MODEL PRESCRIPTIONS:
# KiDS1000 Methodology:
# https://www.pure.ed.ac.uk/ws/portalfiles/portal/188893969/2007.01844v2.pdf, after (E.10)

# Krause2017: https://arxiv.org/pdf/1601.05779.pdf
# about the mass definition, the paper says:
# "Throughout this paper we define halo properties using the over density ∆ = 200 ¯ρ, with ¯ρ the mean matter density"

# mass definition
if hm_recipe == 'KiDS1000':  # arXiv:2007.01844
    c_m = 'Duffy08'  # ! NOT SURE ABOUT THIS
    mass_def = ccl.halos.MassDef200m(c_m=c_m)
    c_M_relation = ccl.halos.concentration.ConcentrationDuffy08(mdef=mass_def)
elif hm_recipe == 'Krause2017':  # arXiv:1601.05779
    c_m = 'Bhattacharya13'  # see paper, after Eq. 1
    mass_def = ccl.halos.MassDef200m(c_m=c_m)
    c_M_relation = ccl.halos.concentration.ConcentrationBhattacharya13(mdef=mass_def)  # above Eq. 12
else:
    raise ValueError('Wrong choice of hm_recipe: it must be either "KiDS1000" or "Krause2017".')

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
if probe == 'WL':
    cov_SS_6D = compute_SSC_PyCCL(cosmo, kernel_A=wil, kernel_B=wil, kernel_C=wil, kernel_D=wil,
                                  ell=ell, tkka=tkka, f_sky=f_sky, integration_method='spline')
elif probe == 'GC':
    cov_SS_6D = compute_SSC_PyCCL(cosmo, kernel_A=wig, kernel_B=wig, kernel_C=wig, kernel_D=wig,
                                  ell=ell, tkka=tkka, f_sky=f_sky, integration_method='qag_quad')

elif probe == '3x2pt':
    cov_SS_3x2pt_dict_10D = {}
    for A, B, C, D in probe_combinations_3x2pt:
        cov_SS_3x2pt_dict_10D[A, B, C, D] = compute_SSC_PyCCL(cosmo,
                                                              kernel_A=probe_wf_dict[A], kernel_B=probe_wf_dict[B],
                                                              kernel_C=probe_wf_dict[C], kernel_D=probe_wf_dict[D],
                                                              ell=ell, tkka=tkka, f_sky=f_sky,
                                                              integration_method='qag_quad')

    # TODO test this by loading the cov_SS_3x2pt_arr_10D from file (and then storing it into a dictionary)

    # symmetrize the matrix:
    LL = probe_ordering[0][0], probe_ordering[0][1]
    GL = probe_ordering[1][0], probe_ordering[1][1]  # ! what if I use LG? check (it should be fine...)
    GG = probe_ordering[2][0], probe_ordering[2][1]
    cov_SS_3x2pt_dict_10D[GL, LL, ...] = cov_SS_3x2pt_dict_10D[LL, GL, ...]
    cov_SS_3x2pt_dict_10D[GG, LL, ...] = cov_SS_3x2pt_dict_10D[LL, GG, ...]
    cov_SS_3x2pt_dict_10D[GG, GL, ...] = cov_SS_3x2pt_dict_10D[GL, GG, ...]

    # stack everything and reshape to 4D
    cov_SS_3x2pt_4D = mm.cov_3x2pt_dict_10D_to_4D(cov_SS_3x2pt_dict_10D, probe_ordering, nbl, zbins, ind, GL_or_LG)

if save_covs:
    if probe == '3x2pt':
        # save as dict
        sio.savemat(
            f'{project_path}/output/covmat/cov_PyCCL_SS_{probe}_nbl{nbl}_ells{ell_recipe}_ellmax{ell_max}_hm_recipe{hm_recipe}_6D.mat',
            cov_SS_3x2pt_dict_10D)
        # save as npy array
        np.save(
            f'{project_path}/output/covmat/cov_PyCCL_SS_{probe}_nbl{nbl}_ells{ell_recipe}_ellmax{ell_max}_hm_recipe{hm_recipe}_4D.npy',
            cov_SS_3x2pt_4D)
    else:
        np.save(
            f'{project_path}/output/covmat/cov_PyCCL_SS_{probe}_nbl{nbl}_ells{ell_recipe}_ellmax{ell_max}_hm_recipe{hm_recipe}_6D.npy',
            cov_SS_6D)

# ! cNG
if compute_cNG:

    if probe == 'WL':
        cov_cNG_6D = compute_cNG_PyCCL(cosmo, kernel_A=wil, kernel_B=wil, kernel_C=wil, kernel_D=wil,
                                       ell=ell, tkka=tkka, f_sky=f_sky, integration_method='spline')
    elif probe == 'GC':
        cov_cNG_6D = compute_cNG_PyCCL(cosmo, kernel_A=wig, kernel_B=wig, kernel_C=wig, kernel_D=wig,
                                       ell=ell, tkka=tkka, f_sky=f_sky, integration_method='spline')
    if save_covs:
        np.save(
            f'{project_path}/output/covmat/cov_PyCCL_cNG_{probe}_nbl{nbl}_ells{ell_recipe}_ellmax{ell_max}_hm_recipe{hm_recipe}_6D.npy',
            cov_cNG_6D)

assert 1 > 2, 'stop here'

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
np.save(f'{project_path}/output/cov_PyCCL_nbl{nbl}_ells{ell_recipe}_2D.npy', cov_SS_2D)

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

import pickle
import sys
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
import yaml
from scipy.special import erf
import ray

ray.shutdown()
ray.init()

# get project directory adn import useful modules
project_path = Path.cwd().parent

sys.path.append(f'../../common_data/common_lib')
import my_module as mm

sys.path.append(f'../../common_data/common_config')
import ISTF_fid_params as ISTF_fid
import mpl_cfg

sys.path.append(f'../../SSC_restructured_v2/bin')
import ell_values as ell_utils

sys.path.append(f'../../cl_v2/lib')
import wf_cl_lib


matplotlib.use('Qt5Agg')
start_time = time.perf_counter()
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)


###############################################################################
###############################################################################
###############################################################################


def compute_SSC_PyCCL(cosmo, kernel_A, kernel_B, kernel_C, kernel_D, ell, tkka, f_sky, integration_method='spline'):
    cov_SSC_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    start_SSC_timer = time.perf_counter()

    for i in range(zbins):
        start = time.perf_counter()
        for j in range(zbins):
            for k in range(zbins):
                for l in range(zbins):
                    cov_SSC_6D[:, :, i, j, k, l] = ccl.covariances.angular_cl_cov_SSC(cosmo, kernel_A[i], kernel_B[j],
                                                                                      ell, tkka,
                                                                                      sigma2_B=None, fsky=f_sky,
                                                                                      cltracer3=kernel_C[k],
                                                                                      cltracer4=kernel_D[l],
                                                                                      ell2=None,
                                                                                      integration_method=integration_method)
        print(f'i-th redshift bins: {i}, computed in  {(time.perf_counter() - start):.2f} s')
    print(f'SSC computed in  {(time.perf_counter() - start_SSC_timer):.2f} s')

    return cov_SSC_6D


def compute_cNG_PyCCL(cosmo, kernel_A, kernel_B, kernel_C, kernel_D, ell, tkka, f_sky, integration_method='spline'):
    cov_cNG_6D = np.zeros((nbl, nbl, zbins, zbins, zbins, zbins))
    start_cNG_timer = time.perf_counter()

    for i in range(zbins):
        start = time.perf_counter()
        for j in range(zbins):
            for k in range(zbins):
                for l in range(zbins):
                    cov_cNG_6D[:, :, i, j, k, l] = ccl.covariances.angular_cl_cov_cNG(cosmo, kernel_A[i], kernel_B[j],
                                                                                      ell=ell, tkka=tkka, fsky=f_sky,
                                                                                      cltracer3=kernel_C[k],
                                                                                      cltracer4=kernel_D[l],
                                                                                      ell2=None,
                                                                                      integration_method=integration_method)
        print(f'i-th redshift bins: {i}, computed in  {(time.perf_counter() - start):.2f} s')
    print(f'cNG computed in {(time.perf_counter() - start_cNG_timer):.2f} s')

    return cov_cNG_6D


def compute_3x2pt_PyCCL(PyCCL_func, cosmo, probe_wf_dict, ell, tkka, f_sky, integration_method,
                        probe_ordering, probe_combinations_3x2pt):
    # TODO finish this function
    cov_SSC_3x2pt_dict_10D = {}
    for A, B, C, D in probe_combinations_3x2pt:
        print('3x2pt: working on probe combination ', A, B, C, D)
        cov_SSC_3x2pt_dict_10D[A, B, C, D] = PyCCL_func(cosmo,
                                                        probe_wf_dict[A], probe_wf_dict[B],
                                                        probe_wf_dict[C], probe_wf_dict[D], ell, tkka,
                                                        f_sky, integration_method)
        np.save(
            f'{project_path}/output/covariance/cov_PyCCL_{which_NG}_3x2pt_{A}{B}{C}{D}_nbl{nbl}_ells{ell_recipe}_ellmax{ell_max}_hm_recipe{hm_recipe}.npy',
            cov_SSC_3x2pt_dict_10D[A, B, C, D])

    # TODO test this by loading the cov_SSC_3x2pt_arr_10D from file (and then storing it into a dictionary)
    # symmetrize the matrix:
    LL = probe_ordering[0][0], probe_ordering[0][1]
    GL = probe_ordering[1][0], probe_ordering[1][1]  # ! what if I use LG? check (it should be fine...)
    GG = probe_ordering[2][0], probe_ordering[2][1]
    # note: the addition is only to have a singe tuple of strings, instead of a tuple of 2 tuples
    cov_SSC_3x2pt_dict_10D[GL + LL] = cov_SSC_3x2pt_dict_10D[LL + GL][...]
    cov_SSC_3x2pt_dict_10D[GG + LL] = cov_SSC_3x2pt_dict_10D[LL + GG][...]
    cov_SSC_3x2pt_dict_10D[GG + GL] = cov_SSC_3x2pt_dict_10D[GL + GG][...]

    return cov_SSC_3x2pt_dict_10D


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


# ! settings
with open('../../exact_SSC/config/config.yml') as f:
    cfg = yaml.safe_load(f)

ell_grid_recipe = cfg['ell_grid_recipe']
probes = cfg['probes']
which_NGs = cfg['which_NGs']
save_covs = cfg['save_covs']
hm_recipe = cfg['hm_recipe']
GL_or_LG = cfg['GL_or_LG']
ell_min = cfg['ell_min']
ell_max = cfg['ell_max']
nbl = cfg['nbl']
zbins = cfg['zbins']
use_ray = cfg['use_ray']  # TODO finish this!
# ! settings

# get number of redshift pairs
zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)

# Create new Cosmology object with a given set of parameters. This keeps track of previously-computed cosmological
# functions
cosmo = wf_cl_lib.instantiate_ISTFfid_PyCCL_cosmo_obj()

################################## Define redshift distribution of sources kernels #####################################
zmin, zmax, dz = 0.001, 2.5, 0.001
z_grid = np.arange(zmin, zmax, dz)  # ! should it start from 0 instead?

# for CosmoLike
# zmin, zmax, zsteps = 0.001, 4., 10_000
# z_grid = np.linspace(zmin, zmax, zsteps)  # ! should it start from 0 instead?

z_median = ISTF_fid.photoz_bins['z_median']

# TODO import these from IST_fid
zbins_edges = np.array([[zmin, 0.418, 0.56, 0.678, 0.789, 0.9, 1.019, 1.155, 1.324, 1.576],
                        [0.418, 0.56, 0.678, 0.789, 0.9, 1.019, 1.155, 1.324, 1.576, zmax]])
# assert (zbins == len(zbins_edges[0])), 'zbins and zbins_edges do not match'

# other useful parameters
n_gal = ISTF_fid.other_survey_specs['n_gal']
survey_area = ISTF_fid.other_survey_specs['survey_area']
f_sky = survey_area * (np.pi / 180) ** 2 / (4 * np.pi)
# n_gal_degsq = n_gal * (180 * 60 / np.pi) ** 2
# sigma_e = ISTF_fid.other_survey_specs['sigma_eps']


fout = ISTF_fid.photoz_pdf['f_out']
cb, zb, sigmab = ISTF_fid.photoz_pdf['c_b'], ISTF_fid.photoz_pdf['z_b'], ISTF_fid.photoz_pdf['sigma_b']
c0, z0, sigma0 = ISTF_fid.photoz_pdf['c_o'], ISTF_fid.photoz_pdf['z_o'], ISTF_fid.photoz_pdf['sigma_o']

nzEuclid = n_gal * (z_grid / z_median * np.sqrt(2)) ** 2 * np.exp(-(z_grid / z_median * np.sqrt(2)) ** 1.5)

nziEuclid = np.array([nzEuclid * 1 / 2 / c0 / cb * (cb * fout *
                                                    (erf((z_grid - z0 - c0 * zbins_edges[0, iz]) / np.sqrt(2) /
                                                         (1 + z_grid) / sigma0) -
                                                     erf((z_grid - z0 - c0 * zbins_edges[1, iz]) / np.sqrt(2) /
                                                         (1 + z_grid) / sigma0)) +
                                                    c0 * (1 - fout) *
                                                    (erf((z_grid - zb - cb * zbins_edges[0, iz]) / np.sqrt(2) /
                                                         (1 + z_grid) / sigmab) -
                                                     erf((z_grid - zb - cb * zbins_edges[1, iz]) / np.sqrt(2) /
                                                         (1 + z_grid) / sigmab))) for iz in range(zbins)])

# normalize nz: this should be the denominator of Eq. (112) of IST:f
for i in range(zbins):
    norm_factor = np.sum(nziEuclid[i, :]) * dz
    nziEuclid[i, :] /= norm_factor


    niz_unnormalized = np.asarray([wf_cl_lib.niz_unnormalized_analytical(z_grid, zbin_idx) for zbin_idx in range(zbins)])
    niz_normalized_arr = wf_cl_lib.normalize_niz_simps(niz_unnormalized, z_grid).T
nzi_v2 = wf_cl_lib.

for zi in range(zbins):
    plt.plot(z_grid, nziEuclid[zi, :], label=f'zbin {zi}')

assert False

# Intrinsic alignment and galaxy bias
IAFILE = np.genfromtxt(project_path / 'input/scaledmeanlum-E2Sa.dat')
FIAzNoCosmoNoGrowth = -1 * 1.72 * 0.0134 * (1 + IAFILE[:, 0]) ** (-0.41) * IAFILE[:, 1] ** 2.17
FIAz = FIAzNoCosmoNoGrowth * (cosmo.cosmo.params.Omega_c + cosmo.cosmo.params.Omega_b) / ccl.growth_factor(cosmo, 1 / (
        1 + IAFILE[:, 0]))

b_array_v2 = wf_cl_lib.build_galaxy_bias_2d_arr(None, None, zbins, z_grid, 'step-wise', plot_bias=False)

assert False

# compute the kernels
wil = [ccl.WeakLensingTracer(cosmo, dndz=(z_grid, nziEuclid[iz]), ia_bias=(IAFILE[:, 0], FIAz), use_A_ia=False)
       for iz in range(zbins)]
wig = [ccl.tracers.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z_grid, nziEuclid[iz]), bias=(z_grid, b_array),
                                      mag_bias=None) for iz in range(zbins)]

# Import fiducial P(k,z)
PkFILE = np.genfromtxt(project_path / 'input/pkz-Fiducial.txt')

# ! XXX are the units correct?
# Populate vectors for z, k [1/Mpc], and P(k,z) [Mpc^3]
zlist = np.unique(PkFILE[:, 0])
k_points = int(len(PkFILE[:, 2]) / len(zlist))
klist = PkFILE[:k_points, 1] * cosmo.cosmo.params.h
z_points = len(zlist)
Pklist = PkFILE[:, 3].reshape(z_points, k_points) / cosmo.cosmo.params.h ** 3

# Create a Pk2D object
a_arr = 1 / (1 + zlist[::-1])
lk_arr = np.log(klist)  # it's the natural log, not log10
Pk = ccl.Pk2D(a_arr=a_arr, lk_arr=lk_arr, pk_arr=Pklist, is_logp=False)

# ! compute cls, just as a test
ells_LL, _ = ell_utils.compute_ells(nbl=30, ell_min=10, ell_max=5000, recipe='ISTF')
cl_LL = wf_cl_lib.cl_PyCCL(wil, wil, ells_LL, zbins, None, cosmo)
# TODO introduce one by one the ingredients from wf_cl_main/lib

assert 1 > 2

# === 3x2pt stuff ===
probe_wf_dict = {
    'L': wil,
    'G': wig
}
probe_ordering = ('LL', f'{GL_or_LG}', 'GG')
probe_idx_dict = {'L': 0, 'G': 1}
# upper diagonal of blocks of the covariance matrix
probe_combinations_3x2pt = (
    (probe_ordering[0][0], probe_ordering[0][1], probe_ordering[0][0], probe_ordering[0][1]),
    (probe_ordering[0][0], probe_ordering[0][1], probe_ordering[1][0], probe_ordering[1][1]),
    (probe_ordering[0][0], probe_ordering[0][1], probe_ordering[2][0], probe_ordering[2][1]),
    (probe_ordering[1][0], probe_ordering[1][1], probe_ordering[1][0], probe_ordering[1][1]),
    (probe_ordering[1][0], probe_ordering[1][1], probe_ordering[2][0], probe_ordering[2][1]),
    (probe_ordering[2][0], probe_ordering[2][1], probe_ordering[2][0], probe_ordering[2][1]))

# ! =============================================== halo model =========================================================
# notebook for mass_relations: https://github.com/LSSTDESC/CCLX/blob/master/Halo-mass-function-example.ipynb
# Cl notebook: https://github.com/LSSTDESC/CCL/blob/v2.0.1/examples/3x2demo.ipynb

# TODO we're not sure about the values of Delta and rho_type
# mass_def = ccl.halos.massdef.MassDef(Delta='vir', rho_type='matter', c_m_relation=name)

# from https://ccl.readthedocs.io/en/latest/api/pyccl.halos.massdef.html?highlight=.halos.massdef.MassDef#pyccl.halos.massdef.MassDef200c

# HALO MODEL PRESCRIPTIONS:
# KiDS1000 Methodology:
# https://www.pure.ed.ac.uk/ws/portalfiles/portal/188893969/2007.01844v2.pdf, after (E.10)

# Krause2017: https://arxiv.org/pdf/1601.05779.pdf
# about the mass definition, the paper says:
# "Throughout this paper we define halo properties using the over density ∆ = 200 ¯ρ, with ¯ρ the mean matter density"

halomod_start_time = time.perf_counter()
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

# it was p_of_k_a=Pk, but it should use the LINEAR power spectrum, so we leave it as None (see documentation:
# https://ccl.readthedocs.io/en/latest/api/pyccl.halos.halo_model.html?highlight=halomod_Tk3D_SSC#pyccl.halos.halo_model.halomod_Tk3D_SSC)
# 🐛 bug fixed: normprof shoud be True
# 🐛 bug fixed?: p_of_k_a=None instead of Pk
tkka = ccl.halos.halo_model.halomod_Tk3D_SSC(cosmo, hmc,
                                             prof1=halo_profile, prof2=None, prof12_2pt=None,
                                             prof3=None, prof4=None, prof34_2pt=None,
                                             normprof1=True, normprof2=True, normprof3=True, normprof4=True,
                                             p_of_k_a=None, lk_arr=lk_arr, a_arr=a_arr, extrap_order_lok=1,
                                             extrap_order_hik=1, use_log=False)
print('trispectrum computed in {:.2f} seconds'.format(time.perf_counter() - halomod_start_time))

# re-define the functions if using ray
# if use_ray:
#     compute_SSC_PyCCL = compute_SSC_PyCCL_ray.remote
#     compute_cNG_PyCCL = compute_cNG_PyCCL_ray.remote


assert 1 > 2

integration_method_dict = {
    'WL': {
        'SSC': 'spline',
        'cNG': 'spline',
    },
    'GC': {
        'SSC': 'qag_quad',
        'cNG': 'qag_quad',
    },
    '3x2pt': {
        'SSC': 'qag_quad',
        'cNG': 'spline',
    }
}

for probe in probes:
    for which_NG in which_NGs:

        assert probe in ['WL', 'GC', '3x2pt'], 'probe must be either WL, GC, or 3x2pt'
        assert which_NG in ['SSC', 'cNG'], 'which_NG must be either SSC or cNG'
        assert ell_grid_recipe in ['ISTF', 'ISTNL'], 'ell_grid_recipe must be either ISTF or ISTNL'

        # === ell values ===
        if ell_grid_recipe == 'ISTF' and nbl != 30:
            print('Warning: ISTF uses 30 ell bins')
        elif ell_grid_recipe == 'ISTNL' and nbl != 20:
            print('Warning: ISTNL uses 20 ell bins')

        if probe == 'WL':
            ell_max = 5000
            kernel = wil
        elif probe == 'GC':
            ell_max = 3000
            kernel = wig
        elif probe == '3x2pt':
            ell_max = 3000
        else:
            raise ValueError('probe must be "WL", "GC" or "3x2pt"')

        ell, delta_ell = compute_ells(nbl, ell_min, ell_max, ell_grid_recipe)

        np.savetxt(f'{project_path}/output/ell_values/ell_values_{probe}.txt', ell)
        np.savetxt(f'{project_path}/output/ell_values/delta_ell_values_{probe}.txt', delta_ell)

        # just a check on the settings
        print(f'\n****************** settings ****************'
              f'\nprobe = {probe}\nwhich_NG = {which_NG}'
              f'\nintegration_method = {integration_method_dict[probe][which_NG]}'
              f'\nwhich_ells = {ell_grid_recipe}\nnbl = {nbl}\nhm_recipe = {hm_recipe}')

        # ! note that the ordering is such that out[i2, i1] = Cov(ell2[i2], ell[i1]). Transpose 1st 2 dimensions??
        # * ok: the check that the matrix symmetric in ell1, ell2 is below
        # print(f'check: is cov_SSC_{probe}[ell1, ell2, ...] == cov_SSC_{probe}[ell2, ell1, ...]?', np.allclose(cov_6D, np.transpose(cov_6D, (1, 0, 2, 3, 4, 5)), rtol=1e-7, atol=0))

        # ! =============================================== compute covs ===============================================

        if which_NG == 'SSC':
            PyCCL_whichNG_funct = compute_SSC_PyCCL
        elif which_NG == 'cNG':
            PyCCL_whichNG_funct = compute_cNG_PyCCL

        if probe in ['WL', 'GC']:
            cov_6D = PyCCL_whichNG_funct(cosmo, kernel_A=kernel, kernel_B=kernel, kernel_C=kernel, kernel_D=kernel,
                                         ell=ell, tkka=tkka, f_sky=f_sky,
                                         integration_method=integration_method_dict[probe][which_NG])
        elif probe == '3x2pt':
            cov_3x2pt_dict_10D = compute_3x2pt_PyCCL(PyCCL_whichNG_funct, cosmo, probe_wf_dict, ell, tkka, f_sky,
                                                     'qag_quad',
                                                     probe_ordering, probe_combinations_3x2pt)

            # stack everything and reshape to 4D
            cov_3x2pt_4D = mm.cov_3x2pt_dict_10D_to_4D(cov_3x2pt_dict_10D, probe_ordering, nbl, zbins, ind,
                                                       GL_or_LG)

        if save_covs:

            filename = f'{project_path}/output/covmat/cov_PyCCL_{which_NG}_{probe}_nbl{nbl}_ells{ell_grid_recipe}' \
                       f'_ellmax{ell_max}_hm_recipe{hm_recipe}'

            if probe in ['WL', 'GC']:
                np.save(f'{filename}_6D.npy', cov_6D)

            elif probe == '3x2pt':
                # save both as dict and as 4D npy array
                with open(f'{filename}_10D.pickle', 'wb') as handle:
                    pickle.dump(cov_3x2pt_dict_10D, handle)

                np.save(f'{filename}_4D.npy', cov_3x2pt_4D)

assert 1 > 2, 'stop here'

ind = np.load(
    f'{project_path.parent}/common_data/ind_files/variable_zbins/{triu_or_tril}_{row_col}-wise/indices_{triu_or_tril}_{row_col}-wise_zbins{zbins}.dat')

# load CosmoLike (Robin) and PySSC
robins_cov_path = '/Users/davide/Documents/Lavoro/Programmi/SSC_paper_jan22/PySSC_vs_CosmoLike/Robin/cov_SS_full_sky_rescaled'
cov_Robin_2D = np.load(robins_cov_path + '/lmax5000_noextrap/davides_reshape/cov_R_WL_SSC_lmax5000_2D.npy')
cov_PySSC_4D = np.load(project_path / 'input/CovMat-ShearShear-SSC-20bins-NL_flag_2_4D.npy')

# reshape
cov_SSC_4D = mm.cov_6D_to_4D(cov_6D, nbl=nbl, npairs=zpairs_auto, ind=ind_LL)
cov_SSC_2D = mm.cov_4D_to_2D(cov_SSC_4D, nbl=nbl, npairs_AB=zpairs_auto, npairs_CD=None, block_index='vincenzo')

cov_Robin_4D = mm.cov_2D_to_4D(cov_Robin_2D, nbl=nbl, npairs=zpairs_auto, block_index='vincenzo')
cov_PySSC_6D = mm.cov_4D_to_6D(cov_PySSC_4D, nbl=nbl, zbins=zbins, probe='LL', ind=ind_LL)

# save PyCCL 2D
np.save(f'{project_path}/output/cov_PyCCL_nbl{nbl}_ells{ell_grid_recipe}_2D.npy', cov_SSC_2D)

# check if the matrics are symmetric in ell1 <-> ell2
print(np.allclose(cov_Robin_4D, cov_Robin_4D.transpose(1, 0, 2, 3), rtol=1e-10))
print(np.allclose(cov_SSC_4D, cov_SSC_4D.transpose(1, 0, 2, 3), rtol=1e-10))
print(np.allclose(cov_PySSC_4D, cov_PySSC_4D.transpose(1, 0, 2, 3), rtol=1e-10))

mm.matshow(cov_6D[:, :, 0, 0, 0, 0], log=True, title='cov_PyCCL_6D')
mm.matshow(cov_PySSC_6D[:, :, 0, 0, 0, 0], log=True, title='cov_PySSC_6D')

# show the various versions
mm.matshow(cov_PySSC_4D[:, :, 0, 0], log=True, title='cov_PySSC_4D')
mm.matshow(cov_Robin_4D[:, :, 0, 0], log=True, title='cov_Robin_4D')
mm.matshow(cov_SSC_4D[:, :, 0, 0], log=True, title='cov_PyCCL_4D')

# compute and plot percent difference (not in log scale)
PyCCL_vs_rob = mm.compare_2D_arrays(cov_SSC_4D[:, :, 0, 0], cov_Robin_4D[:, :, 0, 0], 'cov_PyCCL_4D', 'cov_Robin_4D',
                                    log_arr=True)
PySSC_vs_PyCCL = mm.compare_2D_arrays(cov_SSC_4D[:, :, 0, 0], cov_PySSC_4D[:, :, 0, 0], 'cov_PyCCL_4D',
                                      'cov_PySSC_4D', log_arr=True)

# mm.matshow(rob_vs_PyCCL[:, :, 0, 0], log=False, title='rob_vs_PyCCL [%]')
# mm.matshow(rob_vs_PySSC[:, :, 0, 0], log=False, title='rob_vs_PySSC [%]')

# correlation matrices
corr_PySSC_4D = mm.correlation_from_covariance(cov_PySSC_4D[:, :, 0, 0])
corr_Robin_4D = mm.correlation_from_covariance(cov_Robin_4D[:, :, 0, 0])
corr_PyCCL_4D = mm.correlation_from_covariance(cov_SSC_4D[:, :, 0, 0])

corr_PyCCL_vs_rob = mm.compare_2D_arrays(corr_PyCCL_4D, corr_Robin_4D, 'corr_PyCCL_4D', 'corr_Robin_4D', log_arr=False)
corr_PySSC_vs_PyCCL = mm.compare_2D_arrays(corr_PyCCL_4D, corr_PySSC_4D, 'corr_PyCCL_4D', 'corr_PySSC_4D',
                                           log_arr=False)

print('done')

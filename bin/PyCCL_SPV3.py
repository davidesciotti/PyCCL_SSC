import pdb
import pickle
import sys
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyccl as ccl
import yaml
from joblib import Parallel, delayed
import ray
from tqdm import tqdm

# get project directory adn import useful modules
project_path = Path.cwd().parent

sys.path.append(f'../lib')
import my_module as mm
import ell_values as ell_utils
import wf_cl_lib

sys.path.append('/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/jobs/SPV3_magcut_zcut/config')
import config_SPV3_magcut_zcut as cfg

matplotlib.use('Qt5Agg')
start_time = time.perf_counter()


# plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def initialize_trispectrum(probe_ordering, which_tkka):
    assert which_tkka in ('SSC', 'SSC_linear_bias', 'cNG')

    halomod_start_time = time.perf_counter()

    # ! tk3d_SSC computation, from
    # ! https://github.com/LSSTDESC/CCL/blob/4df2a29eca58d7cd171bc1986e059fd35f425d45/benchmarks/test_covariances.py
    # ! see also https://github.com/LSSTDESC/CCLX/blob/master/Halo-model-Pk.ipynb
    mass_def = ccl.halos.MassDef200m()  # default is (c_m = 'Duffy08')
    concentration = ccl.halos.ConcentrationDuffy08(mass_def)
    halo_mass_func = ccl.halos.MassFuncTinker10(cosmo_ccl, mass_def=mass_def)
    halo_bias_func = ccl.halos.HaloBiasTinker10(cosmo_ccl, mass_def=mass_def)
    halo_profile_nfw = ccl.halos.HaloProfileNFW(concentration)
    halo_profile_hod = ccl.halos.HaloProfileHOD(concentration)  # default has is_number_counts=True
    hm_calculator = ccl.halos.HMCalculator(cosmo_ccl, halo_mass_func, halo_bias_func, mass_def=mass_def)

    halo_profile_dict = {
        'L': halo_profile_nfw,
        'G': halo_profile_hod,
    }

    prof_2pt_dict = {
        ('L', 'L'): ccl.halos.Profile2pt(),
        ('G', 'L'): ccl.halos.Profile2pt(),
        # see again https://github.com/LSSTDESC/CCLX/blob/master/Halo-model-Pk.ipynb
        ('G', 'G'): ccl.halos.Profile2ptHOD(),
    }
    tkka_dict = {}

    # 🐛 bug fixed: it was p_of_k_a=Pk, but it should use the LINEAR power spectrum, so we leave it as None (see documentation:
    # https://ccl.readthedocs.io/en/latest/api/pyccl.halos.halo_model.html?highlight=halomod_Tk3D_SSC#pyccl.halos.halo_model.halomod_Tk3D_SSC)
    # 🐛 bug fixed: normprof shoud be True
    if which_tkka == 'SSC':
        for A, B in probe_ordering:
            for C, D in probe_ordering:
                print(f'Computing tkka {which_tkka} for {A}{B}{C}{D}')
                tkka_dict[A, B, C, D] = ccl.halos.halomod_Tk3D_SSC(cosmo=cosmo_ccl, hmc=hm_calculator,
                                                                   prof1=halo_profile_dict[A],
                                                                   prof2=halo_profile_dict[B],
                                                                   prof3=halo_profile_dict[C],
                                                                   prof4=halo_profile_dict[D],
                                                                   prof12_2pt=prof_2pt_dict[A, B],
                                                                   prof34_2pt=prof_2pt_dict[C, D],
                                                                   normprof1=True, normprof2=True,
                                                                   normprof3=True, normprof4=True,
                                                                   lk_arr=None, a_arr=a_grid_increasing, p_of_k_a=None)

    # TODO finish this, insert the linear bias values and better understand the prof argument
    if which_tkka == 'SSC_linear_bias':
        raise NotImplementedError('halomod_Tk3D_SSC_linear_bias not implemented yet')

        """
        tkka_dict = {}
        for A, B in probe_ordering:
            for C, D in probe_ordering:
                print(f'Computing tkka {which_tkka} for {A}{B}{C}{D}')
                tkka_dict[A, B, C, D] = ccl.halos.halomod_Tk3D_SSC_linear_bias(cosmo=cosmo_ccl,
                                                                                            hmc=hm_calculator,
                                                                                            prof=...,
                                                                                            bias1=1,
                                                                                            bias2=1, 
                                                                                            bias3=1, 
                                                                                            bias4=1,
                                                                                            is_number_counts1=False,
                                                                                            is_number_counts2=False,
                                                                                            is_number_counts3=False,
                                                                                            is_number_counts4=False,
                                                                                            p_of_k_a=None, lk_arr=None,
                                                                                            a_arr=a_grid_increasing_forttka, extrap_order_lok=1,
                                                                                            extrap_order_hik=1,
                                                                                            use_log=False)
        """

    # TODO test tkka for cNG
    elif which_tkka == 'cNG':
        for A, B in probe_ordering:
            for C, D in probe_ordering:
                print(f'Computing tkka {which_tkka} for {A}{B}{C}{D}')
                tkka_dict[A, B, C, D] = ccl.halos.halomod_Tk3D_1h(cosmo=cosmo_ccl, hmc=hm_calculator,
                                                                  prof1=halo_profile_dict[A],
                                                                  prof2=halo_profile_dict[B],
                                                                  prof3=halo_profile_dict[C],
                                                                  prof4=halo_profile_dict[D],
                                                                  prof12_2pt=prof_2pt_dict[A, B],
                                                                  prof34_2pt=prof_2pt_dict[C, D],
                                                                  normprof1=True,
                                                                  normprof2=True,
                                                                  normprof3=True,
                                                                  normprof4=True,
                                                                  lk_arr=None, a_arr=a_grid_increasing,
                                                                  use_log=False)

    print('trispectrum computed in {:.2f} s'.format(time.perf_counter() - halomod_start_time))
    return tkka_dict


def compute_cov_SSC_ccl(cosmo, kernel_A, kernel_B, kernel_C, kernel_D, ell, tkka, f_sky,
                        ind_AB, ind_CD, integration_method='spline'):
    zpairs_AB = ind_AB.shape[0]
    zpairs_CD = ind_CD.shape[0]

    # parallel version:
    start_time = time.perf_counter()

    cov_ssc = Parallel(n_jobs=-1, backend='threading')(
        delayed(ccl.covariances.angular_cl_cov_SSC)(cosmo,
                                                    cltracer1=kernel_A[ind_AB[ij, -2]],
                                                    cltracer2=kernel_B[ind_AB[ij, -1]],
                                                    ell=ell, tkka=tkka,
                                                    sigma2_B=None, fsky=f_sky,
                                                    cltracer3=kernel_C[ind_CD[kl, -2]],
                                                    cltracer4=kernel_D[ind_CD[kl, -1]],
                                                    ell2=None,
                                                    integration_method=integration_method)
        for kl in tqdm(range(zpairs_CD))
        for ij in range(zpairs_AB))

    print(f'parallel version took {(time.perf_counter() - start_time):.2f} s')

    cov_ssc = np.array(cov_ssc).transpose(1, 2, 0).reshape(nbl, nbl, zpairs_AB, zpairs_CD)

    return cov_ssc


def compute_cov_cNG_ccl(cosmo, kernel_A, kernel_B, kernel_C, kernel_D, ell, tkka, f_sky,
                        ind_AB, ind_CD, integration_method='spline'):
    zpairs_AB = ind_AB.shape[0]
    zpairs_CD = ind_CD.shape[0]

    # parallel version:
    start_time = time.perf_counter()
    cov_cng = Parallel(
        n_jobs=-1, backend='threading')(
        delayed(ccl.covariances.angular_cl_cov_cNG)(cosmo,
                                                    tracer1=kernel_A[ind_AB[ij, -2]],
                                                    tracer2=kernel_B[ind_AB[ij, -1]],
                                                    ell=ell, t_of_kk_a=tkka, fsky=f_sky,
                                                    tracer3=kernel_C[ind_CD[kl, -2]],
                                                    tracer4=kernel_D[ind_CD[kl, -1]],
                                                    ell2=None,
                                                    integration_method=integration_method)
        for kl in tqdm(range(zpairs_CD))
        for ij in range(zpairs_AB))
    print(f'parallel version took {(time.perf_counter() - start_time):.2f} s')

    # move ell1, ell2 to first 2 axes and expand the last 2 axes to the number of redshift pairs
    cov_cng = np.array(cov_cng).transpose(1, 2, 0).reshape(nbl, nbl, zpairs_AB, zpairs_CD)

    return cov_cng


def compute_3x2pt_PyCCL(ng_function, cosmo, kernel_dict, ell, tkka_dict, f_sky, integration_method,
                        probe_ordering, ind_dict, output_4D_array=True):
    cov_ng_3x2pt_dict_8D = {}
    for A, B in probe_ordering:
        for C, D in probe_ordering:
            print('3x2pt: working on probe combination ', A, B, C, D)
            cov_ng_3x2pt_dict_8D[A, B, C, D] = ng_function(cosmo=cosmo,
                                                           kernel_A=kernel_dict[A],
                                                           kernel_B=kernel_dict[B],
                                                           kernel_C=kernel_dict[C],
                                                           kernel_D=kernel_dict[D],
                                                           ell=ell, tkka=tkka_dict[A, B, C, D], f_sky=f_sky,
                                                           ind_AB=ind_dict[A + B],
                                                           ind_CD=ind_dict[C + D],
                                                           integration_method=integration_method)

    if output_4D_array:
        return mm.cov_3x2pt_8D_dict_to_4D(cov_ng_3x2pt_dict_8D, probe_ordering)

    return cov_ng_3x2pt_dict_8D


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


# ! POTENTIAL ISSUES:
# 1. input files (WF, ell, a, pk...)
# 2. halo model recipe
# 3. ordering of the resulting covariance matrix
# * fanstastic collection of notebooks: https://github.com/LSSTDESC/CCLX

general_cfg = cfg.general_cfg
covariance_cfg = cfg.covariance_cfg

# ! settings
ell_grid_recipe = 'ISTF'
probes = ('3x2pt',)
which_NGs = ('SSC',)
save_covs = True
test_against_benchmarks = False
GL_or_LG = covariance_cfg['GL_or_LG']
ell_min = general_cfg['ell_min']
ell_max_WL = general_cfg['ell_max_WL']
nbl_WL = general_cfg['nbl_WL_opt']
zbins = general_cfg['zbins']
triu_tril = covariance_cfg['triu_tril']
row_col_major = covariance_cfg['row_col_major']
# z_grid = np.linspace(cfg['z_min_sigma2'], cfg['z_max_sigma2'], cfg['z_steps_sigma2'])
# a_grid_increasing_forttka = (1 / (1 + z_grid))[::-1][::6]
warnings.warn('increase the number of points in the grid, for now Im only testing if this works')
f_sky = covariance_cfg['fsky']
# n_samples_wf = cfg['n_samples_wf']
# get_3xtpt_cov_in_4D = cfg['get_3x2pt_cov_in_4D']
which_pk_vincenzo = 'HMCode2020'
specs = 'ML245-MS245-idIA2-idB3-idM3-idR1'
EP_or_ED = general_cfg['EP_or_ED']
vincenzo_wf_folder = '/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/LiFEforSPV3/InputFiles/InputSSC/Windows'
vincenzo_wf_filename = 'wi{probe:s}-{EP_or_ED:s}{zbins:02d}-{specs:s}.dat'
maglim = 245
A_IA = 0.16
eta_IA = 1.66
# ! settings

# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


# get number of redshift pairs
zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_zpairs(zbins)
ind = mm.build_full_ind(triu_tril, row_col_major, zbins)
ind_auto = ind[:zpairs_auto, :]
ind_cross = ind[zpairs_auto:zpairs_auto + zpairs_cross, :]

assert GL_or_LG == 'GL', 'you should update ind_cross (used in ind_dict) for GL, but we work with GL...'

# ! compute cls, just as a test
ell_grid, _ = ell_utils.compute_ells(nbl_WL, ell_min, ell_max_WL, ell_grid_recipe)
np.savetxt(f'{project_path}/output/ell_values/ell_values_nbl_WL{nbl_WL}.txt', ell_grid)

# Create new Cosmology object with a given set of parameters. This keeps track of previously-computed cosmological
# functions
cosmo_ccl = wf_cl_lib.instantiate_ISTFfid_PyCCL_cosmo_obj()

other_specs = {'EP_or_ED': EP_or_ED,
               'zbins': zbins,
               'magcut_lens': maglim,
               'magcut_source': maglim,
               'specs': specs}
wf_delta = np.genfromtxt(f'{vincenzo_wf_folder}/{vincenzo_wf_filename.format(probe="delta", **other_specs)}')
wf_gamma = np.genfromtxt(f'{vincenzo_wf_folder}/{vincenzo_wf_filename.format(probe="gamma", **other_specs)}')
wf_ia = np.genfromtxt(f'{vincenzo_wf_folder}/{vincenzo_wf_filename.format(probe="ia", **other_specs)}')
wf_mu = np.genfromtxt(f'{vincenzo_wf_folder}/{vincenzo_wf_filename.format(probe="mu", **other_specs)}')

z_grid_kernels = wf_delta[:, 0]
wf_delta = wf_delta[:, 1:]
wf_gamma = wf_gamma[:, 1:]
wf_ia = wf_ia[:, 1:]
wf_mu = wf_mu[:, 1:]

# construct lensing kernel
ia_bias = wf_cl_lib.build_IA_bias_1d_arr(z_grid_kernels, input_z_grid_lumin_ratio=None, input_lumin_ratio=None,
                                         cosmo=cosmo_ccl,
                                         A_IA=A_IA, eta_IA=eta_IA, beta_IA=0, C_IA=None,
                                         growth_factor=None, Omega_m=None, output_F_IA_of_z=False)
wf_lensing = wf_gamma + ia_bias[:, None] * wf_ia
wf_galaxy = wf_delta + wf_mu


for zi in range(zbins):
    wf_lensing[zi] = ccl.tracers.Tracer().add_tracer(cosmo=cosmo_ccl, kernel=(z_grid_kernels, wf_lensing[:, zi]), transfer_ka=None,
                                    transfer_k=None, transfer_a=None, der_bessel=0, der_angles=2,
                                    is_logt=False, extrap_order_lok=0, extrap_order_hik=2)
    wf_delta[zi] = ccl.tracers.Tracer().add_tracer(cosmo=cosmo_ccl, kernel=(z_grid_kernels, wf_delta[:, zi]), transfer_ka=None,
                                    transfer_k=None, transfer_a=None, der_bessel=0, der_angles=1,
                                    is_logt=False, extrap_order_lok=0, extrap_order_hik=2)
    wf_mu[zi] = ccl.tracers.Tracer().add_tracer(cosmo=cosmo_ccl, kernel=(z_grid_kernels, wf_mu[:, zi]), transfer_ka=None,
                                    transfer_k=None, transfer_a=None, der_bessel=0, der_angles=1,
                                    is_logt=False, extrap_order_lok=0, extrap_order_hik=2)

# import the pk and instantiate pyccl pk objects to compute the cls and check againts vincenzo's files

vincenzo_pk_folder = cfg['vincenzo_pk_folder'].format(which_pk_vincenzo=which_pk_vincenzo, is_flat=cfg['is_flat'])
pk_mm_table = np.genfromtxt(
    '/Users/davide/Documents/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/LiFEforSPV3/InputFiles/InputPS/TakaBird/InFiles/Flat/h/PddVsZedLogK-h_6.700e-01.dat')
z_grid_Pk = np.unique(pk_mm_table[:, 0])
k_grid_Pk = np.unique(10 ** pk_mm_table[:, 1])
pk_mm_2d = np.reshape(pk_mm_table[:, 2], (len(z_grid_Pk), len(k_grid_Pk))).T  # I want P(k, z), not P(z, k)

# now compute Pk_gm and Pk_gg
gal_bias = wf_cl_lib.b_of_z_fs2_fit(z_grid_Pk, maglim=cfg['maglim'])
pk_gm_2d = pk_mm_2d * gal_bias
pk_gg_2d = pk_mm_2d * gal_bias ** 2
pk_mm_ccl = ccl.pk2d.Pk2D
pk_flipped_in_z = np.flip(pk_2d, axis=0)
scale_factor_grid_pk = cosmo_lib.z_to_a(z_grid_pk)[::-1]  # flip it
pk2d_pyccl = ccl.pk2d.Pk2D(pkfunc=None, a_arr=scale_factor_grid_pk, lk_arr=np.log(k_grid_pk),
                           pk_arr=pk_flipped_in_z, is_logp=False, cosmo=cosmo)


cl_LL_3D = wf_cl_lib.cl_PyCCL(wf_lensing, wf_lensing, ell_grid, zbins, p_of_k_a=pk_mm, cosmo=cosmo_ccl)
cl_gg_3D = wf_cl_lib.cl_PyCCL(wf_delta, wf_delta, ell_grid, zbins, p_of_k_a=pk_gg, cosmo=cosmo_ccl)
cl_gmu_3D = wf_cl_lib.cl_PyCCL(wf_delta, wf_mu, ell_grid, zbins, p_of_k_a=pk_gmu, cosmo=cosmo_ccl)



assert False, 'stop here'

# source redshift distribution, default ISTF values for bin edges & analytical prescription for the moment
niz_unnormalized_arr = np.asarray(
    [wf_cl_lib.niz_unnormalized_analytical(z_grid, zbin_idx) for zbin_idx in range(zbins)])
niz_normalized_arr = wf_cl_lib.normalize_niz_simps(niz_unnormalized_arr, z_grid).T
n_of_z = niz_normalized_arr

# galaxy bias
galaxy_bias_2d_array = wf_cl_lib.build_galaxy_bias_2d_arr(bias_values=None, z_values=None, zbins=zbins,
                                                          z_grid=z_grid, bias_model=bias_model,
                                                          plot_bias=False)

# IA bias
ia_bias_1d_array = wf_cl_lib.build_IA_bias_1d_arr(z_grid, input_lumin_ratio=None, cosmo=cosmo_ccl,
                                                  A_IA=None, eta_IA=None, beta_IA=None, C_IA=None, growth_factor=None,
                                                  Omega_m=None)

# # ! compute tracer objects
wf_lensing = [ccl.tracers.WeakLensingTracer(cosmo_ccl, dndz=(z_grid, n_of_z[:, zbin_idx]),
                                            ia_bias=(z_grid, ia_bias_1d_array), use_A_ia=False, n_samples=n_samples_wf)
              for zbin_idx in range(zbins)]

wf_galaxy = [ccl.tracers.NumberCountsTracer(cosmo_ccl, has_rsd=False, dndz=(z_grid, n_of_z[:, zbin_idx]),
                                            bias=(z_grid, galaxy_bias_2d_array[:, zbin_idx]),
                                            mag_bias=None, n_samples=n_samples_wf)
             for zbin_idx in range(zbins)]

# the cls are not needed, but just in case:
# cl_LL_3D = wf_cl_lib.cl_PyCCL(wf_lensing, wf_lensing, ell_grid, zbins, p_of_k_a=None, cosmo=cosmo_ccl)
# cl_GL_3D = wf_cl_lib.cl_PyCCL(wf_galaxy, wf_lensing, ell_grid, zbins, p_of_k_a=None, cosmo=cosmo_ccl)
# cl_GG_3D = wf_cl_lib.cl_PyCCL(wf_galaxy, wf_galaxy, ell_grid, zbins, p_of_k_a=None, cosmo=cosmo_ccl)


# notebook for mass_relations: https://github.com/LSSTDESC/CCLX/blob/master/Halo-mass-function-example.ipynb
# Cl notebook: https://github.com/LSSTDESC/CCL/blob/v2.0.1/examples/3x2demo.ipynb
# SSC cov for KiDS: https://github.com/tilmantroester/KiDS-1000xtSZ/blob/master/tools/covariance_NG.py
# covariance ordering stuff
probe_ordering = (('L', 'L'), (GL_or_LG[0], GL_or_LG[1]), ('G', 'G'))

# convenience dictionaries
ind_dict = {
    'LL': ind_auto,
    'GL': ind_cross,
    'GG': ind_auto,
}

probe_idx_dict = {
    'L': 0,
    'G': 1
}

kernel_dict = {
    'L': wf_lensing,
    'G': wf_galaxy
}

integration_method_dict = {
    'LL': {
        'SSC': 'qag_quad',
        'cNG': 'qag_quad',
    },
    'GG': {
        'SSC': 'qag_quad',
        'cNG': 'qag_quad',
    },
    '3x2pt': {
        'SSC': 'qag_quad',
        'cNG': 'qag_quad',
    }
}
# TODO test if qag_quad works for all cases
# integration_method_dict = {ket: qag_quad for key in keys()} (pseudocode)

for probe in probes:
    for which_NG in which_NGs:

        assert probe in ['LL', 'GG', '3x2pt'], 'probe must be either LL, GG, or 3x2pt'
        assert which_NG in ['SSC', 'cNG'], 'which_NG must be either SSC or cNG'
        assert ell_grid_recipe in ['ISTF', 'ISTNL'], 'ell_grid_recipe must be either ISTF or ISTNL'

        if ell_grid_recipe == 'ISTNL' and nbl != 20:
            print('Warning: ISTNL uses 20 ell bins')

        if probe == 'LL':
            kernel = wf_lensing
        elif probe == 'GG':
            kernel = wf_galaxy

        # just a check on the settings
        print(f'\n****************** settings ****************'
              f'\nprobe = {probe}\nwhich_NG = {which_NG}'
              f'\nintegration_method = {integration_method_dict[probe][which_NG]}'
              f'\nwhich_ells = {ell_grid_recipe}\nnbl = {nbl}'
              f'\n********************************************')

        # ! =============================================== compute covs ===============================================

        if which_NG == 'SSC':
            ng_function = compute_cov_SSC_ccl
            tkka_dict = initialize_trispectrum(probe_ordering, which_tkka='SSC')
        elif which_NG == 'cNG':
            ng_function = compute_cov_cNG_ccl
            tkka_dict = initialize_trispectrum(probe_ordering, which_tkka='cNG')
        else:
            raise ValueError('which_NG must be either SSC or cNG')

        if probe in ['LL', 'GG']:
            assert probe[0] == probe[1], 'probe must be either LL or GG'

            kernel_A = kernel_dict[probe[0]]
            kernel_B = kernel_dict[probe[1]]
            kernel_C = kernel_dict[probe[0]]
            kernel_D = kernel_dict[probe[1]]
            ind_AB = ind_dict[probe[0] + probe[1]]
            ind_CD = ind_dict[probe[0] + probe[1]]

            cov_ng_4D = ng_function(cosmo_ccl,
                                    kernel_A=kernel_A, kernel_B=kernel_B,
                                    kernel_C=kernel_C, kernel_D=kernel_D,
                                    ell=ell_grid, tkka_dict=tkka_dict[probe[0], probe[1], probe[0], probe[1]],
                                    f_sky=f_sky,
                                    ind_AB=ind_AB, ind_CD=ind_CD,
                                    integration_method=integration_method_dict[probe][which_NG])

        elif probe == '3x2pt':
            cov_ng_4D = compute_3x2pt_PyCCL(ng_function=ng_function, cosmo=cosmo_ccl,
                                            kernel_dict=kernel_dict,
                                            ell=ell_grid, tkka_dict=tkka_dict, f_sky=f_sky,
                                            probe_ordering=probe_ordering,
                                            ind_dict=ind_dict,
                                            output_4D_array=True,
                                            integration_method=integration_method_dict[probe][which_NG])
        else:
            raise ValueError('probe must be either LL, GG, or 3x2pt')

        cov_ng_2D = mm.cov_4D_to_2D(cov_ng_4D)

        # ! note that the ordering is such that out[i2, i1] = Cov(ell2[i2], ell[i1]). Transpose 1st 2 dimensions??
        # * ok: the check that the matrix symmetric in ell1, ell2 is below
        print(f'check: is cov_SSC_{probe}[ell1, ell2, ...] == cov_SSC_{probe}[ell2, ell1, ...]?')
        np.testing.assert_allclose(cov_ng_4D, np.transpose(cov_ng_4D, (1, 0, 2, 3)), rtol=1e-7, atol=0)
        np.testing.assert_allclose(cov_ng_2D, cov_ng_2D.T, rtol=1e-7, atol=0)

        if save_covs:
            output_folder = f'{project_path}/output/covmat/after_script_update'
            filename = f'cov_PyCCL_{which_NG}_{probe}_nbl{nbl}_ellmax{ell_max}'

            np.savez_compressed(f'{output_folder}/{filename}_4D.npz', cov_ng_4D)
            np.savez_compressed(f'{output_folder}/{filename}_2D.npz', cov_ng_2D)

        if test_against_benchmarks:
            mm.test_folder_content(output_folder, output_folder + 'benchmarks', 'npy', verbose=False, rtol=1e-10)

assert 1 > 2, 'end of script'

print('done')

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

sys.path.append(f'../../common_lib_and_cfg/common_lib')
import my_module as mm
import cosmo_lib as csmlib

sys.path.append(f'../../SSC_restructured_v2/bin')
import ell_values as ell_utils

sys.path.append(f'../../cl_v2/bin')
import wf_cl_lib

matplotlib.use('Qt5Agg')
start_time = time.perf_counter()


# plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)


# ======================================================================================================================
# ======================================================================================================================
# ======================================================================================================================


def initialize_trispectrum(probe_ordering, which_tkka):
    assert which_tkka in ('SSC', 'SSC_linear_bias', 'cNG')
    # 🐛 bug fixed: it was p_of_k_a=Pk, but it should use the LINEAR power spectrum, so we leave it as None (see documentation:
    # https://ccl.readthedocs.io/en/latest/api/pyccl.halos.halo_model.html?highlight=halomod_Tk3D_SSC#pyccl.halos.halo_model.halomod_Tk3D_SSC)
    # 🐛 bug fixed: normprof shoud be True

    tkka_start_time = time.perf_counter()

    # ! tk3d_SSC computation, from
    # ! https://github.com/LSSTDESC/CCL/blob/4df2a29eca58d7cd171bc1986e059fd35f425d45/benchmarks/test_covariances.py
    # ! see also https://github.com/LSSTDESC/CCLX/blob/master/Halo-model-Pk.ipynb
    warnings.warn('check_ 200c or 200m?')
    mass_def = ccl.halos.MassDef200c  # default is (c_m = 'Duffy08')
    concentration = ccl.halos.ConcentrationDuffy08()
    halo_mass_func = ccl.halos.MassFuncTinker10(mass_def=mass_def)
    halo_bias_func = ccl.halos.HaloBiasTinker10(mass_def=mass_def)
    halo_profile_nfw = ccl.halos.HaloProfileNFW(mass_def='200c', concentration=concentration)
    halo_profile_hod = ccl.halos.HaloProfileHOD(mass_def='200c', concentration=concentration)  # default has is_number_counts=True
    hm_calculator = ccl.halos.HMCalculator(mass_function=halo_mass_func, halo_bias=halo_bias_func, mass_def=mass_def)
    tkka_dict = {}

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

    if which_tkka == 'SSC':

        if cfg['use_HOD_for_GCph']:
            # this is the correct way to initialize the trispectrum, but the code does not run.
            # Asked David Alonso about this.
            for A, B in probe_ordering:
                for C, D in probe_ordering:
                    print(f'Computing tkka {which_tkka} for {A}{B}{C}{D}')
                    tkka_dict[A, B, C, D] = ccl.halos.halomod_Tk3D_SSC(cosmo=cosmo_ccl, hmc=hm_calculator,
                                                                       prof=halo_profile_dict[A],
                                                                       prof2=halo_profile_dict[B],
                                                                       prof3=halo_profile_dict[C],
                                                                       prof4=halo_profile_dict[D],
                                                                       prof12_2pt=prof_2pt_dict[A, B],
                                                                       prof34_2pt=prof_2pt_dict[C, D],
                                                                       # normprof1=True, normprof2=True,
                                                                       # normprof3=True, normprof4=True,
                                                                       lk_arr=None, a_arr=a_grid_increasing_for_ttka,
                                                                       p_of_k_a=None)
        else:
            warnings.warn('using the same halo profile (NFW) for all probes, this is not quite correct')
            tkka = ccl.halos.halomod_Tk3D_SSC(cosmo=cosmo_ccl, hmc=hm_calculator,
                                              prof=halo_profile_nfw, prof2=halo_profile_nfw,
                                              prof3=halo_profile_nfw, prof4=halo_profile_nfw,
                                              prof12_2pt=None, prof34_2pt=None,
                                              # normprof1=True, normprof2=True,
                                              # normprof3=True, normprof4=True,
                                              lk_arr=None, a_arr=a_grid_increasing_for_ttka, p_of_k_a=None)
            for A, B in probe_ordering:
                for C, D in probe_ordering:
                    tkka_dict[A, B, C, D] = tkka

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
                                                                  lk_arr=None, a_arr=a_grid_increasing_for_ttka,
                                                                  use_log=False)


    # TODO finish this, insert the linear bias values and better understand the prof argument
    elif which_tkka == 'SSC_linear_bias':
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
                                                                                            a_arr=a_grid_increasing_for_ttka, extrap_order_lok=1,
                                                                                            extrap_order_hik=1,
                                                                                            use_log=False)
        """

    print('trispectrum computed in {:.2f} s'.format(time.perf_counter() - tkka_start_time))
    return tkka_dict


def compute_cov_SSC_ccl(cosmo, kernel_A, kernel_B, kernel_C, kernel_D, ell, tkka, f_sky,
                        ind_AB, ind_CD, integration_method='spline'):
    zpairs_AB = ind_AB.shape[0]
    zpairs_CD = ind_CD.shape[0]

    # parallel version:
    start_time = time.perf_counter()

    cov_ssc = Parallel(n_jobs=-1, backend='threading')(
        delayed(ccl.covariances.angular_cl_cov_SSC)(cosmo,
                                                    tracer1=kernel_A[ind_AB[ij, -2]],
                                                    tracer2=kernel_B[ind_AB[ij, -1]],
                                                    ell=ell, t_of_kk_a=tkka,
                                                    sigma2_B=None, fsky=f_sky,
                                                    tracer3=kernel_C[ind_CD[kl, -2]],
                                                    tracer4=kernel_D[ind_CD[kl, -1]],
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


# ! ====================================================================================================================
# ! ====================================================================================================================
# ! ====================================================================================================================

# * some resources
# fanstastic collection of notebooks: https://github.com/LSSTDESC/CCLX
# notebook for mass_relations: https://github.com/LSSTDESC/CCLX/blob/master/Halo-mass-function-example.ipynb
# Cl notebook: https://github.com/LSSTDESC/CCL/blob/v2.0.1/examples/3x2demo.ipynb
# SSC cov for KiDS: https://github.com/tilmantroester/KiDS-1000xtSZ/blob/master/tools/covariance_NG.py


# ! settings
with open('../../exact_SSC/config/cfg_exactSSC_ISTF.yml') as f:
    cfg = yaml.safe_load(f)
with open(cfg['fiducial_pars_yml_path']) as f:
    cosmo_pars_dict = yaml.safe_load(f)

ell_grid_recipe = cfg['ell_grid_recipe']
probes = cfg['probes']
which_NGs = cfg['which_NGs']
save_covs = cfg['save_covs']
test_against_benchmarks = cfg['test_against_benchmarks']
GL_or_LG = cfg['GL_or_LG']
ell_min = cfg['ell_min']
ell_max = cfg['ell_max']
nbl = cfg['nbl']
zbins = cfg['zbins']
triu_tril = cfg['triu_tril']
row_col_major = cfg['row_col_major']
z_grid = np.linspace(cfg['z_min_sigma2'], cfg['z_max_sigma2'], cfg['z_steps_sigma2'])
a_grid_increasing_for_ttka = csmlib.z_to_a(z_grid)[::-1]
f_sky = csmlib.deg2_to_fsky(cfg['sky_area_deg2'])
n_samples_wf = cfg['n_samples_wf']
get_3xtpt_cov_in_4D = cfg['get_3xtpt_cov_in_4D']
bias_model = cfg['bias_model']
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
ell_grid, _ = ell_utils.compute_ells(nbl, ell_min, ell_max, ell_grid_recipe)
np.savetxt(f'{project_path}/output/ell_values/ell_values_nbl{nbl}.txt', ell_grid)

# Create new Cosmology object with a given set of parameters
cosmo_ccl = wf_cl_lib.instantiate_PyCCL_cosmo_obj(cosmo_pars_dict['Om_m0'], cosmo_pars_dict['Om_b0'],
                                                  cosmo_pars_dict['Om_Lambda0'],
                                                  cosmo_pars_dict['w_0'], cosmo_pars_dict['w_a'],
                                                  cosmo_pars_dict['h_0'],
                                                  cosmo_pars_dict['sigma_8'], cosmo_pars_dict['n_s'],
                                                  cosmo_pars_dict['m_nu'])

# source redshift distribution, default ISTF values for bin edges & analytical prescription for the moment
niz_unnormalized_arr = np.asarray(
    [wf_cl_lib.niz_unnormalized_analytical(z_grid, zbin_idx) for zbin_idx in range(zbins)])
niz_normalized_arr = wf_cl_lib.normalize_niz_simps(niz_unnormalized_arr, z_grid).T
n_of_z = niz_normalized_arr

# galaxy bias
galaxy_bias_2d_array = wf_cl_lib.build_galaxy_bias_2d_arr(bias_values=None, z_values=None, zbins=zbins,
                                                          z_grid=z_grid, bias_model=bias_model,
                                                          plot_bias=True)

# IA bias
ia_bias_1d_array = wf_cl_lib.build_IA_bias_1d_arr(z_grid, input_lumin_ratio=None, cosmo=cosmo_ccl,
                                                  A_IA=cosmo_pars_dict['A_IA'], eta_IA=cosmo_pars_dict['eta_IA'],
                                                  beta_IA=cosmo_pars_dict['beta_IA'], C_IA=cosmo_pars_dict['C_IA'],
                                                  growth_factor=None,
                                                  Omega_m=cosmo_pars_dict['Om_m0'])

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


# covariance ordering stuff
probe_ordering = (('L', 'L'), (GL_or_LG[0], GL_or_LG[1]), ('G', 'G'))
warnings.warn('TESTING, restore probe_ordering')
probe_ordering = (('G', 'L'),)

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

# ! =============================================== compute covs ===============================================
for probe in probes:
    for which_NG in which_NGs:

        assert probe in ['LL', 'GG', '3x2pt'], 'probe must be either LL, GG, or 3x2pt'
        assert which_NG in ['SSC', 'cNG'], 'which_NG must be either SSC or cNG'
        assert ell_grid_recipe in ['ISTF', 'ISTNL'], 'ell_grid_recipe must be either ISTF or ISTNL'

        if probe == 'LL':
            kernel = wf_lensing
        elif probe == 'GG':
            kernel = wf_galaxy

        # just a check on the settings
        print(f'\n****************** settings ****************'
              f'\nprobe = {probe}\nwhich_NG = {which_NG}'
              f'\nintegration_method = {integration_method_dict[probe][which_NG]}'
              f'\nwhich_ells = {ell_grid_recipe}\nnbl = {nbl}\nzbins = {zbins}'
              f'\ncfg["use_HOD_for_GCph"]'
              f'\n********************************************')

        if which_NG == 'SSC':
            ng_function = compute_cov_SSC_ccl
        elif which_NG == 'cNG':
            ng_function = compute_cov_cNG_ccl
        else:
            raise ValueError('which_NG must be either SSC or cNG')

        tkka_dict = initialize_trispectrum(probe_ordering, which_tkka=which_NG)

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
                                    ell=ell_grid, tkka=tkka_dict[probe[0], probe[1], probe[0], probe[1]],
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

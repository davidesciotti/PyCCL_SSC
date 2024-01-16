import sys
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

ROOT = '/home/davide/Documenti/Lavoro/Programmi'
SB_ROOT = f'{ROOT}/Spaceborne'

# project modules
sys.path.append(SB_ROOT)
import bin.ell_values as ell_utils
import bin.cl_preprocessing as cl_utils
import bin.compute_Sijkl as Sijkl_utils
import bin.covariance as covmat_utils
import bin.fisher_matrix as FM_utils
import bin.my_module as mm
import bin.cosmo_lib as csmlib
import bin.wf_cl_lib as wf_cl_lib
import common_cfg.mpl_cfg as mpl_cfg

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

is_number_counts_12 = False
is_number_counts_34 = True
k_units = '1overMpc'
which_su_resp = 'spv3_vincenzo'
probe_block = 'LLGL'

# pyccl responses from halo model
ccl_responses_path = f'{ROOT}/Spaceborne/jobs/SPV3_magcut_zcut_thesis/output/Flagship_2/pyccl_responses'
k_1overMpc_hm = np.genfromtxt(f'{ccl_responses_path}/{probe_block}/k_{k_units}.txt')
a_arr = np.genfromtxt(f'{ccl_responses_path}/{probe_block}/a_arr.txt')
dpk12 = np.load(f'{ccl_responses_path}/{probe_block}/dpk12_is_number_counts{is_number_counts_12}.npy')
dpk34 = np.load(f'{ccl_responses_path}/{probe_block}/dpk34_is_number_counts{is_number_counts_34}.npy')

pk2d = np.load(f'{ccl_responses_path}/{probe_block}/pk2d.npy')
dpk2d = np.load(f'{ccl_responses_path}/{probe_block}/dpk2d.npy')

pk1d = np.load(f'{ccl_responses_path}/{probe_block}/pk1d.npy')
dpk1d = np.load(f'{ccl_responses_path}/{probe_block}/dpk1d.npy')

bA = np.load(f"{ccl_responses_path}/{probe_block}/bA.npy").T
bB = np.load(f"{ccl_responses_path}/{probe_block}/bB.npy").T

assert dpk12.shape == dpk34.shape == dpk2d.shape == pk2d.shape
assert np.allclose(pk2d[:, -1], pk1d, rtol=1e-4, atol=0), 'pk1d should simply be pk2d in a \sim 1'


# SU responses
if which_su_resp == 'istf_davide':
    su_responses_path = f'{ROOT}/exact_SSC/output/ISTF/other_stuff'
    r1_mm_su = np.load(f'{su_responses_path}/r1_mm_k{k_units}.npy')
    # ! I'm not sure if this is in 1/Mpc
    k_1overMpc_su = np.genfromtxt(f'{su_responses_path}/k_grid_responses_{k_units}.txt')
    z_arr_su = np.load(f'{su_responses_path}/../d2ClAB_dVddeltab/z_grid_ssc_integrand_zsteps3000.npy')

elif which_su_resp == 'spv3_vincenzo':
    vincenzo_responses_folder = '/home/davide/Documenti/Lavoro/Programmi/common_data/vincenzo/SPV3_07_2022/LiFEforSPV3/InputFiles/InputSSC/ResFun/HMCodeBar'
    vincenzo_responses_filename = 'resfun-idBM03.dat'
    rAB_of_k = np.genfromtxt(f'{vincenzo_responses_folder}/{vincenzo_responses_filename}')

    log_k_arr = np.unique(rAB_of_k[:, 0])
    k_1overMpc_su = 10 ** log_k_arr
    z_arr_su = np.unique(rAB_of_k[:, 1])

    r1_mm_su = np.reshape(rAB_of_k[:, 2], (len(k_1overMpc_su), len(z_arr_su)))
    r1_gm_su = np.reshape(rAB_of_k[:, 3], (len(k_1overMpc_su), len(z_arr_su)))
    r1_gg_su = np.reshape(rAB_of_k[:, 4], (len(k_1overMpc_su), len(z_arr_su)))

else:
    raise ValueError(f'which_su_resp = {which_su_resp} not recognized')


assert r1_mm_su.shape == (len(k_1overMpc_su), len(
    z_arr_su)), f'R1_mm_su.shape = {r1_mm_su.shape} != ({len(k_1overMpc_su)}, {len(z_arr_su)})'

# cut the arrays to the maximum redshift of the SU responses (much smaller range!)
z_arr_hm = csmlib.a_to_z(a_arr)
z_arr_hm = z_arr_hm[::-1]
zmax_su = np.max(z_arr_su)
z_arr_hm_trimmed = z_arr_hm[z_arr_hm <= zmax_su]

# pick a redshift and get the corresponding index
z = 0
z_idx_su = np.argmin(np.abs(z_arr_su - z))
z_idx_hm = np.argmin(np.abs(z_arr_hm_trimmed - z))

# re-translate z into a for hm
a_hm = 1. / (1. + z_arr_hm_trimmed[z_idx_hm])
a_idx_hm = np.argmin(np.abs(a_arr - a_hm))

z_hm = z_arr_hm_trimmed[z_idx_hm]
z_su = z_arr_su[z_idx_su]

plt.figure()
plt.plot(k_1overMpc_hm, dpk12[:, a_idx_hm] / pk2d[:, a_idx_hm],
         label=f'dpk12/pk2d, a={a_arr[a_idx_hm]}, z_hm={z_hm}', alpha=0.5)
plt.plot(k_1overMpc_hm, dpk34[:, a_idx_hm] / pk2d[:, a_idx_hm],
         label=f'dpk34/pk2d, a={a_arr[a_idx_hm]}, z_hm={z_hm}', ls='--', alpha=0.5)
# plt.plot(k_1overMpc_hm, dpk12[:, a_idx_hm] / pk1d, label=f'dpk12/pk1d, a={a_arr[a_idx_hm]:.2f}, z={z_hm:.2f}', alpha=0.5)
# plt.plot(k_1overMpc_hm, dpk34[:, a_idx_hm] / pk1d, label=f'dpk34/pk1d, a={a_arr[a_idx_hm]:.2f}, z={z_hm:.2f}', ls='--', alpha=0.5)
plt.plot(k_1overMpc_su, r1_mm_su[:, z_idx_su], label=f'R1_mm_su, a={a_arr[a_idx_hm]:.2f}, z={z_su:.2f}', alpha=0.5)
plt.legend()
plt.xlim(1e-2, 2)
plt.ylim(0., 7)
plt.xscale('log')
plt.xlabel('k [1/Mpc]')
plt.ylabel('$\partial \ln P_{mm} / \partial \ln \delta_b$')

import sys
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

project_path = Path.cwd().parent.parent.parent
job_path = Path.cwd().parent

sys.path.append(f'{project_path.parent}/common_data/common_lib')
import my_module as mm
import cosmo_lib as csmlib

sys.path.append(f'{project_path.parent}/common_data/common_config')
import ISTF_fid_params as ISTFfid
import mpl_cfg

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

is_number_counts = False

# pyccl responses from halo model
ccl_responses_path = '../output/pyccl_responses/davide'
dpk12 = np.load(f'{ccl_responses_path}/dpk12_is_number_counts{is_number_counts}.npy')
dpk34 = np.load(f'{ccl_responses_path}/dpk34_is_number_counts{is_number_counts}.npy')
k_1overMpc_hm = np.load(f'{ccl_responses_path}/k_1overMpc.npy')
dpk2d = np.load(f'{ccl_responses_path}/dpk2d.npy')
pk2d = np.load(f'{ccl_responses_path}/pk2d.npy')
pk1d = np.load(f'{ccl_responses_path}/pk1d.npy')
a_arr = np.load(f'{ccl_responses_path}/a_arr.npy')
assert dpk12.shape == dpk34.shape == dpk2d.shape == pk2d.shape


# SU responses
su_responses_path = '../../exact_SSC/output/ISTF/other_stuff'
R1_mm_su = np.load(f'{su_responses_path}/r1_mm_k1overMpc.npy')
k_1overMpc_su = np.genfromtxt(f'{su_responses_path}/k_grid_responses_1overMpc.txt')  # ! I'm not sure if this is in 1/Mpc
z_arr_su = np.load(f'{su_responses_path}/../d2ClAB_dVddeltab/z_grid_ssc_integrand_zsteps3000.npy')

assert R1_mm_su.shape == (len(k_1overMpc_su), len(z_arr_su)), f'R1_mm_su.shape = {R1_mm_su.shape} != ({len(k_1overMpc_su)}, {len(z_arr_su)})'

# cut the arrays to the maximum redshift of the SU responses (much smaller range!)
z_arr_hm = 1. / a_arr - 1.
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
plt.plot(k_1overMpc_hm, dpk12[:, a_idx_hm] / pk2d[:, a_idx_hm], label=f'dpk12, a={a_arr[a_idx_hm]}, z_hm={z_hm}')
plt.plot(k_1overMpc_hm, dpk34[:, a_idx_hm] / pk2d[:, a_idx_hm], label=f'dpk34, a={a_arr[a_idx_hm]}, z_hm={z_hm}', ls='--')
plt.plot(k_1overMpc_hm, dpk12[:, a_idx_hm] / pk1d, label=f'dpk12, a={a_arr[a_idx_hm]:.2f}, z={z_hm:.2f}')
plt.plot(k_1overMpc_hm, dpk34[:, a_idx_hm] / pk1d, label=f'dpk34, a={a_arr[a_idx_hm]:.2f}, z={z_hm:.2f}', ls='--')
plt.plot(k_1overMpc_su, R1_mm_su[:, z_idx_su], label=f'R1_mm_su, a={a_arr[a_idx_hm]:.2f}, z={z_su:.2f}')
plt.legend()
plt.xlim(1e-2, 2)
plt.ylim(0., 7)
plt.xscale('log')
plt.xlabel('k [1/Mpc]')
plt.ylabel('$\partial \ln P_{mm} / \partial \ln \delta_b$')

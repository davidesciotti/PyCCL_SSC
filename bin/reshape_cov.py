import pickle
import sys
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

project_path = Path.cwd().parent

sys.path.append(f'{project_path.parent}/common_data/common_lib')
import my_module as mm
import cosmo_lib as csmlib

sys.path.append(f'{project_path.parent}/common_data/common_config')
import ISTF_fid_params
import mpl_cfg

sys.path.append(f'{project_path}/config')
import PyCCL_config as cfg

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

# ! settings
zbins = cfg.general_cfg['zbins']
nbl = cfg.general_cfg['nbl']
triu_or_tril = cfg.general_cfg['triu_or_tril']
row_col = cfg.general_cfg['row_col']
# ! end settings


# get number of redshift pairs
zpairs_auto, zpairs_cross, zpairs_3x2pt = mm.get_pairs(zbins)

# this is to habe triu_row-wise, which does not seem to be in the ind_files directory...
# This one is probably tril_row_wise or something
ind = np.genfromtxt(f'{project_path.parent}/common_data/ind_files/'
                    f'{triu_or_tril}_{row_col}-wise/indices_{triu_or_tril}_{row_col}-wise_zbins{zbins}.dat', dtype=int)
ind_auto = ind[:zpairs_auto, :]


for probe in ['WL', 'GC']:
    for which_NG in ['SSC', 'cNG']:

        if probe == 'WL':
            ell_max = 5000
        elif probe == 'GC':
            ell_max = 3000

        cov_6D = np.load(f'{project_path}/output/covmat/'
                         f'cov_PyCCL_{which_NG}_{probe}_nbl{nbl}_ellsISTF_ellmax{ell_max}_hm_recipeKrause2017_6D.npy')
        cov_4D = mm.cov_6D_to_4D(cov_6D, nbl, zpairs_auto, ind_auto)
        cov_2D = mm.cov_4D_to_2D(cov_4D, block_index='vincenzo')
        np.save(f'{project_path}/output/covmat/2D/'
                f'cov_PyCCL_{which_NG}_{probe}_nbl{nbl}_ellsISTF_ellmax{ell_max}_hm_recipeKrause2017_2D.npy',
                cov_2D)

probe = '3x2pt'
ell_max = 3000
GL_or_LG = 'GL'
probe_ordering = ('LL', f'{GL_or_LG}', 'GG')
assert GL_or_LG == 'GL', 'double check what you are using in PyCCL_test.py'

for which_NG in ['SSC', ]:
    filename = f'{project_path}/output/covmat/cov_PyCCL_{which_NG}_3x2pt_nbl30_ellsISTF_ellmax{ell_max}_hm_recipeKrause2017'
    with open(f'{filename}_10D.pickle', 'rb') as handle:
        cov_3x2pt_10D = pickle.load(handle)

    cov_3x2pt_4D = mm.cov_3x2pt_dict_10D_to_4D(cov_3x2pt_10D, probe_ordering, nbl, zbins, ind, GL_or_LG)
    cov_3x2pt_2D = mm.cov_4D_to_2D(cov_3x2pt_4D, block_index='vincenzo')
    np.save(f'{project_path}/output/covmat/2D/'
            f'cov_PyCCL_{which_NG}_3x2pt_nbl30_ellsISTF_ellmax{ell_max}_hm_recipeKrause2017_2D.npy', cov_3x2pt_2D)

# TODO
# these are the cross indices, just to build the ind files...
np.indices((zbins, zbins))

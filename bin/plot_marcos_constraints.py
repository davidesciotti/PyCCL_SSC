import sys
import time
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json 

project_path = Path.cwd().parent

sys.path.append(f'{project_path.parent}/common_data/common_lib')
import my_module as mm
import cosmo_lib as csmlib

sys.path.append(f'{project_path.parent}/common_data/common_config')
import ISTF_fid_params as ISTFfid
import mpl_cfg

sys.path.append('/Users/davide/Documents/Lavoro/Programmi/SSC_restructured_v2/bin')
import plots_FM_running as plot_utils

matplotlib.use('Qt5Agg')
plt.rcParams.update(mpl_cfg.mpl_rcParams_dict)
start_time = time.perf_counter()

# load json into dict
json_path = f'{project_path}/config/config_SPV3_magcut_zcut.json'

probe = 'WL'
nparams_toplot = 10

fid_cosmo = np.array(
    (ISTFfid.IA_free['eta_IA'], ISTFfid.primary['Om_b0'], ISTFfid.IA_free['beta_IA'], ISTFfid.primary['n_s'],
     1., ISTFfid.primary['Om_m0'], ISTFfid.primary['sigma_8'], ISTFfid.primary['w_0'],
     ISTFfid.IA_free['A_IA'], ISTFfid.primary['h_0']*100))

# Open the JSON file and read the contents
with open(f'{project_path}/input/marcos_constraints/WL_GS_CCL.json') as json_file:
    WL_GS_CCL = json.load(json_file)
with open(f'{project_path}/input/marcos_constraints/GC_GS_CCL.json') as json_file:
    GC_GS_CCL = json.load(json_file)
with open(f'{project_path}/input/marcos_constraints/WL_GS_SSC.json') as json_file:
    WL_GS_SSC = json.load(json_file)
with open(f'{project_path}/input/marcos_constraints/GC_GS_SSC.json') as json_file:
    GC_GS_SSC = json.load(json_file)

uncert_WL_array = np.zeros((2, len(WL_GS_CCL.keys())))
uncert_GC_array = np.zeros((2, len(GC_GS_CCL.keys())))
for idx, key in enumerate(WL_GS_CCL.keys()):
    uncert_WL_array[0, idx] = WL_GS_CCL[key]
    uncert_WL_array[1, idx] = WL_GS_SSC[key]

for idx, key in enumerate(GC_GS_CCL.keys()):
    uncert_GC_array[0, idx] = GC_GS_CCL[key]
    uncert_GC_array[1, idx] = GC_GS_SSC[key]

uncert_WL_array /= fid_cosmo
uncert_WL_array = np.abs(uncert_WL_array)
# uncert_GC_array /= fid_cosmo
title = 'WL'
cases = ['CCL', 'SSC']
paramnames = list(WL_GS_CCL.keys())
plot_utils.bar_plot(uncert_WL_array[:2, :nparams_toplot], title, cases, nparams=nparams_toplot,
                    param_names_label=paramnames[:nparams_toplot], bar_width=0.12)

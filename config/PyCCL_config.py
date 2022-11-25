import sys
import time
from operator import itemgetter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from matplotlib.cm import get_cmap

project_path = Path.cwd().parent.parent.parent.parent.parent
job_path = Path.cwd().parent.parent.parent

sys.path.append(str(project_path / 'lib'))
import my_module as mm

matplotlib.use('Qt5Agg')

params = {'lines.linewidth': 3.5,
          'font.size': 20,
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
          'mathtext.fontset': 'stix',
          'font.family': 'STIXGeneral',
          'figure.figsize': (8, 8)
          # 'backend': 'Qt5Agg'
          }
plt.rcParams.update(params)
markersize = 10

start_time = time.perf_counter()

general_cfg = {
    'zbins': 10,
    'nbl': 30,
    'triu_or_tril': 'triu',
    'row_col': 'row',
    'ell_recipe': 'ISTF',
    'probes': ('3x2pt',),
    'which_NGs': ('SSC', 'cNG'),
    'save_covs': True,
    'hm_recipe': 'Krause2017',
    'GL_or_LG': 'GL',
    'ell_min': 10,
    'ell_max': 5000,
    'use_ray': False,  # TODO finish this!
}

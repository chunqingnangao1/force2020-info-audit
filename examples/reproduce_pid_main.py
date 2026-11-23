# -*- coding: utf-8 -*-
"""Tutorial: reproduce the paper's PID main table (GR curve) in three steps.

Run from the repository root after `python run_all.py --quick` (or at least
stages 1-3 of it) has produced results/e3_windows.csv.gz:

    python examples/reproduce_pid_main.py

Step 1 loads the sliding-window feature table; step 2 discretises features into
global quantile bins; step 3 decomposes the joint information of each
(linear, nonlinear) pair about the lithofacies target and evaluates a
within-well block-permutation null. Expected output (paper Table 4-2, GR):
unique information of DFA-H beyond mean/variance/lag-1 ~ 0.095 bits,
permutation p at the floor.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(ROOT, 'src', 'pid'))

import numpy as np
import pandas as pd
from run_pid import disc, pid_two_sources, block_permute_labels  # noqa: E402

N_PERM = 200  # paper uses 1000 (GR main analysis: 10,000); 200 keeps the demo quick

# Step 1: window table produced by src/features/extract_windows.py
wd = pd.read_csv(os.path.join(ROOT, 'results', 'e3_windows.csv.gz'))
wd['seg'] = wd['well'] + '|' + wd['formation'].astype(str)
y = pd.factorize(wd['grp6'])[0]
seg = wd['seg'].values

# Step 2: 4-bin global quantile discretisation of each feature
rng = np.random.default_rng(20231121)

# Step 3: PID for the (linear, DFA-H) pairs + permutation null
for lin in ['mean', 'var', 'lag1']:
    s1, ok1 = disc(wd['GR_%s' % lin].values)
    s2, ok2 = disc(wd['GR_H'].values)
    m = (s1 >= 0) & (s2 >= 0)
    s1m, s2m, ym, segm = s1[m], s2[m], y[m], seg[m]
    groups = [np.where(segm == s)[0] for s in np.unique(segm)]
    res = pid_two_sources(s1m, s2m, ym)
    null = [pid_two_sources(s1m, s2m, block_permute_labels(ym, groups, rng))['UI2_wb']
            for _ in range(N_PERM)]
    p = (np.sum(np.asarray(null) >= res['UI2_wb']) + 1) / (N_PERM + 1)
    print('%-5s | H :  unique(H) = %.4f bits   I(Y; pair) = %.3f   perm p = %.4f'
          % (lin, res['UI2_wb'], res['I12'], p))
print('\nInterpretation: the unique information of DFA-H beyond each linear statistic\n'
      'is positive and significant — the nonlinear feature carries facies information\n'
      'no single classical statistic contains (paper Section 4.5.1).')

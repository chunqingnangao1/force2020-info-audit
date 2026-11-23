# -*- coding: utf-8 -*-
"""Review-response experiments A+C: PID at 10k permutations + surrogate-verdict split."""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_pid import disc, pid_two_sources, block_permute_labels, PAIRS  # noqa

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, '..', '..', 'results'))
SEED = 20281122

def run_pid_slice(d, y, seg, n_perm, label):
    rng = np.random.default_rng(SEED)
    rows = []
    for (f1, f2) in PAIRS:
        s1, ok1 = disc(d[f1].values)
        s2, ok2 = disc(d[f2].values)
        m = (s1 >= 0) & (s2 >= 0)
        s1m, s2m, ym, sgm = s1[m], s2[m], y[m], seg[m]
        groups = [np.where(sgm == s)[0] for s in np.unique(sgm)]
        res = pid_two_sources(s1m, s2m, ym)
        nulls = {k: [] for k in ['UI1_wb', 'UI2_wb', 'S_wb']}
        for _ in range(n_perm):
            r = pid_two_sources(s1m, s2m, block_permute_labels(ym, groups, rng))
            for k in nulls:
                nulls[k].append(r[k])
        row = {'slice': label, 'pair': '%s|%s' % (f1, f2), 'n_win': int(m.sum()),
               'UI1_wb': res['UI1_wb'], 'UI2_wb': res['UI2_wb'], 'S_wb': res['S_wb'],
               'R_mmi': res['R_mmi'], 'I12': res['I12']}
        for k in nulls:
            arr = np.asarray(nulls[k])
            row['p_' + k] = float((np.sum(arr >= res[k]) + 1) / (n_perm + 1))
        rows.append(row)
        print('[%s] %s|%s n=%d UI2=%.4f p=%.5f' % (
            label, f1, f2, m.sum(), res['UI2_wb'], row['p_UI2_wb']))
    return rows

def main():
    t0 = time.time()
    wd = pd.read_csv(os.path.join(RESULTS, 'e3_windows.csv.gz'))
    wd['seg'] = wd['well'] + '|' + wd['formation'].astype(str)
    y = pd.factorize(wd['grp6'])[0]
    feats = {'mean': 'GR_mean', 'var': 'GR_var', 'lag1': 'GR_lag1', 'H': 'GR_H', 'sampen': 'GR_sampen'}
    d = pd.DataFrame({k: wd[v] for k, v in feats.items()})
    seg = wd['seg'].values

    rowsA = run_pid_slice(d, y, seg, 10000, 'full_GR_10k')
    pd.DataFrame(rowsA).to_csv(os.path.join(RESULTS, 'e3_pid_10k.csv'), index=False)

    proxy = pd.read_csv(os.path.join(RESULTS, 'proxy_test.csv'))[['seg', 'reject_any']]
    wd2 = wd.merge(proxy, on='seg', how='left')
    keep = wd2['reject_any'].notna()
    rowsC = []
    for flag, lab in [(1.0, 'reject_linear_null'), (0.0, 'no_reject')]:
        sub = wd2[keep & (wd2['reject_any'] == flag)]
        yy = pd.factorize(sub['grp6'])[0]
        dd = pd.DataFrame({k: sub[v] for k, v in feats.items()})
        rowsC += run_pid_slice(dd, yy, sub['seg'].values, 1000, lab)
    pd.DataFrame(rowsC).to_csv(os.path.join(RESULTS, 'e3_pid_rejectsplit.csv'), index=False)
    print('saved e3_pid_10k.csv + e3_pid_rejectsplit.csv  %.0fs' % (time.time() - t0))

if __name__ == '__main__':
    main()

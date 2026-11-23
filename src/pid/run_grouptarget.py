# -*- coding: utf-8 -*-
"""Second-target check: PID for stratigraphic-group classification (task transfer).
Window groups inherit the majority GROUP of their (well, formation) segment.
Output: results/e3_pid_grouptarget.csv"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_pid import disc, pid_two_sources, block_permute_labels, PAIRS  # noqa

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
RESULTS = os.path.join(ROOT, 'results')
SEED = 20311123

def run_pid_target(wd, y, seg, n_perm, curve='GR'):
    rng = np.random.default_rng(SEED)
    feats = {'mean': '%s_mean' % curve, 'var': '%s_var' % curve, 'lag1': '%s_lag1' % curve,
             'H': '%s_H' % curve, 'sampen': '%s_sampen' % curve}
    d = pd.DataFrame({k: wd[v] for k, v in feats.items()})
    rows = []
    # Segment-level permutation design: GROUP is constant within a segment, so
    # within-segment permutation cannot break the association; permute the segment->group map instead
    seg_codes, seg_uniq = pd.factorize(seg)
    seg_group_label = np.array([y[seg_codes == s][0] for s in range(len(seg_uniq))])
    for (f1, f2) in PAIRS:
        s1, ok1 = disc(d[f1].values)
        s2, ok2 = disc(d[f2].values)
        m = (s1 >= 0) & (s2 >= 0)
        s1m, s2m, ym = s1[m], s2[m], y[m]
        sm = seg_codes[m]
        res = pid_two_sources(s1m, s2m, ym)
        nulls = {'UI2_wb': [], 'UI1_wb': []}
        for _ in range(n_perm):
            perm_lab = seg_group_label[rng.permutation(len(seg_group_label))]
            yp = perm_lab[sm]
            r = pid_two_sources(s1m, s2m, yp)
            nulls['UI2_wb'].append(r['UI2_wb'])
            nulls['UI1_wb'].append(r['UI1_wb'])
        row = {'pair': '%s|%s' % (f1, f2), 'n_win': int(m.sum()),
               'I1': res['I1'], 'I2': res['I2'], 'I12': res['I12'],
               'UI1_wb': res['UI1_wb'], 'UI2_wb': res['UI2_wb'], 'R_mmi': res['R_mmi']}
        for k, arr in nulls.items():
            row['p_' + k] = float((np.sum(np.asarray(arr) >= res[k]) + 1) / (n_perm + 1))
        rows.append(row)
        print('[group] %s|%s n=%d UI2=%.4f p=%.4f' % (f1, f2, m.sum(), res['UI2_wb'], row['p_UI2_wb']))
    return rows

def main():
    t0 = time.time()
    wd = pd.read_csv(os.path.join(RESULTS, 'e3_windows.csv.gz'))
    raw = pd.read_csv(os.path.join(ROOT, 'data', 'clean', 'wells_raw.csv.gz'))
    # (well|formation) -> majority GROUP (string keys)
    raw_ok = raw[raw['FORMATION'].notna() & raw['GROUP'].notna()].copy()
    raw_ok['key'] = raw_ok['WELL'] + '|' + raw_ok['FORMATION']
    seg_group = raw_ok.groupby('key')['GROUP'].agg(lambda s: s.mode().iloc[0])
    wd['seg'] = wd['well'] + '|' + wd['formation'].astype(str)
    wd['group'] = (wd['well'] + '|' + wd['formation']).map(seg_group)
    keep = wd['group'].notna()
    wd = wd[keep]
    y = pd.factorize(wd['group'])[0]
    print('windows:', len(wd), '| groups:', wd['group'].nunique())
    print(wd['group'].value_counts().to_string())
    rows = run_pid_target(wd, y, wd['seg'].values, 1000)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'e3_pid_grouptarget.csv'), index=False)
    print('saved e3_pid_grouptarget.csv  %.0fs' % (time.time() - t0))

if __name__ == '__main__':
    main()

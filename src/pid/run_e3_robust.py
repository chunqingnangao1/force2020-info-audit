# -*- coding: utf-8 -*-
"""Robustness suite R1-R4 (12-class, binning, balanced MI, XGBoost increments)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_pid import disc, mi_disc, pid_two_sources, block_permute_labels, PAIRS  # noqa

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, '..', '..', 'results'))
SEED = 20241121

def mi_weighted(x, y, w):
    x, y = x.astype(int), y.astype(int)
    nx, ny = x.max() + 1, y.max() + 1
    joint = np.bincount(x * ny + y, weights=w, minlength=nx * ny).reshape(nx, ny)
    joint /= joint.sum()
    px = joint.sum(1, keepdims=True)
    py = joint.sum(0, keepdims=True)
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log2(joint[mask] / (px @ py)[mask])))

def main():
    wd = pd.read_csv(os.path.join(RESULTS, 'e3_windows.csv.gz'))
    wd['seg'] = wd['well'] + '|' + wd['formation'].astype(str)
    rows = []

    # ---- R1: official 12-class sensitivity (GR, 300 permutations) ----
    d = wd[['GR_mean', 'GR_var', 'GR_lag1', 'GR_H', 'GR_sampen', 'code', 'seg']].dropna(subset=['code'])
    y12 = d['code'].astype(int).values
    seg = d['seg'].values
    rng = np.random.default_rng(SEED)
    fmap = {'mean': d['GR_mean'].values, 'var': d['GR_var'].values,
            'lag1': d['GR_lag1'].values, 'H': d['GR_H'].values, 'sampen': d['GR_sampen'].values}
    for (f1, f2) in PAIRS:
        s1, _ = disc(fmap[f1]); s2, _ = disc(fmap[f2])
        m = (s1 >= 0) & (s2 >= 0)
        s1, s2, yy12 = s1[m], s2[m], y12[m]
        segf = seg[m]
        sgg = [np.where(segf == s)[0] for s in np.unique(segf)]
        res = pid_two_sources(s1, s2, yy12)
        keys = ['UI1_wb', 'UI2_wb', 'S_wb']
        nulls = {k: [] for k in keys}
        for _ in range(300):
            r = pid_two_sources(s1, s2, block_permute_labels(yy12, sgg, rng))
            for k in keys:
                nulls[k].append(r[k])
        row = {'item': 'R1_12class', 'curve': 'GR', 'pair': '%s|%s' % (f1, f2),
               'I1_12': res['I1'], 'I2_12': res['I2'], 'I12_12': res['I12'],
               'UI1_12': res['UI1_wb'], 'UI2_12': res['UI2_wb'], 'S_12': res['S_wb']}
        for k in keys:
            arr = np.asarray(nulls[k])
            row['p_' + k] = float((np.sum(arr >= res[k]) + 1) / 301)
        rows.append(row)
        print('R1 %s|%s done' % (f1, f2))

    # ---- R2: bin-count sensitivity k in {4,8} (point estimates) ----
    y6_all = wd.dropna(subset=['grp6'])
    y6 = pd.factorize(y6_all['grp6'])[0]
    for k in [4, 8]:
        for (f1, f2) in PAIRS:
            s1, _ = disc(fmap[f1], k=k); s2, _ = disc(fmap[f2], k=k)
            m = (s1 >= 0) & (s2 >= 0)
            res = pid_two_sources(s1[m], s2[m], y6[m])
            rows.append({'item': 'R2_kbin%d' % k, 'curve': 'GR', 'pair': '%s|%s' % (f1, f2),
                         'I1': res['I1'], 'I2': res['I2'], 'I12': res['I12'],
                         'UI1_wb': res['UI1_wb'], 'UI2_wb': res['UI2_wb'], 'S_wb': res['S_wb']})
        print('R2 k=%d done' % k)

    # ---- R3: class-balanced weighted MI ----
    for (f1, f2) in PAIRS:
        s1, _ = disc(fmap[f1]); s2, _ = disc(fmap[f2])
        m = (s1 >= 0) & (s2 >= 0)
        s1, s2, y6m = s1[m], s2[m], y6[m]
        cnt = np.bincount(y6m)
        w = (1.0 / cnt)[y6m]
        w *= len(w) / w.sum()
        I1 = mi_weighted(s1, y6m, w)
        I2 = mi_weighted(s2, y6m, w)
        I12 = mi_weighted(s1 * 4 + s2, y6m, w)
        rows.append({'item': 'R3_balancedMI', 'curve': 'GR', 'pair': '%s|%s' % (f1, f2),
                     'I1_bal': I1, 'I2_bal': I2, 'I12_bal': I12,
                     'UI1_mmi_bal': I12 - I2, 'UI2_mmi_bal': I12 - I1})
        print('R3 %s|%s done' % (f1, f2))

    # ---- R4: XGBoost increments (per-fold class remapping) ----
    sets = {
        'L': sum([['%s_%s' % (c, f) for c in ['GR', 'RHOB', 'DTC']]
                  for f in ['mean', 'var', 'lag1']], []),
        'F': ['%s_H' % c for c in ['GR', 'RHOB', 'DTC']],
        'C': ['%s_sampen' % c for c in ['GR', 'RHOB', 'DTC']],
    }
    sets['L+F'] = sets['L'] + sets['F']
    sets['L+C'] = sets['L'] + sets['C']
    sets['L+F+C'] = sets['L'] + sets['F'] + sets['C']
    dd = wd.dropna(subset=['grp6'])
    y = dd['grp6'].values
    wells = sorted(dd['well'].unique())
    g = dd['well'].values
    rng2 = np.random.default_rng(SEED)
    sizes = {w: int((g == w).sum()) for w in wells}
    fold_of, loads = {}, np.zeros(5, dtype=int)
    for idx in rng2.permutation(len(wells)):
        w = wells[int(idx)]
        k = int(np.argmin(loads))
        fold_of[w] = k
        loads[k] += sizes[w]
    for name, feats in sets.items():
        X = dd[feats].astype(float).values
        ok = ~np.isnan(X).any(axis=1)
        X, yy, gg = X[ok], y[ok], g[ok]
        ycodes, yuni = pd.factorize(yy)
        oof = np.full(len(X), -1)
        for k in range(5):
            fold_arr = np.array([fold_of[w] for w in gg])
            te, tr = fold_arr == k, fold_arr != k
            present = sorted(set(ycodes[tr]))
            cmap = {c: i for i, c in enumerate(present)}
            clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                tree_method='hist', n_jobs=-1, random_state=SEED)
            clf.fit(X[tr], np.vectorize(cmap.get)(ycodes[tr]))
            oof[te] = np.vectorize({i: c for c, i in cmap.items()}.__getitem__)(clf.predict(X[te]))
        f1m = f1_score(ycodes, oof, average='macro')
        rows.append({'item': 'R4_xgb_increment', 'curve': '-', 'pair': name, 'macro_f1': float(f1m)})
        print('R4 %-8s macroF1=%.3f' % (name, f1m))

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'e3_robustness.csv'), index=False)
    print('saved results/e3_robustness.csv (%d rows)' % len(rows))

if __name__ == '__main__':
    main()

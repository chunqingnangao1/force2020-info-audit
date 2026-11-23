# -*- coding: utf-8 -*-
"""Prediction-increment cross-check (logistic, leave-wells-out)."""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, '..', '..', 'results'))
SEED = 20231121
CURVES = ['GR', 'RHOB', 'DTC']

def main():
    wd = pd.read_csv(os.path.join(RESULTS, 'e3_windows.csv.gz'))
    wd = wd.dropna(subset=['grp6'])
    y = wd['grp6'].values
    wells = sorted(wd['well'].unique())
    g = wd['well'].values

    sets = {
        'L': sum([['%s_%s' % (c, f) for c in CURVES] for f in ['mean', 'var', 'lag1']], []),
        'C': ['%s_sampen' % c for c in CURVES],
        'F': ['%s_H' % c for c in CURVES],
    }
    sets['L+C'] = sets['L'] + sets['C']
    sets['L+F'] = sets['L'] + sets['F']
    sets['L+C+F'] = sets['L'] + sets['C'] + sets['F']
    sets['best_L(lag1_GR)'] = ['GR_lag1']
    sets['F(H_GR)'] = ['GR_H']
    sets['C(se_GR)'] = ['GR_sampen']

    rng = np.random.default_rng(SEED)
    sizes = {w: int((g == w).sum()) for w in wells}
    order = rng.permutation(len(wells))
    fold_of, loads = {}, np.zeros(5, dtype=int)
    for idx in order:
        w = wells[int(idx)]
        k = int(np.argmin(loads))
        fold_of[w] = k
        loads[k] += sizes[w]

    rows = []
    for name, feats in sets.items():
        X = wd[feats].astype(float).values
        ok = ~np.isnan(X).any(axis=1)
        X, yy, gg = X[ok], y[ok], g[ok]
        oof = np.full(len(X), None, dtype=object)
        fold_arr = np.array([fold_of[w] for w in gg])
        per_fold = []
        for k in range(5):
            te, tr = fold_arr == k, fold_arr != k
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            clf = LogisticRegression(max_iter=3000, C=1.0)
            clf.fit((X[tr] - mu) / sd, yy[tr])
            oof[te] = clf.predict((X[te] - mu) / sd)
            per_fold.append(f1_score(yy[te], oof[te], average='macro'))
        f1 = f1_score(yy, oof, average='macro')
        rows.append({'set': name, 'n_feat': len(feats), 'macro_f1': float(f1),
                     'fold_sd': float(np.std(per_fold)),
                     'folds': ','.join('%.3f' % v for v in per_fold)})
        print('%-16s macroF1=%.3f ± %.3f' % (name, f1, np.std(per_fold)))
    base = [r['macro_f1'] for r in rows if r['set'] == 'L'][0]
    for r in rows:
        r['delta_vs_L'] = r['macro_f1'] - base
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'e3_predict_increment.csv'), index=False)
    print('saved results/e3_predict_increment.csv')

if __name__ == '__main__':
    main()

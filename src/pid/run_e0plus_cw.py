# -*- coding: utf-8 -*-
"""E0-style full-feature baseline + nonlinear ablation + per-class F1
(with the feature-leak assertion and class-name alignment fixes)."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_pid import *  # noqa (also wraps stdout)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
RESULTS = os.path.join(ROOT, 'results')
SEED = 20301122
CURVES = ['GR', 'RHOB', 'DTC']
NBASE = ['GR', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'NPHI', 'DTC', 'DTS', 'PEF']

def engineer(df):
    out = df.copy()
    for c in NBASE:
        s = df[c] if c in df else pd.Series(np.nan, index=df.index)
        for w in (5, 11):
            out['%s_mean%d' % (c, w)] = s.rolling(w, center=True, min_periods=1).mean()
            out['%s_std%d' % (c, w)] = s.rolling(w, center=True, min_periods=1).std()
        out['%s_diff' % c] = s.diff()
    return out

def main():
    t0 = time.time()
    wd = pd.read_csv(os.path.join(RESULTS, 'e3_windows.csv.gz')).dropna(subset=['grp6'])
    raw = pd.read_csv(os.path.join(ROOT, 'data', 'clean', 'wells_raw.csv.gz'))
    raw_idx = {}
    for w, g in raw.groupby('WELL'):
        raw_idx[w] = (g['DEPTH'].values, g)
    rows = []
    for _, r in wd.iterrows():
        darr, gdf = raw_idx[r['well']]
        j = np.searchsorted(darr, r['depth'])
        j = int(np.clip(j, 1, len(darr) - 1))
        if abs(darr[j] - r['depth']) > 0.31:
            continue
        rec = {'WELL': r['well'], 'grp6': r['grp6'],
               'GR_H': r['GR_H'], 'GR_sampen': r['GR_sampen'],
               'RHOB_H': r['RHOB_H'], 'DTC_H': r['DTC_H']}
        for c in NBASE:
            rec[c] = gdf[c].iloc[j]
        rows.append(rec)
    anchor = pd.DataFrame(rows)
    parts = [engineer(g) for _, g in anchor.groupby('WELL')]
    anchor = pd.concat(parts, ignore_index=True)
    print('anchored windows:', len(anchor), ' (%.0fs)' % (time.time() - t0))

    e0_feats = [c for c in anchor.columns if c in NBASE or
                any(c == '%s_%s' % (b, s) for b in NBASE
                    for s in ['mean5', 'mean11', 'std5', 'std11', 'diff'])]
    nl_H = ['GR_H', 'RHOB_H', 'DTC_H']
    sets = {'E0': e0_feats,
            'E0+H': e0_feats + nl_H,
            'E0+H+SE': e0_feats + nl_H + ['GR_sampen']}
    assert not (set(nl_H) & set(e0_feats)), 'feature leak into E0 baseline'

    y = anchor['grp6'].values
    wells = sorted(anchor['WELL'].unique())
    g = anchor['WELL'].values
    rng = np.random.default_rng(SEED)
    sizes = {w: int((g == w).sum()) for w in wells}
    fold_of, loads = {}, np.zeros(5, dtype=int)
    for idx in rng.permutation(len(wells)):
        w = wells[int(idx)]
        k = int(np.argmin(loads))
        fold_of[w] = k
        loads[k] += sizes[w]
    out = []
    for name, feats in sets.items():
        X = anchor[feats].astype(float).values
        yc = pd.factorize(y)[0]
        oof = np.full(len(X), -1)
        per_fold = []
        for k in range(5):
            fa = np.array([fold_of[w] for w in g])
            te, tr = fa == k, fa != k
            present = sorted(set(yc[tr]))
            cmap = {c: i for i, c in enumerate(present)}
            inv = {i: c for c, i in cmap.items()}
            clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                tree_method='hist', n_jobs=-1, random_state=SEED)
            clf.fit(X[tr], np.vectorize(cmap.get)(yc[tr]))
            pred = np.vectorize(inv.__getitem__)(clf.predict(X[te]))
            oof[te] = pred
            per_fold.append(f1_score(yc[te], pred, average='macro', zero_division=0))
        out.append({'set': name, 'macro_f1': float(f1_score(yc, oof, average='macro', zero_division=0)),
                    'folds': str([round(v, 3) for v in per_fold]),
                    'fold_mean': float(np.mean(per_fold)), 'fold_sd': float(np.std(per_fold))})
        print('%-9s macroF1=%.3f folds=%s' % (name, out[-1]['macro_f1'], [round(v, 3) for v in per_fold]))
    pd.DataFrame(out).to_csv(os.path.join(RESULTS, 'e3_e0plus_nonlinear.csv'), index=False)

    # ---- per-class F1: L vs L+F (logistic, class-name-aligned codes) ----
    L = sum([['%s_%s' % (c, f)] for c in CURVES for f in ['mean', 'var', 'lag1']], [])
    LF = L + ['%s_H' % c for c in CURVES]
    class_names = sorted(wd['grp6'].unique())
    preds = {}
    rows2 = []
    for name, feats in [('L', L), ('L+F', LF)]:
        X = wd[feats].astype(float).values
        ok = ~np.isnan(X).any(axis=1)
        X, yy, gg = X[ok], y[ok], g[ok]
        yc2 = pd.Categorical(yy, categories=class_names).codes
        oof = np.full(len(X), -1)
        for k in range(5):
            fa = np.array([fold_of[w] for w in gg])
            te, tr = fa == k, fa != k
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            clf = LogisticRegression(max_iter=3000, C=1.0)
            clf.fit((X[tr] - mu) / sd, yc2[tr])
            oof[te] = clf.predict((X[te] - mu) / sd)
        preds[name] = (yc2, oof)
    for cls_i, cls in enumerate(class_names):
        f1_L = f1_score(preds['L'][0] == cls_i, preds['L'][1] == cls_i, zero_division=0)
        f1_LF = f1_score(preds['L+F'][0] == cls_i, preds['L+F'][1] == cls_i, zero_division=0)
        n = int((preds['L'][0] == cls_i).sum())
        rows2.append({'class': cls, 'n': n, 'f1_L': f1_L, 'f1_LF': f1_LF, 'delta': f1_LF - f1_L})
        print('%-11s n=%4d  F1_L=%.3f  F1_LF=%.3f  D=%+.3f' % (cls, n, f1_L, f1_LF, f1_LF - f1_L))
    pd.DataFrame(rows2).to_csv(os.path.join(RESULTS, 'e3_perclass_increment.csv'), index=False)
    print('saved e3_e0plus_nonlinear.csv + e3_perclass_increment.csv  %.0fs' % (time.time() - t0))

if __name__ == '__main__':
    main()

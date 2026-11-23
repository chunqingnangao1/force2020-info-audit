# -*- coding: utf-8 -*-
"""E0 baseline training: leave-wells-out 5-fold XGBoost with per-fold class remapping."""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CLEAN = os.path.join(ROOT, 'data', 'clean')
RESULTS = os.path.join(ROOT, 'results')
os.makedirs(RESULTS, exist_ok=True)

LITHOLOGY_KEYS = {
    30000: 'sandstone', 65030: 'sandstone-shale interl.', 65000: 'shale',
    80000: 'marl', 74000: 'dolomite', 70000: 'limestone', 70032: 'chalk',
    88000: 'halite', 86000: 'anhydrite', 99000: 'tuff', 90000: 'coal',
    93000: 'bituminous shale',
}
CLASSES = sorted(LITHOLOGY_KEYS)
NBASE = ['GR', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'NPHI', 'DTC', 'DTS', 'PEF']
SEED = 2020

def engineer(g):
    out = g.copy()
    for c in NBASE:
        s = g[c] if c in g else pd.Series(np.nan, index=g.index)
        for w in (5, 11):
            out['%s_mean%d' % (c, w)] = s.rolling(w, center=True, min_periods=1).mean()
            out['%s_std%d' % (c, w)] = s.rolling(w, center=True, min_periods=1).std()
        out['%s_diff' % c] = s.diff()
    return out

def main():
    t0 = time.time()
    df = pd.read_csv(os.path.join(CLEAN, 'wells_raw.csv.gz'))
    df = df.dropna(subset=['LITH_CODE']).copy()
    df['Y'] = df['LITH_CODE'].astype(int)
    wells = sorted(df['WELL'].unique())
    print('wells:', len(wells), 'rows:', len(df))
    parts = [engineer(df[df['WELL'] == w]) for w in wells]
    df = pd.concat(parts, ignore_index=True)
    feats = [c for c in df.columns if c not in ('DEPTH', 'WELL', 'LITH_CODE', 'Y',
                                                'GROUP', 'FORMATION', 'LITH_CONF')]
    X = df[feats].astype(float).values
    y = df['Y'].values
    g = df['WELL'].values
    rng = np.random.default_rng(SEED)
    sizes = {w: int((g == w).sum()) for w in wells}
    fold_of, loads = {}, np.zeros(5, dtype=int)
    for idx in rng.permutation(len(wells)):
        w = wells[int(idx)]
        k = int(np.argmin(loads))
        fold_of[w] = k
        loads[k] += sizes[w]
    oof_pred = np.full(len(df), -1)
    for k in range(5):
        te_wells = [w for w in wells if fold_of[w] == k]
        te = np.isin(g, te_wells)
        tr = ~te
        present = [c for c in CLASSES if np.any(y[tr] == c)]
        cmap = {c: i for i, c in enumerate(present)}
        inv = {i: c for c, i in cmap.items()}
        clf = XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8,
                            tree_method='hist', n_jobs=-1, random_state=SEED,
                            eval_metric='mlogloss')
        clf.fit(X[tr], np.vectorize(cmap.get)(y[tr]))
        oof_pred[te] = np.vectorize(inv.__getitem__)(clf.predict(X[te]))
        print('fold %d done (%d wells, %d rows, %d classes) %.0fs' %
              (k, len(te_wells), int(te.sum()), len(present), time.time() - t0))
    macro = f1_score(y, oof_pred, labels=CLASSES, average='macro', zero_division=0)
    print('\n=== OOF macro-F1: %.4f ===' % macro)
    rep = classification_report(y, oof_pred, labels=CLASSES,
                                target_names=[LITHOLOGY_KEYS[c] for c in CLASSES],
                                zero_division=0, output_dict=True)
    cm = confusion_matrix(y, oof_pred, labels=CLASSES)
    pd.DataFrame(cm, index=[LITHOLOGY_KEYS[c] for c in CLASSES],
                 columns=[LITHOLOGY_KEYS[c] for c in CLASSES]).to_csv(
        os.path.join(RESULTS, 'e0_confusion_matrix.csv'))
    with open(os.path.join(RESULTS, 'e0_metrics.json'), 'w', encoding='utf-8') as fh:
        import json
        json.dump({'macro_f1': float(macro), 'per_class': rep,
                   'n_wells': len(wells), 'n_rows': int(len(df))}, fh, ensure_ascii=False, indent=1)
    print('saved results/e0_metrics.json, e0_confusion_matrix.csv')

if __name__ == '__main__':
    main()

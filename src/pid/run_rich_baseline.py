# -*- coding: utf-8 -*-
"""Review-response R4: same-window-scale rich linear baseline + class-balanced prediction.
Outputs: results/e4_rich_windows.csv.gz, e4_same_scale.csv, e4_balanced_pred.csv"""
import sys, io, os, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
RESULTS = os.path.join(ROOT, 'results')
CLEAN = os.path.join(ROOT, 'data', 'clean')
SEED = 20411123
CURVES = ['GR', 'RHOB', 'DTC']
N_W, STRIDE = 512, 256
PURE, LAB_MIN = 0.6, 0.5
GROUP6 = {30000: 'sand', 65030: 'interl', 65000: 'shale', 93000: 'shale',
          70000: 'carb', 70032: 'carb', 80000: 'marl_dolo', 74000: 'marl_dolo',
          88000: 'evap_other', 86000: 'evap_other', 99000: 'evap_other', 90000: 'evap_other'}

def rich_stats(w):
    ok = w[~np.isnan(w)]
    if len(ok) < 0.8 * len(w) or len(ok) < 100:
        return None
    out = {'mean': float(ok.mean()), 'var': float(ok.var())}
    for L in range(1, 6):
        out['lag%d' % L] = float(np.corrcoef(ok[:-L], ok[L:])[0, 1]) if np.std(ok) > 0 else np.nan
    out['skew'] = float(sps.skew(ok))
    out['kurt'] = float(sps.kurtosis(ok))
    return out

def main():
    t0 = time.time()
    raw = pd.read_csv(os.path.join(CLEAN, 'wells_raw.csv.gz'))
    raw = raw[raw['FORMATION'].notna()]
    win = pd.read_csv(os.path.join(RESULTS, 'e3_windows.csv.gz'))
    # window centres and nonlinear features per well (sorted by depth) for the join
    nl = win[['well', 'depth', 'GR_H', 'RHOB_H', 'DTC_H', 'GR_sampen', 'grp6']].copy()
    nl_by_well = {w: g.sort_values('depth') for w, g in nl.groupby('well')}

    recs = []
    for wname, g in raw.groupby('WELL'):
        if wname not in nl_by_well:
            continue
        g = g.sort_values('DEPTH')
        nlg = nl_by_well[wname]
        ndepth = nlg['depth'].values
        for form, seg in g.groupby('FORMATION', sort=False):
            n = len(seg)
            if n < N_W:
                continue
            lith = seg['LITH_CODE'].values.astype(float)
            depth = seg['DEPTH'].values.astype(float)
            vals = {c: seg[c].values.astype(float) for c in CURVES}
            for i in range(0, n - N_W + 1, STRIDE):
                lab = lith[i:i + N_W]
                lab_ok = lab[~np.isnan(lab)]
                if len(lab_ok) < LAB_MIN * N_W:
                    continue
                codes, cnts = np.unique(lab_ok.astype(int), return_counts=True)
                code = int(codes[np.argmax(cnts)])
                if cnts.max() / len(lab_ok) < PURE:
                    continue
                center = float(depth[i + N_W // 2])
                j = np.searchsorted(ndepth, center)
                j = int(np.clip(j, 1, len(ndepth) - 1))
                if abs(ndepth[j] - center) > 0.35 or abs(ndepth[j - 1] - center) < abs(ndepth[j] - center):
                    j = j - 1
                if abs(ndepth[j] - center) > 0.35:
                    continue
                rec = {'well': wname, 'grp6': nlg['grp6'].iloc[j],
                       'GR_H': nlg['GR_H'].iloc[j], 'RHOB_H': nlg['RHOB_H'].iloc[j],
                       'DTC_H': nlg['DTC_H'].iloc[j], 'GR_sampen': nlg['GR_sampen'].iloc[j]}
                okall = True
                for c in CURVES:
                    st = rich_stats(vals[c][i:i + N_W])
                    if st is None:
                        okall = False
                        break
                    for k, v in st.items():
                        rec['%s_%s' % (c, k)] = v
                if okall:
                    recs.append(rec)
    wd = pd.DataFrame(recs).dropna()
    print('rich windows:', len(wd), ' (%.0fs)' % (time.time() - t0))
    wd.to_csv(os.path.join(RESULTS, 'e4_rich_windows.csv.gz'), index=False)

    y = wd['grp6'].values
    wells = sorted(wd['well'].unique())
    g = wd['well'].values
    rng = np.random.default_rng(SEED)
    sizes = {w: int((g == w).sum()) for w in wells}
    fold_of, loads = {}, np.zeros(5, dtype=int)
    for idx in rng.permutation(len(wells)):
        w = wells[int(idx)]
        k = int(np.argmin(loads))
        fold_of[w] = k
        loads[k] += sizes[w]

    rich = ['%s_%s' % (c, k) for c in CURVES
            for k in ['mean', 'var', 'lag1', 'lag2', 'lag3', 'lag4', 'lag5', 'skew', 'kurt']]
    sets = {'L-rich(27)': rich,
            'L-rich+H': rich + ['GR_H', 'RHOB_H', 'DTC_H'],
            'L-rich+H+SE': rich + ['GR_H', 'RHOB_H', 'DTC_H', 'GR_sampen'],
            'L(9, main)': sum([['%s_%s' % (c, f) for c in CURVES]
                              for f in ['mean', 'var', 'lag1']], [])}

    out = []
    yc, yuni = pd.factorize(y)
    for name, feats in sets.items():
        X = wd[feats].astype(float).values
        oof = np.full(len(X), -1)
        per_fold = []
        for model in ['logistic', 'xgb']:
            for k in range(5):
                fa = np.array([fold_of[w] for w in g])
                te, tr = fa == k, fa != k
                present = sorted(set(yc[tr]))
                cmap = {c: i for i, c in enumerate(present)}
                inv = {i: c for c, i in cmap.items()}
                if model == 'logistic':
                    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
                    clf = LogisticRegression(max_iter=3000, C=1.0)
                    clf.fit((X[tr] - mu) / sd, np.vectorize(cmap.get)(yc[tr]))
                    pred = clf.predict((X[te] - mu) / sd)
                else:
                    clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                        tree_method='hist', n_jobs=-1, random_state=SEED)
                    clf.fit(X[tr], np.vectorize(cmap.get)(yc[tr]))
                    pred = clf.predict(X[te])
                oof[te] = np.vectorize(inv.__getitem__)(pred) if model == 'xgb' else \
                    np.vectorize(inv.__getitem__)(pred)
            per_fold.append(f1_score(yc, oof, average='macro', zero_division=0))
            if model == 'xgb':
                pass
        # record per model
        for model in ['logistic', 'xgb']:
            oof = np.full(len(X), -1)
            for k in range(5):
                fa = np.array([fold_of[w] for w in g])
                te, tr = fa == k, fa != k
                present = sorted(set(yc[tr]))
                cmap = {c: i for i, c in enumerate(present)}
                inv = {i: c for c, i in cmap.items()}
                if model == 'logistic':
                    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
                    clf = LogisticRegression(max_iter=3000, C=1.0)
                    clf.fit((X[tr] - mu) / sd, np.vectorize(cmap.get)(yc[tr]))
                    pred = clf.predict((X[te] - mu) / sd)
                else:
                    clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                        tree_method='hist', n_jobs=-1, random_state=SEED)
                    clf.fit(X[tr], np.vectorize(cmap.get)(yc[tr]))
                    pred = clf.predict(X[te])
                oof[te] = np.vectorize(inv.__getitem__)(pred)
            f1m = f1_score(yc, oof, average='macro', zero_division=0)
            out.append({'set': name, 'model': model, 'n_feat': len(feats), 'macro_f1': float(f1m)})
            print('%-12s %-9s macroF1=%.3f' % (name, model, f1m))
    pd.DataFrame(out).to_csv(os.path.join(RESULTS, 'e4_same_scale.csv'), index=False)

    # ---- B: class-balanced XGBoost (L vs L+F), per-class F1 ----
    L = sum([['%s_%s' % (c, f) for c in CURVES] for f in ['mean', 'var', 'lag1']], [])
    LF = L + ['%s_H' % c for c in CURVES]
    sw = compute_sample_weight('balanced', yc)
    rows = []
    for name, feats in [('L_bal', L), ('L+F_bal', LF)]:
        X = wd[feats].astype(float).values
        oof = np.full(len(X), -1)
        for k in range(5):
            fa = np.array([fold_of[w] for w in g])
            te, tr = fa == k, fa != k
            present = sorted(set(yc[tr]))
            cmap = {c: i for i, c in enumerate(present)}
            inv = {i: c for c, i in cmap.items()}
            clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                                tree_method='hist', n_jobs=-1, random_state=SEED)
            clf.fit(X[tr], np.vectorize(cmap.get)(yc[tr]), sample_weight=sw[tr])
            oof[te] = np.vectorize(inv.__getitem__)(clf.predict(X[te]))
        f1m = f1_score(yc, oof, average='macro', zero_division=0)
        print('%-9s balanced-XGB macroF1=%.3f' % (name, f1m))
        for ci, cn in enumerate(yuni):
            f1c = f1_score(yc == ci, oof == ci, zero_division=0)
            rows.append({'set': name, 'class': cn, 'n': int((yc == ci).sum()), 'f1': float(f1c)})
            print('    %-11s n=%4d F1=%.3f' % (cn, (yc == ci).sum(), f1c))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'e4_balanced_pred.csv'), index=False)
    print('saved e4_rich/same_scale/balanced  %.0fs' % (time.time() - t0))

if __name__ == '__main__':
    main()

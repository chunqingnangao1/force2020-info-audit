# -*- coding: utf-8 -*-
"""n=256 mid-scale window sensitivity: extraction + PID (GR pairs) + increment L vs L+F."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_pid import disc, pid_two_sources, block_permute_labels, PAIRS  # noqa (wraps stdout)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
RESULTS = os.path.join(ROOT, 'results')
CLEAN = os.path.join(ROOT, 'data', 'clean')
SEED = 20321123
N_W, STRIDE = 256, 128

GROUP6 = {30000: 'sand', 65030: 'interl', 65000: 'shale', 93000: 'shale',
          70000: 'carb', 70032: 'carb', 80000: 'marl_dolo', 74000: 'marl_dolo',
          88000: 'evap_other', 86000: 'evap_other', 99000: 'evap_other', 90000: 'evap_other'}

def feat_one(w):
    ok = w[~np.isnan(w)]
    if len(ok) < 0.8 * len(w) or len(ok) < 100:
        return (np.nan,) * 5
    mean, var = float(ok.mean()), float(ok.var())
    c1 = float(np.corrcoef(ok[:-1], ok[1:])[0, 1]) if np.std(ok) > 0 else np.nan
    try:
        import nolds
        H = float(nolds.dfa(ok, order=1)) - 1.0
        se = float(nolds.sampen(ok, emb_dim=2, tolerance=0.2 * float(np.std(ok))))
    except Exception:
        H, se = np.nan, np.nan
    return mean, var, c1, H, se

def pid_pairs(sub, n_perm=1000):
    rng = np.random.default_rng(SEED)
    yy = pd.factorize(sub['grp6'])[0]
    segv = sub['seg'].values
    rows = []
    for (f1, f2) in PAIRS:
        s1, _ = disc(sub[f1].values)
        s2, _ = disc(sub[f2].values)
        m = (s1 >= 0) & (s2 >= 0)
        s1m, s2m, ym, sgm = s1[m], s2[m], yy[m], segv[m]
        if len(ym) < 200:
            continue
        groups = [np.where(sgm == s)[0] for s in np.unique(sgm)]
        res = pid_two_sources(s1m, s2m, ym)
        arr = []
        for _ in range(n_perm):
            r = pid_two_sources(s1m, s2m, block_permute_labels(ym, groups, rng))
            arr.append(r['UI2_wb'])
        arr = np.asarray(arr)
        rows.append({'pair': '%s|%s' % (f1, f2), 'n_win': int(m.sum()),
                     'UI2_wb': res['UI2_wb'], 'UI1_wb': res['UI1_wb'], 'R_mmi': res['R_mmi'],
                     'I12': res['I12'],
                     'p_UI2_wb': float((np.sum(arr >= res['UI2_wb']) + 1) / (n_perm + 1))})
        print('[n256] %s|%s n=%d UI2=%.4f p=%.4f' % (f1, f2, m.sum(), res['UI2_wb'], rows[-1]['p_UI2_wb']))
    return rows

def increment(sub):
    y = sub['grp6'].values
    wells = sorted(sub['well'].unique())
    g = sub['well'].values
    rng = np.random.default_rng(SEED)
    sizes = {w: int((g == w).sum()) for w in wells}
    fold_of, loads = {}, np.zeros(5, dtype=int)
    for idx in rng.permutation(len(wells)):
        w = wells[int(idx)]
        k = int(np.argmin(loads))
        fold_of[w] = k
        loads[k] += sizes[w]
    L = ['mean', 'var', 'lag1']
    LF = L + ['H']
    out = []
    for name, feats in [('L(GR)', L), ('L+F(GR)', LF)]:
        X = sub[feats].astype(float).values
        ok = ~np.isnan(X).any(axis=1)
        X, yy, gg = X[ok], pd.factorize(y[ok])[0], g[ok]
        oof = np.full(len(X), -1)
        for k in range(5):
            fa = np.array([fold_of[w] for w in gg])
            te, tr = fa == k, fa != k
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            clf = LogisticRegression(max_iter=3000, C=1.0)
            clf.fit((X[tr] - mu) / sd, yy[tr])
            oof[te] = clf.predict((X[te] - mu) / sd)
        f1m = f1_score(yy, oof, average='macro', zero_division=0)
        out.append({'set': name, 'macro_f1': float(f1m)})
        print('[n256-inc] %-8s macroF1=%.3f' % (name, f1m))
    return out

def main():
    t0 = time.time()
    cache = os.path.join(RESULTS, 'e3_windows_n256.csv.gz')
    if os.path.exists(cache):
        wd = pd.read_csv(cache)
        print('loaded cached n256 windows:', len(wd))
    else:
        raw = pd.read_csv(os.path.join(CLEAN, 'wells_raw.csv.gz'))
        raw = raw[raw['FORMATION'].notna()]
        out = []
        for wname, g in raw.groupby('WELL'):
            g = g.sort_values('DEPTH')
            for form, seg in g.groupby('FORMATION', sort=False):
                n = len(seg)
                if n < N_W:
                    continue
                vals = seg['GR'].values.astype(float)
                lith = seg['LITH_CODE'].values.astype(float)
                depth = seg['DEPTH'].values.astype(float)
                for i in range(0, n - N_W + 1, STRIDE):
                    lab = lith[i:i + N_W]
                    lab_ok = lab[~np.isnan(lab)]
                    if len(lab_ok) < 0.5 * N_W:
                        continue
                    codes, cnts = np.unique(lab_ok.astype(int), return_counts=True)
                    code = int(codes[np.argmax(cnts)])
                    if cnts.max() / len(lab_ok) < 0.6:
                        continue
                    m, v, l1, H, se = feat_one(vals[i:i + N_W])
                    out.append({'well': wname, 'formation': form, 'grp6': GROUP6[code],
                                'depth': float(depth[i + N_W // 2]),
                                'mean': m, 'var': v, 'lag1': l1, 'H': H, 'sampen': se})
        wd = pd.DataFrame(out)
        print('n256 windows:', len(wd), ' wells:', wd['well'].nunique(), ' (%.0fs)' % (time.time() - t0))
    wd['seg'] = wd['well'] + '|' + wd['formation'].astype(str)
    print('n256 windows:', len(wd), ' wells:', wd['well'].nunique(), ' (%.0fs)' % (time.time() - t0))
    wd.to_csv(os.path.join(RESULTS, 'e3_windows_n256.csv.gz'), index=False)

    rows = pid_pairs(wd)
    rows = [{'kind': 'pid'} | r for r in rows]
    rows += [{'kind': 'increment'} | r for r in increment(wd)]
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'e3_n256_all.csv'), index=False)
    print('saved e3_windows_n256.csv.gz + e3_n256_all.csv  %.0fs' % (time.time() - t0))

if __name__ == '__main__':
    main()

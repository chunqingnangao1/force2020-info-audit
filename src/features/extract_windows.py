# -*- coding: utf-8 -*-
"""Per-well sliding-window feature extraction (nolds.sampen uses the `tolerance` kwarg)."""
import sys, io, os, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
CLEAN = os.path.join(ROOT, 'data', 'clean')
RESULTS = os.path.join(ROOT, 'results')

N_W, STRIDE = 512, 256
CURVES = ['GR', 'RHOB', 'DTC']
PURE, LAB_MIN = 0.6, 0.5
NAN_MAX = 0.2
GROUP6 = {30000: 'sand', 65030: 'interl', 65000: 'shale', 93000: 'shale',
          70000: 'carb', 70032: 'carb', 80000: 'marl_dolo', 74000: 'marl_dolo',
          88000: 'evap_other', 86000: 'evap_other', 99000: 'evap_other', 90000: 'evap_other'}

def feat_one(w):
    ok = w[~np.isnan(w)]
    if len(ok) < (1 - NAN_MAX) * len(w) or len(ok) < 100:
        return (np.nan,) * 5
    mean, var = float(ok.mean()), float(ok.var())
    c1 = float(np.corrcoef(ok[:-1], ok[1:])[0, 1]) if len(ok) > 2 and np.std(ok) > 0 else np.nan
    try:
        import nolds
        H = float(nolds.dfa(ok, order=1)) - 1.0
    except Exception:
        H = np.nan
    try:
        import nolds
        se = float(nolds.sampen(ok, emb_dim=2, tolerance=0.2 * float(np.std(ok))))
    except Exception:
        se = np.nan
    return mean, var, c1, H, se

def process_well(args):
    wname, g = args
    out = []
    g = g.sort_values('DEPTH')
    for form, seg in g[g['FORMATION'].notna()].groupby('FORMATION', sort=False):
        vals = {c: seg[c].values.astype(float) for c in CURVES}
        lith = seg['LITH_CODE'].values.astype(float)
        n = len(seg)
        for i in range(0, n - N_W + 1, STRIDE):
            lab = lith[i:i + N_W]
            lab_ok = lab[~np.isnan(lab)]
            if len(lab_ok) < LAB_MIN * N_W:
                continue
            codes, cnts = np.unique(lab_ok.astype(int), return_counts=True)
            code = int(codes[np.argmax(cnts)])
            frac = float(cnts.max()) / len(lab_ok)
            if frac < PURE:
                continue
            rec = {'well': wname, 'formation': form,
                   'depth': float(seg['DEPTH'].values[i + N_W // 2]),
                   'code': code, 'grp6': GROUP6[code], 'frac': frac}
            for c in CURVES:
                m, v, l1, H, se = feat_one(vals[c][i:i + N_W])
                rec['%s_mean' % c] = m
                rec['%s_var' % c] = v
                rec['%s_lag1' % c] = l1
                rec['%s_H' % c] = H
                rec['%s_sampen' % c] = se
            out.append(rec)
    return out

def main():
    from multiprocessing import Pool
    t0 = time.time()
    df = pd.read_csv(os.path.join(CLEAN, 'wells_raw.csv.gz'))
    df = df[df['FORMATION'].notna()]
    wells = sorted(df['WELL'].unique())
    print('wells:', len(wells), 'rows in-segment:', len(df))
    jobs = [(w, df[df['WELL'] == w]) for w in wells]
    all_rows = []
    with Pool(processes=6) as pool:
        for k, rows in enumerate(pool.imap_unordered(process_well, jobs)):
            all_rows.extend(rows)
            if (k + 1) % 20 == 0:
                print('wells %d/%d (%.0fs) windows %d' % (k + 1, len(wells), time.time() - t0, len(all_rows)))
    wd = pd.DataFrame(all_rows)
    wd.to_csv(os.path.join(RESULTS, 'e3_windows.csv.gz'), index=False)
    print('windows total:', len(wd))
    print(wd['grp6'].value_counts().to_string())
    for c in CURVES:
        print('%s H valid %.3f | sampen valid %.3f' % (c, wd['%s_H' % c].notna().mean(), wd['%s_sampen' % c].notna().mean()))

if __name__ == '__main__':
    main()

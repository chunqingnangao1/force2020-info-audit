# -*- coding: utf-8 -*-
"""E2 synthetic-stratigraphy identifiability (equal-mean, equal-variance,
different-correlation intra-bed texture)."""
import sys, io, os, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import mannwhitneyu
import nolds

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, '..', '..', 'results'))

STEP_M = 0.152
N_COL = 8192
N_W = 512
STRIDE = 256
PURE = 0.6
RUNS = 4
SEED = 20221121
MID_GR = 75.0
SIG_TEX = 10.0
H_TEX = 0.8

STRUCTURES = [('markov', 0.5), ('markov', 0.8), ('markov', 0.95),
              ('fbm', 0.3), ('fbm', 0.5), ('fbm', 0.7)]
THICK_MODES = ['logn', 'pow1.5']
CONTRASTS = [0.0, 120.0]
SMOOTH_M = [0.0, 0.3]
SNR_GRID = [20, 10**9]

def fbm(H, n, rng):
    m = 1 << (int(np.ceil(np.log2(2 * n))) + 2)
    f = np.fft.rfft(rng.standard_normal(m))
    freq = np.fft.rfftfreq(m)
    freq[0] = freq[1]
    f = f / (freq ** (H + 0.5))
    s = np.fft.irfft(f, m)[:n]
    s -= s.mean()
    return s / s.std() if s.std() > 0 else s

def sample_thickness(mode, rng):
    out, total = [], 0
    while total < N_COL:
        if mode == 'logn':
            t = rng.lognormal(mean=np.log(8.0), sigma=0.8)
        else:
            t = 2.0 * rng.pareto(1.5) + 2.0
        out.append(int(max(2, round(t))))
        total += out[-1]
    return out

def facies_sequence(path, param, n_beds, rng):
    if path == 'markov':
        s, cur = [], (0 if rng.random() < 0.5 else 1)
        for _ in range(n_beds):
            s.append(cur)
            if rng.random() > param:
                cur = 1 - cur
        return np.array(s)
    y = fbm(param, n_beds, rng)
    return (y > np.median(y)).astype(int)

def build_column(path, param, tmode, contrast, rng):
    th = sample_thickness(tmode, rng)
    fac_beds = facies_sequence(path, param, len(th), rng)
    fac = np.repeat(fac_beds, th)[:N_COL]
    tex_white = rng.standard_normal(N_COL)
    tex_fbm = fbm(H_TEX, N_COL, rng)
    tex = np.where(fac == 1, tex_fbm * SIG_TEX, tex_white * SIG_TEX)
    bed_val = MID_GR + fac * (contrast / 120.0) * 30.0
    return bed_val + tex, fac

def forward(x, smooth_m, snr_db, rng):
    if smooth_m > 0:
        x = gaussian_filter1d(x, sigma=smooth_m / STEP_M)
    if snr_db < 10**8:
        x = x + rng.standard_normal(len(x)) * np.sqrt(np.var(x) / (10 ** (snr_db / 10)))
    return x

def window_table(x, fac):
    rows = []
    for i in range(0, len(x) - N_W + 1, STRIDE):
        w = x[i:i + N_W]
        c1 = np.corrcoef(w[:-1], w[1:])[0, 1] if w.std() > 0 else np.nan
        try:
            H = float(nolds.dfa(w, order=1)) - 1.0
        except Exception:
            H = np.nan
        try:
            se = float(nolds.sampen(w, emb_dim=2, tolerance=0.2 * float(w.std())))
        except Exception:
            se = np.nan
        rows.append({'mean': float(w.mean()), 'var': float(w.var()),
                     'lag1': float(c1) if np.isfinite(c1) else np.nan,
                     'H': H, 'sampen': se, 'frac': float(fac[i:i + N_W].mean())})
    return pd.DataFrame(rows)

def auc(pos, neg, vals):
    p, n = vals[pos], vals[neg]
    p, n = p[~np.isnan(p)], n[~np.isnan(n)]
    if len(p) < 5 or len(n) < 5:
        return np.nan
    return float(mannwhitneyu(p, n, alternative='two-sided').statistic / (len(p) * len(n)))

def auc_matched_lag1(dfw):
    d = dfw.dropna(subset=['lag1', 'H']).copy()
    labs = (d['frac'] > PURE).astype(int) - (d['frac'] < 1 - PURE).astype(int)
    d = d[labs != 0]
    d['lab'] = (labs > 0).astype(int)
    if d['lab'].nunique() < 2 or len(d) < 20:
        return np.nan
    d['bin'] = pd.qcut(d['lag1'], 4, labels=False, duplicates='drop')
    stats = []
    for b, g in d.groupby('bin'):
        pos, neg = g[g['lab'] == 1]['H'], g[g['lab'] == 0]['H']
        if len(pos) >= 3 and len(neg) >= 3:
            stats.append(mannwhitneyu(pos, neg, alternative='two-sided').statistic / (len(pos) * len(neg)))
    return float(np.mean(stats)) if stats else np.nan

def main():
    rng_master = np.random.default_rng(SEED)
    t0 = time.time()
    rows = []
    total = len(STRUCTURES) * len(THICK_MODES) * len(CONTRASTS) * len(SMOOTH_M) * len(SNR_GRID)
    done = 0
    for path, param in STRUCTURES:
        for tmode in THICK_MODES:
            for contrast in CONTRASTS:
                for sm in SMOOTH_M:
                    for snr in SNR_GRID:
                        accs = {}
                        for _ in range(RUNS):
                            rng = np.random.default_rng(rng_master.integers(0, 2**63 - 1))
                            x0, fac = build_column(path, param, tmode, contrast, rng)
                            x = forward(x0, sm, snr, rng)
                            d = window_table(x, fac)
                            pure = (d['frac'] > PURE) | (d['frac'] < 1 - PURE)
                            dp = d[pure]
                            if len(dp) >= 10 and ((dp['frac'] > PURE).sum() >= 3) and ((dp['frac'] < 1 - PURE).sum() >= 3):
                                pos, neg = dp['frac'] > PURE, dp['frac'] < 1 - PURE
                                for f in ['mean', 'var', 'lag1', 'H', 'sampen']:
                                    accs.setdefault('auc_' + f, []).append(auc(pos, neg, dp[f].values))
                            accs.setdefault('H_win_mean', []).append(float(np.nanmean(d['H'])))
                            for f in ['var', 'lag1', 'H']:
                                sub = d[['frac', f]].dropna()
                                accs.setdefault('corr_frac_' + f, []).append(
                                    float(sub.corr().iloc[0, 1]) if len(sub) > 10 else np.nan)
                            accs.setdefault('auc_H_mlag1', []).append(auc_matched_lag1(d))
                        row = {'path': path, 'param': param, 'tmode': tmode,
                               'contrast': contrast, 'smooth_m': sm, 'snr_db': snr}
                        for k, v in accs.items():
                            row[k] = float(np.nanmean(v)) if len(v) else np.nan
                        row['n_pure_runs'] = int(len(accs.get('auc_H', [])))
                        rows.append(row)
                        done += 1
                        if done % 24 == 0:
                            pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'e2v2_identifiability.csv'), index=False)
                            print('grid %d/%d (%.0fs)' % (done, total, time.time() - t0))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'e2v2_identifiability.csv'), index=False)
    print('saved e2v2_identifiability.csv (%d rows) %.0fs' % (len(rows), time.time() - t0))

if __name__ == '__main__':
    main()

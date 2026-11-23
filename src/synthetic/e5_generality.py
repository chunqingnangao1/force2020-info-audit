# -*- coding: utf-8 -*-
"""E5 geological generality map: sweep the geological parameter space and compute the
nonlinear information share per cell (bed-thickness median {4,8,16} samples x facies
contrast {0,30,60} gAPI x sequence persistence H {0.3,0.5,0.7}).
Metric: UI_mmi(H beyond (mean,lag1)) / I(Y; mean,lag1,H).
Output: results/e5_generality.csv"""
import sys, io, os, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import nolds

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
RESULTS = os.path.join(ROOT, 'results')

N_COL = 8192
N_W = 512
STRIDE = 256
SMOOTH_M = 0.15
RUNS = 4
SEED = 20511123
MID_GR, SIG_TEX, H_TEX = 75.0, 10.0, 0.8

def fbm(H, n, rng):
    m = 1 << (int(np.ceil(np.log2(2 * n))) + 2)
    f = np.fft.rfft(rng.standard_normal(m))
    freq = np.fft.rfftfreq(m)
    freq[0] = freq[1]
    f = f / (freq ** (H + 0.5))
    s = np.fft.irfft(f, m)[:n]
    s -= s.mean()
    return s / s.std() if s.std() > 0 else s

def gen_column(H_seq, t_med, contrast, rng):
    th, total = [], 0
    while total < N_COL:
        t = int(max(2, round(rng.lognormal(mean=np.log(t_med), sigma=0.5))))
        th.append(t)
        total += t
    y = fbm(H_seq, len(th), rng)
    fac_beds = (y > np.median(y)).astype(int)
    fac = np.repeat(fac_beds, th)[:N_COL]
    tex = np.where(fac == 1, fbm(H_TEX, N_COL, rng) * SIG_TEX, rng.standard_normal(N_COL) * SIG_TEX)
    bed_val = MID_GR + fac * contrast
    return gaussian_filter1d(bed_val + tex, sigma=SMOOTH_M / 0.152), fac

def win_feats(x, fac):
    rows = []
    for i in range(0, len(x) - N_W + 1, STRIDE):
        w = x[i:i + N_W]
        frac = fac[i:i + N_W].mean()
        if frac > 0.7 or frac < 0.3:
            rows.append({'mean': w.mean(), 'lag1': np.corrcoef(w[:-1], w[1:])[0, 1],
                         'H': nolds.dfa(w, order=1) - 1.0,
                         'y': int(frac > 0.5)})
    return pd.DataFrame(rows)

def mi_bins(*arrs):
    """Joint MI (bits) after 4-quantile binning of each continuous input."""
    cols = [pd.qcut(pd.Series(a).rank(method='first'), 4, labels=False).values for a in arrs[:-1]]
    y = arrs[-1].astype(int)
    code = np.zeros(len(y), dtype=int)
    for c in cols:
        code = code * 4 + c
    return mi_disc(code, y)

def mi_disc(x, y):
    nx = x.max() + 1
    ny = y.max() + 1
    joint = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny).astype(float)
    n = joint.sum()
    if n == 0:
        return 0.0
    px = joint.sum(1, keepdims=True) / n
    py = joint.sum(0, keepdims=True) / n
    pij = joint / n
    mask = pij > 0
    return max(0.0, float(np.sum(pij[mask] * np.log2(pij[mask] / (px @ py)[mask]))))

def main():
    rng_master = np.random.default_rng(SEED)
    rows = []
    t0 = time.time()
    grid = [(t, c, h) for t in [4, 8, 16] for c in [0, 30, 60] for h in [0.3, 0.5, 0.7]]
    for gi, (t_med, contrast, H_seq) in enumerate(grid):
        accs = {'I_full': [], 'I_lin': [], 'auc_H': []}
        for _ in range(RUNS):
            rng = np.random.default_rng(rng_master.integers(0, 2**63 - 1))
            x, fac = gen_column(H_seq, t_med, contrast, rng)
            d = win_feats(x, fac)
            if len(d) < 15 or d['y'].nunique() < 2:
                continue
            I_full = mi_bins(d['mean'].values, d['lag1'].values, d['H'].values, d['y'].values)
            I_lin = mi_bins(d['mean'].values, d['lag1'].values, d['y'].values)
            pos = d[d['y'] == 1]['H'].values
            neg = d[d['y'] == 0]['H'].values
            from scipy.stats import mannwhitneyu
            if len(pos) >= 3 and len(neg) >= 3:
                accs['auc_H'].append(mannwhitneyu(pos, neg, alternative='two-sided').statistic /
                                      (len(pos) * len(neg)))
            accs['I_full'].append(I_full)
            accs['I_lin'].append(I_lin)
        I_f = float(np.mean(accs['I_full'])) if accs['I_full'] else np.nan
        I_l = float(np.mean(accs['I_lin'])) if accs['I_lin'] else np.nan
        UI_H = I_f - I_l  # MMI unique information of H beyond (mean, lag-1)
        rows.append({'t_med': t_med, 'contrast': contrast, 'H_seq': H_seq,
                     'I_full': I_f, 'I_lin': I_l, 'UI_H': UI_H,
                     'nonlin_share': UI_H / I_f if I_f and I_f > 0 else np.nan,
                     'auc_H': float(np.mean(accs['auc_H'])) if accs['auc_H'] else np.nan})
        print('%d/27 t=%2d c=%2d H=%.1f | I_full=%.3f I_lin=%.3f UI_H=%.3f share=%.2f auc=%.2f (%.0fs)' % (
            gi + 1, t_med, contrast, H_seq, I_f, I_l, UI_H,
            rows[-1]['nonlin_share'] if np.isfinite(rows[-1]['nonlin_share']) else -1,
            rows[-1]['auc_H'] if np.isfinite(rows[-1]['auc_H']) else -1, time.time() - t0))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, 'e5_generality.csv'), index=False)
    print('\nsaved e5_generality.csv  %.0fs' % (time.time() - t0))

if __name__ == '__main__':
    main()

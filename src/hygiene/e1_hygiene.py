# -*- coding: utf-8 -*-
"""E1 validity audit grid (DFA differencing and levels-minus-one conventions)."""
import sys, io, os, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import nolds

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, '..', '..', 'results'))
os.makedirs(RESULTS, exist_ok=True)
STEP_M = 0.152
RUNS = 20
SEED = 20201120
H_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
N_GRID = [64, 128, 256, 512, 1024]
SMOOTH_M = [0.0, 0.15, 0.3, 0.6]
SNR_GRID = [5, 10, 20, 10**9]

def fbm(H, n, rng):
    m = 1 << (int(np.ceil(np.log2(2 * n))) + 2)
    f = np.fft.rfft(rng.standard_normal(m))
    freq = np.fft.rfftfreq(m)
    freq[0] = freq[1]
    f = f / (freq ** (H + 0.5))
    s = np.fft.irfft(f, m)[:n]
    s -= s.mean()
    sd = s.std()
    return s / sd if sd > 0 else s

def ar1(n, phi, rng):
    e = rng.standard_normal(n + 200)
    x = np.empty(n + 200)
    x[0] = e[0]
    for i in range(1, n + 200):
        x[i] = phi * x[i - 1] + e[i]
    x = x[200:]
    return (x - x.mean()) / x.std()

def degrade(s, smooth_m, snr_db, rng):
    out = s
    if smooth_m > 0:
        out = gaussian_filter1d(out, sigma=smooth_m / STEP_M)
        sd = out.std()
        if sd > 0:
            out = out / sd
    if np.isfinite(snr_db) and snr_db < 10**8:
        p_sig = np.var(out)
        out = out + rng.standard_normal(len(out)) * np.sqrt(p_sig / (10 ** (snr_db / 10)))
    return out

def safe(fn, x):
    try:
        with np.errstate(all='ignore'):
            v = fn(x)
        return float(v) if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

def main():
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    rows = []
    total = len(H_GRID) * len(N_GRID) * len(SMOOTH_M) * len(SNR_GRID)
    done = 0
    for H in H_GRID:
        for n in N_GRID:
            for sm in SMOOTH_M:
                for snr in SNR_GRID:
                    acc = {'dfa': [], 'rs': [], 'sampen': [], 'lyap': [], 'd2': []}
                    for _ in range(RUNS):
                        s = fbm(H, n, rng)
                        x = degrade(s, sm, snr, rng)
                        acc['dfa'].append(safe(lambda z: nolds.dfa(np.diff(z), order=1), x))
                        acc['rs'].append(safe(nolds.hurst_rs, x))
                        acc['sampen'].append(safe(lambda z: nolds.sampen(z, emb_dim=2, tolerance=0.2 * float(np.std(z))), x))
                        if n <= 512:
                            acc['lyap'].append(safe(nolds.lyap_r, x))
                            acc['d2'].append(safe(lambda z: nolds.corr_dim(z, emb_dim=5), x))
                    dfa_arr = np.asarray(acc['dfa'])
                    row = {'H_nominal': H, 'n': n, 'smooth_m': sm, 'snr_db': snr,
                           'dfa_bias': float(np.nanmean(dfa_arr) - H),
                           'dfa_rmse': float(np.sqrt(np.nanmean((dfa_arr - H) ** 2))),
                           'sampen_mean': float(np.nanmean(acc['sampen'])),
                           'lyap_fp_rate': float(np.mean(np.asarray(acc['lyap']) > 0)) if acc['lyap'] else np.nan,
                           'lyap_mean': float(np.nanmean(acc['lyap'])) if acc['lyap'] else np.nan,
                           'd2_mean': float(np.nanmean(acc['d2'])) if acc['d2'] else np.nan}
                    rows.append(row)
                    done += 1
                    if done % 48 == 0:
                        pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'e1_grid.csv'), index=False)
                        print('grid %d/%d (%.0fs)' % (done, total, time.time() - t0))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'e1_grid.csv'), index=False)

    ar_rows = []
    for n in [256, 512]:
        for sm in SMOOTH_M:
            for snr in SNR_GRID:
                fps, vals = [], []
                for _ in range(RUNS):
                    x = degrade(ar1(n, 0.9, rng), sm, snr, rng)
                    v = safe(nolds.lyap_r, x)
                    vals.append(v)
                    fps.append(np.isfinite(v) and v > 0)
                ar_rows.append({'model': 'AR1(phi=0.9)', 'n': n, 'smooth_m': sm, 'snr_db': snr,
                                'lyap_fp_rate': float(np.mean(fps)),
                                'lyap_mean': float(np.nanmean(vals))})
    pd.DataFrame(ar_rows).to_csv(os.path.join(RESULTS, 'e1_ar1_lyap.csv'), index=False)
    print('saved e1_grid.csv + e1_ar1_lyap.csv  %.0fs' % (time.time() - t0))

if __name__ == '__main__':
    main()

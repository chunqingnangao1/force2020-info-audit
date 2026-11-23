# -*- coding: utf-8 -*-
"""E1b DFA dual-convention comparison grid (n up to 2048, 12 reps, checkpointed)."""
import sys, io, os, time, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import nolds

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, '..', '..', 'results'))
STEP_M = 0.152
SNR_GRID = [5, 10, 20, 10**9]
SMOOTH_M = [0.0, 0.15, 0.3, 0.6]
H_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
N_GRID = [64, 128, 256, 512, 1024, 2048]
RUNS = 12
SEED = 20201120

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

def degrade(s, smooth_m, snr_db, rng):
    out = s
    if smooth_m > 0:
        out = gaussian_filter1d(out, sigma=smooth_m / STEP_M)
        sd = out.std()
        if sd > 0:
            out = out / sd
    if snr_db < 10**8:
        out = out + rng.standard_normal(len(out)) * np.sqrt(np.var(out) / (10 ** (snr_db / 10)))
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
                    diff_est, lvl_est = [], []
                    for _ in range(RUNS):
                        x = degrade(fbm(H, n, rng), sm, snr, rng)
                        diff_est.append(safe(lambda z: nolds.dfa(np.diff(z), order=1), x))
                        lvl_est.append(safe(lambda z: nolds.dfa(z, order=1) - 1.0, x))
                    rows.append({
                        'H_nominal': H, 'n': n, 'smooth_m': sm, 'snr_db': snr,
                        'dfa_diff_bias': float(np.nanmean(diff_est) - H),
                        'dfa_diff_rmse': float(np.sqrt(np.nanmean((np.asarray(diff_est) - H) ** 2))),
                        'dfa_lvl_bias': float(np.nanmean(lvl_est) - H),
                        'dfa_lvl_rmse': float(np.sqrt(np.nanmean((np.asarray(lvl_est) - H) ** 2))),
                    })
                    done += 1
                    if done % 48 == 0:
                        pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'e1_dfa_conventions.csv'), index=False)
                        print('grid %d/%d (%.0fs)' % (done, total, time.time() - t0))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'e1_dfa_conventions.csv'), index=False)
    print('saved e1_dfa_conventions.csv  %.0fs' % (time.time() - t0))

if __name__ == '__main__':
    main()

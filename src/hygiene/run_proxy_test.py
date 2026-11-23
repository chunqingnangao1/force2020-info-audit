# -*- coding: utf-8 -*-
"""Surrogate test: IAAFT x199, statistics = SampEn + third moment of increments,
two-sided empirical p."""
import sys, io, os, time, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import nolds

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
RESULTS = os.path.join(ROOT, 'results')
MAX_PTS = 600
MIN_PTS = 120
N_SURR = 199
ALPHA = 0.01
SEED = 20271122

def iaaft_surrogates(x, n_surr, rng, iters=30):
    n = len(x)
    target_amp = np.abs(np.fft.rfft(x))
    sorted_x = np.sort(x)
    surrs = []
    for _ in range(n_surr):
        y = rng.permutation(x)
        for _ in range(iters):
            fx = np.fft.rfft(y)
            phases = np.exp(1j * np.angle(fx))
            y = np.fft.irfft(target_amp * phases, n)
            ranks = np.argsort(np.argsort(y))
            y = sorted_x[ranks]
        surrs.append(y)
    return np.asarray(surrs)

def sampen_safe(x):
    try:
        v = float(nolds.sampen(x, emb_dim=2, tolerance=0.2 * float(np.std(x))))
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

def tir_stat(x):
    d = np.diff(x)
    sd = d.std()
    return float((d ** 3).mean() / sd ** 3) if sd > 0 else np.nan

def two_sided_p(orig, surr):
    s = np.asarray(surr)
    n_le = int(np.sum(s <= orig))
    n_ge = int(np.sum(s >= orig))
    return float(2 * min(n_le, n_ge) + 1) / (len(s) + 1)

def main():
    rng = np.random.default_rng(SEED)
    wd = pd.read_csv(os.path.join(RESULTS, 'e3_windows.csv.gz'))
    raw = pd.read_csv(os.path.join(ROOT, 'data', 'clean', 'wells_raw.csv.gz'))
    wd['seg'] = wd['well'] + '|' + wd['formation'].astype(str)
    segs = wd.groupby('seg').agg(well=('well', 'first'), formation=('formation', 'first'))
    print('segments: %d' % len(segs))
    rows = []
    t0 = time.time()
    for i, (segid, s) in enumerate(segs.iterrows()):
        r = raw[(raw['WELL'] == s['well']) & (raw['FORMATION'] == s['formation'])]
        r = r.dropna(subset=['GR']).sort_values('DEPTH')
        gr = r['GR'].values.astype(float)
        if len(gr) > MAX_PTS:
            idx = np.sort(rng.choice(len(gr), MAX_PTS, replace=False))
            gr = gr[idx]
        gr = gr[np.isfinite(gr)]
        if len(gr) < MIN_PTS or np.std(gr) < 1e-6:
            continue
        x = (gr - gr.mean()) / gr.std()
        surrs = iaaft_surrogates(x, N_SURR, rng)
        o_se = sampen_safe(x)
        s_se = np.array([sampen_safe(sy) for sy in surrs])
        s_se = s_se[np.isfinite(s_se)]
        if not np.isfinite(o_se) or len(s_se) < N_SURR * 0.8:
            continue
        p_se = two_sided_p(o_se, s_se)
        o_tir = tir_stat(x)
        s_tir = np.array([tir_stat(sy) for sy in surrs])
        p_tir = two_sided_p(o_tir, s_tir)
        rows.append({'seg': segid, 'well': s['well'], 'formation': s['formation'],
                     'n': len(x), 'sampen_orig': o_se, 'p_sampen': p_se,
                     'tir_orig': o_tir, 'p_tir': p_tir,
                     'reject_sampen': int(p_se < ALPHA), 'reject_tir': int(p_tir < ALPHA),
                     'reject_any': int(p_se < ALPHA or p_tir < ALPHA)})
        if (i + 1) % 20 == 0:
            pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'proxy_test.csv'), index=False)
            done = pd.DataFrame(rows)
            print('segs %d/%d (%.0fs) reject_any=%.2f' % (i + 1, len(segs), time.time() - t0, done['reject_any'].mean()))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, 'proxy_test.csv'), index=False)
    out = {'n_segments': len(df), 'alpha': ALPHA, 'n_surrogates': N_SURR,
           'reject_rate_sampen': float(df['reject_sampen'].mean()),
           'reject_rate_tir': float(df['reject_tir'].mean()),
           'reject_rate_any': float(df['reject_any'].mean())}
    print(json.dumps(out, indent=1))
    with open(os.path.join(RESULTS, 'proxy_test_summary.json'), 'w') as fh:
        json.dump(out, fh, indent=1)

if __name__ == '__main__':
    main()

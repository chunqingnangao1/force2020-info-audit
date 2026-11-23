# -*- coding: utf-8 -*-
"""E4b composite-kernel intervention: K = s_smooth*RBF + s_rough*fBm."""
import sys, os, time, json, warnings
# stdout wrapping is handled by the imported run_e4_kernel (double wrap closes the buffer)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_e4_kernel import fbm_cov, rbf_cov

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
RESULTS = os.path.join(ROOT, 'results')
MAX_PTS = 600
MIN_PTS = 120
ITERS = 100
LR = 0.03
SEED = 20261122
JITTER = 1e-5

def fit_composite(t, v):
    tt = torch.as_tensor(t).double().reshape(-1, 1)
    y = torch.as_tensor(v).double()
    N = len(y)
    eye = torch.eye(N, dtype=torch.float64)
    rawH = torch.nn.Parameter(torch.tensor(-0.5))
    lo, hi = 0.05, 0.95
    Hf = lambda r: lo + (hi - lo) * torch.sigmoid(r)
    raw_ls_sm = torch.nn.Parameter(torch.tensor(np.log(0.3)))
    log_s_sm = torch.nn.Parameter(torch.tensor(0.0))
    log_s_ro = torch.nn.Parameter(torch.tensor(np.log(0.5)))
    log_nz = torch.nn.Parameter(torch.tensor(np.log(0.3)))
    opt = torch.optim.Adam([rawH, raw_ls_sm, log_s_sm, log_s_ro, log_nz], lr=LR)
    loss = None
    for it in range(ITERS):
        opt.zero_grad()
        K0 = torch.nn.functional.softplus(log_s_sm) * rbf_cov(tt, torch.exp(raw_ls_sm)) \
            + fbm_cov(tt, Hf(rawH))
        M = K0 + (torch.exp(log_nz) + 1e-4) * eye + JITTER * eye
        try:
            Lc = torch.linalg.cholesky(M)
        except Exception:
            M = M + 1e-3 * eye
            Lc = torch.linalg.cholesky(M)
        a = y[:, None]
        quad = (torch.cholesky_solve(a, Lc) * a).sum()
        logdet = 2.0 * torch.log(torch.diagonal(Lc)).sum()
        loss = (0.5 * (quad + logdet + N * np.log(2 * np.pi))) / N
        loss.backward()
        opt.step()
    return {'H_rough': float(Hf(rawH)),
            'trend_share': float(torch.nn.functional.softplus(log_s_sm)) /
                           (float(torch.nn.functional.softplus(log_s_sm)) + float(torch.nn.functional.softplus(log_s_ro))),
            'ls_smooth': float(torch.exp(raw_ls_sm)), 'noise': float(torch.exp(log_nz)),
            'll': float(-loss.detach())}

def main():
    rng = np.random.default_rng(SEED)
    wd = pd.read_csv(os.path.join(RESULTS, 'e3_windows.csv.gz'))
    raw = pd.read_csv(os.path.join(ROOT, 'data', 'clean', 'wells_raw.csv.gz'))
    wd['seg'] = wd['well'] + '|' + wd['formation'].astype(str)
    segs = wd.groupby('seg').agg(well=('well', 'first'), formation=('formation', 'first'),
                                 H_slid=('GR_H', 'mean'), n_win=('GR_H', 'count'))
    segs = segs[segs['n_win'] >= 5]
    print('segments: %d' % len(segs))
    rows = []
    t0 = time.time()
    for i, (segid, s) in enumerate(segs.iterrows()):
        r = raw[(raw['WELL'] == s['well']) & (raw['FORMATION'] == s['formation'])]
        r = r.dropna(subset=['GR', 'LITH_CODE']).sort_values('DEPTH')
        if len(r) < MIN_PTS:
            continue
        if len(r) > MAX_PTS:
            idx = np.sort(rng.choice(len(r), MAX_PTS, replace=False))
            r = r.iloc[idx]
        gr = r['GR'].values.astype(float)
        if np.std(gr) < 1e-6 or len(r) < MIN_PTS:
            continue
        gr_z = (gr - gr.mean()) / max(gr.std(), 1e-9)
        d = r['DEPTH'].values.astype(float)
        if np.isnan(d).any() or (np.nanmax(d) - np.nanmin(d)) <= 0:
            d = np.arange(len(r), dtype=float)
        t = ((d - d.min()) / max(d.max() - d.min(), 1e-9))[:, None]
        rec = {'seg': segid, 'well': s['well'], 'H_slid': float(s['H_slid']), 'n_win': int(s['n_win'])}
        try:
            rec.update(fit_composite(t, gr_z))
        except Exception as e:
            rec['H_rough'] = np.nan
        rows.append(rec)
        if (i + 1) % 20 == 0:
            pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'e4b_composite.csv'), index=False)
            print('segs %d/%d (%.0fs)' % (i + 1, len(segs), time.time() - t0))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, 'e4b_composite.csv'), index=False)
    ok = df.dropna(subset=['H_rough', 'H_slid'])

    def corr_boot(x, y, B=500):
        r = float(np.corrcoef(x, y)[0, 1])
        boots = []
        for _ in range(B):
            pick = rng.choice(ok['well'].unique(), len(ok['well'].unique()), replace=True)
            idx = np.concatenate([np.where(ok['well'].values == w)[0] for w in pick])
            boots.append(np.corrcoef(x[idx], y[idx])[0, 1])
        lo, hi = np.nanpercentile(boots, [2.5, 97.5])
        return r, float(lo), float(hi)

    r_rough, lo_r, hi_r = corr_boot(ok['H_rough'].values, ok['H_slid'].values)
    r_ts, lo_t, hi_t = corr_boot(ok['trend_share'].values, ok['H_slid'].values)
    out = {'n_segments': len(ok),
           'composite_r_Hrough_vs_Hslid': r_rough, 'composite_boot95': [lo_r, hi_r],
           'pure_fbm_r_reference': -0.21,
           'r_trendshare_vs_Hslid': r_ts, 'trendshare_boot95': [lo_t, hi_t],
           'median_H_rough': float(ok['H_rough'].median()),
           'median_trend_share': float(ok['trend_share'].median()),
           'interpretation': ('trend share explains the negative correlation' if r_rough > -0.10
                              else 'negative correlation persists beyond trend handling')}
    print(json.dumps(out, indent=1))
    with open(os.path.join(RESULTS, 'e4b_summary.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    print('saved e4b_composite.csv (%d segs) %.0fs' % (len(df), time.time() - t0))

if __name__ == '__main__':
    main()

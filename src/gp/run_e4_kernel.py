# -*- coding: utf-8 -*-
"""E4 GP kernel encoding: pure-torch per-channel exact GP, regression + classification tracks."""
import sys, io, os, time, json, warnings
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch

torch.set_num_threads(4)
import gpytorch.likelihoods

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
RESULTS = os.path.join(ROOT, 'results')
MAX_PTS = 600
MIN_PTS = 120
ITERS = 120
LR = 0.03
SEED = 20251121
JITTER = 1e-5

def fbm_cov(t, H):
    a = (t.abs() ** (2 * H)).squeeze(-1)
    d = torch.cdist(t, t).clamp_min(0.0)
    return 0.5 * (a[:, None] + a[None, :] - d ** (2 * H))

def rbf_cov(t, ls):
    d2 = torch.cdist(t, t) ** 2
    return torch.exp(-d2 / (2 * ls ** 2))

def fit_segment(t, y_codes, kind):
    """Classification track: Dirichlet transform + per-channel exact GP, shared-fBm kernel."""
    N = len(y_codes)
    like = gpytorch.likelihoods.DirichletClassificationLikelihood(
        torch.as_tensor(y_codes).long(), learn_additional_noise=True)
    alpha = like.transformed_targets.detach().clone().double()
    C = alpha.shape[0]
    init_noise = like.noise.detach().double()
    mn = max(float(init_noise.mean()), 1e-4)
    tt = torch.as_tensor(t).double().reshape(-1, 1)
    if kind == 'fbm':
        rawH = torch.nn.Parameter(torch.tensor(-0.5))
        lo, hi = 0.05, 0.95
        sig = lambda r: lo + (hi - lo) * torch.sigmoid(r)
        logs_c = torch.nn.Parameter(torch.log(torch.full((C,), mn)))
        params = [rawH, logs_c]
    else:
        raw_ls = torch.nn.Parameter(torch.tensor(np.log(0.1)))
        raw_scale_c = torch.nn.Parameter(torch.zeros(C))
        logs_c = torch.nn.Parameter(torch.log(torch.full((C,), mn)))
        params = [raw_ls, raw_scale_c, logs_c]
    opt = torch.optim.Adam(params, lr=LR)
    eye = torch.eye(N, dtype=torch.float64)
    losses = []
    for it in range(ITERS):
        opt.zero_grad()
        if kind == 'fbm':
            K0 = fbm_cov(tt, sig(rawH))
        else:
            K0 = torch.nn.functional.softplus(raw_scale_c)[..., None, None] * \
                rbf_cov(tt, torch.exp(raw_ls))
        nll = 0.0
        for c in range(C):
            Kc = K0 if kind == 'fbm' else K0[c]
            M = Kc + (torch.exp(logs_c[c]) + 1e-4) * eye + JITTER * eye
            Lc = torch.linalg.cholesky(M)
            a = alpha[c][:, None]
            quad = (torch.cholesky_solve(a, Lc) * a).sum()
            logdet = 2.0 * torch.log(torch.diagonal(Lc)).sum()
            nll = nll + 0.5 * (quad + logdet + N * np.log(2 * np.pi))
        loss = nll / C
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        mus = []
        for c in range(C):
            Kc = K0 if kind == 'fbm' else K0[c]
            M = Kc + (torch.exp(logs_c[c]) + 1e-4) * eye + JITTER * eye
            Lc = torch.linalg.cholesky(M)
            mu = alpha[c][:, None] - torch.exp(logs_c[c]) * \
                torch.cholesky_solve(alpha[c][:, None], Lc)
            mus.append(mu.squeeze(1))
        mu = torch.stack(mus)
        acc = float((mu.argmax(0) == torch.as_tensor(y_codes)).double().mean())
    out = {'train_acc': acc, 'll': -losses[-1] * C}
    if kind == 'fbm':
        out['H_fbm'] = float(sig(rawH))
    else:
        out['rbf_ls'] = float(torch.exp(raw_ls))
    return out

def fit_segment_reg(t, v, kind, iters=ITERS):
    """Regression track: GP on the GR values directly (commensurable with sliding DFA)."""
    tt = torch.as_tensor(t).double().reshape(-1, 1)
    y = torch.as_tensor(v).double()
    N = len(y)
    eye = torch.eye(N, dtype=torch.float64)
    if kind == 'fbm':
        rawH = torch.nn.Parameter(torch.tensor(-0.5))
        lo, hi = 0.05, 0.95
        sig = lambda r: lo + (hi - lo) * torch.sigmoid(r)
        raw_ls_c = torch.nn.Parameter(torch.tensor(np.log(0.3)))
        params = [rawH, raw_ls_c]
    else:
        raw_ls = torch.nn.Parameter(torch.tensor(np.log(0.1)))
        raw_scale = torch.nn.Parameter(torch.tensor(0.0))
        raw_ls_c = torch.nn.Parameter(torch.tensor(np.log(0.3)))
        params = [raw_ls, raw_scale, raw_ls_c]
    opt = torch.optim.Adam(params, lr=0.02)
    loss = None
    for it in range(iters):
        opt.zero_grad()
        if kind == 'fbm':
            K0 = fbm_cov(tt, sig(rawH))
        else:
            K0 = torch.exp(raw_scale) * rbf_cov(tt, torch.exp(raw_ls))
        sig2 = torch.exp(raw_ls_c)
        M = K0 + (sig2 + 1e-4) * eye + 1e-5 * eye
        try:
            Lc = torch.linalg.cholesky(M)
        except Exception:
            return {'%s_ls_c' % kind: float('nan')}
        a = y[:, None]
        quad = (torch.cholesky_solve(a, Lc) * a).sum()
        logdet = 2.0 * torch.log(torch.diagonal(Lc)).sum()
        loss = (0.5 * (quad + logdet + N * np.log(2 * np.pi))) / N
        loss.backward()
        opt.step()
    out = {'%s_ls_c' % kind: float(torch.exp(raw_ls_c).detach())}
    if kind == 'fbm':
        out['H_fbm_reg'] = float(sig(rawH))
    else:
        out['rbf_ls_reg'] = float(torch.exp(raw_ls))
        out['rbf_scale_reg'] = float(torch.exp(raw_scale))
    return out

def main():
    n_wells = int(sys.argv[1]) if len(sys.argv) > 1 else None
    rng = np.random.default_rng(SEED)
    wd = pd.read_csv(os.path.join(RESULTS, 'e3_windows.csv.gz'))
    raw = pd.read_csv(os.path.join(ROOT, 'data', 'clean', 'wells_raw.csv.gz'))
    wd['seg'] = wd['well'] + '|' + wd['formation'].astype(str)
    segs = wd.groupby('seg').agg(
        well=('well', 'first'), formation=('formation', 'first'),
        H_slid=('GR_H', 'mean'), n_win=('GR_H', 'count'))
    segs = segs[segs['n_win'] >= 5]
    wells = sorted(segs['well'].unique())
    if n_wells:
        wells = wells[:n_wells]
    segs = segs[segs['well'].isin(wells)]
    print('segments: %d, wells: %d' % (len(segs), len(wells)))

    grp6map = {30000: 'sand', 65030: 'interl', 65000: 'shale', 93000: 'shale',
               70000: 'carb', 70032: 'carb', 80000: 'marl_dolo', 74000: 'marl_dolo',
               88000: 'evap_other', 86000: 'evap_other', 99000: 'evap_other', 90000: 'evap_other'}
    rows = []
    t0 = time.time()
    for i, (segid, s) in enumerate(segs.iterrows()):
        r = raw[(raw['WELL'] == s['well']) & (raw['FORMATION'] == s['formation'])]
        r = r.dropna(subset=['GR', 'LITH_CODE']).sort_values('DEPTH')
        codes = r['LITH_CODE'].astype(int).map(grp6map)
        keep = codes.notna().values
        r, codes = r[keep], codes[keep].values
        if len(r) < MIN_PTS:
            continue
        if len(r) > MAX_PTS:
            idx = np.sort(rng.choice(len(r), MAX_PTS, replace=False))
            r, codes = r.iloc[idx], codes[idx]
        y_uni, y_codes = np.unique(codes, return_inverse=True)
        if len(y_uni) < 2:
            continue
        d = r['DEPTH'].values.astype(float)
        if np.isnan(d).any() or (np.nanmax(d) - np.nanmin(d)) <= 0:
            d = np.arange(len(r), dtype=float)
        t = ((d - d.min()) / max(d.max() - d.min(), 1e-9))[:, None]
        gr = r['GR'].values.astype(float)
        gr_z = (gr - gr.mean()) / max(gr.std(), 1e-9)
        rec = {'seg': segid, 'well': s['well'], 'formation': s['formation'],
               'n_pts': len(r), 'n_class': len(y_uni), 'H_slid': float(s['H_slid']),
               'n_win': int(s['n_win'])}
        try:
            import nolds
            rec['H_seg_dfa'] = float(nolds.dfa(gr_z, order=1)) - 1.0
        except Exception:
            rec['H_seg_dfa'] = np.nan
        try:
            rec.update(fit_segment_reg(t, gr_z, 'fbm', iters=80))
        except Exception as e:
            rec['H_fbm_reg'] = np.nan
        try:
            rec.update(fit_segment_reg(t, gr_z, 'rbf', iters=80))
        except Exception:
            rec['rbf_ls_reg'] = np.nan
        try:
            rec.update(fit_segment(t, y_codes, 'fbm'))
        except Exception as e:
            rec['H_fbm'] = np.nan
        try:
            rec.update(fit_segment(t, y_codes, 'rbf'))
        except Exception:
            rec['rbf_ls'] = np.nan
        rows.append(rec)
        if (i + 1) % 20 == 0:
            pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'e4_kernel_fits.csv'), index=False)
            print('segs %d/%d (%.0fs)' % (i + 1, len(segs), time.time() - t0))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, 'e4_kernel_fits.csv'), index=False)
    ok = df.dropna(subset=['H_fbm_reg', 'H_slid'])
    if len(ok) > 10:
        def corr_boot(x, y):
            r = float(np.corrcoef(x, y)[0, 1])
            boots = []
            for _ in range(500):
                pick = rng.choice(ok['well'].unique(), len(ok['well'].unique()), replace=True)
                idx = np.concatenate([np.where(ok['well'].values == w)[0] for w in pick])
                boots.append(np.corrcoef(x[idx], y[idx])[0, 1])
            lo, hi = np.nanpercentile(boots, [2.5, 97.5])
            return r, float(lo), float(hi)
        r_reg, lo, hi = corr_boot(ok['H_fbm_reg'].values, ok['H_slid'].values)
        r_dfa = None
        if ok['H_seg_dfa'].notna().sum() > 10:
            ok2 = ok.dropna(subset=['H_seg_dfa'])
            r_dfa = float(np.corrcoef(ok2['H_seg_dfa'], ok2['H_slid'])[0, 1])
        r_alpha = None
        if ok['H_fbm'].notna().sum() > 10:
            ok3 = ok.dropna(subset=['H_fbm'])
            r_alpha = float(np.corrcoef(ok3['H_fbm'], ok3['H_slid'])[0, 1])
        out = {'n_segments': len(ok), 'H3a_primary_r_HfbmReg_vs_Hslid': r_reg,
               'H3a_boot95': [lo, hi], 'sanity_r_HsegDfa_vs_Hslid': r_dfa,
               'contrast_r_HfbmAlpha_vs_Hslid': r_alpha,
               'median_H_fbm_reg': float(ok['H_fbm_reg'].median()),
               'median_H_slid': float(ok['H_slid'].median())}
        print(json.dumps(out, indent=1))
        with open(os.path.join(RESULTS, 'e4_samelatent.json'), 'w') as fh:
            json.dump(out, fh, indent=1)
    print('saved e4_kernel_fits.csv (%d segs) %.0fs' % (len(df), time.time() - t0))

if __name__ == '__main__':
    main()

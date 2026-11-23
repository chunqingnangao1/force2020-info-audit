# -*- coding: utf-8 -*-
"""PID core: WB I_min (unweighted specific information), bincount 3D tables,
pre-grouped segment indices for the block-permutation nulls."""
import sys, io, os, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import numpy as np
import pandas as pd

K_BIN = 4
N_PERM = 1000
BLOCK = 8
SEED = 20231121
PAIRS = [('mean', 'H'), ('var', 'H'), ('lag1', 'H'),
         ('mean', 'sampen'), ('lag1', 'sampen'), ('H', 'sampen')]

def disc(x, k=K_BIN):
    x = np.asarray(x, dtype=float)
    ok = ~np.isnan(x)
    q = np.full(len(x), -1)
    if ok.sum() > k:
        q[ok] = pd.qcut(pd.Series(x[ok]).rank(method='first'), k, labels=False).values
    return q, ok

def mi_disc(x, y):
    x = x.astype(int)
    y = y.astype(int)
    nx, ny = x.max() + 1, y.max() + 1
    joint = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny).astype(float)
    n = joint.sum()
    if n == 0:
        return 0.0
    px = joint.sum(1, keepdims=True) / n
    py = joint.sum(0, keepdims=True) / n
    pij = joint / n
    mask = pij > 0
    mi = float(np.sum(pij[mask] * np.log2(pij[mask] / (px @ py)[mask])))
    nz_x = (joint.sum(1) > 0).sum()
    nz_y = (joint.sum(0) > 0).sum()
    mi += (nz_x - 1) * (nz_y - 1) / (2 * n * np.log(2))
    return max(0.0, mi)

def mi_disc_joint(s1, s2, y):
    s12 = s1.astype(int) * (s2.max() + 1) + s2.astype(int)
    return mi_disc(s12, y)

def i_min_wb(s1, s2, y):
    """Williams-Beer I_min: per-outcome minimum of the (unweighted) specific information
    over joint-source states with p(y|s) > 0."""
    s1 = s1.astype(int); s2 = s2.astype(int); y = y.astype(int)
    n1, n2 = s1.max() + 1, s2.max() + 1
    ny = y.max() + 1
    n = len(y)
    idx = (s1 * n2 + s2) * ny + y
    c3 = np.bincount(idx, minlength=n1 * n2 * ny).astype(float).reshape(n1, n2, ny)
    p_s = c3.sum(axis=2) / n
    p_y = c3.sum(axis=(0, 1)) / n
    p_y_given_s = np.divide(c3, (p_s * n)[..., None], out=np.zeros_like(c3),
                            where=(p_s[..., None] * n) > 0)
    p_exp = np.tensordot(p_s, p_y_given_s, axes=([0, 1], [0, 1]))
    R = 0.0
    for yi in range(ny):
        if p_y[yi] <= 0:
            continue
        mask = c3[..., yi] > 0
        if not mask.any():
            continue
        spec = np.log2(p_y_given_s[..., yi][mask] / max(p_exp[yi], 1e-300))
        R += p_y[yi] * spec.min()
    return max(0.0, float(R))

def pid_two_sources(s1, s2, y):
    I1 = mi_disc(s1, y)
    I2 = mi_disc(s2, y)
    I12 = mi_disc_joint(s1, s2, y)
    R_wb = i_min_wb(s1, s2, y)
    ui1_wb = I1 - R_wb
    ui2_wb = I2 - R_wb
    s_wb = I12 - R_wb - ui1_wb - ui2_wb
    r_mmi = min(I1, I2)
    ui1_mmi = I12 - I2
    ui2_mmi = I12 - I1
    s_mmi = I12 - I1 - I2 + r_mmi
    return {'I1': I1, 'I2': I2, 'I12': I12,
            'R_wb': R_wb, 'UI1_wb': ui1_wb, 'UI2_wb': ui2_wb, 'S_wb': s_wb,
            'R_mmi': r_mmi, 'UI1_mmi': ui1_mmi, 'UI2_mmi': ui2_mmi, 'S_mmi': s_mmi}

def block_permute_labels(y, seg_groups, rng, block=BLOCK):
    y2 = y.copy()
    for idx in seg_groups:
        if len(idx) <= block:
            y2[idx] = y[rng.permutation(idx)]
        else:
            y2[idx] = np.roll(y[idx], int(rng.integers(1, len(idx))))
    return y2

def analyze_curve(wd, curve, n_perm=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    feats = sorted(set(p[0] for p in PAIRS) | set(p[1] for p in PAIRS))
    cols = ['%s_%s' % (curve, f) for f in feats]
    d = wd[cols + ['grp6', 'seg']].dropna(subset=['grp6']).copy()
    y_raw = d['grp6'].astype('category')
    y_codes, y_uni = pd.factorize(y_raw)
    disc_map, ok_map = {}, np.ones(len(d), dtype=bool)
    for f, col in zip(feats, cols):
        q, ok = disc(d[col].values)
        disc_map[f] = q
        ok_map &= ok
    d = d[ok_map]
    y = y_codes[ok_map]
    seg_groups = [np.where(d['seg'].values == s)[0] for s in np.unique(d['seg'].values)]
    disc_map = {f: q[ok_map] for f, q in disc_map.items()}
    rows = []
    for (f1, f2) in PAIRS:
        s1, s2 = disc_map[f1], disc_map[f2]
        res = pid_two_sources(s1, s2, y)
        keys = ['UI1_wb', 'UI2_wb', 'S_wb', 'R_wb']
        nulls = {k: [] for k in keys}
        for _ in range(n_perm):
            yp = block_permute_labels(y, seg_groups, rng)
            r = pid_two_sources(s1, s2, yp)
            for k in keys:
                nulls[k].append(r[k])
        out = {'curve': curve, 'pair': '%s|%s' % (f1, f2), 'k_class': len(y_uni)}
        for k in keys:
            arr = np.asarray(nulls[k])
            out['p_' + k] = float((np.sum(arr >= res[k]) + 1) / (n_perm + 1))
        out.update(res)
        rows.append(out)
    return rows, y_uni

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--curves', default='GR,RHOB,DTC')
    ap.add_argument('--nperm', type=int, default=N_PERM)
    args = ap.parse_args()
    HERE_ = os.path.dirname(os.path.abspath(__file__))
    wd = pd.read_csv(os.path.join(os.path.dirname(HERE_), '..', 'results', 'e3_windows.csv.gz'))
    wd['seg'] = wd['well'] + '|' + wd['formation'].astype(str)
    all_rows = []
    for curve in args.curves.split(','):
        t0 = time.time()
        rows, y_uni = analyze_curve(wd, curve, n_perm=args.nperm)
        all_rows.extend(rows)
        print('curve %s done (%.0fs)' % (curve, time.time() - t0))
    res = pd.DataFrame(all_rows)
    for q in ['p_UI1_wb', 'p_UI2_wb', 'p_S_wb', 'p_R_wb']:
        p = res[q].values
        order = np.argsort(p)
        ranked = p[order]
        qv = np.minimum.accumulate((ranked * len(p) / (np.arange(len(p)) + 1))[::-1])[::-1]
        out = np.empty(len(p))
        out[order] = np.clip(qv, 0, 1)
        res[q.replace('p_', 'q_')] = out
    res.to_csv(os.path.join(os.path.dirname(HERE_), '..', 'results', 'e3_pid_tables.csv'), index=False)
    print(res[['curve', 'pair', 'UI1_wb', 'UI2_wb', 'R_mmi', 'p_UI2_wb']].round(3).to_string(index=False))

if __name__ == '__main__':
    main()

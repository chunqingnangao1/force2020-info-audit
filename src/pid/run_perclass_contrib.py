# -*- coding: utf-8 -*-
"""Per-facies MI contribution breakdown (which facies drive each feature's label info).
I_c = sum_x p(x, c) log2[p(c|x) / p(c)], summing over c to I(X;Y).
Output: results/e3_perclass_contrib.csv
"""
import sys, os
# stdout wrapping is handled by the imported run_pid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_pid import disc  # noqa

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, '..', '..', 'results'))
FEATS = ['GR_mean', 'GR_lag1', 'GR_H', 'GR_sampen', 'RHOB_H', 'DTC_H']

def per_class_contrib(x, y):
    x = x.astype(int)
    y = y.astype(int)
    nx, ny = x.max() + 1, y.max() + 1
    joint = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny).astype(float)
    n = joint.sum()
    px = joint.sum(1) / n
    py = joint.sum(0) / n
    out = np.zeros(ny)
    for c in range(ny):
        mask = joint[:, c] > 0
        pcx = joint[mask, c] / px[mask]  # p(c|x)
        contrib = np.sum((joint[mask, c] / n) * np.log2(pcx / max(py[c], 1e-300)))
        out[c] = contrib
    return out

def main():
    wd = pd.read_csv(os.path.join(RESULTS, 'e3_windows.csv.gz')).dropna(subset=['grp6'])
    y, classes = pd.factorize(wd['grp6'], sort=True)
    rows = []
    for f in FEATS:
        s, ok = disc(wd[f].values)
        contrib = per_class_contrib(s[ok], y[ok])
        total = contrib.sum()
        for ci, cls in enumerate(classes):
            rows.append({'feature': f, 'class': cls, 'n': int((y[ok] == ci).sum()),
                         'MI_contrib_bits': float(contrib[ci]),
                         'share': float(contrib[ci] / total) if total > 0 else np.nan})
        print('%-10s total I=%.3f bits | shares: %s' % (
            f, total, {c: round(float(s), 3) for c, s in
                       zip(classes, contrib / max(total, 1e-9))}))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'e3_perclass_contrib.csv'), index=False)
    print('saved results/e3_perclass_contrib.csv')

if __name__ == '__main__':
    main()

# -*- coding: utf-8 -*-
"""F1流程图 / F4滑窗示意 / F9有效性热图 / F10配对折点（恢复版）"""
import sys, io, os, ast
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'axes.unicode_minus': False})
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

# ===================== F1 工作流程总览 =====================
fig, ax = plt.subplots(figsize=(7.2, 2.6))
ax.axis('off')
boxes = [
    ('FORCE 2020\n118 wells\n0.152 m, 12 facies', '#EAF2F8'),
    ('1 Validity audit\nsynthetic fBm/AR(1)\nvalidity domains', '#D5E8D4'),
    ('2 Identifiability\nsynthetic stratigraphy\nphase diagram', '#D5E8D4'),
    ('3 Information accounting\nPID + prediction\nincrement (FORCE)', '#DAE8FC'),
    ('4 Kernel correspondence\nGP fBm encoding\nvs sliding features', '#DAE8FC'),
    ('Verdict:\ninformative\nbut uneconomical', '#F8CECC'),
]
w, h, y0 = 0.138, 0.52, 0.24
xs = [0.008 + i * 0.162 for i in range(5)]
xs.append(0.008 + 5 * 0.162 + 0.02)
for i, (txt, c) in enumerate(boxes):
    ax.add_patch(FancyBboxPatch((xs[i], y0), w, h, boxstyle='round,pad=0.008',
                                fc=c, ec='0.35', lw=0.9))
    ax.text(xs[i] + w / 2, y0 + h / 2, txt, ha='center', va='center', fontsize=6.6)
for i in range(5):
    ax.add_patch(FancyArrowPatch((xs[i] + w + 0.005, y0 + h / 2),
                                 (xs[i + 1] - 0.006, y0 + h / 2),
                                 arrowstyle='-|>', mutation_scale=10, color='0.3', lw=1.0))
ax.text(0.5, 0.95, 'FORCE 2020 information-accounting pipeline: audit before claim',
        ha='center', fontsize=9)
ax.text(0.5, 0.06, 'synthetic ground truth first (1–2)  →  real-data accounting on a strong baseline (3)  '
                   '→  encoding correspondence (4)', ha='center', fontsize=7.2, color='0.35')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'F1_overview.png'), dpi=300)
plt.close(fig)
print('F1 saved')

# ===================== F4 滑窗示意 =====================
raw = pd.read_csv(os.path.join(HERE, 'data', 'clean', 'wells_raw.csv.gz'))
e3w = pd.read_csv(os.path.join(R, 'e3_windows.csv.gz'))
well, form = '15_9-13', 'Utsira Fm.'
r = raw[(raw['WELL'] == well) & (raw['FORMATION'] == form)].dropna(
    subset=['GR', 'LITH_CODE']).sort_values('DEPTH')
seg = e3w[(e3w['well'] == well) & (e3w['formation'] == form)].sort_values('depth')
d = r['DEPTH'].values.astype(float)
gr = r['GR'].values.astype(float)
d0, d1 = d.min(), d.max()
pad = 5.0
fig, ax = plt.subplots(figsize=(3.4, 4.6))
ax.plot(gr, d, color='0.25', lw=0.7)
ax.set_xlim(0, max(150, np.nanpercentile(gr, 99)))
ax.set_ylim(d1 + pad, d0 - pad)
for _, wrow in seg.iterrows():
    z0, z1 = wrow['depth'] - 256 * 0.152 / 2, wrow['depth'] + 256 * 0.152 / 2
    pure = wrow['frac'] > 0.6 or wrow['frac'] < 0.4
    ax.add_patch(Rectangle((0, z0), ax.get_xlim()[1], z1 - z0,
                           fc='#2E7D32' if pure else '#B0B0B0',
                           alpha=0.16 if pure else 0.28, lw=0))
ax.text(0.96 * ax.get_xlim()[1], 1250, 'green = pure\nwindows (kept)\n\ngrey = mixed\n(dropped)',
        ha='right', va='center', fontsize=7,
        bbox=dict(fc='white', ec='0.6', alpha=0.9, pad=2))
ax.set_xlabel('GR (gAPI)')
ax.set_ylabel('depth (m)')
ax.set_title('F4  %s / %s\nsliding windows (n=512, stride 256, %d kept)' % (
    well, form, len(seg)), fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'F4_windows.png'), dpi=300)
plt.close(fig)
print('F4 saved')

# ===================== F9 DFA 双口径热图 =====================
cv = pd.read_csv(os.path.join(R, 'e1_dfa_conventions.csv'))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
for ax, (col, ttl) in zip(axes, [
        ('dfa_diff_rmse', '(a) differencing convention'),
        ('dfa_lvl_rmse', '(b) levels-minus-one convention')]):
    sub = cv[(cv['snr_db'] == 20) & (cv['smooth_m'] == 0.15)]
    piv = sub.pivot_table(index='H_nominal', columns='n', values=col)
    im = ax.imshow(piv.values, origin='lower', cmap='RdYlGn_r', vmin=0, vmax=0.7,
                   aspect='auto')
    ax.set_xticks(range(len(piv.columns)), ['%d' % c for c in piv.columns], fontsize=7)
    ax.set_yticks(range(len(piv.index)), ['%.1f' % i for i in piv.index], fontsize=7)
    ax.set_xlabel('window length n (samples)')
    ax.set_ylabel('nominal H')
    ax.set_title(ttl, fontsize=9)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label('RMSE (H units)', fontsize=8)
    cb.ax.tick_params(labelsize=7)
    ok = piv.values < 0.10
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            if ok[i, j]:
                ax.text(j, i, '·', ha='center', va='center', fontsize=7, color='k')
fig.suptitle('DFA-H estimate error under the two conventions (SNR 20 dB, smoothing 0.15 m); '
             'dots mark cells with RMSE < 0.10', fontsize=8.5)
fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(os.path.join(FIG, 'F9_validity_heatmap.png'), dpi=300)
plt.close(fig)
print('F9 saved')

# ===================== F10 配对折点 =====================
inc = pd.read_csv(os.path.join(R, 'e3_predict_increment.csv'))
e0 = pd.read_csv(os.path.join(R, 'e3_e0plus_nonlinear.csv'))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
for ax, (base_set, cmp_set, title) in zip(axes, [
        ('L', 'L+F', 'logistic: L vs L+F'),
        ('E0', 'E0+H', 'XGBoost full base: E0 vs E0+H')]):
    df = inc if base_set == 'L' else e0
    b = np.array(ast.literal_eval(df[df['set'] == base_set].iloc[0]['folds']))
    f = np.array(ast.literal_eval(df[df['set'] == cmp_set].iloc[0]['folds']))
    lim = [min(b.min(), f.min()) - 0.02, max(b.max(), f.max()) + 0.02]
    ax.plot(lim, lim, 'k:', lw=0.8)
    for k in range(5):
        ax.plot([b[k], f[k]], [k + 1, k + 1], color='0.8', lw=0.8, zorder=1)
        ax.scatter(b[k], k + 1, color='#4878CF', s=22, zorder=2)
        ax.scatter(f[k], k + 1, color='#EE854A', s=22, zorder=2)
    ax.set_yticks(range(1, 6))
    ax.set_xlim(lim)
    ax.set_xlabel('per-fold macro-F1')
    ax.set_ylabel('fold')
    ax.set_title(title + '\n(paired folds; no consistent direction)', fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'F10_paired_folds.png'), dpi=300)
plt.close(fig)
print('F10 saved')

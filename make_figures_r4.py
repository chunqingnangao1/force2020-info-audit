# -*- coding: utf-8 -*-
"""R4新增图：特征相关矩阵（线性冗余视角）+ 逐岩相特征分布（小提琴）
+ 复原F8井展示 + F11 PID示意（事故前已画但用户未见过）"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle

plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'axes.unicode_minus': False})
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

GROUP6 = {30000: 'sand', 65030: 'interl', 65000: 'shale', 93000: 'shale',
          70000: 'carb', 70032: 'carb', 80000: 'marl_dolo', 74000: 'marl_dolo',
          88000: 'evap_other', 86000: 'evap_other', 99000: 'evap_other', 90000: 'evap_other'}
COL6 = {'sand': '#F4D03F', 'interl': '#82E0AA', 'shale': '#5D6D7E', 'carb': '#AED6F1',
        'marl_dolo': '#BB8FCE', 'evap_other': '#F1948A'}
CURVES = ['GR', 'RHOB', 'DTC']
FEATS = ['mean', 'var', 'lag1', 'H', 'sampen']

wd = pd.read_csv(os.path.join(R, 'e3_windows.csv.gz'))

# ===================== F12 特征相关矩阵（Spearman，族块标注） =====================
cols = ['%s_%s' % (c, f) for c in CURVES for f in FEATS]
sub = wd[cols].dropna()
corr = sub.corr(method='spearman')
fig, ax = plt.subplots(figsize=(7.2, 6.0))
im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)
labels = [c.replace('_', '\n') for c in cols]
ax.set_xticks(range(len(cols)), labels, fontsize=6.5)
ax.set_yticks(range(len(cols)), labels, fontsize=6.5)
for i in range(len(cols)):
    for j in range(len(cols)):
        ax.text(j, i, '%.2f' % corr.values[i, j], ha='center', va='center',
                fontsize=5.5, color='k' if abs(corr.values[i, j]) < 0.75 else 'w')
# 族块边界：线性(0-8) | H(9-11) | sampen(12-14)
for b in [8.5, 11.5]:
    ax.axhline(b, color='k', lw=1.4)
    ax.axvline(b, color='k', lw=1.4)
ax.text(4, -0.9, 'linear statistics', ha='center', fontsize=8, fontweight='bold')
ax.text(10.5, -0.9, 'DFA-H', ha='center', fontsize=8, fontweight='bold')
ax.text(13.5, -0.9, 'SampEn', ha='center', fontsize=8, fontweight='bold')
fig.colorbar(im, fraction=0.045, label='Spearman ρ')
ax.set_title('Cross-feature correlation of sliding-window features (n = %d windows)\n'
             'linear-family blocks vs nonlinear families — the linear view of redundancy that '
             'PID decomposes' % len(sub), fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'F12_corr_matrix.png'), dpi=300)
plt.close(fig)
print('F12 saved')

# ===================== F13 逐岩相特征分布（小提琴） =====================
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9))
order = ['shale', 'sand', 'interl', 'carb', 'evap_other', 'marl_dolo']
for ax, feat, ttl in zip(axes, ['GR_H', 'GR_sampen', 'GR_mean'],
                         ['sliding DFA-H', 'sliding SampEn', 'window mean GR']):
    data = [wd[wd['grp6'] == g][feat].dropna().values for g in order]
    parts = ax.violinplot(data, showmedians=True, widths=0.85)
    for pc, g in zip(parts['bodies'], order):
        pc.set_facecolor(COL6[g])
        pc.set_alpha(0.7)
    for k in ['cmins', 'cmaxes', 'cbars', 'cmedians']:
        parts[k].set_color('0.3')
        parts[k].set_linewidth(0.8)
    ax.set_xticks(range(1, len(order) + 1),
                  [g if g != 'evap_other' else 'evap+oth' for g in order], fontsize=7, rotation=30)
    ax.set_title(ttl, fontsize=8.5)
    if feat == 'GR_mean':
        ax.set_ylabel('GR (gAPI)')
fig.suptitle('Per-facies distributions of the GR window features (n = %d)' % len(wd),
             fontsize=8.5, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'F13_facies_violin.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print('F13 saved')

# ===================== F8 井多轨展示（复原） =====================
raw = pd.read_csv(os.path.join(HERE, 'data', 'clean', 'wells_raw.csv.gz'))
well = '15_9-13'
rw = raw[raw['WELL'] == well].sort_values('DEPTH')
ww = wd[wd['well'] == well].sort_values('depth')
fig, axes = plt.subplots(1, 4, figsize=(7.2, 6.4), sharey=True,
                         gridspec_kw={'width_ratios': [2.2, 0.5, 1.4, 1.4], 'wspace': 0.06})
ax = axes[0]
ax.plot(rw['GR'].values, rw['DEPTH'].values, color='0.25', lw=0.5)
ax.set_xlim(0, 160)
ax.set_xlabel('GR (gAPI)')
ax.set_ylabel('depth (m)')
ax.set_ylim(rw['DEPTH'].max() + 20, rw['DEPTH'].min() - 20)
ax.invert_yaxis()
ax = axes[1]
lab = rw['LITH_CODE'].map(GROUP6)
for g, c in COL6.items():
    m = (lab == g).values
    ax.fill_betweenx(rw['DEPTH'].values[m], 0, 1, color=c, linewidth=0)
ax.set_xticks([])
ax.set_xlabel('facies', fontsize=8)
ax.set_xlim(0, 1)
ax = axes[2]
ax.scatter(ww['GR_H'], ww['depth'], s=8, c=[COL6[g] for g in ww['grp6']], edgecolors='none')
ax.set_xlabel('sliding DFA-H')
ax.axvline(0, color='k', lw=0.5)
ax = axes[3]
ax.scatter(ww['GR_sampen'], ww['depth'], s=8, c=[COL6[g] for g in ww['grp6']], edgecolors='none')
ax.set_xlabel('sliding SampEn')
handles = [plt.Line2D([], [], marker='s', ls='', color=c, label=g) for g, c in COL6.items()]
fig.legend(handles=handles, loc='lower center', ncol=6, fontsize=7, frameon=False)
fig.suptitle('Well %s: GR log, facies column, and sliding-window nonlinear features (n=512, '
             'stride 256)' % well, fontsize=8.5, y=0.985)
fig.tight_layout(rect=[0, 0.035, 1, 0.97])
fig.savefig(os.path.join(FIG, 'F8_well_multiwell.png'), dpi=300)
plt.close(fig)
print('F8 saved')

# ===================== F11 PID 示意（复原） =====================
fig, ax = plt.subplots(figsize=(4.4, 3.0))
ax.axis('off')
ax.add_patch(Ellipse((0.38, 0.55), 0.62, 0.86, fc='#4878CF', alpha=0.25, ec='#4878CF'))
ax.add_patch(Ellipse((0.62, 0.55), 0.62, 0.86, fc='#EE854A', alpha=0.25, ec='#EE854A'))
ax.text(0.22, 0.62, 'UI₁\nunique\n(linear side)', ha='center', fontsize=9)
ax.text(0.78, 0.62, 'UI₂\nunique\n(nonlinear side)', ha='center', fontsize=9)
ax.text(0.50, 0.62, 'R\nredundant', ha='center', fontsize=9)
ax.text(0.50, 0.16, 'S synergistic', ha='center', fontsize=9, color='#6C3483')
ax.text(0.38, 0.97, 'source S₁ (linear stat.)', ha='center', fontsize=9)
ax.text(0.62, 0.97, 'source S₂ (nonlinear feat.)', ha='center', fontsize=9)
ax.text(0.88, 0.62, '→ Y\n(lithofacies)', fontsize=10, ha='left')
ax.annotate('', xy=(0.86, 0.60), xytext=(0.76, 0.60),
            arrowprops=dict(arrowstyle='-|>', color='k'))
ax.set_title('Two-source partial information decomposition (bits about Y)', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'F11_pid_schematic.png'), dpi=300)
plt.close(fig)
print('F11 saved')

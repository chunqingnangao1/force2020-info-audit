# -*- coding: utf-8 -*-
"""主图版绘制（恢复版）：F2卫生审计 / F3相图 / F5 PID / F6稳健性 / F7同源性"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9, 'axes.linewidth': 0.8,
                     'font.family': 'DejaVu Sans', 'axes.unicode_minus': False})

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, 'results')
FIG = os.path.join(HERE, 'figures')
os.makedirs(FIG, exist_ok=True)

# ===================== F2 卫生审计三面板 =====================
g = pd.read_csv(os.path.join(R, 'e1_grid.csv'))
ar = pd.read_csv(os.path.join(R, 'e1_ar1_lyap.csv'))
cv = pd.read_csv(os.path.join(R, 'e1_dfa_conventions.csv'))

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
ax = axes[0]
sub = g[(g['n'] == 512) & (g['snr_db'] == 10)]
piv = sub.pivot_table(index='H_nominal', columns='smooth_m', values='lyap_fp_rate')
im = ax.imshow(piv.values, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
ax.set_xticks(range(len(piv.columns)), ['0', '0.15', '0.30', '0.60'], fontsize=7)
ax.set_yticks(range(len(piv.index)), ['%.1f' % i for i in piv.index], fontsize=7)
ax.set_xlabel('smoothing σ (m)', fontsize=8)
ax.set_ylabel('nominal H', fontsize=8)
ax.set_title('(a) λmax false-positive rate\n(fBm truth, n=512, SNR 10dB)', fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.046)

ax = axes[1]
sub = cv[(cv['snr_db'] == 20) & (cv['smooth_m'] == 0.15)]
for H, mk in zip([0.3, 0.5, 0.7], ['o', 's', '^']):
    ss = sub[sub['H_nominal'] == H].sort_values('n')
    ax.plot(ss['n'], ss['dfa_diff_bias'], mk + '-', label='diff, H=%.1f' % H, ms=4, lw=1)
    ax.plot(ss['n'], ss['dfa_lvl_bias'], mk + '--', label='levels−1, H=%.1f' % H, ms=4, lw=1)
hb, lb = ax.get_legend_handles_labels()
ax.legend(hb, lb, fontsize=5.5, ncol=2, loc='lower left')
ax.axhline(0, color='k', lw=0.6)
ax.axhspan(-0.08, 0.08, color='0.85', zorder=0)
ax.set_xscale('log', base=2)
ax.set_xlabel('window length n (samples)')
ax.set_ylabel('DFA-H bias')
ax.set_title('(b) convention comparison\n(SNR 20dB, σ=0.15m; grey=±0.08)', fontsize=8)

ax = axes[2]
sub = ar[ar['n'] == 512].pivot_table(index='smooth_m', columns='snr_db', values='lyap_fp_rate')
sub.columns = ['5', '10', '20', 'none']
x = np.arange(len(sub.columns))
ax.bar(x - 0.2, sub.loc[0.0], 0.4, label='σ=0.00 m')
ax.bar(x + 0.2, sub.loc[0.3], 0.4, label='σ=0.30 m')
ax.axhline(0.05, color='r', ls=':', lw=0.8)
ax.set_xticks(x, sub.columns, fontsize=7)
ax.set_xlabel('SNR (dB)', fontsize=8)
ax.set_ylim(0, 1.05)
ax.set_ylabel('λmax FP rate on AR(1)')
ax.set_title('(c) linear null has no chaos:\nFP rate by noise/smoothing', fontsize=8)
ax.legend(fontsize=6, loc='center left', bbox_to_anchor=(1.01, 0.5))
fig.tight_layout(rect=[0, 0.14, 1, 1])
fig.legend(hb, lb, loc='lower center', ncol=3, fontsize=6.5)
fig.savefig(os.path.join(FIG, 'F2_hygiene.png'), dpi=300)
plt.close(fig)
print('F2 saved')

# ===================== F3 可辨识性相图 =====================
e2 = pd.read_csv(os.path.join(R, 'e2v2_identifiability.csv'))
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax = axes[0]
sub = e2[(e2['contrast'] == 0.0) & (e2['smooth_m'] == 0.3) & (e2['snr_db'] == 10**9)
         & (e2['tmode'] == 'logn') & (e2['path'] == 'fbm')]
params = sorted(sub['param'].unique())
w = 0.35
x = np.arange(len(params))
for k, (col, lab) in enumerate([('auc_lag1', 'lag-1 ACF (linear)'), ('auc_H', 'DFA-H (fractal)')]):
    ax.bar(x + (k - 0.5) * w, sub.set_index('param').loc[params, col], w, label=lab)
ax.axhline(0.5, color='k', ls=':', lw=0.8)
ax.set_xticks(x, ['seq-H=%.1f' % v for v in params], fontsize=7)
ax.set_ylim(0, 1.02)
ax.set_ylabel('AUC (sand vs shale windows)')
ax.set_title('(a) structure-only info survives\nsmoothing iff long-memory (contrast=0)', fontsize=8)
ax.legend(fontsize=7, loc='lower right')
ax = axes[1]
sub2 = e2[(e2['snr_db'] == 10**9) & (e2['smooth_m'] == 0.3)]
fbm = sub2[sub2['path'] == 'fbm'].pivot_table(index='param', columns='tmode', values='auc_H')
mk = sub2[sub2['path'] == 'markov'].pivot_table(index='param', columns='tmode', values='auc_H')
x = np.arange(len(fbm.index))
ax.plot(x, fbm['logn'], 'o-', label='fBm path (long-memory)')
ax.plot(x, mk['logn'], 's--', label='Markov path')
ax.axhline(0.5, color='k', ls=':', lw=0.8)
ax.set_xticks(x, ['seq-H=%.1f' % i for i in fbm.index], fontsize=7)
ax.set_ylim(0.3, 1.02)
ax.set_xlabel('sequence-organization H / persistence')
ax.set_ylabel('AUC of DFA-H')
ax.set_title('(b) smoothing kills short memory,\nspares long memory (σ=0.3m)', fontsize=8)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'F3_identifiability.png'), dpi=300)
plt.close(fig)
print('F3 saved')

# ===================== F5 PID 主图 =====================
pid = pd.read_csv(os.path.join(R, 'e3_pid_tables.csv'))
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
for ax, cur in zip(axes, ['GR', 'RHOB', 'DTC']):
    sub = pid[pid['curve'] == cur]
    y = np.arange(len(sub))
    ax.barh(y + 0.22, sub['UI1_wb'], 0.4, label='unique: linear side', color='#4878CF')
    ax.barh(y - 0.22, sub['UI2_wb'], 0.4, label='unique: nonlinear side', color='#EE854A')
    ax.scatter(sub['R_mmi'], y, marker='|', color='k', label='redundancy (MMI)', zorder=5)
    ax.set_yticks(y, sub['pair'], fontsize=6.5)
    ax.axvline(0, color='k', lw=0.6)
    ax.set_xlabel('bits')
    ax.set_title('(%s) %s' % ('abc'[['GR', 'RHOB', 'DTC'].index(cur)], cur), fontsize=8)
    if cur == 'GR':
        ax.legend(fontsize=6, loc='upper center', bbox_to_anchor=(0.5, -0.16))
fig.suptitle('PID: unique information beyond the paired feature (permutation p at floor, q(FDR)≤0.001)',
             fontsize=8, y=1.02)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'F5_pid.png'), dpi=300, bbox_inches='tight')
plt.close(fig)
print('F5 saved')

# ===================== F6 稳健性 =====================
rb = pd.read_csv(os.path.join(R, 'e3_robustness.csv'))
fig, ax = plt.subplots(figsize=(3.6, 2.6))
r1 = rb[rb['item'] == 'R1_12class']
r3 = rb[rb['item'] == 'R3_balancedMI']
pairs = sorted(set(r1['pair']))
main = pid[(pid['curve'] == 'GR')]
m6 = {r['pair']: r['UI2_wb'] for _, r in main.iterrows()}
x = np.arange(len(pairs))
ax.bar(x - 0.27, [m6.get(p, np.nan) for p in pairs], 0.27, label='6-class (main)')
ax.bar(x, [float(r1[r1['pair'] == p]['UI2_12'].iloc[0]) for p in pairs], 0.27, label='12-class')
bal = {r['pair']: r['UI2_mmi_bal'] for _, r in r3.iterrows()}
ax.bar(x + 0.27, [bal.get(p, np.nan) for p in pairs], 0.27, label='class-balanced (MMI)')
ax.set_xticks(x, [p.replace('|', '\n') for p in pairs], fontsize=7)
ax.set_ylabel('unique info of DFA-H (bit)')
ax.set_title('robustness of H unique info (GR)', fontsize=8)
ax.legend(fontsize=6)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'F6_robustness.png'), dpi=300)
plt.close(fig)
print('F6 saved')

# ===================== F7 同源性散点 =====================
k = pd.read_csv(os.path.join(R, 'e4_kernel_fits.csv'))
ok = k.dropna(subset=['H_fbm_reg', 'H_slid'])
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
ax = axes[0]
sc = ax.scatter(ok['H_slid'], ok['H_fbm_reg'], c=np.log10(ok['n_win'] + 1), s=14, cmap='viridis')
r = float(np.corrcoef(ok['H_fbm_reg'], ok['H_slid'])[0, 1])
ax.set_xlabel('sliding-window DFA-H (per segment mean)')
ax.set_ylabel('GP fBm kernel H (regression track)')
ax.set_title('(a) H3a rejected: r = %.2f, CI [%.2f, %.2f]' % (r, -0.40, -0.05), fontsize=8)
fig.colorbar(sc, ax=ax, label='log10(windows)', fraction=0.046)
lim = [min(ok['H_slid'].min(), ok['H_fbm_reg'].min()), max(ok['H_slid'].max(), ok['H_fbm_reg'].max())]
ax.plot(lim, lim, 'k:', lw=0.7)
ax = axes[1]
ok2 = k.dropna(subset=['H_fbm', 'H_slid'])
ax.scatter(ok2['H_slid'], ok2['H_fbm'], s=14, color='#888')
ra = float(np.corrcoef(ok2['H_fbm'], ok2['H_slid'])[0, 1])
ax.set_xlabel('sliding-window DFA-H (texture)')
ax.set_ylabel('GP fBm kernel H (classification track)')
ax.set_title('(b) boundary-geometry H: r = %.2f ≈ 0\n(two different latent quantities)' % ra, fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'F7_samelatent.png'), dpi=300)
plt.close(fig)
print('F7 saved')

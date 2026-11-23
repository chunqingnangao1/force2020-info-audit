# -*- coding: utf-8 -*-
"""F14 地质普适性图谱：非线性信息占比热图（3个H_seq面板 × 层厚×对比度）"""
import sys, io, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'axes.unicode_minus': False})
HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, 'results', 'e5_generality.csv'))

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
for ax, H in zip(axes, [0.3, 0.5, 0.7]):
    sub = df[df['H_seq'] == H].pivot_table(index='t_med', columns='contrast', values='nonlin_share')
    im = ax.imshow(sub.values, origin='lower', cmap='viridis', vmin=-0.005, vmax=0.085,
                   aspect='auto')
    ax.set_xticks(range(3), ['0', '30', '60'], fontsize=8)
    ax.set_yticks(range(3), ['0.6', '1.2', '2.4'], fontsize=8)
    ax.set_xlabel('facies contrast (gAPI)')
    ax.set_ylabel('median bed thickness (m)')
    ax.set_title('sequence H = %.1f' % H, fontsize=9)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, '%.2f' % sub.values[i, j], ha='center', va='center',
                    fontsize=7.5, color='white' if sub.values[i, j] < 0.055 else 'black')
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cb.set_label('nonlinear information share (upper bound)', fontsize=8)
fig.suptitle('Geological generality map: DFA-H unique-information share beyond (mean, lag-1), '
             'never above 8% across the geological grid', fontsize=8.5)
fig.savefig(os.path.join(HERE, 'figures', 'F14_generality_map.png'), dpi=300,
            bbox_inches='tight')
plt.close(fig)
print('F14 saved')

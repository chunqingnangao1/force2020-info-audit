# -*- coding: utf-8 -*-
"""F3 PID schematic (rebuilt after the tmpconv accident; fixes title/label overlap)."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch

plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'axes.unicode_minus': False})

HERE = os.path.dirname(os.path.abspath(__file__))

fig, ax = plt.subplots(figsize=(7.2, 4.8))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# title band kept clear of everything below (y >= 0.90)
ax.text(0.5, 0.965, 'Two-source partial information decomposition (bits about Y)',
        ha='center', va='center', fontsize=10.5)
ax.text(0.30, 0.875, 'source S$_1$ (linear feat.)', ha='center', va='center', fontsize=9)
ax.text(0.70, 0.875, 'source S$_2$ (nonlinear feat.)', ha='center', va='center', fontsize=9)

# venn ellipses
left = Ellipse((0.32, 0.53), width=0.46, height=0.60, fc='#DAE8FC', ec='0.35',
               lw=1.0, alpha=0.55, zorder=1)
right = Ellipse((0.60, 0.53), width=0.46, height=0.60, fc='#FFE6CC', ec='0.35',
                lw=1.0, alpha=0.55, zorder=2)
ax.add_patch(left)
ax.add_patch(right)

ax.text(0.185, 0.58, 'UI$_1$\nunique\n(linear side)', ha='center', va='center',
        fontsize=8.5, zorder=3)
ax.text(0.735, 0.58, 'UI$_2$\nunique\n(nonlinear side)', ha='center', va='center',
        fontsize=8.5, zorder=3)
ax.text(0.46, 0.66, 'R\nredundant', ha='center', va='center', fontsize=8.5, zorder=3)
ax.text(0.46, 0.33, 'S\nsynergistic', ha='center', va='center', fontsize=8.5,
        color='#2E7D32', zorder=3)

# joint information -> Y
ax.add_patch(FancyArrowPatch((0.85, 0.53), (0.97, 0.53), arrowstyle='-|>',
                             mutation_scale=14, color='0.25', lw=1.2))
ax.text(0.912, 0.60, 'joint $I$', ha='center', va='center', fontsize=8, color='0.25')
ax.text(0.912, 0.44, 'Y\n(lithofacies)', ha='center', va='center', fontsize=8.5)

fig.tight_layout()
for out in [os.path.join(HERE, 'figures', 'F3_pid.png'),
            os.path.join(HERE, 'submission', 'figs', 'F3_pid.png')]:
    fig.savefig(out, dpi=300)
    print('saved', out)

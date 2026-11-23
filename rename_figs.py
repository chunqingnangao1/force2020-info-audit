# -*- coding: utf-8 -*-
"""Rename submission figs to the names referenced in main.tex."""
import os, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, 'submission', 'figs')
MAP = {
    'F4_windows.png': 'F2_windows.png',
    'F11_pid_schematic.png': 'F3_pid.png',
    'F9_validity_heatmap.png': 'F4_validity.png',
    'F2_hygiene.png': 'F5_audit.png',
    'F3_identifiability.png': 'F6_ident.png',
    'F12_corr_matrix.png': 'F7_corr.png',
    'F8_well_multiwell.png': 'F8_well.png',
    'F5_pid.png': 'F9_pid.png',
    'F13_facies_violin.png': 'F10_violin.png',
    'F10_paired_folds.png': 'F11_folds.png',
    'F6_robustness.png': 'F12_robust.png',
    'F7_samelatent.png': 'F13_kernel.png',
    'F14_generality_map.png': 'F14_generality.png',
}
# 两阶段重命名避免覆盖冲突
tmp = {}
for old, new in MAP.items():
    src = os.path.join(FIGS, old)
    tmpf = os.path.join(FIGS, '_tmp_' + new)
    if os.path.exists(src):
        shutil.move(src, tmpf)
        tmp[new] = tmpf
for new, tmpf in tmp.items():
    final = os.path.join(FIGS, new)
    if os.path.exists(final):
        os.remove(final)
    shutil.move(tmpf, final)
    print('renamed ->', new)
print('figs now:', sorted(os.listdir(FIGS)))

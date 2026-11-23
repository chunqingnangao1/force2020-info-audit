# -*- coding: utf-8 -*-
"""Setting-stratified PID: rerun GR main pairs within three FORCE depositional-setting
clusters (Cenozoic mudstone-dominated / Jurassic marine / older carbonate-evaporitic),
answering 'does the two-sided conclusion vary across geological settings' with real data.
Output: results/e3_pid_settings.csv
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_pid import disc, pid_two_sources, block_permute_labels, PAIRS  # noqa

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, '..', '..', 'results'))
SEED = 20331123

SETTINGS = {
    'cenozoic_mudstone': ['HORDALAND GP.', 'ROGALAND GP.', 'NORDLAND GP.'],
    'jurassic_marine': ['VIKING GP.', 'SHETLAND GP.', 'DUNLIN GP.', 'TYNE GP.', 'HEGRE GP.'],
    'older_carb_evap': ['ZECHSTEIN GP.', 'CROMER KNOLL GP.', 'VESTLAND GP.',
                        'ROTLIEGENDES GP.', 'BAAT GP.', 'BOKNFJORD GP.'],
}

def main():
    t0 = time.time()
    wd = pd.read_csv(os.path.join(RESULTS, 'e3_windows.csv.gz'))
    raw = pd.read_csv(os.path.abspath(os.path.join(HERE, '..', '..', 'data', 'clean', 'wells_raw.csv.gz')))
    raw_ok = raw[raw['FORMATION'].notna() & raw['GROUP'].notna()].copy()
    raw_ok['key'] = raw_ok['WELL'] + '|' + raw_ok['FORMATION']
    seg_group = raw_ok.groupby('key')['GROUP'].agg(lambda s: s.mode().iloc[0])
    wd['seg'] = wd['well'] + '|' + wd['formation'].astype(str)
    wd['group'] = (wd['well'] + '|' + wd['formation']).map(seg_group)
    wd = wd[wd['group'].notna()]

    def setting_of(g):
        for name, groups in SETTINGS.items():
            if g in groups:
                return name
        return 'other'

    wd['setting'] = wd['group'].map(setting_of)
    rng = np.random.default_rng(SEED)
    feats = {'mean': 'GR_mean', 'var': 'GR_var', 'lag1': 'GR_lag1',
             'H': 'GR_H', 'sampen': 'GR_sampen'}
    d = pd.DataFrame({k: wd[v] for k, v in feats.items()})
    d['grp6'] = wd['grp6'].values
    d['seg'] = wd['seg'].values
    d['setting'] = wd['setting'].values
    rows = []
    for setting, sub in d.groupby('setting'):
        yy = pd.factorize(sub['grp6'])[0]
        segv = sub['seg'].values
        seg_groups = [np.where(segv == s)[0] for s in np.unique(segv)]
        for (f1, f2) in PAIRS:
            s1, _ = disc(sub[f1].values)
            s2, _ = disc(sub[f2].values)
            m = (s1 >= 0) & (s2 >= 0)
            s1m, s2m, ym, sgm = s1[m], s2[m], yy[m], segv[m]
            if len(ym) < 150:
                continue
            groups = [np.where(sgm == s)[0] for s in np.unique(sgm)]
            res = pid_two_sources(s1m, s2m, ym)
            arr = []
            for _ in range(500):
                r = pid_two_sources(s1m, s2m, block_permute_labels(ym, groups, rng))
                arr.append(r['UI2_wb'])
            arr = np.asarray(arr)
            rows.append({'setting': setting, 'pair': '%s|%s' % (f1, f2), 'n_win': int(m.sum()),
                         'UI_H': res['UI2_wb'],
                         'p_UI_H': float((np.sum(arr >= res['UI2_wb']) + 1) / 501)})
            print('[%s] %s|%s n=%d UI_H=%.4f p=%.4f' % (setting, f1, f2, m.sum(),
                  res['UI2_wb'], rows[-1]['p_UI_H']))
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, 'e3_pid_settings.csv'), index=False)
    print('saved e3_pid_settings.csv  %.0fs' % (time.time() - t0))

if __name__ == '__main__':
    main()

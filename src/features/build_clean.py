# -*- coding: utf-8 -*-
"""FORCE 2020 LAS -> cleaned table (with the depth-coalesce fix):
- depth = row-wise coalesce(DEPT, DEPTH_MD) (DEPTH_MD has NaN segments in 117 wells)
- NaN-depth rows are barred from formation labels (searchsorted on NaN is unreliable)
- lithology codes reconciled against the 12-class dictionary; formation-top xlsx join
"""
import sys, io, os, glob, json, collections
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import lasio
import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, '..', '..', 'data'))
LASDIR = os.path.join(DATA, 'las')
CLEAN = os.path.join(DATA, 'clean')
os.makedirs(CLEAN, exist_ok=True)

LITHOLOGY_KEYS = {
    30000: 'sandstone', 65030: 'sandstone-shale interl.', 65000: 'shale',
    80000: 'marl', 74000: 'dolomite', 70000: 'limestone', 70032: 'chalk',
    88000: 'halite', 86000: 'anhydrite', 99000: 'tuff', 90000: 'coal',
    93000: 'bituminous shale',
}
BASE_CURVES = ['GR', 'RSHA', 'RMED', 'RDEP', 'RHOB', 'NPHI', 'DTC', 'DTS', 'PEF']

def load_tops(fn):
    wb = openpyxl.load_workbook(os.path.join(DATA, fn), read_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = []
    for i, r in enumerate(ws.iter_rows(values_only=True)):
        if i == 0 or r[0] is None or r[1] is None or r[5] is None:
            continue
        try:
            md = float(r[5])
        except (TypeError, ValueError):
            continue
        rows.append({'WELL': str(r[0]).strip().replace('/', '_'), 'surface': str(r[1]).strip(), 'MD': md})
    wb.close()
    return pd.DataFrame(rows)

def assign_strat(df, tops, colname):
    w = df['WELL'].iloc[0]
    t = tops[tops['WELL'] == w].sort_values('MD')
    if len(t) == 0:
        df[colname] = np.nan
        return df
    edges = t['MD'].values
    labels = t[colname.lower()].values
    idx = np.searchsorted(edges, df['DEPTH'].values, side='right') - 1
    assigned = np.where(idx >= 0, labels[np.clip(idx, 0, len(labels) - 1)], np.nan)
    assigned[np.isnan(df['DEPTH'].values)] = np.nan
    df[colname] = assigned
    return df

def main():
    # fetch stratigraphy spreadsheets first if missing
    import urllib.request
    for fn in ['NPD_Lithostratigraphy_member_formations_all_wells.xlsx',
               'NPD_Lithostratigraphy_groups_all_wells.xlsx',
               'lithology%20scoring%20matrix%20cost%20function.xlsx']:
        dst = os.path.join(DATA, fn.replace('%20', ' '))
        if not os.path.exists(dst):
            urllib.request.urlretrieve(
                'https://zenodo.org/records/4351156/files/%s?download=1' % fn, dst)
            print('downloaded', dst)

    form_tops = load_tops('NPD_Lithostratigraphy_member_formations_all_wells.xlsx')
    group_tops = load_tops('NPD_Lithostratigraphy_groups_all_wells.xlsx')
    form_tops['formation'] = form_tops['surface'].str.replace(' Top', '', regex=False)
    group_tops['group'] = group_tops['surface'].str.replace(' Top', '', regex=False)

    # extract LAS archive if missing
    las_files = glob.glob(os.path.join(LASDIR, '**', '*.las'), recursive=True)
    if not las_files:
        import zipfile
        os.makedirs(LASDIR, exist_ok=True)
        with zipfile.ZipFile(os.path.join(DATA, 'LAS_files_Force_2020.zip')) as z:
            z.extractall(LASDIR)
        las_files = glob.glob(os.path.join(LASDIR, '**', '*.las'), recursive=True)
    print('LAS files:', len(las_files))

    records, code_counter = [], collections.Counter()
    for f in las_files:
        wname = os.path.splitext(os.path.basename(f))[0]
        try:
            las = lasio.read(f)
        except Exception as e:
            print('READ FAIL', wname, str(e)[:80])
            continue
        curves = {c.mnemonic.upper(): c for c in las.curves}
        dept = np.asarray(curves['DEPT'].data, dtype=float) if 'DEPT' in curves else None
        dmd = np.asarray(curves['DEPTH_MD'].data, dtype=float) if 'DEPTH_MD' in curves else None
        if dept is not None:
            depth = np.where(np.isfinite(dept), dept, dmd if dmd is not None else np.nan)
        else:
            depth = dmd
        df = pd.DataFrame({'DEPTH': depth})
        for bc in BASE_CURVES:
            df[bc] = np.asarray(curves[bc].data, dtype=float) if bc in curves else np.nan
        lith_col = curves.get('FORCE_2020_LITHOFACIES_LITHOLOGY')
        conf_col = curves.get('FORCE_2020_LITHOFACIES_CONFIDENCE')
        df['LITH_CODE'] = np.asarray(lith_col.data, dtype=float) if lith_col is not None else np.nan
        df['LITH_CONF'] = np.asarray(conf_col.data, dtype=float) if conf_col is not None else np.nan
        df['WELL'] = wname
        assign_strat(df, form_tops, 'FORMATION')
        assign_strat(df, group_tops, 'GROUP')
        known = df['LITH_CODE'].dropna().astype(int)
        code_counter.update(known.value_counts().to_dict())
        records.append(df)

    allw = pd.concat(records, ignore_index=True)
    print('total rows:', len(allw), '| wells:', allw['WELL'].nunique())
    unknown = {int(c): int(n) for c, n in code_counter.items() if int(c) not in LITHOLOGY_KEYS}
    print('unknown codes:', dict(sorted(unknown.items())))
    for bc in BASE_CURVES + ['LITH_CODE', 'FORMATION', 'GROUP']:
        print('  %-10s %.3f' % (bc, allw[bc].notna().mean()))
    allw.to_csv(os.path.join(CLEAN, 'wells_raw.csv.gz'), index=False)
    with open(os.path.join(CLEAN, 'litho_audit.json'), 'w', encoding='utf-8') as fh:
        json.dump({'code_counts': {str(int(k)): int(v) for k, v in code_counter.items()},
                   'unknown_codes': unknown}, fh, ensure_ascii=False, indent=1)
    print('saved data/clean/wells_raw.csv.gz')

if __name__ == '__main__':
    main()

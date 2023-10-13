#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
|===============================================================|
| File: Vis.py                                           |
| Project: WasteAndMaterialFootprint-MacroStudy                 |
| Repository: www.github.com/Stew-McD/WasteAndMaterialFootprint-MacroStudy|
| Description: <<description>>                                  |
|---------------------------------------------------------------|
| File Created: Thursday, 28th September 2023 12:54:49 pm       |
| Author: Stewart Charles McDowall                              |
| Email: s.c.mcdowall@cml.leidenuniv.nl                         |
| Github: Stew-McD                                              |
| Company: CML, Leiden University                               |
|---------------------------------------------------------------|
| Last Modified: Friday, 29th September 2023 8:25:34 pm         |
| Modified By: Stewart Charles McDowall                         |
| Email: s.c.mcdowall@cml.leidenuniv.nl                         |
|---------------------------------------------------------------|
|License: The Unlicense                                         |
|===============================================================|
'''

from pathlib import Path
import pandas as pd

import plotly.graph_objects as go

from user_settings import title, project_name, dir_results


file_all = list(Path(dir_results).glob(f'*cookedresults_df.csv'))
file_top = list(Path(dir_results).glob(f'*topactivities_df.csv'))

df_all = pd.read_csv(file_all[0], index_col=None, sep=';')
df_top = pd.read_csv(file_top[0], index_col=None, sep=';')

cols_meta = ['name', 'database', 'code', 'prod_category', 'prod_sub_category', 'location', 'reference product', 'unit', 'model','pathway', 'subpathway','year',]
cols_waste = [x for x in df_all.columns if 'waste' in x]
cols_material = [x for x in df_all.columns if '(demand)' in x]
cols_methods = [x for x in df_all.columns if x not in cols_meta + cols_waste + cols_material]
cols_to_normalize = [x for x in df_all.columns if x not in cols_meta]

def normalize_to_2020(group):
    cols_to_normalize = [x for x in group.columns if x not in cols_meta]
    ref_value = group[group['year'] == 2020]
    if not ref_value.empty:
        group[cols_to_normalize] = group[cols_to_normalize].div(ref_value[cols_to_normalize].values)
    return group
    

cols_meta_cut = ['name', 'model','pathway', 'subpathway','year',]

dbs = list(df_all['database'].unique())[1:]


batt = df_all[
    df_all['name'].str.contains('market for battery, Li-ion, NMC811', case=False, na=False) &
    df_all['database'].isin(dbs) 
]

batt_mat = batt[cols_meta_cut + cols_material]


df = batt_mat
combinations = df[['model', 'pathway', 'subpathway']].drop_duplicates()

df = df.groupby(['model', 'pathway', 'subpathway'], group_keys=False).apply(normalize_to_2020)

# drop columns with all nan
df = df.dropna(axis=1, how='all')

max = df[cols_material]


# Create the plot
fig = go.Figure()

cols = [x for x in df.columns if x not in cols_meta]

for _, row in combinations.iterrows():
    filtered_df = df_all[
        (df_all['model'] == row['model']) &
        (df_all['pathway'] == row['pathway']) &
        (df_all['subpathway'] == row['subpathway'])
    ]
    for col in cols:
        fig.add_trace(go.Scatter(x=filtered_df['year'],
                                 y=filtered_df[col],
                                 mode='dots',
                                 name=f"{row['model']}|{row['pathway']}|{row['subpathway']} {col}"))

fig.show()



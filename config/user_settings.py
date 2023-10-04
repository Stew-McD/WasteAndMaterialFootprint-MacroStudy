#!/usr/bin/env python3

'''
|===============================================================|
| File: user_settings.py                                        |
| Project: WasteAndMaterialFootprint-MacroStudy                 |
| Repository: www.github.com/Stew-McD/WasteAndMaterialFootprint-MacroStudy|
| Description: <<description>>                                  |
|---------------------------------------------------------------|
| File Created: Thursday, 28th September 2023 1:27:06 pm        |
| Author: Stewart Charles McDowall                              |
| Email: s.c.mcdowall@cml.leidenuniv.nl                         |
| Github: Stew-McD                                              |
| Company: CML, Leiden University                               |
|---------------------------------------------------------------|
| Last Modified: Friday, 29th September 2023 8:27:09 pm         |
| Modified By: Stewart Charles McDowall                         |
| Email: s.c.mcdowall@cml.leidenuniv.nl                         |
|---------------------------------------------------------------|
|License: The Unlicense                                         |
|===============================================================|
'''
#%%
import os
import numpy as np
from pathlib import Path
import bw2data as bd


title = 'markets'
project_name = 'WMF-SSP125_cutoff'

if project_name not in bd.projects:
    print(f'Project {project_name} not found, exiting...')
    exit(0)

else:
    print(f'Project {project_name} found, continuing...')
    bd.projects.set_current(project_name)


database_names = None # you could also specify a list of databases here

# WMF was the prefix for all biosphere databases processed with the WasteAndMaterialFootprint tool
if not database_names:
    exclude = ["biosphere", 'WMF-'] # add to here if you want
    database_names = sorted([x for x in bd.databases if not any(e in x for e in exclude)])


# Define filters for activities of interest
# Can leave as an empty list or include specific names to filter.

# Uncommenting a name (e.g. 'battery production') will include it in the filter.
names_filter = [
    'market for',
    # 'battery production',
]

# Specify CPC (Central Product Classification) numbers to include. (integers)
cpc_num_filter = [
    # 46420, 
    # 46410, 
    # -1 # if there is no CPC number
]

# Specify keywords to exclude from the activities.
exclude_filter = [
    # 'recovery',
    # 'Treatment', 
    # 'disposal', 
    # 'waste', 
    # 'services', 
    # 'Waste', 
    # 'Site preparation', 
    # 'Construction of'
]

locations_filter = [
    'GLO', 
    'RoW', 
    'World',
]

units_filter = [
    'kilogram', 
    'cubic meter',
]

filters = {
    "names": names_filter,
    "CPC_num": cpc_num_filter,
    "exclude": exclude_filter,
    "locations": locations_filter,
    "units": units_filter
}


# choose methods

# Filter methods to select those of interest
methods_all =  np.unique([x[0] for x in bd.methods.list])
methods_waste = [x for x in bd.methods.list if "Waste Footprint" in x[0]]
methods_material = [x for x in bd.methods.list if "Material Demand Footprint" in x[0]]

METHOD_KEYWORDS = [
    # "Ecological Footprint",
    "Crustal Scarcity Indicator 2020",
    'ReCiPe 2016 v1.03, endpoint (H) no LT',
    # 'Cumulative Energy Demand (CED)',
    # 'Cumulative Exergy Demand (CExD)',
]

methods_other = [x for x in bd.methods.list if any(e in x[0] for e in METHOD_KEYWORDS)]

methods = methods_waste + methods_material + methods_other
# methods = methods_other


# %% DIRECTORY PATHS
# Set the paths (to the data, logs, and the results

# Get the directory of the main script
cwd = Path.cwd()
# Get the path one level up
dir_root = cwd.parents[0]

    # Set up the data directories
dir_data = dir_root / 'data' / title
dir_tmp = dir_data / 'tmp' / project_name
dir_logs = dir_data / 'logs' / project_name
dir_results = dir_data / 'results' / project_name

dirs = [dir_data, dir_tmp, dir_logs, dir_results]

for DIR in dirs:
    if not os.path.isdir(DIR): 
        os.makedirs(DIR)
        
activities_list = dir_data / f"activities_list_merged_{project_name}_{title}.csv"
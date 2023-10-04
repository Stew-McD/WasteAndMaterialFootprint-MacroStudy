#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
|===============================================================|
| File: main.py                                                 |
| Project: WasteAndMaterialFootprint-MacroStudy                 |
| Repository: www.github.com/Stew-McD/WasteAndMaterialFootprint-MacroStudy|
| Description: <<description>>                                  |
|---------------------------------------------------------------|
| File Created: Thursday, 28th September 2023 12:54:52 pm       |
| Author: Stewart Charles McDowall                              |
| Email: s.c.mcdowall@cml.leidenuniv.nl                         |
| Github: Stew-McD                                                |
| Company: CML, Leiden University                               |
|---------------------------------------------------------------|
| Last Modified: Thursday, 28th September 2023 1:15:22 pm       |
| Modified By: Stewart Charles McDowall                         |
| Email: s.c.mcdowall@cml.leidenuniv.nl                         |
|---------------------------------------------------------------|
|License: The Unlicense                                         |
|===============================================================|
'''

import os
import sys
from pathlib import Path

    # Set the working directory to the location of this script
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

    # Add the cwd to the Python path
cwd = Path.cwd()
sys.path.insert(0, str(cwd))

    # Add the root and config dirs to the Python path
dir_root = cwd.parents[0]
dir_config = dir_root / 'config'
sys.path.insert(0, str(dir_config))

    # Import custom modules (from the root dir)
from FilterActivities import GetActivitiesMP, MergeActivities 
from Calculations import LCIA, MergeResults 
from Processing import Raw2Cooked, ExtractTopActivities

from user_settings import title, project_name, database_names, activities_list

print(f'{"="*80}')
print('   \t\t\t*** Starting calculations ***')
print(f'{"="*80}\n')

print(f"\n\n*** Calculating the {title} activities' LCIAs \
for the following databases:\n\t" + '\n\t'.join(database_names))

#%% FILTER ACTIVITIES

# filter activities from databases
GetActivitiesMP(database_names, project_name, title)
# merge activities from all databases
MergeActivities(database_names, project_name, title)
    

#%% RUN CALCULATIONS
LCIA(activities_list, project_name, title, limit=100)

# merge results from all databases
# combined_raw_csv, combined_raw_pickle = \
#     MergeResults(project_name, title)

# #%% PROCESS RESULTS

# combined_cooked_csv, combined_cooked_pickle = \
#     Raw2Cooked(activities_list, combined_raw_pickle)

# # extract top activities
# top_csv, top_pickle = \
#     ExtractTopActivities(combined_cooked_pickle)

print("\n\n*** Finished calculations for the following databases:\
      \n\t" + '\n\t'.join(database_names))




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
import cowsay

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

from user_settings import title, project_name, database_names, verbose

print('   \t\t\t*** Starting script main.py ***')
print(f'{"."*80}\n')

if verbose:
    print(f"\n*** Calculating the '{title}' activities' LCIAs for the following databases:\n\t" + '\n\t'.join(database_names))

#%% FILTER ACTIVITIES

# # filter activities from databases
# GetActivitiesMP(database_names, project_name, title)
# # # merge activities from all databases
# MergeActivities(database_names, project_name, title)
    

# # #%% RUN CALCULATIONS
# LCIA()

# merge results from all databases

# MergeResults()

# #%% PROCESS RESULTS


Raw2Cooked()

# # extract top activities
ExtractTopActivities(n_top=1)

print("\n", flush=True)
print(f"\n\n{'@'*80}")
print("\n\n*** Finished calculations for the following databases:\
      \n\t" + '\n\t'.join(database_names))
print(f"\n{'@'*80}\n\n")

cowsay.turtle("GREAT SUCCESS!!")







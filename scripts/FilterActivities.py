#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
|===============================================================|
| File: FilterActivities.py                                     |
| Project: scripts                                              |
| Repository: www.github.com/Stew-McD/scripts                   |
| Description: <<description>>                                  |
|---------------------------------------------------------------|
| File Created: Thursday, 28th September 2023 12:54:49 pm       |
| Author: Stewart Charles McDowall                              |
| Email: s.c.mcdowall@cml.leidenuniv.nl                         |
| Github: Stew-McD                                              |
| Company: CML, Leiden University                               |
|---------------------------------------------------------------|
| Last Modified: Friday, 29th September 2023 11:52:43 am        |
| Modified By: Stewart Charles McDowall                         |
| Email: s.c.mcdowall@cml.leidenuniv.nl                         |
|---------------------------------------------------------------|
|License: The Unlicense                                         |
|===============================================================|
'''

import os
import re
import bw2data as bd
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count

from user_settings import dir_tmp, dir_data, filters

num_cpus = int(os.getenv('SLURM_CPUS_PER_TASK', default=cpu_count()))

def worker(args):
    return GetActivities(*args)

def GetActivitiesMP(database_names, project_name, title):
    args_list = [(database_name, project_name, title) for database_name in database_names]
    
    with Pool(num_cpus) as pool:
        pool.map(worker, args_list)

def GetActivities(database_name, project_name, title):

    print("\n** Getting activities list from database:", database_name)
    # set the project
    bd.projects.set_current(project_name)
    db = bd.Database(database_name)

    acts_all = pd.DataFrame([x.as_dict() for x in db])
    
    if 'activity type' not in acts_all.columns:
        acts_all.rename(columns={'type': 'activity type'}, inplace=True)
    
    columns = [
        'code',
        'name',
        'unit',
        'location',
        'activity type',
        'reference product',
        'classifications',
        'database',
        # 'production amount',
        'price'
    ]
    try:
        acts_all = acts_all[columns]
    except KeyError as e:
        missing_columns = re.findall(r"\'(.*?)\'", str(e))
        columns = list(set(columns) - set(missing_columns))
        print(f"\n\t** Warning: Column(s) '{missing_columns}' not found in DataFrame '{database_name}'. Removing from selection. **\n")
        acts_all = acts_all[columns]
    
    print(f"\t{database_name}: # of activities before filtering:", len(acts_all))
    
    # pull out and look at the categories 
    print("\t* Extracting classification data")
    # Define the function to extract values
    def extract_values(row):
        isic_num, isic_name, cpc_num, cpc_name = '', '', '', ''
        
        if isinstance(row["classifications"], list):
            for classification in row["classifications"]:
                if "ISIC" in classification[0]:
                    split_values = classification[1].split(":")
                    isic_num = int(split_values[0].strip())
                    isic_name = ":".join(split_values[1:]).strip()
                elif "CPC" in classification[0]:
                    split_values = classification[1].split(":")
                    cpc_num = int(split_values[0].strip())
                    cpc_name = ":".join(split_values[1:]).strip()

        return pd.Series([isic_num, isic_name, cpc_num, cpc_name], index=['ISIC_num', 'ISIC_name', 'CPC_num', 'CPC_name'])

    # Use the apply method to apply the function row-wise and join the results to the original DataFrame
    acts_all[['ISIC_num', 'ISIC_name', 'CPC_num', 'CPC_name']] = acts_all.apply(extract_values, axis=1)
    
    # acts_all.replace("", np.nan, inplace=True)
    acts_all['CPC_num'] = acts_all['CPC_num'].replace('',-1).astype(int)
    acts_all['ISIC_num'] = acts_all['ISIC_num'].replace('',-1).astype(int)
    # Drop the original "classifications" column
    acts_all = acts_all.drop("classifications", axis=1)

    print(f'** {database_name} **' )
    # Extracting and processing unique ISIC names
    isic_names = acts_all["ISIC_name"].unique()
    isic_names = [name for name in isic_names if isinstance(name, str) and name]
    isic_names.sort()
    print("\t# of ISIC categories:", len(isic_names))

    # Extracting and processing unique CPC names
    cpc_names = acts_all["CPC_name"].unique().tolist()
    cpc_names = [name for name in cpc_names if isinstance(name, str) and name]
    cpc_names.sort()
    print("\t# of CPC categories:", len(cpc_names))

    # Filter activities in database to select those of interest
    print("\t* Filtering activities")

    acts = filter_dataframe(acts_all, filters)
    
    print("\t# of activities after filtering:", len(acts))

    # look at the categories in the activities
    isic = acts.ISIC_name.unique()
    isic.sort()
    print("\t# of ISIC categories after filtering:", len(isic))
    cpc =  acts.CPC_name.unique().tolist()
    cpc.sort()
    print("\t# of CPC categories after filtering:", len(cpc))

# assign product categories and sub-categories based on CPC and ISIC codes
    acts["prod_category"] = ""
    acts["prod_sub_category"] = ""
    for i, j in acts.iterrows():

        cpc = str(acts.at[i, "CPC_num"])
        if len(cpc) < 5:
            cpc += "0"*(5-len(cpc))
        cpc = int(cpc)

        if (cpc in range(0,2000) or cpc in range (3000, 4000)):
            acts.at[i, "prod_category"] = "AgriForeAnim"
            acts.at[i, "prod_sub_category"] = "Agricultural and forestry products"
        if (cpc in range(2000,3000) or cpc in range (4000, 5000)):
            acts.at[i, "prod_category"] = "AgriForeAnim"
            acts.at[i, "prod_sub_category"] = "Live animal, fish and their products"
        if cpc in range(11000,18000):
            acts.at[i, "prod_category"] = "OreMinFuel"
            acts.at[i, "prod_sub_category"] = "Ores, minerals & fuels"
        if cpc in range(18000,19000):
            acts.at[i, "prod_category"] = "Chemical"
            acts.at[i, "prod_sub_category"] = "Chemical products"
        if cpc in range(21000,24000):
            acts.at[i, "prod_category"] = "ProcBio"
            acts.at[i, "prod_sub_category"] = "Food & beverages, animal feed"
        if cpc in range(26000,28200):
            acts.at[i, "prod_category"] = "ProcBio"
            acts.at[i, "prod_sub_category"] = "Textile"
        if cpc in range(31000,32000):
            acts.at[i, "prod_category"] = "ProcBio"
            acts.at[i, "prod_sub_category"] = "Wood, straw & cork"
        if cpc in range(32000,33000):
            acts.at[i, "prod_category"] = "ProcBio"
            acts.at[i, "prod_sub_category"] = "Pulp & paper"
        if cpc in range(33000,34000):
            acts.at[i, "prod_category"] = "OreMinFuel"
            acts.at[i, "prod_sub_category"] = "Ores, minerals & fuels"
        if cpc in range(34000,36000):
            acts.at[i, "prod_category"] = "Chemical"
            acts.at[i, "prod_sub_category"] = "Chemical products"
        if cpc in range(34700,34800):
            acts.at[i, "prod_category"] = "PlastRub"
            acts.at[i, "prod_sub_category"] = "Plastics & rubber products"
        if cpc in range(35500,37000):
            acts.at[i, "prod_category"] = "PlastRub"
            acts.at[i, "prod_sub_category"] = "Plastics & rubber products"
        if cpc in range(37000,38000):
            acts.at[i, "prod_category"] = "GlasNonMetal"
            acts.at[i, "prod_sub_category"] = "Glass and other non-metallic products"
        if cpc in range(39000,40000):
            acts.at[i, "prod_category"] = "AgriForeAnim"
            acts.at[i, "prod_sub_category"] = "Agricultural and forestry products"
        if cpc in range(40000,42000):
            acts.at[i, "prod_category"] = "MetalAlloy"
            acts.at[i, "prod_sub_category"] = "Basic metals & alloys, their semi-finished products"
        if cpc in range(42000,43000):
            acts.at[i, "prod_category"] = "ProcBio"
            acts.at[i, "prod_sub_category"] = "Food & beverages, animal feed"
        if cpc in range(43000,49000):
            acts.at[i, "prod_category"] = "MachElecTrans"
            acts.at[i, "prod_sub_category"] = "Metal/electronic equipments and parts"
        if cpc in range(49000,49400):
            acts.at[i, "prod_category"] = "MachElecTrans"
            acts.at[i, "prod_sub_category"] = "Transport vehicles"
        if cpc in range(49000,49915):
            acts.at[i, "prod_category"] = "MachElecTrans"
            acts.at[i, "prod_sub_category"] = "Transport vehicles"
        if cpc in range(49941,50000):
            acts.at[i, "prod_category"] = "MachElecTrans"
            acts.at[i, "prod_sub_category"] = "Metal/electronic equipments and parts"
        if cpc in range(60000,70000):
            acts.at[i, "prod_category"] = "OreMinFuel"
            acts.at[i, "prod_sub_category"] = "Ores, minerals & fuels"
        if cpc == 38100: #wooden furniture
            acts.at[i, "prod_category"] = "ProcBio"
            acts.at[i, "prod_sub_category"] = "Wood, straw & cork"
        if cpc == 38450: #fishing stuff
            acts.at[i, "prod_category"] = "ProcBio"
            acts.at[i, "prod_sub_category"] = "Textile"

    # save to a file for each database
    f = dir_tmp / f"activities_list_from_{db.name}_{title}.csv"
    print("\n\tSaved activities list to csv:\n\t", f)
    acts.to_csv(f, sep=";", index=False)
    

def MergeActivities(database_names, project_name, title):

    print("\n** Merging activities lists from all selected databases **\n\t" + '\n\t'.join(database_names))
    
    files = [f"activities_list_from_{x}_{title}.csv" for x in database_names]
    database_names_string = "_".join(database_names)
    
    paths = [dir_tmp / f for f in files]

    df_merged = pd.read_csv(paths[0], sep=';')
    df_merged = df_merged.reset_index(drop=True)

    if len(paths) > 1:
        for f in paths[1:]:
            df = pd.read_csv(f, sep=';')
            df_merged = pd.concat([df_merged, df], axis=0, ignore_index=True)
    
        
    file_name = dir_data / f"activities_list_merged_{project_name}_{title}.csv"
    df_merged.to_csv(file_name, sep=';', index=False)
    print("\nSaved combined activities list to csv:\n\t", file_name)
    
    return


def filter_dataframe(df, filters):
    conditions = []
    # Name condition
    if filters["names"]:
        conditions.append(df['name'].str.contains('|'.join(filters["names"])))
    
    # Location condition
    if filters["locations"]:
        conditions.append(df['location'].isin(filters["locations"]))
    
    # Unit condition
    if filters["units"]:
        conditions.append(df['unit'].isin(filters["units"]))

    # Exclude CPC condition
    if filters["exclude"]:
        conditions.append(~df['CPC_name'].str.contains('|'.join(filters["exclude"])).fillna(False))

    # Exclude ISIC condition
    if filters["exclude"]:
        conditions.append(~df['ISIC_name'].str.contains('|'.join(filters["exclude"])).fillna(False))
    
    # CPC number condition
    if "CPC_num" in filters and filters["CPC_num"]:
            if not all(isinstance(num, (int)) for num in filters["CPC_num"]):
                raise ValueError("All CPC numbers should be of type int")
            conditions.append(df['CPC_num'].isin(filters["CPC_num"]))
            
    # ISIC number condition
    if "ISIC_num" in filters and filters["ISIC_num"]:
            if not all(isinstance(num, (int)) for num in filters["ISIC_num"]):
                raise ValueError("All ISIC numbers should be of type int")
            conditions.append(df['ISIC_num'].isin(filters["ISIC_num"]))

    # Combine conditions using logical AND
    combined_condition = conditions[0]
    for condition in conditions[1:]:
        combined_condition &= condition

    return df[combined_condition].reset_index(drop=True)



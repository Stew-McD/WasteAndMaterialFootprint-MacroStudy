#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 23:30:57 2023

@author: stew
"""
import pandas as pd
from datetime import datetime, timedelta
import os
from multiprocessing import cpu_count
from pathos.multiprocessing import ProcessingPool as Pool


import bw2data as bd
import bw2calc as bc

from user_settings import methods, database_names, dir_tmp, dir_logs

num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', os.environ.get('SLURM_JOB_CPUS_PER_NODE', cpu_count())))


def LCIA(activities_list, project_name, title, limit=None):

    start = datetime.now()
# read list of activities to be calculated from csv produced by MergeActivities()
    print(f'\n {"="*80}')
    print('   \t\t\t*** Starting LCA calculations ***')
    print(f'{"="*80}\n')
    print(f"\n** Reading activities list from file \n   {activities_list}")
    data = pd.read_csv(activities_list, sep=";")
    data.drop(columns=['ISIC_num', 'CPC_num'], inplace=True)
    
    bd.projects.set_current(project_name)
    db_names = list(data.database.unique())
    
    estimated_time_minutes = len(data) * len(methods) * 0.00028
    finish_time = start + timedelta(minutes=estimated_time_minutes)

    if finish_time.date() == datetime.now().date():
        display_time = f"{finish_time.strftime('%H:%M')} today"
    elif finish_time.date() == (datetime.now() + timedelta(days=1)).date():
        display_time = f"{finish_time.strftime('%H:%M')} tomorrow"
    else:
        # For dates further in the future, you might want to display the actual date
        display_time = finish_time.strftime('%Y-%m-%d %H:%M')
    
    print(f"""
 
    * Using {len(database_names)} databases
       in project: {project_name}

    * Using packages:
    - bw2data: {bd.__version__}
    - bw2calc: {bc.__version__}

    * Number of activities to be calculated: {len(data)}
    * Number of methods to be used: {len(methods)}
    * Total number of calculations: {len(data) * len(methods)}
    
    * Estimated calculation time: {round(estimated_time_minutes, 2)} minutes
    * Should be finished at {display_time}
    """)
    print(f'{"="*80}\n')
    
        
        
    def LCIA_singledatabase(args):
        i, db_name = args
        db = bd.Database(db_name)
# Select activities from the correct database
        acts = data[data.database == db.name]
        acts = acts.reset_index(drop=True)
        
# For testing, if you want, you can set a limit on the number of activities to be calculated
        print(f'**********   db_name: {db_name} *********')
        try:
            if limit is not None:
                print(f"\n=== Limiting number of activities calculated to: {limit} ===")
                acts = acts.sample(n=limit)
                acts = acts.reset_index(drop=True)
        except NameError:
            limit = len(acts)
            print(f"Limit not defined, all {limit} activities will be calculated")

        
# Start LCA calculations

        print("\n*** Starting LCA calculations ***")
        print("\n * Using db:", db_name, "\nin project:", project_name,
            "\n\n * Using packages: ",
            "\n\t bw2data" ,bd.__version__,
            "\n\t bw2calc" , bc.__version__)
        print("\n * Number of activities to be calculated:", len(acts))
        print(" * Number of methods to be used:", len(methods))
        print(" * Total number of calculations:", len(acts)*len(methods))
        print(" * Estimated calculation time:", round(len(acts)*len(methods)*0.00028, 2), " minutes") 
        
        # 0.00028 minutes per calculation (on my machine)
        print("\n * Using methods: ")
        for method in sorted(set([x[0] for x in methods])): 
            print("\t", method)
        
        # Run first calculation
        try:
            act = db.get(acts.code.sample().values[0])
            lca = bc.LCA({act : 1}, method=methods[0])
            lca.lci()
            lca.lcia()
            
        except NameError:
            print("Error: 'lca' or 'act' not defined")
            exit(0)

        # Repeat calculations for the rest of the activities
        print("\nRunning 'lca.redo_lci & lca.redo_lcia'...\n")
        print(" (printing only scores above 1)")

        results_dic = {}
        for j, act in acts.iterrows():
            code = act.code
            print(code + " : " + act['name'])

        # repeat calculations for each activity
            lca.redo_lci()
            act = db.get(code)
            dic = act.as_dict()

        # remove unnecessary keys from dictionary
            cut = [
                'comment',
                'classifications',
                'activity',
                'filename',
                'synonyms',
                'parameters',
                'authors',
                'type',
                'flow',
                'production amount'
            ]
            for c in cut:
                try:
                    dic.pop(c)
                except KeyError:
                    pass
                
            # repeat calculations for each method
            for k, m in enumerate(methods):
                lca.switch_method(m)
                lca.redo_lcia({act : 1})

                score = lca.score
                #top_acts = lca.top_activities()

                dic.update({lca.method[2] : score})
                #dic.update({"top_activities_"+lca.method[2] : top_acts})

                # print info about calculations
                if score > 1:
                    print(f"{db_name} {i+1}/{len(db_names)} "
                        f"Act. {j+1: >2} of {len(acts): <2} "
                        f"Met. {k+1: >2} of {len(methods):<2} |"
                        f" Score: {score:.1e}:"
                        f" {m[2].split('_')[-1]: <10}/ {dic['unit']: <10}\t |"
                        f" '{dic['name']: <10}' |"
                        f"  with method: {lca.method[2]: <5}")

                results_dic.update({dic["code"]: dic})
                    
        finish = datetime.now()
        duration = finish - start
        outcome = (f"\n\n** {finish.strftime('%Y-%m-%d %H:%M:%S')} -- * Completed {len(acts)*len(methods)} LCIA calculations: " f"{len(acts)} activities and {len(methods)} methods from {db_name} in: {str(duration).split('.')[0]}")
        print(outcome)

        # write to log file
        log_file_path = dir_logs / f"{db_name}_{title}_log.txt"
        with open(log_file_path, "a") as f:
            f.write(outcome)
                
                
    args_list = [(i, database_name) for i, database_name in enumerate(database_names)]
    # Use multiprocessing to parallelize the work
    with Pool(num_cpus) as pool:
        pool.map(LCIA_singledatabase, args_list)

    # print info about the combined calculations for the db
    finish = datetime.now()
    duration = finish - start
    outcome = (f"\n\n** {finish.strftime('%Y-%m-%d %H:%M:%S')} -- * Completed {len(data)*len(methods)} LCIA calculations: "f"{len(data)} activities and {len(methods)} methods from \n {print(*database_names)} \n in: {str(duration).split('.')[0]}")
    
    print(outcome)

    # write to log file
    log_file_path = dir_logs / f"{title}_log.txt"
    with open(log_file_path, "a") as f:
        f.write(outcome)

    # convert results dictionary to dataframe and transpose
    results_df = pd.DataFrame.from_dict(results_dic).T

    # save individual db results as pickle
    pickle_path = dir_tmp / f"{db_name}_{title}_rawresults_df.pickle"
    results_df.to_pickle(pickle_path)

    # save individual db results as csv
    csv_path = dir_tmp / f"{db_name}_{title}_rawresults_df.csv"
    results_df.to_csv(csv_path, sep=";")

    
    return


def MergeResults(project_name, title):
    print(f"\n** Merging results from :\n {project_name} : {title}")
    
    # get paths to individual results
    files = [dir_tmp / f"{db_name}_{title}_rawresults_df.csv" for db_name in database_names]
    
    # merge results from multiple databases
    df_merged = pd.read_csv(files[0], sep=';')
    if len(files) > 1:
        for f in files[1:]:
            df = pd.read_csv(f, sep=';')
            df_merged = pd.concat([df_merged, df], axis=0, ignore_index=True)
    df_merged = df_merged.reset_index(drop=True)
    df_merged.drop("Unnamed: 0", axis=1, inplace=True)
    
    # save combined results as pickle and csv
    combined_raw_csv = dir_tmp / f"{title}_combined_rawresults_df.csv"
    combined_raw_pickle = dir_tmp / f"{title}_combined_rawresults_df.pickle"
    df_merged.to_csv(combined_raw_csv, sep=';', index=False)
    df_merged.to_pickle(combined_raw_pickle)
    print(f"\n*Saved combined activities list \n\tto csv: {combined_raw_csv}\n\tand pickle: {combined_raw_pickle}")

    
    return combined_raw_csv, combined_raw_pickle

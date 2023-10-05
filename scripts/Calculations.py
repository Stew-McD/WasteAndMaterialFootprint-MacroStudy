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
from tqdm import tqdm
from termcolor import colored
from multiprocessing import Lock
import bw2data as bd
import bw2calc as bc
from time import sleep
from user_settings import methods, database_names, project_name, limit, verbose, dir_tmp, dir_logs, activities_list, title

num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', os.environ.get('SLURM_JOB_CPUS_PER_NODE', cpu_count())))
print_lock = Lock()

def LCIA():

    start = datetime.now()
# read list of activities to be calculated from csv produced by MergeActivities()
    print(f'\n {"-"*80}')
    print('   \t\t\t*** Starting LCA calculations ***')
    print(f'{"-"*80}\n')
    print(f"** Reading activities list from file \n   {activities_list}")
    data = pd.read_csv(activities_list, sep=";")
    data.drop(columns=['ISIC_num', 'CPC_num'], inplace=True)
    
    bd.projects.set_current(project_name)
    
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
    * Running calculations in {len(database_names)} databases in project: {project_name}
       
    * Using packages:
    - bw2data: {bd.__version__}
    - bw2calc: {bc.__version__}

    * Number of activities to be calculated: {len(data)}
    * Number of methods to be used: {len(methods)}
    * Total number of calculations: {len(data) * len(methods)}
    
    * Estimated calculation time: {round(estimated_time_minutes, 2)} minutes
    * Should be finished at {display_time}
    """)
    print('Using methods:')
    
    for method in sorted(set([x[0] for x in methods])): 
        print("\t", method)
    
    if limit:
        print(f"\n %%% Limiting number of activities calculated per database to: {limit} %%%")
    print(f'\n{"="*80}\n')
    
    args_list = [(i, database_name, data, project_name, limit) for i, database_name in enumerate(database_names)]
    # Use multiprocessing to parallelize the work
    
    with Pool(num_cpus) as pool:
        pool.map(LCIA_singledatabase, args_list)
        pool.close()  # Close the pool to any new tasks
        pool.join()   # Wait for all worker processes to finish
        
        # print info about the combined calculations for the db
        finish = datetime.now()
        duration = finish - start
        outcome = (f"\n\n** {finish.strftime('%Y-%m-%d %H:%M:%S')} -- * Completed {len(data)*len(methods)} LCIA calculations: "
            f"{len(data)} activities and {len(methods)} methods in: {duration.total_seconds()} seconds")

        with print_lock:
            print(outcome, flush=True)

        # write to log file
        log_file_path = dir_logs / f"{title}_log.txt"
        try:
            with open(log_file_path, "a") as f:
                f.write(outcome)
        except Exception as e:
            print(f"Error writing to log file: {e}")


    return

def LCIA_singledatabase(args):
    
    i, database_name, data, project_name, limit  = args
    if verbose:
        print(f'**********   database_name: {database_name} *********')
    start_single = datetime.now()
    
    db = bd.Database(database_name)
# Select activities from the correct database
    acts = data[data.database == db.name]
    acts = acts.reset_index(drop=True)
    
    
# For testing, if you want, you can set a limit on the number of activities to be calculated
    try:
        acts = acts.sample(n=limit)
        acts = acts.reset_index(drop=True)
        if verbose:
            print(f"Limiting number of activities calculated to: {limit}")
    except NameError:
        limit = len(acts)
        if verbose:
            print(f"Limit not defined, all {limit} activities will be calculated")

# This progress bar tracks the processing of each activity-method pair for this worker
    if not verbose:
        desc = f"\t {i:02} - {database_name}"
        # Make the description a fixed width 
        formatted_desc = "{:<80.80}".format(desc)
        # Colorize the description
        colorized_desc = colored(formatted_desc, 'white')

        bar_format = "{l_bar}\033[95m{bar:40}\033[0m| {n}/{total} - [{percentage:3.0f}%]"
        pbar = tqdm(total=len(acts), leave=True, mininterval=0.5, 
                    position=i, desc=colorized_desc, ncols=140, bar_format=bar_format)
        
# Start LCA calculations (around 0.00028 minutes per calculation (on my machine))
    if verbose:
        print(
            "\n*** Starting LCA calculations ***",
            f"\n * Using db: {database_name} in project: {project_name}",
            f"\n\n * Using packages:",
            f"\n\tbw2data {bd.__version__}",
            f"\n\tbw2calc {bc.__version__}",
            f"\n * Number of activities to be calculated: {len(acts)}",
            f" * Number of methods to be used: {len(methods)}",
            f" * Total number of calculations: {len(acts) * len(methods)}",
            f" * Estimated calculation time: {round(len(acts) * len(methods) * 0.00028, 2)} minutes",
            sep=''
        )
 
    # Run first calculation
    try:
        act = db.get(acts.code.sample().values[0])
        lca = bc.LCA({act : 1}, method=methods[0])
        lca.lci()
        lca.lcia()
        
    except Exception as e:
        print(f"Error {e}: 'lca' or 'act' not defined")

    # Repeat calculations for the rest of the activities
    if verbose:
        print("\nRunning 'lca.redo_lci & lca.redo_lcia'...\n")
        print(" (printing only scores above 1)")

    results_dic = {}
    for j, act in acts.iterrows():
        code = act.code
        if verbose: 
            print(database_name, code + " : " + act['name'])

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
            if verbose:
                if abs(score) > 1:
                    print(f"{database_name} {i+1}/{len(database_names)} "
                        f"Act.{j+1: >2}/{len(acts): <2} "
                        f"Met.{k+1: >2}/{len(methods): <2} |"
                        f" Score: {score:.1e}:"
                        f" {m[2].split('_')[-1]: <10}/ {dic['unit']: <10}\t |"
                        f" '{dic['name']: <10}' |"
                        #f"  with method: {lca.method[2]: <5}"
                        )
                    
            results_dic.update({dic["code"]: dic}) # add results to dictionary
        
        pbar.update(1) # update progress bar
    
    # get information for logs
    finish = datetime.now()
    duration = finish - start_single
    outcome = (f"\n\n** {finish.strftime('%Y-%m-%d %H:%M:%S')} -- * Completed {len(acts)*len(methods)} LCIA calculations: " f"{len(acts)} activities and {len(methods)} methods from {database_name} in: {str(duration).split('.')[0]}")
    
    if verbose:
        print(outcome)

    # write to log file
    log_file_path = dir_logs / f"{database_name}_{title}_log.txt"
    with open(log_file_path, "a") as f:
        f.write(outcome)
            
    # convert results dictionary to dataframe and transpose
    results_df = pd.DataFrame.from_dict(results_dic).T

    # save individual db results as pickle
    pickle_path = dir_tmp / f"{database_name}_{title}_rawresults_df.pickle"
    results_df.to_pickle(pickle_path)

    # save individual db results as csv
    csv_path = dir_tmp / f"{database_name}_{title}_rawresults_df.csv"
    results_df.to_csv(csv_path, sep=";")
    
    pbar.close() # close progress bar
       
    return

def MergeResults(project_name, title):
    print(f"\n** Merging results from :\n {project_name} : {title}")
    
    # get paths to individual results
    files = [dir_tmp / f"{database_name}_{title}_rawresults_df.csv" for database_name in database_names]
    
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

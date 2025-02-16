#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contributions_Graph_Traversa
l
Created on Sun Nov 27 10:29:02 2022
@author: SC-McD
"""

# %% 0. IMPORTS

import os
import sys
import json
import pickle
import pandas as pd
from datetime import datetime

import bw2calc as bc
import bw2data as bd
import bw2analyzer as ba

from SankeyGraph import Graph

# %% 1. Setup 
method_keyword = "Waste Footprint"
title = 'market_selection'
data_dir = os.getcwd() + "/data/"
tmp = data_dir +"tmp/"
top_acts_dir = data_dir + "/Contributions_GT/"
if not os.path.isdir(top_acts_dir): os.mkdir(top_acts_dir)

# %% 2. Load lists of top activities from previous calculations

f = [x for x in os.listdir(tmp) if "combined_topactivities_df.pickle" in x][0]
df = pd.read_pickle(tmp+f).reset_index(drop=True)
db_names = df.database.unique()

# Iterate over databases
def GraphTraversalCalculations():
    for h, db_name in enumerate(db_names):
        project_name = "WasteFootprint_"+db_name
        acts = df[df.database == db_name]
        acts = acts.drop_duplicates(subset=['code']).reset_index(drop=True).sort_values(by=['name'], ascending=True)

        bd.projects.set_current(project_name)
        db = bd.Database(db_name)

        methods_waste = []
        for method in list(bd.methods):
            if method_keyword == method[0]:
                methods_waste.append(method)

        #methods = [x for x in methods_waste if x[1] == "waste_total_combined" or x[1] == "waste_hazardous_combined"]
        methods = sorted(methods_waste)
        print("\n\n***** Starting LCA GraphTraversal calculations *****")
        print("\n** Using methods:")
        print(*methods, sep = "\n")
        print("\n** Using activities:", )
        print(*acts["name"].values, sep = "\n")
        results_list = []
    # %% 2. RUN LCA CALCULATIONS AND GRAPH TRAVERSAL
        
        limit = acts.shape[0]
        start = datetime.now()
        cut_off = 0.02
        max_calcs = 2000
    # iterate over activities
        for i, code in enumerate(acts.code):
            if i < limit:    
                act_object = db.get(code)
                dic = act_object.as_dict()
                print("\n** Database: ",db_name,h+1,"/",len(db_names), "\t* cutoff = ", cut_off, "\t* max_calcs = ", max_calcs)
                print("*** Activity:", i+1,"/",limit,",", act_object.as_dict()["name"], "," , code)
                print("*** Top for: ", acts[acts['code'] == code]['top'].values, 'in category:', acts[acts['code'] == code]['prod_category'].values)
                print("**** Duration: "+ str(datetime.now()-start)+"\n")
                
    # iterate over methods
                for j, m in enumerate(methods):
                    
                    fu = {act_object : 1}

                    
                    try:
                        results = bc.GraphTraversal().calculate(
                        fu, m, cutoff=cut_off, max_calc=max_calcs)
                        print("\tMethod:",j+1,"/",len(methods),": ","\t{:.2e} kg/kg".format(results["lca"].score),":\t", m[2])
                        ab = Graph()
                        ab.new_graph(results)
                        results_json = ab.json_data
                        
# save json data as individual files
                        file_path = top_acts_dir+"{}-{}-{}-sankeydata.json".format(db_name, code, m[2])
                        json_file = json.dumps(results_json)
                        with open(file_path, "w") as f:
                            f.write(json_file)
                        
                        results_list.append(json_file)
                            
                    except Exception as e:
                        print("*X* oh no! for:",m[2], ",", str(e))
                        pass
            
    # save all results as a pickle 
        with open(os.path.join(data_dir, "GraphTraversalJSONS.pickle"), "wb") as f:
            pickle.dump(results_list, f)         
                    

    # print info about the combined calculations for the db
        finish = datetime.now()
        duration = finish - start
        outcome = ("\n\n** {} -- * Completed {} LCIA Contribution GraphTraversal calculations: in: {} with databases {}".format(finish.strftime("%Y-%m-%d %H:%M:%S"), db_names, i*len(db_names)*len(methods), str(duration).split(".")[0]))
        print(outcome)
            
    # write to log file
        with open(os.path.join(top_acts_dir, title+"_GraphTraversal_log.txt"), "a") as f:
            f.write(outcome)
    return

GraphTraversalCalculations()
# %% 3. LOAD JSON FILES TO PROCESS
def LoadJSONs():
    
    activities_file = [x for x in os.listdir(tmp) if "combined_topactivities_df.pickle" in x][0]
    df = pd.read_pickle(tmp+activities_file).reset_index(drop=True)
    db_names = df.database.unique()
    db_name = "cutoff391"

    json_dir = os.path.join(os.getcwd(), "data/Contributions_GT")
    jsons = [x for x in os.listdir(json_dir) if db_name in x]
    jsons = [x for x in jsons if ""]

    json_file = os.path.join(json_dir, jsons[0])
    with open(json_file) as json_file:
        data = json.load(json_file)
    return

# # %% 5. SET LAYOUT DICTIONARY
# colour_list = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
#                 'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
#                 'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
#                 'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
#                 'rgb(188, 189, 34)', 'rgb(23, 190, 207)',
#                 'rgb(141,211,199)','rgb(255,255,179)',
#                 'rgb(190,186,218)','rgb(251,128,114)',
#                 'rgb(128,177,211)','rgb(253,180,98)',
#                 'rgb(179,222,105)','rgb(252,205,229)',
#                 'rgb(217,217,217)','rgb(188,128,189)',
#                 'rgb(204,235,197)','rgb(255,237,111)']

# from webcolors import rgb_to_name

# c_list = [x.replace('rgb(', '').replace(")","") for x in colour_list]




# from scipy.spatial import KDTree
# from webcolors import CSS3_HEX_TO_NAMES, hex_to_rgb

# def convert_rgb_to_names(rgb_tuple):

#     # a dictionary of all the hex and their respective names in css3
#     css3_db = CSS3_HEX_TO_NAMES
#     names = []
#     rgb_values = []
#     for color_hex, color_name in css3_db.items():
#         names.append(color_name)
#         rgb_values.append(hex_to_rgb(color_hex))

#     kdt_db = KDTree(rgb_values)
#     distance, index = kdt_db.query(rgb_tuple)
#     return f'closest match: {names[index]}'

# c_names = {}
# for c in c_list:
#     n = convert_rgb_to_names(tuple(eval(c)))
#     c_names.update({c:n})

# # %% 6. MAKE SANKEY
# def sankey(json_data):
#     import plotly.graph_objects as go
#     import plotly.io as pio
#     pio.renderers.default = 'browser'

#     ed = pd.DataFrame()
#     """" GET DATA FROM DATA DICT """
#     #ed['edge_amounts'] = [edge['amount'] for edge in json_data['edges']]
#     #ed['edge_impacts'] = [edge['impact'] for edge in json_data['edges']]
#     ed['edge_ind_norm'] = [edge['ind_norm'] for edge in json_data['edges']]
#     ed['edge_products'] = [edge['product'] for edge in json_data['edges']]
#     ed['edge_from'] = [edge['source_id'] for edge in json_data['edges']]
#     ed['edge_to'] = [edge['target_id'] for edge in json_data['edges']]
#     #ed['edge_name'] = [edge['location'] for edge in json_data['edges']]


#     #ed['edge_tooltips'] = [edge['tooltip'] for edge in json_data['edges']]
#     #ed['edge_units'] = [edge['unit'] for edge in json_data['edges']]

#     no = pd.DataFrame()
#     #no['node_amounts'] = [node['amount'] for node in json_data['nodes']]
#     #no['node_impacts'] = [node['cum'] for node in json_data['nodes']]
#     #no['node_impacts_norm'] = [node['cum_norm'] for node in json_data['nodes']]
#     no['node_names'] = [node['name'] for node in json_data['nodes']]
#     #no['node_products'] = [node['product'] for node in json_data['nodes']]
#     no['edge_to'] = [node['id'] for node in json_data['nodes']]


#     df = pd.merge(no, ed, on="edge_to", how="inner")

#     """ SOURCE AND TARGET IDS FROM BW2 MUST BE CONVERTED FROM 32 BIT HEXADECIMAL TO INTEGERS! """
#     m = (10**43)  # because the numbers were to big (computer says no) - not ideal. better with some kind of key matching from index
#     ed['edge_from_int'] = [int(x, 32)//m for x in ed.edge_from]
#     ed['edge_to_int'] = [int(x, 32)//m for x in ed.edge_to]

#     df.edge_to = [int(x, 32)//m for x in df.edge_to]
#     df.edge_from = [int(x, 32)//m for x in df.edge_from]
#     #df['from= [int(x, 32)//m for x in df.edge_from_x]


#     # MAKE THE SANKEY THING. FINALLY...
#     fig = go.Figure(data=[go.Sankey(

#         valueformat=".3f",
#         valuesuffix=" "+json_data["nodes"][0]["LCIA_unit"],

#         # define nodes
#         node=dict(
#                   #pad=15,
#                   #thickness=25,
#                   #line=dict(color="black", width=0.5),
#                   label=df.node_names,
#                   #color="white",
#                   #customdata = ed.edge_products,
#                   #hovertemplate='Node %{customdata} has total value %{value}<extra></extra>',

#                   ),

#         # add links
#         link=dict(
#             source=df.edge_from,
#             target=df.edge_to,
#             value=df.edge_ind_norm,        #
#             #label=ed.edge_products,
#             #customdata = [ed.edge_products],
#             #hovertemplate='%{value}<extra></extra> from: %{target.customdata} to %{customdata}',
#             #color =  colour_list[-len(ed.edge_products):]
#             #, hovertemplate=edge_tooltips
#         ))])

#     # UPDATE LAYOUT
#     tit = act.as_dict()["name"]
#     tit_sub = json_data['title']
#     #fig.update_layout(title_text="Waste demand flows for: {}".format(title_text), font_size=16)
#     #fig.update_layout(title_text=fig.update_layout(title_text="Waste demand flows for: % {tit} <br><sup> % {tit_sub}</sup>", font_size=16))
#     fig.update_layout(paper_bgcolor="rgb(0,0,0,0.2)")


#     # SAVE FIGURE AND SHOW IN BROWSER
#     name = "test"
#     unit_long = "unit"
#     figures = os.path.join(os.getcwd(), "figures")
#     name_file = name.replace(".", "").replace(" ", "").lower()

#     sank_dir = figures + '/sankeys/'
#     if not os.path.isdir(sank_dir): os.makedirs(sank_dir)

#     fig.write_html(
#         sank_dir + "WasteDemand_Sankey_{}_{}.html".format(unit_long, name_file))
#     fig.write_image(
#         sank_dir + "WasteDemand_Sankey_{}_{}.svg".format(unit_long, name_file))
#     fig.show()

# ab = Graph()
# ab.new_graph(data)
# json_data = ab.json_data
# sankey(json_data)
# # %% WRITING THE SANKEYS DOES NOT SEEM TO WORK WELL

# import string
# import random
# lowercase_letters = string.ascii_lowercase
# def lowercase_word(): #The function responsible for generating the word
#     word = '' #The variable which will hold the random word
#     random_word_length = random.randint(1,10) #The random length of the word
#     while len(word) != random_word_length: #While loop
#         word += random.choice(lowercase_letters) #Selects a random character on each iteration
#     return word #Returns the word
# random_word = lowercase_word()

# labs = []
# for x in range(len(ed["edge_products"])):#range(len(ed.edge_products)):
#     w = lowercase_word()
#     labs.append(w)

# import plotly.graph_objects as go
# import plotly.io as pio
# pio.renderers.default = 'browser'

# """" GET DATA FROM DATA DICT """

# # MAKE THE SANKEY THING. FINALLY...
# fig = go.Figure(data=[go.Sankey(

#     #valueformat=".2f",
#     #valuesuffix=data["nodes"][0]["LCIA_unit"],

#     # define nodes
#     node=dict(
#               #pad=15,
#               #thickness=25,
#               #line=dict(color="black", width=0.5),
#               #label=list(ed.edge_from_int)[0:5]
#               # customdata = no.node_names,
#               # hovertemplate='Node %{customdata} has total value %{value}<extra></extra>'
#               #customdata=node_products,
#               ),

#     # add links
#     link=dict(
#         source=list(ed.edge_to_int)[0:5],
#         target=list(ed.edge_from_int)[0:5],
#         value=list(ed.edge_impacts)[0:5]      #

#         #customdata = [ed.edge_products],
#         #hovertemplate='%{value}<extra></extra> from: %{target.customdata} to %{customdata}',
#         #color =  colour_list[-len(ed.edge_products):]
#         #, hovertemplate=edge_tooltips
#     ))])

# # UPDATE LAYOUT
# tit = act.as_dict()["name"]
# tit_sub = json_data['title']
# #fig.update_layout(title_text="Waste demand flows for: {}".format(title_text), font_size=16)
# #fig.update_layout(title_text=fig.update_layout(title_text="Waste demand flows for: % {tit} <br><sup> % {tit_sub}</sup>", font_size=16))
# # fig.update_layout(paper_bgcolor="rgb(0,0,0,0.1)")


# # SAVE FIGURE AND SHOW IN BROWSER
# # name = "test"
# # unit_long = "unit"
# # figures = os.path.join(os.getcwd(), "figures") %% 2. RUN LCA CALCULATION AND GRAPH TRAVERSAL



# #%%


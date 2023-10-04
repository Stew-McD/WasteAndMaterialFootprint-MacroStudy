#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 10:29:02 2022

@author: stew
"""
import pickle
import json
import plotly.io as pio
import plotly.graph_objects as go
from datetime import datetime
import os
import sys
import bw2calc as bc
import bw2data as bd
import bw2analyzer as ba
import pandas as pd

#ContributionAnalysis().annotated_top_processes(lca)

method_keyword = "Waste Footprint"
title = 'market_selection'
data_dir = os.getcwd() + "/data/"
tmp = data_dir +"tmp/"
top_acts_dir = data_dir + "/top_activities/"
if not os.path.isdir(top_acts_dir): os.mkdir(top_acts_dir)

# %% Load lists of top activities
f = [x for x in os.listdir(tmp) if "combined_topactivities_df.pickle" in x][0]
df = pd.read_pickle(tmp+f).reset_index(drop=True)

db_names = df.database.unique()
for db_name in df.database.unique():
    project_name = "WasteFootprint_"+db_name
    acts = df[df.database == db_name]
    acts = acts.drop_duplicates(subset=['code']).reset_index(drop=True)

    bd.projects.set_current(project_name)
    db = bd.Database(db_name)

    methods_waste = []
    for method in list(bd.methods):
        if method_keyword == method[0]:
            methods_waste.append(method)

    # methods = [x for x in methods_waste if x[1] == "waste_total_combined" or x[1] == "waste_hazardous_combined"]
    methods = methods_waste
    print("\n** Using methods:\n", methods)
    print("\n** Using activities:\n", acts["name"].values)

    #%% LCA GraphTraversal calculations


    # Repeat calculations for the rest of the activities
    results_dics = []
    limit = acts.shape[1]
    start = datetime.now()
    #methods = [x for x in methods_waste if x[2] == "waste_total_liquid"]

    act = db.get(acts.code[0])
    lca = bc.LCA({act : 1}, method=methods[0])
    lca.lci()
    lca.lcia()

    top_all = pd.DataFrame()
    print("\n***** Starting LCA top activities calculations")

    for i, code in enumerate(acts.code):
            
            act_object = db.get(code)
            dic = act_object.as_dict()
            print(i, code, act_object.as_dict()["name"])
            for j, m in enumerate(methods):
                lca.switch_method(m)
                lca.redo_lcia({act_object : 1})
                
                top_acts = pd.DataFrame(lca.top_activities(), columns=["waste", "dem", "object"])
                
                top_acts['name'] = top_acts.object.apply(lambda x: x.as_dict()["name"])
                top_acts['code'] = top_acts.object.apply(lambda x: x.as_dict()["code"])
                top_acts['score'] = lca.score
                top_acts['act_type'] = top_acts.object.apply(lambda x: x.as_dict()["activity type"])
                top_acts['prod_amount'] = top_acts.object.apply(lambda x: x.as_dict()['production amount'])
                top_acts['reference_product'] = top_acts.object.apply(lambda x: x.as_dict()['reference product'])
                top_acts['activity_name'] = dic["name"]
                top_acts['activity_code'] = dic["name"]
                top_acts['database'] = dic["database"]
                top_acts["method"] = m[2]
                top_acts = top_acts[top_acts.act_type == "ordinary transforming activity"]
                top_acts = top_acts[top_acts.prod_amount != -1]
                top_acts = top_acts.reset_index(drop=True)
                top_one = top_acts.sort_values('waste', ascending = False).head(1)["name"].reset_index(drop=True)
                try:
                    acts.iat[i, j] = top_one[0]
                    print("** Top activity for : "+m[2]+" : "+dic["name"]+" : "+top_one[0])
                except:
                    print("** Top activity for : "+m[2]+" : "+dic["name"]+" : unknown")
                top_acts = top_acts.drop(columns=["object"])

                top_all = pd.concat([top_all, top_acts], axis=0)

    # print info about the combined calculations for the db
    finish = datetime.now()
    duration = finish - start
    outcome = ("\n\n** {} -- * Completed {} LCIA Top Activity calculations: in: {}".format(finish.strftime("%Y-%m-%d %H:%M:%S"), i, str(duration).split(".")[0]))
    print(outcome)
        
    # write to log file
    with open(os.path.join(tmp, title+"TopActs_log.txt"), "a") as f:
        f.write(outcome)
            
    # convert results dictionary to dataframe
            
            
    # save individual db results as pickle
    pickle_path = top_acts_dir + db_name+'_'+title+"_TopActsTopActs_df.pickle"
    top_all.to_pickle(os.path.join(tmp, pickle_path))
            
    # save individual db results as csv
    csv_path = pickle_path.replace(".pickle", ".csv")
    top_all.to_csv(csv_path, sep=";")


#%% 2. RUN LCA CALCULATION AND GRAPH TRAVERSAL

#%%


#%% 3. DEFINE CLASS GRAPH() TO GET DATA FOR SANKEY


# #%% OPTIONAL: LOAD FILES TO PROCESS
# # file_base = "sankey_data"
# # tmp = os.path.join(os.getcwd(), "data/tmp")

# # jsons = [x for x in os.listdir(tmp) if ".json" in x]

# # json_file = os.path.join(tmp, jsons[0])
# # with open(json_file) as json_file:
# #     data = json.load(json_file)


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


# #%% 3. DEFINE CLASS GRAPH() TO GET DATA FOR SANKEY

# """ thanks to the AB people for this, it's much more elegant than what I had written"""


# class Graph():

#     import bw2calc as bc
#     import bw2data as bd

#     def new_graph(self, data):
#         self.json_data = Graph.get_json_data(data)

#     @staticmethod
#     def get_json_data(data) -> str:
#         """Transform bw.Graphtraversal() output to JSON data."""
#         lca = data["lca"]
#         lca_score = lca.score
#         lcia_unit = bd.Method(lca.method).metadata["unit"]
#         demand = list(lca.demand.items())[0]
#         reverse_activity_dict = {v: k for k, v in lca.activity_dict.items()}

#         build_json_node = Graph.compose_node_builder(
#             lca_score, lcia_unit, demand[0])
#         build_json_edge = Graph.compose_edge_builder(
#             reverse_activity_dict, lca_score, lcia_unit)

#         valid_nodes = (
#             (bd.get_activity(reverse_activity_dict[idx]), v)
#             for idx, v in data["nodes"].items() if idx != -1
#         )
#         valid_edges = (
#             edge for edge in data["edges"]
#             if all(i != -1 for i in (edge["from"], edge["to"]))
#         )

#         json_data = {
#             "nodes": [build_json_node(act, v) for act, v in valid_nodes],
#             "edges": [build_json_edge(edge) for edge in valid_edges],
#             "title": Graph.build_title(demand, lca_score, lcia_unit),
#             "max_impact": max(abs(n["cum"]) for n in data["nodes"].values()),
#         }
#         #print("JSON DATA (Nodes/Edges):", len(nodes), len(edges))
#         # print(json_data)
#         return json_data

#     @staticmethod
#     def build_title(demand: tuple, lca_score: float, lcia_unit: str) -> str:
#         act, amount = demand[0], demand[1]
#         if type(act) is tuple:
#             act = bd.get_activity(act)
#         format_str = ("Reference flow: {:.2g} {} {} | {} | {} <br>"
#                       "Total impact: {:.2g} {}")
#         return format_str.format(
#             amount,
#             act.get("unit"),
#             act.get("reference product") or act.get("name"),
#             act.get("name"),
#             act.get("location"),
#             lca_score, lcia_unit,
#         )

#     @staticmethod
#     def compose_node_builder(lca_score: float, lcia_unit: str, demand: tuple):
#         """Build and return a function which processes activities and values
#         into valid JSON documents.

#         Inspired by https://stackoverflow.com/a/7045809
#         """

#         def build_json_node(act, values: dict) -> dict:
#             return {
#                 "db": act.key[0],
#                 "id": act.key[1],
#                 "product": act.get("reference product") or act.get("name"),
#                 "name": act.get("name"),
#                 "location": act.get("location"),
#                 "amount": values.get("amount"),
#                 "LCIA_unit": lcia_unit,
#                 "ind": values.get("ind"),
#                 "ind_norm": values.get("ind") / lca_score,
#                 "cum": values.get("cum"),
#                 "cum_norm": values.get("cum") / lca_score,
#                 # if act == demand else identify_activity_type(act),
#                 "class": "demand"
#             }

#         return build_json_node

#     @staticmethod
#     def compose_edge_builder(reverse_dict: dict, lca_score: float, lcia_unit: str):
#         """Build a function which turns graph edges into valid JSON documents.
#         """

#         def build_json_edge(edge: dict) -> dict:
#             p = bd.get_activity(reverse_dict[edge["from"]])
#             from_key = reverse_dict[edge["from"]]
#             to_key = reverse_dict[edge["to"]]
#             return {
#                 "source_id": from_key[1],
#                 "target_id": to_key[1],
#                 "amount": edge["amount"],
#                 "product": p.get("reference product") or p.get("name"),
#                 "impact": edge["impact"],
#                 "ind_norm": edge["impact"] / lca_score,
#                 "unit": lcia_unit,
#                 "tooltip": '<b>{}</b> ({:.2g} {})'
#                            '<br>{:.3g} {} ({:.2g}%) '.format(
#                     lcia_unit, edge["amount"], p.get("unit"),
#                     edge["impact"], lcia_unit, edge["impact"] / lca_score * 100,
#                 )
#             }

#         return build_json_edge



# # %% 4. RUN GRAPH()
# # start = datetime.now()
# # for cat, df in dfs_top.items():
# #     for code in df.code:
# #         print(cat, code)
# #         ab = Graph()
# #         ab.new_graph(data)
# #         data = ab.json_data
# #         sankey(data)
# #         finish = datetime.now()
# #         duration = finish - start
# #         print(duration)
# #%%



# # %% OPTIONAL: TRY TO SAVE DATA
# # import json

# # data2save = data
# # file_base = "sankey_data"
# # path = os.path.join(os.getcwd(), "data/tmp", file_base)
# # path_file = path + ".json"

# # json_file = json.dumps(data2save)
# # with open(path_file, "w+") as f:
# #     f.write(json_file)


# #%% OPTIONAL: LOAD FILES TO PROCESS
# # file_base = "sankey_data"
# # tmp = os.path.join(os.getcwd(), "data/tmp")

# # jsons = [x for x in os.listdir(tmp) if ".json" in x]

# # json_file = os.path.join(tmp, jsons[0])
# # with open(json_file) as json_file:
# #     data = json.load(json_file)


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
# #fig.update_layout(title_text="Waste demand flows for: {}".fo
# # name_file = name.replace(".", "").replace(" ", "").lower()



# fig.show()

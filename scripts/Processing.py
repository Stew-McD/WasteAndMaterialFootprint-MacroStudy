#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
|===============================================================|
| File: Processing.py                                           |
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

import pandas as pd
import numpy as np
from pathlib import Path

from user_settings import dir_results

def Raw2Cooked(activities_list, combined_raw_pickle):
    """Convert raw data to processed data."""
    
    def load_data():
        """Load data from the provided sources."""
        print("\n* Loading activity list with metadata from previous script")
        acts_all = pd.read_csv(activities_list, sep=";").reset_index(drop=True)
        acts = acts_all[['name', 'database', 'code', 'prod_category', 'prod_sub_category', 'location']]
        print("\n* Loading raw calculation results")
        df_results = pd.read_pickle(combined_raw_pickle).reset_index(drop=True)
        df = pd.merge(acts, df_results, how='outer', on=['code', 'name', 'database'])
        return df_results

    def process_columns(df):
        """Process and adjust column names and data."""
        df = df.drop(columns=['activity type'])
        capital_columns = [col for col in df.columns if col[0].isupper()]
        df.rename(columns={col: col + " (demand)" for col in capital_columns}, inplace=True)
        return df

    def remove_unwanted_rows(df):
        """Remove rows without waste and zeros."""
        no_waste = df[(df.waste_total_solid + df.waste_total_liquid) == 0]
        print("\n* No waste found for activities: \n", no_waste.name.values)
        df = df[(df.waste_total_solid + df.waste_total_liquid) != 0]
        cols_meta = ['name', 'prod_category', 'prod_sub_category', 'unit', 'code', 'location', 'database', 'reference product']
        zero_sum = df[df.columns.difference(cols_meta)].loc[:, (df == 0).any(axis=0)].columns.to_list()
        print("\n* Removing columns with zero sum: \n", zero_sum)
        df = df.drop(zero_sum, axis=1)
        return df, cols_meta

    def adjust_units_and_calculations(df, cols_meta):
        """Adjust units and perform relevant calculations."""
        
        # Replace zeros with NaN for subsequent calculations
        df = df.replace({0: np.nan})

        # Add columns for each end-of-life category as a percentage of total waste
        cols_percentage, cols_waste, cols_solid, cols_liquid = [], [], [], []
        for i in ["hazardous", "non-hazardous", "landfill", "recycling", "incineration", "open-burning", "digestion", 'composting', 'radioactive']:
            for unit in ['_solid', '_liquid']:
                col_name = "waste_" + i + unit
                if col_name in df.columns:
                    cols_percentage.append(col_name + "_per")
                    df[col_name + "_per"] = 100 * df[col_name].divide(df["waste_total" + unit].replace({0: np.nan}))
                    cols_waste.append(col_name)
                    if 'solid' in unit:
                        cols_solid.append(col_name)
                    else:
                        cols_liquid.append(col_name)

        # Convert units
        df[cols_solid] = df.apply(lambda x: x[cols_solid] / 1000 if x['unit'] == 'cubic meter' else x[cols_solid], axis=1)
        df[cols_liquid] = df.apply(lambda x: x[cols_liquid] * 1000 if x['unit'] == 'kilogram' else x[cols_liquid], axis=1)

        # Add and calculate new columns
        df['waste_total'] = df.waste_total_solid + df.waste_total_liquid
        df['waste_haz_tot'] = df.waste_hazardous_liquid + df.waste_hazardous_solid
        df['waste_haz_tot_per'] = df.waste_haz_tot.divide(df.waste_total)
        df['waste_circ'] = (df.waste_composting_solid + df.waste_digestion_solid + df.waste_recycling_solid).div(df.waste_total)
        cols_categorised = [x for x in cols_waste if 'tot' not in x and "hazardous" not in x and "radioactive" not in x]
        df['waste_categorised'] = df[cols_categorised].sum(axis=1)
        df['waste_uncategorised_per'] = 100 * (df.waste_total - df.waste_categorised).div(df.waste_total)
        df = df.drop('waste_categorised', axis=1)

        cols_waste = [col for col in df.columns if 'waste' in col]
        cols_material = [col for col in df.columns if '(demand)' in col]
        not_methods = cols_material + cols_meta + cols_waste
        cols_methods = [col for col in df.columns if col not in not_methods]

        return df, cols_percentage, cols_waste, cols_solid, cols_liquid, cols_categorised, cols_methods, cols_material

    def initialize_column_sets(df, cols_meta, cols_total, cols_waste, cols_solid, cols_liquid, cols_methods, cols_percentage, cols_material):
        """Initialize column sets based on existing dataframe and metadata columns."""
        cols_t = cols_meta + cols_total
        cols_w = cols_meta + cols_waste
        cols_s = cols_meta + cols_solid
        cols_l = cols_meta + cols_liquid
        cols_m = cols_meta + cols_methods
        cols_d = cols_meta + cols_material
        cols_per = cols_total + cols_percentage + cols_meta

        cols_per_df_liq = [x for x in cols_per if "solid" not in x]
        cols_per_df_sol = [x for x in cols_per if "liquid" not in x]
        return cols_t, cols_w, cols_s, cols_l, cols_m, cols_d, cols_per, cols_per_df_liq, cols_per_df_sol 

    def save_processed_data(df):
        """Save processed data."""
        combined_cooked_pickle_str = str(combined_raw_pickle).replace("raw", "cooked").replace('tmp', 'results')
        combined_cooked_pickle = Path(combined_cooked_pickle_str)
        df.to_pickle(combined_cooked_pickle)
        combined_cooked_csv_str = combined_cooked_pickle_str.replace("pickle", "csv")
        combined_cooked_csv = Path(combined_cooked_csv_str)
        df.to_csv(combined_cooked_csv, sep=';', index=False)
        print(f"\nCooked results saved to:\n- pickle: {combined_cooked_pickle}\n- csv: {combined_cooked_csv}")
        return combined_cooked_csv, combined_cooked_pickle

    print("\n** Starting processing **")

    # Load data
    df = load_data()

    # Process columns
    df = process_columns(df)

    # Remove unwanted rows and retrieve metadata columns
    df, cols_meta = remove_unwanted_rows(df)

    # Adjust units and perform relevant calculations
    (df, cols_percentage, cols_waste, cols_solid, cols_liquid, cols_categorised, cols_methods, cols_material) = adjust_units_and_calculations(df, cols_meta)

    # Define total columns list
    cols_total = [
        'waste_total', 'waste_haz_tot', 'waste_haz_tot_per', 
        'waste_circ', 'waste_uncategorised_per'
    ]

    # Initialize various column sets
    (cols_t, cols_w, cols_s, cols_l, cols_m, cols_d,
    cols_per, cols_per_df_liq, cols_per_df_sol) = initialize_column_sets(
        df, cols_meta, cols_total, cols_waste, cols_solid, cols_liquid, cols_methods, cols_percentage, cols_material
    )

    return save_processed_data(df)


#%% extract top activities

from pathlib import Path

def ExtractTopActivities(combined_cooked_pickle, n_top=1):
    """Extract top activities for each category and waste/material indicator.
    
    Parameters:
        combined_cooked_pickle (str): Path to the processed pickle file.
        n_top (int, optional): Number of top activities per category. Default is 1.
    """
    
    print(f"\n** Extracting top {n_top} activities for each category and waste/material indicator...")
    
    df = pd.read_pickle(combined_cooked_pickle)

    # Define criteria for sorting
    criteria = {
        'waste_total': 'desc',
        'waste_total_solid': 'desc',
        'waste_total_liquid': 'desc',
        'waste_haz_tot': 'desc',
        'waste_haz_tot_per': 'desc',
        'waste_circ': 'desc'
    }
    criteria.update({col: 'desc' for col in df.columns if "(demand)" in col})

    dbs = df.database.unique()
    df_top_all = pd.DataFrame()

    for db in dbs:
        topacts = {}
        for key, order in criteria.items():
            direction = False if order == 'desc' else True
            topacts[key] = df[df.database == db].sort_values(key, ascending=direction).groupby('prod_category').head(n_top)

        df_top = pd.concat([value.assign(top=key) for key, value in topacts.items()], axis=0)
        
        df_top_all = pd.concat([df_top_all, df_top], axis=0).reset_index(drop=True)

    cooked_path = Path(combined_cooked_pickle)
    file_top_pickle = cooked_path.with_name(f"{cooked_path.stem.replace('cookedresults', 'topactivities')}.pickle")
    file_top_csv = file_top_pickle.with_suffix('.csv')

    df_top_all.to_pickle(file_top_pickle)
    df_top_all.to_csv(file_top_csv, sep=";", index=False)
    
    print (f"\n* Top activities saved to: \ncsv - {file_top_csv}\npickle - {file_top_pickle}")
    
    return file_top_csv, file_top_pickle




#%%

#df_compare_tot =df_compare_tot[df_compare_tot.waste_total_solid_cutoff31 >= 0.1]
#df_compare_tot =df_compare_tot[df_compare_tot.waste_total_solid_cutoff31 < 1000] # cuts off gold and platinum

# df.plot.scatter(x='waste_total_solid', y='waste_total_liquid')
# df.plot.scatter(x='waste_total_liquid', y='water use: human health no LT')
# df.plot.scatter(x='waste_total_solid', y='natural resources no LT')
# df.plot.scatter(x='haz', y='ecosystem quality no LT')
# df.plot.scatter(x='haz', y='human health no LT')
# df.plot.scatter(x='haz', y='circ')
# df.plot.scatter(x='circ', y='natural resources no LT')

# df_waste_per = df[cols_per]
# df_waste_per_sol = df[cols_per_df_sol]
# df_waste_per_sol_cut = df_waste_per_sol[df_waste_per_sol.waste_total_solid >= 0.01]
# df_waste_per_liq = df[cols_per_df_liq]



# prod_cats = df.prod_category.dropna().sort_values().unique().tolist()
# #['AgriForeAnim', 'OreMinFuel', 'Chemical', 'ProcBio', 'PlastRub', 'GlasNonMetal', 'MetalAlloy', 'MachElecTrans']


# df_waste = df[cols_w] #.drop(columns=(cols_liquid+cols))
# cols = df_waste
# df_waste = df_waste.sort_values(by='waste_total_solid', ascending=False)


# df_waste.mean(skipna=True, numeric_only=True)
# stats = df_waste_per[cols_per].describe(include='all').transpose()


# top_sol = df.sort_values('waste_total_solid', ascending = False).groupby('prod_category').head(1)
# top_liq = df.sort_values('waste_total_liquid', ascending = False).groupby('prod_category').head(1)
# top_haz = df.sort_values('haz', ascending = False).groupby('prod_category').head(1)
# top_haz_liq = df.sort_values('waste_hazardous_liquid', ascending = False).groupby('prod_category').head(1)
# top_haz_sol = df.sort_values('waste_hazardous_solid', ascending = False).groupby('prod_category').head(1)
# top_haz_liq_per = df.sort_values('waste_hazardous_liquid_per', ascending = False).groupby('prod_category').head(1)
# top_haz_sol_per = df.sort_values('waste_hazardous_solid_per', ascending = False).groupby('prod_category').head(1)


# #%%%
# import os
# import plotly.express as px
# import plotly.io as pio
# from plotly.subplots import make_subplots
# import plotly.graph_objects as go
# import numpy as np

# pio.renderers.default='browser'

# figures = os.getcwd() + "/figures"
# if not os.path.isdir(figures): os.mkdir(figures)

# tables = os.getcwd() + "/tables"
# if not os.path.isdir(tables): os.mkdir(tables)

# colour_set = px.colors.qualitative.Vivid + px.colors.qualitative.Antique
# colours = {}
# for i, c in enumerate(prod_cats):
#     colours.update({c:colour_set[i]})



# fig1 = px.scatter(df, x='waste_total_solid', y='waste_total_liquid', hover_data=["name"], color='prod_category', log_x=True, log_y=True, trendline='ols', trendline_scope='overall')
# fig2 = px.scatter(df, x='waste_total_liquid', y='water use: human health no LT', hover_data=["name"], color='prod_category', log_x=True, log_y=True, trendline='ols', trendline_scope='overall')
# fig3 = px.scatter(df, x='waste_total_solid', y='natural resources no LT', hover_data=["name"], color='prod_category', log_x=True, log_y=True, trendline='ols', trendline_scope='overall')
# fig4 = px.scatter(df, x='haz_tot', y='ecosystem quality no LT', hover_data=["name"], color='prod_category', log_x=True, log_y=True, trendline='ols', trendline_scope='overall')
# fig1 = px.scatter(df, x='haz_tot', y='human health no LT', hover_data=["name"], color='prod_category', log_x=True, log_y=True, trendline='ols', trendline_scope='overall')
# fig6 = px.scatter(df, x='haz_tot', y='circ', hover_data=["name"], color='prod_category', log_x=True, log_y=True, trendline='ols', trendline_scope='overall')
# fig7 = px.scatter(df, x='circ_inv', y='natural resources no LT', hover_data=["name"], color='prod_category', log_x=True, log_y=True, trendline='ols', trendline_scope='overall')
# fig8 = px.scatter(df, x='waste_total', y='natural resources no LT', hover_data=["name"], color='prod_category', log_x=True, log_y=True, trendline='ols', trendline_scope='overall')

# # fig1 = px.scatter(df, x='waste_total_solid', y='waste_total_liquid')
# # fig2 = df.plot.scatter(x='waste_total_liquid', y='water use: human health no LT')
# # fig3 = df.plot.scatter(x='waste_total_solid', y='natural resources no LT')
# # fig4 = df.plot.scatter(x='haz', y='ecosystem quality no LT')
# # fig1 = df.plot.scatter(x='haz', y='human health no LT')
# # fig6 = df.plot.scatter(x='haz', y='circ')
# # fig7 = df.plot.scatter(x='circ', y='natural resources no LT')

# fig1.show()
# fig2.show()
# fig3.show()
# fig4.show()
# fig1.show()
# fig6.show()
# fig7.show()
# fig8.show()
# fig9.show()

# #%%



# fig = px.box(
#     df,
#     x='tot_log',
#     y='prod_category',
#     points='all',
#     notched=True,
#     hover_data=["name"],

#     color='prod_category',
#     )

# fig.show()


#%%


# fig = go.Figure()

# fig.add_trace(go.Scatter(



# for i, d in enumerate(df_units):

#     fig = make_subplots(rows=2, cols=1, subplot_titles=("Solid waste", "Liquid waste"), vertical_spacing = 0.18, )
#     traces = []
#     m_sol_names = [x.replace("_", " ").capitalize() for x in methods_sol]

#     for j, (ms, unit_w, name_w) in enumerate([(methods_sol, "kg", 'solid'), (methods_liq, 'm3', 'liquid')]):

#         for k, m in enumerate(ms):

#             m_name = m.split("_")[1]
#             fig.add_scatter(
#                  x=d[m]
#                 ,y=[m_name]*len(d[m])
#                 ,name=m_name + " ("+unit_w+")"
#                 ,mode='lines+text+markers'
#                 ,marker=dict(color=colours[m_name])
#                 ,line=dict(width=0.2)
#                 # ,text=d["name"]
#                 # ,texttemplate = "%{text}<br>(%{a:.2f}, %{b:.2f}, %{c:.2f})"
#                 # ,textfont = { 'size': 2, 'color': "Black"}
#                 # ,textposition="top right"
#                 ,showlegend=True
#                 #,legendgroup=name_w
#                 ,legendgrouptitle=dict(text="Waste Category")
#                 #,legendrank=-1
#                 #,text= m_sol_names[k] +'\n from: ' + d['name'] + "\n, ref. prod: " + d["reference product"]
#                 ,customdata = np.stack(([m_name + " ("+unit_w+")"]*len(d[m]), d["reference product"], d["name"], d.index, d["unit"], d["location"]), axis=-1)
#                 #,hovertext= 'Waste'
#                  ,hoverinfo = 'all'
#                 ,hovertemplate =
#                 '<b>Activity</b>: %{customdata[2]}, (%{customdata[1]})'+
#                 '<br><b>Waste %{customdata[0]} per %{customdata[4]} </b>: %{x}'+
#                 '<br><b>Reference Product</b>: %{customdata[1]}'+
#                 '<br><b>Code</b>: %{customdata[3]}'+
#                 '<extra></extra>'
#                 ,xhoverformat=".3e"
#                 ,row = j+1
#                 ,col = 1

#                 )
           # traces.append(trace)




        #sfig= go.Figure(data=traces)

    #fig.append_trace(sfig, row=j, col=1)
"""
    fig.update_layout(template="none" #["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"]:
    ,title_text = 'Specific waste production of every "ordinary transforming activity" with production unit "{}" in EcoInvent {} (GLO/RoW)'.format(prod_units[i],db_name)
    ,title_font_size = 26
    ,title_x = 0.01
    ,title_y = 0.98
    #,margin_pad = 0.2
    ,xaxis_automargin=True
    ,yaxis_automargin=True
    #,hovermode="x unified"
    ,yaxis1= dict(type='category', tickfont_size=12, showgrid=True, zeroline=True, autorange="reversed")
    ,xaxis1= dict(tickfont_size=10, showgrid=True, zeroline=True, title_font_size=13, title_text="Waste produced per {} of production (kilograms)".format(prod_units[i]))
    ,yaxis2= dict(type='category', tickfont_size=12, showgrid=True, zeroline=True, autorange="reversed")
    ,xaxis2= dict(tickfont_size=10, showgrid=True, zeroline=True, title_font_size=13, title_text="Waste produced per {} of production (cubic meters)".format(prod_units[i]))
    ,legend_y = 0.1
    ,legend_tracegroupgap = 220
    ,font_family="Open Sans",
      )
    #fig.update_yaxes(type='category')
    fig.write_html(figures + "/WasteDemand_LCIA_AllProcesses_{}_{}_gl.html".format(db_name, prod_units[i]))
    fig.show()


    # fig.add_trace(
    #     go.Bar(
    #         go.Bar(
    #             x=[df_plot.index],
    #             #x=[plot_df.category, plot_df.tick],
    #             y=abs(df_plot.values),
    #             name="{} waste demand for Li-ion battery production".format(unit_long.capitalize())  ),
    #     ), row=1, col=1+1
    #     )

    fig = px.bar(df_plot, text_auto=True, barmode="relative",

            labels={
                          "variable":"Waste category",
                          "index" : "Battery production",
                          "value" : ("Waste Demand ("+ unit_fig + sup +")")
                    },
            title="{} waste demand for Li-ion battery production".format(unit_long.capitalize()))
    # fig.update_xaxes(title_font=dict(size=20)
    # fig.update_yaxes(title_font=dict(size=20))
    # fig.update_traces(textposition='inside', textfont_size=20)
    # fig.update_layout(showlegend=True)
    # fig.update_layout(font=dict(size=20))

    fig.show()
    fig.write_html(figures + "/WasteDemand_categorised_{}.html".format(unit_long))
    fig.write_image(figures + "/WasteDemand_categorised_{}.svg".format(unit_long))
    #fig.write_image(figures, "/WasteDemand_categorised_{}.png".format(unit))
    fig.update_layout(title_text='Waste Demand for Batteries in EcoInvent 3.8')
    #df = dfTranspose(df)
    df_plot.to_csv(tables + "/WasteDemand_categorised_{}.csv".format(unit_long))
    df_plot.to_excel(tables + "/WasteDemand_categorised_{}.xlsx".format(unit_long))

print("\nTables & figures have been saved as */WasteDemand_categorised_* ")


"""


# #%%%
# df_waste.pop('code_cutoff39')
# df_waste.pop('code_cutoff31')

# df_AgriForeAnim = df_waste[df_waste.prod_category == 'AgriForeAnim']
# df_OreMinFuel = df_waste[df_waste.prod_category == 'OreMinFuel']
# df_Chemical = df_waste[df_waste.prod_category == 'Chemical']
# df_ProcBio = df_waste[df_waste.prod_category == 'ProcBio']
# df_PlastRub = df_waste[df_waste.prod_category == 'PlastRub']
# df_GlasNonMetal = df_waste[df_waste.prod_category == 'GlasNonMetal']
# df_MetalAlloy = df_waste[df_waste.prod_category == 'MetalAlloy']
# df_MachElecTrans = df_waste[df_waste.prod_category == 'MachElecTrans']

# df_AgriForeAnim_per = df_waste_per[df_waste_per.prod_category == 'AgriForeAnim']
# df_OreMinFuel_per = df_waste_per[df_waste_per.prod_category == 'OreMinFuel']
# df_Chemical_per = df_waste_per[df_waste_per.prod_category == 'Chemical']
# df_ProcBio_per = df_waste_per[df_waste_per.prod_category == 'ProcBio']
# df_PlastRub_per = df_waste_per[df_waste_per.prod_category == 'PlastRub']
# df_GlasNonMetal_per = df_waste_per[df_waste_per.prod_category == 'GlasNonMetal']
# df_MetalAlloy_per = df_waste_per[df_waste_per.prod_category == 'MetalAlloy']
# df_MachElecTrans_per = df_waste_per[df_waste_per.prod_category == 'MachElecTrans']

# df_AgriForeAnim_per_sol = df_waste_per_sol[df_waste_per_sol.prod_category == 'AgriForeAnim']
# df_OreMinFuel_per_sol = df_waste_per_sol[df_waste_per_sol.prod_category == 'OreMinFuel']
# df_Chemical_per_sol = df_waste_per_sol[df_waste_per_sol.prod_category == 'Chemical']
# df_ProcBio_per_sol = df_waste_per_sol[df_waste_per_sol.prod_category == 'ProcBio']
# df_PlastRub_per_sol = df_waste_per_sol[df_waste_per_sol.prod_category == 'PlastRub']
# df_GlasNonMetal_per_sol = df_waste_per_sol[df_waste_per_sol.prod_category == 'GlasNonMetal']
# df_MetalAlloy_per_sol = df_waste_per_sol[df_waste_per_sol.prod_category == 'MetalAlloy']
# df_MachElecTrans_per_sol = df_waste_per_sol[df_waste_per_sol.prod_category == 'MachElecTrans']

# df_AgriForeAnim_per_liq = df_waste_per_liq[df_waste_per_liq.prod_category == 'AgriForeAnim']
# df_OreMinFuel_per_liq = df_waste_per_liq[df_waste_per_liq.prod_category == 'OreMinFuel']
# df_Chemical_per_liq = df_waste_per_liq[df_waste_per_liq.prod_category == 'Chemical']
# df_ProcBio_per_liq = df_waste_per_liq[df_waste_per_liq.prod_category == 'ProcBio']
# df_PlastRub_per_liq = df_waste_per_liq[df_waste_per_liq.prod_category == 'PlastRub']
# df_GlasNonMetal_per_liq = df_waste_per_liq[df_waste_per_liq.prod_category == 'GlasNonMetal']
# df_MetalAlloy_per_liq = df_waste_per_liq[df_waste_per_liq.prod_category == 'MetalAlloy']
# df_MachElecTrans_per_liq = df_waste_per_liq[df_waste_per_liq.prod_category == 'MachElecTrans']

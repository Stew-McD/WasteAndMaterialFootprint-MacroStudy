#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 15:20:07 2023

@author: stew
"""

print("\n======== Running WasteFootprint visualisation script ========")

# %% 0. Import libraries
print("\n==== Importing libraries ====")

from decimal import ROUND_FLOOR
import warnings
warnings.filterwarnings('ignore', lineno=1544)
warnings.filterwarnings('ignore', lineno=1752)
# warnings.filterwarnings('ignore', category=Type["SettingWithCopyWarning"])

from tabulate import tabulate        
import cowsay
     
import os                                
import pandas as pd
pd.options.mode.chained_assignment = None
# type: ignore
import numpy as np
import itertools

import dash
from dash import html
from dash import dcc

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default='chromium'

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import statsmodels.api as sm

from datetime import datetime
start = datetime.now()
# %% 1. Set up directories
print("\n==== Setting up directories ====")
os.chdir(os.path.dirname(__file__))

# import os
# dir_figures = os.getcwd() + "/figures/"
# if not os.path.isdir(dir_figures): os.mkdir(dir_figures)

dir_results =  os.getcwd() + "/results/"
if not os.path.isdir(dir_results): os.mkdir(dir_results)

dir_results_plotly_scatter =  dir_results + "plotly_scatter/"
if not os.path.isdir(dir_results_plotly_scatter): os.mkdir(dir_results_plotly_scatter)

dir_results_plotly_box =  dir_results + "plotly_box/"
if not os.path.isdir(dir_results_plotly_box): os.mkdir(dir_results_plotly_box)

dir_corr = dir_results + "correlation_matrix/"
if not os.path.exists(dir_corr): os.mkdir(dir_corr)
    
tables = dir_results + "tables/"
if not os.path.isdir(tables): os.mkdir(tables)
    
#function for printing nice tables in the output 
  
def pprint_df(dframe):
    print(tabulate(dframe, headers='keys', tablefmt='psql', showindex=False))
    
# %% 2. LOAD DATA

print("\n==== Processing data ====")

path = "/home/stew/code/gh/WasteAndMaterialFootprint-MacroStudy/data/markets/results/WMFootprint-SSP2LT-cutoff/markets_combined_cookedresults_df.pickle"

act_selection = "market-selection"
df = pd.read_pickle(path)
print("\n*Loaded data from: ", path)

# setup colour dictionary for the different product categories
#fig = px.colors.qualitative.swatches()
#fig.show()
colour_brewer = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']
colour_set = colour_brewer + colour_brewer #px.colors.qualitative.Plotly + px.colors.qualitative.Bold + px.colors.qualitative.Vivid # px.colors.qualitative.Antique
colour_dic = {}
for i, category in enumerate(df.prod_sub_category.unique().tolist()+df.prod_category.unique().tolist()):
    
    colour_dic.update({category: colour_set[i]})
    #print(i, category, colour_set[i])
    
df["colour_prod_category"] = df.prod_category.map(colour_dic)
df["colour_prod_sub_category"] = df.prod_sub_category.map(colour_dic)
df['price_ratio_total_waste_solid'] = df.waste_total_solid.div(df.price)
df['price_ratio_total_waste_liquid'] = df.waste_total_liquid.div(df.price)

db_names = df.database.unique().tolist()
db_name = "cutoff391" #["con391", "cutoff391", "apos391"]
df = df[df.database == db_name].replace(np.nan, 0)
df.columns = [x.replace("/", " and ") for x in df.columns]
df = df.replace("Textiles", "Textile")
print("\n* Processing database: ", db_name)

# define sets of columns for the datasets + the columns for the metadata
cols = sorted([x for x in df.columns if x != "colour"])
cols_waste = [x for x in cols if ('waste_' in x and '_per' not in x and "log" not in x)]
cols_solid = [x for x in cols_waste if 'solid' in x]
cols_liquid = [x for x in cols_waste if 'liquid' in x]
cols_total = [x for x in cols_waste if 'total' in x]
cols_meta = ['name', 'prod_category', 'prod_sub_category', 'unit', 'code', 'location', 'database', 'reference product', 'price', 'currency', 'colour_cat', 'colour_sub_cat', "colour_prod_category", "colour_prod_sub_category"]
cols_percentage = [x for x in cols if '_per' in x]
cols_methods = [x for x in cols if (x not in cols_waste) and (x not in cols_meta) and (x not in cols_percentage) and "colour" not in x]

cols_t = cols_meta + cols_total
cols_w = cols_meta + cols_waste
cols_s = cols_meta + cols_solid
cols_l = cols_meta + cols_liquid
cols_r = cols_meta + cols_methods
cols_per = cols_total +  cols_percentage + cols_meta

print("\n+++ Waste categories as a percentage of total (liq/sol) waste:\n")
cats_liq = df[cols_liquid].sum(axis=0).sort_values(ascending=False)
cats_liq = 100*cats_liq/cats_liq.waste_total_liquid
cats_sol = df[cols_solid].sum(axis=0).sort_values(ascending=False)
cats_sol = 100*cats_sol/cats_sol.waste_total_solid
cats_waste = pd.concat([cats_liq, cats_sol], axis=0)
cats_waste_significant = cats_waste #.drop(["waste_non_hazardous_solid", "waste_non_hazardous_liquid", "waste_digestion_solid", "waste_open_burning_solid", "waste_composting_solid", 'waste_landfill_liquid'])
print(cats_waste.to_markdown())
print("\n These 'important' waste categories will be used for visualisation:")
cats_waste_significant_cols = cats_waste_significant.index.tolist()
print(*cats_waste_significant_cols, sep="\n")

# make some other dataframes for comparison of the different databases and methods
df_cutoff = df[df.database == "cutoff391"]
df_apos = df[df.database == "apos391"]
df_con = df[df.database == "con391"]

df_apos_coff = df_cutoff.merge(df_apos, how="inner", on=["name","prod_category"], suffixes=("_coff", "_apos"))[["name", "waste_total_solid_coff", "waste_total_solid_apos"]]
df_apos_coff["apos_coff"] = (df_apos_coff.waste_total_solid_apos - df_apos_coff.waste_total_solid_coff).div(df_apos_coff.waste_total_solid_coff)
df_apos_coff.apos_coff.describe()

df_con_coff = df_cutoff.merge(df_con, how="inner", on=["name", "prod_category"], suffixes=("_coff", "_con"))[["name", "waste_total_solid_coff", "waste_total_solid_con"]]
df_con_coff["con_coff"] = (df_con_coff.waste_total_solid_con - df_con_coff.waste_total_solid_coff).div(df_con_coff.waste_total_solid_coff)
df_con_coff.con_coff.describe()
df_con_coff.waste_total_solid_coff.corr(df_con_coff.waste_total_solid_con)
 
# df_apos_coff["cutoff_apos"] = df.loc["waste_total_solid"]-df[df.database == "apos391"]["waste_total_solid"]
# df.cutoff_apos.describe()

# %% 3a. CORRELATION ANALYSIS

print("\n==== Performing correlation analysis ====\n")
cols_waste_corr = cats_waste_significant_cols

""" ["waste_total_solid", "waste_total_liquid", 'waste_circ', "waste_hazardous_liquid", "waste_hazardous_solid", "waste_categorised",'waste_landfill_solid','waste_open_burning_solid','waste_incineration_liquid','waste_incineration_solid']
 """
# make a list of all possible combinations of variables between waste and LCIA methods
tuples_wasteVlcia = [(x,y) for x in cols_waste_corr for y in (cols_methods+["price"])]

#list(itertools.combinations(cols_methods+cols_waste_corr+["price"],2))

#[(x,y) for x in cols_waste_corr for y in (cols_methods+["price"])]

# adjust to percentages
df['waste_circ'] = df.waste_circ.multiply(100)
df["haz_ratio"] = df.waste_circ.multiply(100)

# make a dataframe of all variable correlations in waste vs lcia methods
correlations_all = []
print("\n+++ Calculating correlations: waste methods vs. LCIA methods +++\n")
for i, t in enumerate(tuples_wasteVlcia):
    df_corr = df #> 0.01]
    corr = df_corr[t[0]].corr(df_corr[t[1]])
    correlations_all.append([(t[0],t[1]), t[0],t[1],round(corr,3)])
    
df_corr_all = pd.DataFrame(correlations_all, columns=["tuple", "waste_method", 'method', 'correlation']).reset_index(drop=True)
corr_min = 0.9
df_corr = df_corr_all[df_corr_all.correlation >= corr_min].reset_index(drop=True).sort_values(by="correlation", ascending=False)

print("** Significant correlations: {}, correlation cutoff = {}\n".format(len(df_corr), corr_min))
print(df_corr.drop("tuple", axis=1).to_markdown())
significant_columns = sorted(list(df_corr.waste_method.unique())+list(df_corr.method.unique()))
print("\n*** Unique variables of significance:\n", len(significant_columns))
print(*significant_columns, sep="\n")

file_name = dir_corr+ "waste_vs_methods_correlations_cut.csv"
df_corr.drop("tuple", axis=1).to_csv(file_name) ; print("\n **** Saved cut correlation table to: {}".format(file_name))
file_name = dir_corr+ "waste_vs_methods_correlations_all.csv"
df_corr_all.drop("tuple", axis=1).to_csv(file_name) ; print("*** Saved full correlation table to: {}".format(file_name))

# do the same for waste vs waste methods
tuples_wasteVwaste = list(itertools.combinations(cols_waste_corr + ["price"],2)) #[[x,y] for x in cols_waste_corr for y in cols_waste_corr]
waste_correlations = []
waste_correlations_all = []

print("\n=== Calculating correlations: waste methods vs. waste methods ===\n")
for i, t in enumerate(tuples_wasteVwaste):
    df_waste_corr = df #[df[t[0]] > 0.00]
    corr = df_waste_corr[t[0]].corr(df_waste_corr[t[1]])
    waste_correlations_all.append([(t[0],t[1]), t[0],t[1],round(corr,3)])
            
waste_correlations_all = sorted(waste_correlations_all)
df_waste_corr_all = pd.DataFrame(waste_correlations_all, columns=["tuple", "waste_method", 'method', 'correlation']).reset_index(drop=True)
corr_min = 0.9
df_waste_corr = df_waste_corr_all[df_waste_corr_all.correlation >= corr_min].reset_index(drop=True).sort_values(by="correlation", ascending=False)

# print some info about the correlations
print("** Significant correlations: {}, correlation cutoff = {}, \n".format(len(df_waste_corr), corr_min))
print(df_waste_corr.drop("tuple", axis=1).to_markdown())
significant_waste_columns = sorted(pd.concat([df_waste_corr.waste_method, df_waste_corr.method], axis=0).unique())
print("\n*** Unique variables of significance:\n", len(significant_waste_columns))
print(*significant_waste_columns, sep="\n")
print("")

file_name = dir_corr+ "{}_{}_waste_vs_waste_correlations_all.csv".format(db_name, act_selection)
df_waste_corr_all.drop("tuple", axis=1).to_csv(file_name)
print("**** Saved full waste method correlation table to: {}".format(file_name))

file_name = dir_corr+ "{}_{}_waste_vs_waste_correlations_cut.csv".format(db_name, act_selection)
df_waste_corr.drop("tuple", axis=1).to_csv(file_name)
print("**** Saved cut waste method correlation table to: {}".format(file_name))
# %% 3b. DEFINE FUNCTION FOR FACET GRID CORRELATION PLOT IN SEABORN

def MakeCorrelationFacetPlot(db_name, act_selection, df_plot=df[significant_columns], axis_type="linear", title="title"):
        
    df = df_plot.copy()    
    print("\n\n==== Running MakeCorrelationFacetPlot() ====")
    sns.set(style='white', font_scale=1)
    # make the figure titles and labels for the facet grid better
    df.columns = df.columns.str.replace("waste_", "")
    df.columns = df.columns.str.replace("_solid", " (s)")
    df.columns = df.columns.str.replace("_liquid", " (l)")
    df.columns = df.columns.str.replace("hazardous", "hazard.")
    df.columns = df.columns.str.replace("open_burning", "open-burn")
    df.columns = df.columns.str.replace("incineration", "incin.")
    df.columns = df.columns.str.replace("categorised", "defined EoL ")
    dim = df.shape
    
    # min max scaling so that the log transformation is possible
    if axis_type == "ln":
        df = (1+df-df.min())/(df.max()-df.min())
        df = np.log(df)
        
    # make the facet grid
    print("** Making {}x{} facet grid ({}) for {}".format(dim[1],dim[1],axis_type, title))
    print("* Be patient, there are {:.2e} data points.....".format(dim[1]**2*dim[0]**2))
    print("* Plotting the grid")
    g = sns.PairGrid(df,  aspect=1, diag_sharey=False)
    print("* Plotting the lower regression plots")
    g.map_lower(sns.regplot, lowess=True, ci=None, line_kws={'color': 'black'})
    print("* Plotting the diagonal plots")
    g.map_diag(sns.distplot, kde_kws={'color': 'black'})
    
    # define function for making the correlation dots at in the figure
    print("* Plotting the upper correlation plots")
    def corrdot(*args, **kwargs):
        corr_r = args[0].corr(args[1], 'pearson')
        corr_text = f"{corr_r:2.2f}".replace("0.", ".")
        ax = plt.gca()
        ax.set_axis_off()
        marker_size = abs(corr_r) * 10000
        ax.scatter([.5], [.5], marker_size, [corr_r], alpha=0.5, cmap="coolwarm",
                vmin=-1, vmax=1, transform=ax.transAxes)
        font_size = abs(corr_r) * 40 + 5
        ax.annotate(corr_text, [.5, .5,],  xycoords="axes fraction",
                    ha='center', va='center', fontsize=font_size)
    
    g.map_upper(corrdot)

    # make titles for the diagonal plots
    print("* Making titles for the diagonal plots")
    for ax, col in zip(np.diag(g.axes), df.columns):
        ax.set_title(col, y=0.60, fontsize=40)
        #ax.set(xlim=(0,df[col].max()))
    # for ax, title in zip(g.axes, df.columns):
    #     ax.set_title(title)
        #ax.text(0.85, 0.85,title[:10], fontsize=1) #add text

    # adjust the formatting of the figure
    print("* Adjusting the formatting of the figure")
    g.fig.subplots_adjust(top=0.80, wspace=0.2, hspace=0.2)
    g.fig.suptitle(t="{}: correlation matrix ({}) for {} activities in EI {}".format(title, axis_type, len(df), db_name), size=(4*dim[1]), y=0.85)


    # save the figures
    print("* Saving the figure as pdf")
    file_name = dir_corr + "{}_{}_{}_correlation_matrix_{}.pdf".format(act_selection, db_name, title.replace(" ", "-"), axis_type)
    g.savefig(file_name); print("\tSaved figure as pdf: {}".format(file_name))
    print("* Saving the figure as png")
    file_name = file_name.replace("pdf","png")
    g.savefig(file_name, ) ; print("\tSaved figure as png: {}".format(file_name))
    
# %% 3c. MAKE FACET GRID CORRELATION PLOTS
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11111

significant_columns += ["waste_circ"]
significant_columns += ['waste_uncategorised_per']
cols_waste += ['waste_uncategorised_per']
significant_columns = sorted(list(set(significant_columns)), reverse=True)
# MakeCorrelationFacetPlot(db_name, act_selection, df_plot=df[significant_columns], axis_type="linear", title="Waste methods vs LCIA methods")
# MakeCorrelationFacetPlot(db_name, act_selection, df_plot=df[significant_columns], axis_type="ln", title="Waste methods vs LCIA methods")

# # MakeCorrelationFacetPlot(db_name, act_selection, df_plot=df[significant_waste_columns],  axis_type="linear", title="Waste methods vs waste methods")
# MakeCorrelationFacetPlot( db_name, act_selection,  df_plot = df[significant_waste_columns],axis_type="ln", title="Waste methods vs waste methods")


# %% 4a. DEFINE FUNCTION FOR MAKING COMBINED SCATTER PLOTS IN PLOTLY

# define a function to make a scatter plot in plotly with a regression line

 
 #%% 4b. MAKE COMBINED SCATTER PLOTS IN PLOTLY
def MakePlotlyScatterCombined(group='prod_sub_category', folder="CombinedPlots", marker_size=10, df_plot=df, type_axis="log", act_selection = "market-selection"):


    def corr_annotation(x, y):
        pearsonr = stats.pearsonr(x, y)
        return 'r = {:.2f} (p = {:.3f})'.format(pearsonr[0], pearsonr[1])
    
    print("\n======= Running MakePlotlyScatterCombined() ========\n")

    type_x=type_axis
    type_y=type_axis
    
    buttonsX = []
    buttonsY = []
    buttons_size = []
    figs = {}
    # marker_size=group
    # if type(marker_size) == str:
    #     marker_size_int = df_plot[marker_size]
    #     marker_size_int = 8 + 40*marker_size_int/max(marker_size_int)    
    
    fig = px.scatter(df_plot, y="price", x='waste_total_solid', 
                         color=group,
                         #size = "price",
                         hover_data ={"name": True, 
                                      "prod_category":True, 
                                      "prod_sub_category": True,
                                      "price": ":.2f",
                                      "waste_total_solid": ":.2e"
                                      },
                         log_y=True, 
                         log_x=True,
                         hover_name= "name"
    )
    
    Xname = fig.layout.xaxis.title.text.replace("waste_", "WasteFootprint: ").replace("_", " ").capitalize()
    Yname = fig.layout.yaxis.title.text.replace("waste_", "WasteFootprint: ").replace("_", " ").capitalize()
    DBname = list(df_plot.database.unique())[0]
    GROUPname = group.replace("prod", "Product").replace("_", " ").capitalize()
    fig.update_layout(margin=dict(t=100), 
                          title=dict(text="<b>Comparison of WasteFootprint and LCIA methods</b><br> for {} {} activities in the EcoInvent database: {}".format(len(df_plot), act_selection, DBname), 
                          font=dict(size=24)))
                        #   legend_traceorder = 'reversed',
                          
    fig.update_xaxes(title_font=dict(size=30), title=Xname)
    fig.update_yaxes(title_font=dict(size=30), title=Yname)
    # df_plot["marker_size"] = 8 + 100*df_plot['price']/max(df_plot['price'])
    
    fig.update_traces(marker=dict(size=12, opacity=0.5))
    
    
    for col in sorted((cols_methods+cols_waste+["price"]),reverse=True):
        

        ax_name = col.replace("waste_", "WasteFootprint: ").replace("_", " ").capitalize()

        ax_unit = "€/kg" if ax_name == "Price" else "kg/kg"
        col_col = "colour_"+group
        buttonsX.append(dict(method='update',
                        label='x = {}'.format(ax_name),
                        args=[{'x': [df_plot[col]]},                           
                              {'xaxis.title': "{} ({})".format(ax_name, ax_unit)},
                            {"color":[df_plot[group]]}
                            ])                              
                        )
        
        buttonsY.append(dict(method='update',
                label='y = {}'.format(ax_name),
                args=[{'y': [df_plot[col].values]},
                      {'yaxis.title': "{} ({})".format(ax_name, ax_unit)},
                      {"color":[df_plot[group]]}
                      ]
                      ))
        
        # "trendline":["ols"],
        #               'marker.color=': [df_plot[col_col]],
        #               "trendline_scope":["overall"]
                      
        # buttons_size.append(dict(method='relayout',
        #         label='Size = {}'.format(ax_name),
        #         args = [{'marker.size': [df_plot['marker_size']]},
        #                 {'marker.sizemode': 'area'}]
        #         ))

     #args=[{"x": [x2], "y": [y1], "trendline":["ols"], "trendline_scope":["overall"]}
                           
    buttons_log = [dict(method='relayout',
                        label="Scale: ln", 
                        args = [{'yaxis.type':'log', 'xaxis.type':'log'}],
                        ),
                   dict(method='relayout',
                        label="Scale: linear", 
                        args = [{'yaxis.type':'linear', 'xaxis.type':'linear'}],
                        )
                        
                   ]
    
    buttonsTL = [dict(method='update',
                        label="Trendline",
                        args=[{"trendline":"ols"} ]                    
                        )]

    # fig_all.add_annotation(dict(text=corr_annotation(df_plot[]),
    #                 showarrow=False, 
    #                 yref='paper', xref='paper',
    #                 x=0.99, y=0.95))
    fig.update_layout(updatemenus=[
        # dict(buttons=buttonsTL, x=1.0, y=1.1, yanchor='bottom',direction="down", showactive=True, type='buttons'),
        dict(buttons=buttonsX, x=1.0, y=1.05, xanchor='right',  yanchor='bottom', showactive=True, direction="down"),
        dict(buttons=buttonsY, x=1.0, y=1.0,  showactive=True, yanchor="bottom",direction="down"),
        dict(buttons=buttons_log, x=0.7, y=1.0, xanchor="right", yanchor="bottom", type='buttons', showactive=True)
        ])
    
        # pad={"r": 10, "t": 10})
    # fig_all.update_layout(updatemenus=[dict(buttons=buttonsY, direction='down', x=0.5, y=1.15)])
  
# show the plot and save it
    #fig.show()
    if folder != "":
        folder = folder + "/"
        if not os.path.exists(dir_results_plotly_scatter + folder):
            os.makedirs(dir_results_plotly_scatter + folder)
            
    file_name = dir_results_plotly_scatter + folder + "{}_Combined_{}_{}-{}.html".format(act_selection, DBname, type_x, type_y).replace(" ", "-").replace("/","+").replace(":", "-")
    
    fig.show() 
    fig.write_html(file_name)#, auto_open=True)
    print("* Saved interactive figure: ", file_name)
    fig.write_json(file_name.replace(".html", ".json"))

# adjust the legend size and save the static figure
    fig.update_layout(legend=dict(font=dict(size=16)))
    file_name_static = file_name.replace(".html", ".png")
    fig.write_image(file_name_static, width=1920, height=1080)
    print("* Saved static figure: ", file_name_static)
   

MakePlotlyScatterCombined()
#%% 4c. DEFINE FUNCTION FOR MAKING SINGLE SCATTER PLOTS IN PLOTLY

def MakePlotlyScatter(X, Y, group_scat="prod_sub_category", folder="", marker_size=8, df_plot=df, type_x="log", type_y="log"):
    
    group = group_scat
    print("\n======= Running MakePlotlyScatter() ========\n")
    fig = go.Figure()
    Xname = X.replace("waste_", "").replace("_", " ")#.capitalize()
    Yname = Y.replace("waste_", "").replace("_", " ")#.capitalize()
    DBname = list(df_plot.database.unique())[0]
#!!!! NEED A DICTIONARY OF UNITS FOR EACH METHOD
    Xunit = "€/kg" if Xname == "price" else "kg/kg"
    Yunit = "€/kg" if Yname == "price" else "kg/kg"
    for g in list(df_plot[group].unique()):
        df_g = df_plot[df_plot[group] == g]
            
        trace = go.Scatter(x=df_g[X], y=df_g[Y] 
                        ,mode="markers" 
                        ,name=g
                        ,marker=dict(size=10, opacity=0.6, line=dict(width=0.1), color=colour_dic[g])
                            #hover_name="name"
                            ,customdata = np.stack((df_g.name,
                                                    [Xname]*len(df_g),
                                                    [Yname]*len(df_g),
                                                    df_g.prod_sub_category, 
                                                    df_g.prod_category,
                                                    df_g.price), axis=1)
                            ,hovertemplate=
                                            "<b>%{customdata[0]}</b><br>" +
                                            "%{customdata[1]} : %{x:.2e}<br>" +
                                            "%{customdata[2]} : %{y:.2e}<br>" +
                                            "Category: %{customdata[3]}<br>" +
                                            "Sub-category: %{customdata[4]}<br>" +
                                            "Price: (2005EUR) %{customdata[5]:.2e}" +
                                            "<extra></extra>",
                        
                        )
                        
        fig.add_trace(trace)
    # add a regression line from the OLS model to each group
        try:
            X_reg = df_g[X]
            Y_reg = df_g[Y]
            model = sm.OLS(Y_reg,X_reg).fit()
            #print(results.summary())
            #print("Parameters: ",Y_reg.name,"=", model.params.index[0], "*" ,round(model.params[0],2))
            R2 = model.rsquared
            #print("R2: ", R2)
            fig.add_trace(go.Scatter(x=X_reg, y=model.predict(),
                                    visible="legendonly",
                                        name='    OLS: y = {:.2e} * x , R2 = {:.4e}'.format(model.params[0], R2),
                                        mode='lines', line=dict(color=colour_dic[g], width=0.5)))
        except RuntimeWarning:
            print("Could not add regression line for group", g) 
        
# add a regression line from the OLS model for the whole thing
    X_reg = df_plot[X]
    Y_reg = df_plot[Y]
    model = sm.OLS(Y_reg,X_reg).fit()
    #print(results.summary())
    R2 = model.rsquared
    print("Parameters: y = x*",round(model.params[0],2), "R^2", R2, "\n")
    
    # fig.add_trace(go.Scatter(x=X_reg, y=model.predict(), 
    #                             name='Overall OLS fit:<br> y = {} * x , R2 = {}'.format(round(model.params[0],2), R2),
    #                             mode='lines', line=dict(color='black', width=1)))
    
    if type_x == type_y :
        # reg = LinearRegression().fit(np.vstack(X.values), Y.values)
        # best_fit = reg.predict(np.vstack(X.values))
        # fig.add_trace(go.Scatter(x=X, y=best_fit, name='Regression Fit', mode='lines', line=dict(color='red', width=1)))
        try:
            fig.add_trace(go.Scatter(x=X_reg, y=model.predict(), 
                                name='    Overall OLS fit: y = {:.2e} * x , R2 = {:.3e}'.format(model.params[0], R2),
                                mode='lines', line=dict(color='red', width=1)))
        except Exception as e:
            print(e)
            print("Error in regression line for:", X, Y)
        
# update the plot format
    fig.update_layout(
        title_text="WasteFootprint: {} vs. {} for {} {} activities in the EcoInvent database: {}".format(Xname, Yname, len(df_plot), act_selection, DBname), title_font=dict(size=30)
    )
# update the plot axes with the correct units and scale
    fig.update_xaxes(title_text="{} ({}) - {}".format(Xname.capitalize(), Xunit, type_x), type= type_x, title_font=dict(size=30))
    fig.update_yaxes(title_text="{} ({}) - {}".format(Yname.capitalize(), Yunit, type_y), type=type_y, title_font=dict(size=30))

# show the plot and save it
    #fig.show()
    if folder != "":
        folder = folder + "/"
        if not os.path.exists(dir_results_plotly_scatter + folder):
            os.makedirs(dir_results_plotly_scatter + folder)
            
    file_name = dir_results_plotly_scatter + folder + "{}R2_{}_{}_vs_{}_{}_{}-{}.html".format(round(R2,4), act_selection, Xname, Yname, DBname, type_x, type_y).replace(" ", "-").replace("/","+").replace(":", "-")
    
    
    fig.write_html(file_name)#, auto_open=True)
    print("* Saved interactive figure: ", file_name)
    fig.write_json(file_name.replace(".html", ".json"))

# adjust the legend size and save the static figure
    fig.update_layout(legend=dict(font=dict(size=16)))
    file_name_static = file_name.replace(".html", ".png")
    fig.write_image(file_name_static, width=1920, height=1080)
    print("* Saved static figure: ", file_name_static)

    # add to dictionary of figures
    figs.update({(Xname, Yname): fig})
    group
    return  fig #, file_name

#%% 4b. MAKE SINGLE SCATTER PLOTS IN PLOTLY: the method vs method and waste vs waste data 
# #!!!!!!!!!!!
figs = {}
for i, t in df_waste_corr_all.iterrows():
    print("\n\n====== Figure", i+1,"of",len(df_waste_corr_all),": ", t.tuple[0], "vs.", t.tuple[1])
    #if i > 0: break
    for type_axis in ["log"]:#, "linear"]:
        X = t.tuple[0]
        Y = t.tuple[1]
        fig = MakePlotlyScatter(X, Y, df_plot=df, group_scat="prod_sub_category", type_x=type_axis, type_y=type_axis, folder="WasteVSWaste")
        figs.update({t.tuple: fig})

def figures_to_html(figs, filename=dir_results + "all_scatters.html"):
    with open(filename, 'w') as dashboard:
        dashboard.write("<html><head></head><body>" + "\n")
        for k, v in figs.items():
            inner_html = v.to_html().split('<body>')[1].split('</body>')[0]
            dashboard.write(inner_html)
        dashboard.write("</body></html>" + "\n")
    print("Written {} scatters to a combined html file:\n{}".format(len(figs.items()),filename))

#figures_to_html(figs)


for i, t in df_corr_all.iterrows():
    print("\n\n====== Figure", i+1,"of",len(df_corr),": ", t.tuple[0], "vs.", t.tuple[1])
    for type_axis in ["log"]:#, "linear"]:
        X = t.tuple[0]
        Y = t.tuple[1]
        MakePlotlyScatter(X, Y, folder="WasteVSMethod", df_plot=df, group_scat="prod_sub_category", type_x=type_axis, type_y=type_axis)      



 
#%% 5a. DEFINE THE FUNCTION FOR MAKING SINGLE PLOTLY BOXPLOTS

def MakePlotlyBoxPlot(group_box="", X="waste_total_solid", df_plot=df, folder="", type_x='log', act_selection = "market-selection"):
    group = group_box
    marker_size="price"
    print("\n======= Running MakePlotlyBoxPlot() ========\n")
    fig = go.Figure()
    #df_plot = df_plot.replace(0, 0.0001)
    Xname = X.replace("waste_", "").replace("_", " ")#.capitalize()
    DBname =df_plot.database.unique()[0]
    Xunit = "kg"
    if "price" in Xname:
        Xunit = "€"
    if "ratio" in Xname:
        Xunit = "kg/kg*€" 
       
    for g in sorted(list(df_plot[group].unique()), reverse=True):
        df_g = df_plot[df_plot[group] == g]
        y_label = group.replace("_", " ").capitalize().replace("Prod", "Product")
        trace = go.Box(x=df_g[X]
                            #,boxpoints='all'
                            #,notched=True
                            ,marker=dict(size=8, opacity=0.7, line=dict(width=0.1))
                            #,fillcolor=colour_dic[g]
                            ,line=dict(color=colour_dic[g], width=1)
                            #,hover_name="name",
                            ,name=g
                            ,customdata = np.stack((df_g.name,
                                                    [Xname]*len(df_g),
                                                    df_g.waste_total_solid,
                                                    df_g.waste_total_liquid,
                                                    df_g.prod_sub_category, 
                                                    df_g.prod_category,
                                                    df_g.price,
                                                    ), axis=1)
                            ,hovertemplate=
                                            "<b>%{customdata[0]}</b><br>" +
                                            "%{customdata[1]} : %{x:.2e}<br>" +
                                            "Total (sol): %{customdata[2]:.2e}<br>" +
                                            "Total (liq): %{customdata[3]:.2e}<br>" +
                                            "Category: %{customdata[4]}<br>" +
                                            "Sub-category: %{customdata[5]}<br>" +
                                            "Price: (2005EUR) %{customdata[6]:.2e}" +
                                            "<extra></extra>",
                           
                           )               
        fig.add_trace(trace)
        # add the dots, scaled by whatever the user wants
        if type(marker_size) == str:
            marker_size_int = df_g[marker_size]
            marker_size_int = 8 + 40*marker_size_int/max(marker_size_int)                
            trace = go.Scatter(x=df_g[X]
                                ,y=df_g[group]
                                ,mode="markers"
                                ,marker=dict(size=marker_size_int, opacity=0.4, line=dict(width=0.1), color=colour_dic[g])
                                ,legendgroup=marker_size
                                #,hover_name="name",
                                ,legendgrouptitle=dict(text="Product category and Price (2005EUR)")
                                ,name="    " + marker_size
                                #,visible="legendonly"
                                ,customdata = np.stack((df_g.name,
                                                        [Xname]*len(df_g),
                                                        [Xunit]*len(df_g),
                                                        df_g.waste_total_solid,
                                                        df_g.waste_total_liquid,
                                                        df_g.prod_sub_category, 
                                                        df_g.prod_category,
                                                        df_g.price), axis=1)
                                ,hovertemplate=
                                                "<b>%{customdata[0]}</b><br>" +
                                                "%{customdata[1]} (%{customdata[2]}): %{x:.2e}<br>" +
                                                "Total (sol): %{customdata[3]:.2e}<br>" +
                                                "Total (liq): %{customdata[4]:.2e}<br>" +
                                                "Category: %{customdata[5]}<br>" +
                                                "Sub-category: %{customdata[6]}<br>" +
                                                "Price (2005EUR): %{customdata[7]:.2e]}" +
                                                "<extra></extra>"
                            
                            )
            
            fig.add_trace(trace)
        
# update the plot format
    fig.update_layout(template="seaborn",
        title_text="WasteFootprint: {} vs. {} <br>    for {} {} activities in the EcoInvent database: {}".format(Xname, y_label, len(df_plot), act_selection, DBname), title_font=dict(size=30),
        legend={'traceorder': 'reversed'}
    )
# update the plot axes with the correct units and scale
    fig.update_xaxes(title_text="{} ({}) - {}".format(Xname.capitalize(), Xunit, type_x), type= type_x, title_font=dict(size=24))
    fig.update_yaxes(title_text="{}".format(y_label), title_font=dict(size=24))


    buttonsX = []
    buttonsY = []
    for col in sorted((cols_methods+cols_waste+["price"]),reverse=True):
        

        ax_name = col.replace("waste_", "WasteFootprint: ").replace("_", " ").capitalize()

        ax_unit = "€/kg" if ax_name == "Price" else "kg/kg"
        col_col = "colour_"+group
        
        
        buttonsX.append(dict(method='update',
                        label='x = {}'.format(ax_name),
                        args=[{'x': [df_plot[col]]},                           
                              {'xaxis.title': "{} ({})".format(ax_name, ax_unit)}
                            # {"color":[df_plot[group]]}
                            ])                              
                        )
        
        # buttonsY.append(dict(method='update',
        #         label='y = {}'.format(ax_name),
        #         args=[{'y': [df_plot[col].values]},
        #               {'yaxis.title': "{} ({})".format(ax_name, ax_unit)},
        #               {"color":[df_plot[group]]}
        #               ]

    
    
    
    
    buttons_log = [dict(method='relayout',
                        label="Scale: ln", 
                        args = [{'xaxis.type':'log'},
                                {'yxais.type':'log'}],
                        ),
                   dict(method='relayout',
                        label="linear", 
                        args = [{'xaxis.type':'linear'},
                                {'yxais.type':'linear'}],
                        )
                   ]
    
    fig.update_layout(updatemenus=[
        # dict(buttons=buttonsTL, x=1.0, y=1.1, yanchor='bottom',direction="down", showactive=True, type='buttons'),
        #  dict(buttons=buttonsX, x=1.0, y=1.05, xanchor='right',  yanchor='bottom', showactive=True, direction="down"),
        dict(buttons=buttons_log, x=0.95, y=0.95, xanchor="right", yanchor="bottom", type='buttons', showactive=True)
        ])

# show the plot and save it
    #fig.show()
    if folder != "":
        folder = folder + "/"
        if not os.path.exists(dir_results_plotly_box + folder):
            os.makedirs(dir_results_plotly_box + folder)
    
    #fig.show()    
    file_name = dir_results_plotly_box + folder + "{}_{}_vs_{}_{}_{}.html".format(act_selection, Xname, group, DBname, type_x).replace(" ", "-").replace("/","+").replace(":", "-")
    
    
    fig.write_html(file_name)#, auto_open=True)
    print("* Saved interactive figure: \n", file_name)
    fig.write_json(file_name.replace(".html", ".json"))

# adjust the legend size and save the static figure
    fig.update_layout(showlegend=False)
    file_name_static = file_name.replace(".html", ".png")
    fig.write_image(file_name_static, width=1920, height=1080)
    print("* Saved static figure: \n", file_name_static)
    return  #, file_name


#%% 5b. RUN THE FUNCTION FOR MAKING PLOTLY BOXPLOTS
def MakePlotlyBoxPlotCombined(df, X, group, act_selection, type_x, folder):
    
    print("\n======= Running MakePlotlyBoxPlot() ========\n")
    # fig = go.Figure()
    # #df_plot = df_plot.replace(0, 0.0001)
    # # Xname = X.replace("waste_", "").replace("_", " ")#.capitalize()
    # DBname =df_plot.database.unique()[0]
    # Xunit = "kg"
    # if "price" in Xname:
    #     Xunit = "€"
    # if "ratio" in Xname:
    #     Xunit = "kg/kg*€" 
    
    df_plot = df
    df_plot[group] = df_plot[group].apply(lambda x: x.replace("Textiles", "Textile"))

    y_label = group.replace("_", " ").capitalize().replace("Prod", "Product")
    fig = go.Figure()
    Xname = X.replace("waste_", "").replace("_", " ")#.capitalize()
    DBname =df_plot.database.unique()[0]
    Xunit = "kg"
    if "price" in Xname:
        Xunit = "€"
    if "ratio" in Xname:
        Xunit = "kg/kg*€"
    traces = []
    buttonsX = []
    for X in sorted(cols_waste+["price"], reverse=True):
        Xname = X.replace("waste_", "").replace("_", " ")#.capitalize()
        y_label = group.replace("_", " ").capitalize().replace("Prod", "Product")
        trace = go.Scatter(x=df_plot[X], y=df_plot[group], mode='markers'
                            #,boxpoints='all'
                            #,notched=True
                            ,marker=dict(size=8, opacity=0.7, line=dict(width=0.1))
                            #,fillcolor=colour_dic[g]
                            #,hover_name="name",
                            ,visible='legendonly'
                            ,name=Xname
                            ,showlegend=True
                            
                            ,customdata = np.stack((df_plot.name,
                                                    [Xname]*len(df_plot),
                                                    df_plot.waste_total_solid,
                                                    df_plot.waste_total_liquid,
                                                    df_plot.prod_sub_category, 
                                                    df_plot.prod_category), axis=1)
                            ,hovertemplate=
                                            "<b>%{customdata[0]}</b><br>" +
                                            "%{customdata[1]} : %{x:.2e}<br>" +
                                            "Total (sol): %{customdata[2]:.2e}<br>" +
                                            "Total (liq): %{customdata[3]:.2e}<br>" +
                                            "Category: %{customdata[4]}<br>" +
                                            "Sub-category: %{customdata[5]}" +
                                            "<extra></extra>")              
        # fig.add_trace(trace)
        # traces.append(trace)
    

        ax_unit = "€/kg" if Xname == "Price" else "kg/kg"
        col_col = "colour_"+group
        
        buttonsX.append(dict(method='restyle',
                        label="x = {}".format(Xname),
                        args=[{'visible': True}, [i for i,x in enumerate(traces) if x.name == X]]
                            )                              
                        )

    for g in sorted(cols_waste+["price"], reverse=True):
    # for g in sorted(list(df_plot[group].unique()), reverse=True):
        df_g = df_plot[df_plot[group] == g]
        y_label = group.replace("_", " ").capitalize().replace("Prod", "Product")
        trace = go.Box(x=df_g[g], y=cols_waste
                            ,boxpoints='all'
                            #,notched=True
                            ,marker=dict(size=8, opacity=0.7, line=dict(width=0.1))
                            #,fillcolor=colour_dic[g]
                            #,line=dict(color=colour_dic[g], width=1)
                            #,hover_name="name",
                            #,name=g
                            ,customdata = np.stack((df_g.name,
                                                    [Xname]*len(df_g),
                                                    df_g.waste_total_solid,
                                                    df_g.waste_total_liquid,
                                                    df_g.prod_sub_category, 
                                                    df_g.prod_category), axis=1)
                            ,hovertemplate=
                                            "<b>%{customdata[0]}</b><br>" +
                                            "%{customdata[1]} : %{x:.2e}<br>" +
                                            "Total (sol): %{customdata[2]:.2e}<br>" +
                                            "Total (liq): %{customdata[3]:.2e}<br>" +
                                            "Category: %{customdata[4]}<br>" +
                                            "Sub-category: %{customdata[5]}" +
                                            "<extra></extra>",
                            
                            )
        fig.add_trace(trace)          
    
    
    # Xname = fig.layout.xaxis.title.text.replace("waste_", "WasteFootprint: ").replace("_", " ").capitalize()
    # Yname = fig.layout.yaxis.title.text.replace("waste_", "WasteFootprint: ").replace("_", " ").capitalize()
    DBname = list(df_plot.database.unique())[0]
    GROUPname = group.replace("prod", "Product").replace("_", " ").capitalize()
    fig.update_layout(margin=dict(t=100), 
                          title=dict(text="<b>WasteFootprint and LCIA methods results</b><br> for {} {} activities in the EcoInvent database: {}".format(len(df_plot), act_selection, DBname), 
                          font=dict(size=30)),
                          legend_title=GROUPname,
                        #   legend_traceorder = 'reversed',
                          )
    fig.update_xaxes(title_font=dict(size=30), title=Xname)
    # fig.update_yaxes(title_font=dict(size=30), title=Yname)
# # update the plot format
#     fig.update_layout(template="seaborn",
#         title_text="WasteFootprint: {} vs. {} <br>    for {} {} activities in the EcoInvent database: {}".format(y_label, len(df_plot), act_selection, DBname), title_font=dict(size=30)
#     )
# # update the plot axes with the correct units and scale
    # fig.update_xaxes(title_text="{} ({}) - {}".format(Xname.capitalize(), Xunit, type_x), type= type_x, title_font=dict(size=24))
    # fig.update_yaxes(title_text="{}".format(y_label), title_font=dict(size=24))

        # buttonsY.append(dict(method='update',
        #         label='y = {}'.format(ax_name),
        #         args=[{'y': [df_plot[col].values]},
        #               {'yaxis.title': "{} ({})".format(ax_name, ax_unit)},
        #               {"color":[df_plot[group]]}
        #               ]

    
    buttons_log = [dict(method='relayout',
                        label="Scale: ln", 
                        args = [{'xaxis.type':'log'}],
                        ),
                   dict(method='relayout',
                        label="Scale: linear", 
                        args = [{'xaxis.type':'linear'}],
                        )
                   ]
    
    fig.update_layout(updatemenus=[
        # dict(buttons=buttonsTL, x=1.0, y=1.1, yanchor='bottom',direction="down", showactive=True, type='buttons'),
        dict(buttons=buttonsX, x=1.0, y=1.05, xanchor='right',  yanchor='bottom', showactive=True, direction="down"),
        dict(buttons=buttons_log, x=1.1, y=1.0, xanchor="right", yanchor="bottom", type='buttons', showactive=True)
        ])

# show the plot and save it
    #fig.show()
    if folder != "":
        folder = folder + "/"
        if not os.path.exists(dir_results_plotly_box + folder):
            os.makedirs(dir_results_plotly_box + folder)
    
    # fig.show()    
    file_name = dir_results_plotly_box + folder + "{}_{}_combined_{}_{}.html".format(act_selection, group, DBname, type_x).replace(" ", "-").replace("/","+").replace(":", "-")
    
    fig.write_html(file_name)#, auto_open=True)
    print("* Saved interactive figure: \n", file_name)
    
    fig.write_json(file_name.replace(".html", ".json"))

# adjust the legend size and save the static figure
    fig.update_layout(legend=dict(font=dict(size=16)))
    file_name_static = file_name.replace(".html", ".png")
    #fig.write_image(file_name_static, width=1920, height=1080)
    print("* Saved static figure: \n", file_name_static)
    
    
    return  #, file_name

MakePlotlyBoxPlotCombined(df, act_selection = "market-selection", type_x="log    ", X="waste_total_solid",  group="prod_sub_category", folder="CombinedPlots")
#%% Make the box plots
df = df.reset_index(drop=True)
for i, col in enumerate(df[cols_waste].columns.to_list()):
    print("\n\n====== Figure", i+1,"of",len(df[cols_waste].columns),": ", col)
    for g in ["prod_sub_category", "prod_category"]:
            MakePlotlyBoxPlot(group_box=g, X=col, df_plot=df, folder="Waste", type_x='log',act_selection = "market-selection")

for i, col in enumerate(df[cols_methods].columns.to_list()):
    print("\n\n====== Figure", i+1,"of",len(df[cols_methods].columns),": ", col)
    for g in ["prod_sub_category", "prod_category"]:
        MakePlotlyBoxPlot(group_box=g, X=col, df_plot=df, folder="Methods", type_x='log', act_selection = "market-selection")
        
price_cols = [x for x in df.columns.to_list() if "price" in x]
for i, col in enumerate(df[price_cols].columns.to_list()):
    print("\n\n====== Figure", i+1,"of",len(df[price_cols].columns),": ", col)
    for g in ["prod_sub_category", "prod_category"]:
        MakePlotlyBoxPlot(group_box=g, X=col, df_plot=df, folder="Price", type_x='log', act_selection = "market-selection")
            
# %% END

end = datetime.now()
duration = end - start

cowsay.turtle("Finished Visualisation! Yay! \n Duration: {} ".format(duration))
"""

fig = px.box(
    df,
    x='price_ratio_total_solid',
    y='prod_category',
    points='all',
    notched=True,
    hover_data=["name"],
    color='prod_category',
    labels={'waste_total_solid':"Total solid waste production (kg/kg)",
            'prod_category':"Product Category",
            },
    title="Solid Waste Footprint for ~1500 activities in the EcoInvent 3.9.1 {} database".format(db_name),
    log_x=True
    )
fig.write_html('figures/BOX_TotalSolidWaste_by_ProductCategory.html')
fig.show()

df['price_ratio_total_solid'] = df.waste_total_solid.div(df.price)
df["price_scaled"] = np.sqrt(df.price)
df["solid_total_scaled"] = np.sqrt(df.waste_total_solid)
fig = px.scatter(
    df,
    x='solid_total_scaled',
    y='prod_category',
    size='price_scaled',
    size_max=100,
    hover_data=["name"],
    color='prod_category',
    labels={'waste_total_solid':"Total solid waste production (kg/kg)",
            'prod_category':"Product Category",
            },
    title="Solid Waste Footprint for ~1500 activities in the EcoInvent 3.9.1 {} database".format(db_name),
    log_x=False
    )
fig.write_html('figures/BOX_TotalSolidWaste_by_ProductCategory.html')
fig.show()


fig2 = px.box(
    df,
    x='waste_circ',
    y='prod_category',
    points='all',
    notched=True,
    hover_data=["name"],
    color='prod_category',
    labels={'circ':"Circularity Ratio (%) [(recycled + composted + digested)/total]",
            'prod_category':"Product Category",
            },
    title="Circularity Ratio (%) by product category for ~1500 activities in the EcoInvent 3.9 {} database".format(db_name)
    )

fig2.write_html('figures/BOX_CircularityRatio_by_ProductCategory.html')
fig2.show()


df['waste_categorised_per'] = 100 - df.waste_uncategorised_per
fig3 = px.box(
    df,
    x=df['waste_categorised_per'],
    y='prod_category',
    points='all',
    notched=True,
    hover_data=["name"],
    color='prod_category',
    labels={'waste_uncategorised_per':"Percentage of total waste that has been classified",
            'prod_category':"Product Category",
            },
    title="Percentage of total waste that has been classified ~1500 activities in the EcoInvent 3.9 {} database".format(db_name)
    )

fig3.write_html('figures/BOX_WasteCategorised%_by_ProductCategory.html')
fig3.show()







#%% seaborn


# set template
sns.set_theme(style='whitegrid',
              context='notebook',
              font='sans-serif',
              font_scale=1
              )


settings = ({'text.usetex' : False}
            ,{'font.sans-serif' :'STIXGeneral'})
plt.style.use(settings)

# plot_settings = {'ytick.labelsize': 16,
#                         'xtick.labelsize': 16,
#                         'font.size': 22,
#                         'figure.figsize': (10, 5),
#                         'axes.titlesize': 22,
#                         'axes.labelsize': 18,
#                         'lines.linewidth': 2,
#                         'lines.markersize': 3,
#                         'legend.fontsize': 11,
#                         'mathtext.fontset': 'stix',
#                         'font.family': 'STIXGeneral'}
# plt.style.use(plot_settings)

tuples_box = [('haz_tot_per', 'prod_category'),
          ('circ', 'prod_category'),
          ('waste_uncategorised_per', 'prod_category')
          ]



#%% make box plots
for t in tuples_box:
    x = t[0]
    y = t[1]
    tit = 'BOX_'+x+"_vs_"+y
    f, ax = plt.subplots(figsize=(7, 6))

    sns.boxplot(data=df, x=x, y=y, linewidth=1 ,whis=[0, 100], width=.6, palette="colorblind", showfliers = False)
    sns.stripplot(data=df, x=x, y=y, size=1, color=".3", linewidth=1)

    #ax.set(xscale="log")
    ax.set_xlabel(x)
    ax.set_xlim(right=round(df[x].max()/10)*10)
    if x == 'circ':
        ax.set(xlabel="Circularity Ratio (%) [(recycled + composted + digested)/total]")
        ax.set(title="Circularity Ratio (%) by product category")

    if x == 'haz_tot_per':
        ax.set(xlabel="Hazardous waste as a percentage of total waste")
        ax.set(title="Hazardous waste (%) by product category")

    if x == 'waste_uncategorised_per':
        ax.set(xlabel="Uncategorised waste as a percentage of total waste")
        ax.set(title="Uncategorised waste (%) by product category")

    ax.xaxis.grid(True)
    ax.yaxis.grid(True)

    ax.set(ylabel="")
    #ax.ticklabel_format(style='sci', axis='x')
    sns.despine(trim=True, left=True)

    f.set_tight_layout(True)
    f.savefig(dir_figures +tit+'.svg')



#%% dash stuff
# app = dash.Dash()

# def scat():
#     # Function for creating line chart showing Google stock prices over time
#     fig = go.Figure([go.Scatter(x = df['date'], y = df['GOOG'],\
#                      line = dict(color = 'firebrick', width = 4), name = 'Google')
#                      ])
#     fig.update_layout(title = 'Prices over time',
#                       xaxis_title = 'Dates',
#                       yaxis_title = 'Prices'
#                       )
#     return fig
 """
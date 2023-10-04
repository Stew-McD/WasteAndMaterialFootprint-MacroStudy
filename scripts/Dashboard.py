from dash import Dash, dash_table, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import plotly.io as pio

Scheme = dash_table.Format.Scheme



pio.renderers.default = 'browser'

# Initialize the dash app


# Read the data
# df = pd.read_pickle('batteries_results_cooked.pickle').T
# df = df.applymap(lambda x: x[0])
# df.to_csv('batteries_results_cooked.csv')

df = pd.read_csv('batteries_results_cooked.csv', index_col=0)
df.drop('waste_digestion_cubicmeter', axis=1, inplace=True)

df.index.set_names('Battery', inplace=True)

df.columns = df.columns.str.replace('waste_', '')

acts = df.index.to_list()
cats = df.columns.to_list()
cats_liq = [x for x in cats if '_cubicmeter' in x]
cats_sol = [x for x in cats if '_kilogram' in x]
df_liq = pd.DataFrame(df, columns=cats_liq)
df_sol = pd.DataFrame(df, columns=cats_sol)

df_liq = df_liq.sort_values(by='total_cubicmeter', ascending=False)
df_sol = df_sol.sort_values(by='total_kilogram', ascending=False)

df_sol.columns = df_sol.columns.str.replace('_kilogram','')
df_liq.columns = df_liq.columns.str.replace('_cubicmeter', '')

df_liq_cat = df_liq.loc[:, ~df_liq.columns.isin(['total', 'non-hazardous', 'hazardous'])]
df_sol_cat = df_sol.loc[:, ~df_sol.columns.isin(['total', 'non-hazardous', 'hazardous'])]

df_liq_cat['undefined EoL'] = df_liq.total - df_liq_cat.sum(axis=1)
df_sol_cat['undefined EoL'] = df_sol.total - df_sol_cat.sum(axis=1)

df_liq_cat = df_liq_cat.div(df_liq.total, axis=0)
df_sol_cat = df_sol_cat.div(df_sol.total, axis=0)

# df_liq_cat_norm = df_liq_cat.applymap(lambda x: round(x, 2))
# df_sol_cat = df_sol_cat.applymap(lambda x: round(x, 2))


# make the figure
fig_liq = px.bar(df_liq.total,
                    text_auto='.2f',
                    barmode="relative",
                    labels={
                        "variable":"Waste category",
                        "index" : "Battery",
                        "value" : "Waste Footprint(kg/kg)"
                            },
                    title="Liquid WasteFootprint for Li-ion battery production"
                    )

fig_sol =  px.bar(df_sol.total,
                    text_auto='.2f',
                    barmode="relative",
                    labels={
                        "variable":"Waste category",
                        "index" : "Battery",
                        "value" : "Percentage of Waste Footprint(kg/kg)"
                            },
                    title="Solid WasteFootprint for Li-ion battery production"
                    )

fig_sol_cat = px.bar(df_sol_cat, 
             text_auto='.1%', 
             barmode="relative",
             labels={
                    "variable":"Waste category",
                    "index" : "Battery",
                    "value" : "Waste Footprint(kg/kg)"
                        },
                title="End of Life categorisation for the Solid WasteFootprint of Li-ion battery production"
             )

fig_liq_cat = px.bar(df_liq_cat, 
             text_auto='.1%', 
             barmode="relative",
             labels={
                    "variable":"Waste category",
                    "index" : "Battery",
                    "value" : "EoL categorisation "
                        },
                title="End of Life categorisation for the Liquid WasteFootprint of Li-ion battery production"
             )                       


# %%Define the styles
""" colors = {
    'background': '#111111',
    'text': '#7FDBFF'
} """
""" fig.update_layout(
    plot_bgcolor=colors['background'],
    paper_bgcolor=colors['background'],
    font_color=colors['text']
) """

#%%

# Make the figure
# fig_liq_cat.show()                        
# fig_sol_cat.show()
#Update the figure layout
row = html.Div(
    [
        dbc.Row(dbc.Col(html.Div("A single column"))),
        dbc.Row(
            [
                dbc.Col(html.Div("One of three columns")),
                dbc.Col(html.Div("One of three columns")),
                dbc.Col(html.Div("One of three columns")),
            ]
        ),
    ]
)




app = Dash(__name__)
# Set up the layout of the app

format = dash_table.Format.Format()
formatted = format.precision(value=2).si_prefix(None).scheme('e')
# formatted_percent = format.precision(value=1).scheme(Scheme.percentage)
formatted_percent = {'locale': {}, 'nully': '', 'prefix': None, 'specifier': '.2%'}

app.layout = html.Div(children=[
    html.H1(children='The supply-chain WasteFootprint of Batteries',
            style={
            'textAlign': 'center',
            #'color': colors['text']
        }),
    
    html.Div(children='The specific waste production (kg/kg) for different battery technologies in the EcoInvent 3.9 database.'),
    
    dcc.Graph(
        id = 'fiq_sol',
        figure=fig_sol
    ),
    dash_table.DataTable(      
        id="tab_sol",
        columns=[
            {
                "name": i,
                "id": i,
                "type": "numeric",  # Required!
                "format": formatted
            }
            for i in df_sol.reset_index().columns
        ],
        data=df_sol.reset_index().to_dict('records'),
        editable=True,
        fill_width=False
    ),
    
    dcc.Graph(
        id='fiq_sol_cat',
        figure=fig_sol_cat
    ),
    dash_table.DataTable(      
        id="tab_sol_cat",
        columns=[
            {
                "name": i,
                "id": i,
                "type": "numeric",  # Required!
                "format": formatted_percent
            }
            for i in df_sol_cat.reset_index().columns
        ],
        data=df_sol_cat.reset_index().to_dict('records'),
        editable=True,
        fill_width=False
    ),
    dcc.Graph(
        id = 'fiq_liq',
        figure = fig_liq
    ),
        dash_table.DataTable(      
        id="format_table",
        columns=[
            {
                "name": i,
                "id": i,
                "type": "numeric",  # Required!
                "format": formatted
            }
            for i in df_liq.reset_index().columns
        ],
        data=df_liq.reset_index().to_dict('records'),
        editable=True,
        fill_width=False
    ),
    dcc.Graph(
        id='fig_liq_cat',
        figure=fig_liq_cat
    ),
    dash_table.DataTable(      
        id="tab_liq_cat",
        columns=[
            {
                "name": i,
                "id": i,
                "type": "numeric",  # Required!
                "format": formatted_percent
            }
            for i in df_liq_cat.reset_index().columns
        ],
        data=df_liq_cat.reset_index().to_dict('records'),
        editable=True,
        fill_width=False
    )
])
                      
                      
# Run the app
app.run(debug=True)
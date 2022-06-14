#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import snpy

from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import html, dcc
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform

# Initial Setup
app = DashProxy(prevent_initial_callbacks=True, transforms=[MultiplexerTransform()])
server = app.server


# In[2]:


params_df = pd.read_csv('sn_parameters.csv')

z = params_df.z.values
mag_cosmo = cosmo.distmod(z).value - 18
params_df["residuals"] = params_df.Jmax.values - mag_cosmo

init_sn_name = params_df.name.values[0]


# In[3]:


#-------
# Graphs

colors = {'background': '#111111',
          'text': '#7FDBFF'}

# Hubble Diagram
fig1 = px.scatter(params_df, x="z", y="Jmax", error_y="Jmax_err", hover_data=['name'], width=700, height=600)
fig1.update_traces(marker=dict(size=10,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

fig1.update_layout({#'plot_bgcolor': colors['background'],
                    #'paper_bgcolor': colors['background'],
                    'font': {'color': colors['text'],
                             'size': 20},
                   })

fig2 = px.scatter(params_df, x="z", y="residuals", error_y="Jmax_err", hover_data=['name'], width=700, height=600)
fig2.update_traces(marker=dict(size=10,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

fig2.update_layout({#'plot_bgcolor': colors['background'],
                    #'paper_bgcolor': colors['background'],
                    'font': {'color': colors['text'],
                             'size': 20},
                   })

graph1 = dcc.Graph(id='hd', figure=fig1)
graph2 = dcc.Graph(id='res', figure=fig2)
graph3 = dcc.Graph(id='test', figure=fig2)

#-------
# Layout

dropdown = dcc.Dropdown(params_df['name'].unique(),
                        id='sn_name')

app.layout = html.Div(children=[

                    # Title
                    html.H1(children='Hubble Diagram',
                            style={'textAlign': 'center'}
                    ),
    
                    html.Div([
                        html.Div(dropdown),
                    ], style={'width': '30%'}),

                    html.Div([
                        html.Div(graph1),
                    ], style={'width': '49%', 'display': 'inline-block'}),
    
                    html.Div([
                        html.Div(graph3)
                    ], style={'display': 'inline-block'}),

])

#----------
# Callbacks

def plot_sn(sn_name):
    sn_file = os.path.join('data', f'{sn_name}_snpy.txt')
    sn = snpy.import_lc(sn_file)

    time = sn.data['g'].MJD
    mag = sn.data['g'].mag

    fig = px.scatter(x=time, y=mag, title=f'SN {sn_name}')
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(margin={'l': 40, 'b': 20, 't': 40, 'r': 0}, hovermode='closest')

    return fig

@app.callback(
    Output('test', 'figure'),
    Input('sn_name', 'value'),)
def update_sn_graph_dropdown(sn_name):
    fig = plot_sn(sn_name)
    return fig

@app.callback(
    Output('test', 'figure'),
    Input('hd', 'clickData'),)
def update_sn_graph_click(clickData):
    # clickData is a dictionary
    sn_name = clickData['points'][0]['customdata'][0]
    fig = plot_sn(sn_name)
    return fig

#-----------------
# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=False)


# In[ ]:


#-------
# Graphs

colors = {'background': '#111111',
          'text': '#7FDBFF'}

# Hubble Diagram
fig1 = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.01)

fig1.add_trace(go.Scatter(x=params_df.z, y=params_df.Jmax, text=params_df.name, hoverinfo='text'),
              row=1, col=1)

fig1.add_trace(go.Scatter(x=params_df.z, y=params_df.residuals, text=params_df.name, hoverinfo='text'),
              row=2, col=1)

fig1.update_xaxes(title_text='z', row=2, col=1)
fig1.update_yaxes(title_text="Jmax", row=1, col=1)
fig1.update_yaxes(title_text="Residual", row=2, col=1)
fig1.update_layout(height=800, width=1000)
fig1.update_traces(mode="markers")



fig2 = px.scatter(params_df, x="z", y="residuals", error_y="Jmax_err", hover_data=['name'], width=700, height=600)
fig2.update_traces(marker=dict(size=10,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))

fig2.update_layout({#'plot_bgcolor': colors['background'],
                    #'paper_bgcolor': colors['background'],
                    'font': {'color': colors['text'],
                             'size': 20},
                   })

graph1 = dcc.Graph(id='hd', figure=fig1)
graph2 = dcc.Graph(id='res', figure=fig2)
graph3 = dcc.Graph(id='test', figure=fig2)

#-------
# Layout

dropdown = dcc.Dropdown(params_df['name'].unique(),
                        #value=init_sn_name,
                        id='sn_name')

app.layout = html.Div(children=[

                    # Title
                    html.H1(children='Hubble Diagram',
                            style={'textAlign': 'center'}
                    ),
    
                    html.Div([
                        html.Div(dropdown),
                    ], style={'width': '30%'}),

                    html.Div([
                        html.Div(graph1),
                    ], style={'width': '49%', 'display': 'inline-block'}),
    
                    html.Div([
                        html.Div(graph3)
                    ], style={'display': 'inline-block'}),

])

#----------
# Callbacks

def plot_sn(sn_name):
    sn_file = os.path.join('data', f'{sn_name}_snpy.txt')
    sn = snpy.import_lc(sn_file)

    time = sn.data['g'].MJD
    mag = sn.data['g'].mag

    fig = px.scatter(x=time, y=mag, title=f'SN {sn_name}')
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(margin={'l': 40, 'b': 20, 't': 40, 'r': 0}, hovermode='closest')

    return fig

@app.callback(
    Output('test', 'figure'),
    Input('sn_name', 'value'),)
def update_sn_graph_dropdown(sn_name):
    fig = plot_sn(sn_name)
    return fig

@app.callback(
    Output('test', 'figure'),
    Input('hd', 'clickData'),)
def update_sn_graph_click(clickData):
    # clickData is a dictionary
    sn_name = clickData['points'][0]['text']
    fig = plot_sn(sn_name)
    return fig

#-----------------
# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=False)


# In[ ]:





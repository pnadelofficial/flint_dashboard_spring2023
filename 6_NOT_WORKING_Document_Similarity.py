import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import networkx as nx
from datasets import load_dataset
import ast
from streamlit_plotly_events import plotly_events
import plotly.io as pio
pio.templates.default = "plotly"

@st.cache_resource
def get_data():
    graph = pickle.load(open('./data/semanticGraph_4-4-23.p','rb'))
    dataset = load_dataset("pnadel/michgovparsed8_16")
    temp = dataset['train'].to_pandas()
    for col in ['From', 'Sent', 'To', 'Cc', 'Subject', 'Attachment']:
        temp[col] = temp[col].apply(ast.literal_eval)
    df = temp.drop(temp.loc[temp.Body == 'nan'].index)
    return graph, df
graph,df = get_data()

# functions
def get_graph_data(G):

    # displays only the largest component of the graph
    components = list(G.subgraph(c) for c in nx.connected_components(G))
    G = components[0]
    
    # removes nodes with few connections
    min_connections = 5
    trimmed_graph = nx.Graph()                                                                                                                                     
    trimmed_graph_edges = filter(lambda x: G.degree()[x[0]] > min_connections and G.degree()[x[1]] > min_connections, G.edges())
    trimmed_graph.add_edges_from(trimmed_graph_edges)
    G = trimmed_graph

    pos = nx.drawing.layout.spring_layout(G)
    nx.set_node_attributes(G, pos, 'pos')

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = G.nodes[edge[0]]['pos']
        x1, y1 = G.nodes[edge[1]]['pos']
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = G.nodes[node]['pos']
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Electric',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        # creates the hover-over text that pops up on the graph after a mouse event
        if len(df["Subject"].iloc[adjacencies[0]]) != 0:
          node_text.append(f'{str(df["Subject"].iloc[adjacencies[0]][0][0])} has '+str(len(adjacencies[1]))+' connection(s)')
        elif len(df["Body"].iloc[adjacencies[0]]) >= 20:
          node_text.append(f'{str(df["Body"].iloc[adjacencies[0]][:20])} has '+str(len(adjacencies[1]))+' connection(s)')
        else:
          node_text.append(f'{str(df["Body"].iloc[adjacencies[0]])} has '+str(len(adjacencies[1]))+' connection(s)')
 

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text 

    return edge_trace, node_trace

# def display_text()

edge_trace, node_trace = get_graph_data(graph)
plot_df = pd.DataFrame({'x':node_trace['x'], 'y':node_trace['y'], 'connections':node_trace['marker']['color'], 'text':list(node_trace['text'])})

fig = px.scatter(plot_df, x='x', y='y', color='connections', hover_data=['text'])
fig.add_trace(edge_trace)
fig.data = (fig.data[1],fig.data[0])

fig_selected = plotly_events(fig, select_event=True)

st.write(fig_selected)

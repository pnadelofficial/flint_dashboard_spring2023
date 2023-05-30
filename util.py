import plotly.graph_objects as go
import streamlit as st
import gdown

@st.cache_resource
def get_data():
    # dit logits
    dit_logits_url = 'https://drive.google.com/file/d/1ICc_RtUcZYZrZUndLHAVWLk5MK3P5xic/view?usp=sharing'
    dit_logits_out = 'data/dit_logits_embedded.csv'
    gdown.download(dit_logits_url, dit_logits_out, quiet=True, fuzzy=True)
    # sem search index
    fl_index_url = 'https://drive.google.com/drive/folders/1BAJNUpOOVCYHsjoQUf_zv5yipOFfldZU?usp=sharing'
    fl_index_out = 'data/fl_index'
    gdown.download_folder(url=fl_index_url, output=fl_index_out, quiet=True, use_cookies=False)
    # lookup data for threads
    metadata_for_threads_url = 'https://drive.google.com/file/d/11axKr60BUpe3nJW69RElIOa4CqHrSi2r/view?usp=sharing'
    metadata_for_threads_out = 'data/metadata_for_threads.csv'
    gdown.download(metadata_for_threads_url, metadata_for_threads_out, quiet=True, fuzzy=True)
    return None

def plot_network(G, title):
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
            colorscale='YlGnBu',
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
        node_text.append(f'{str(adjacencies[0])} has '+str(len(adjacencies[1]))+' connection(s)')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                 layout=go.Layout(
                    title=f'{title}',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

def plot_igraph_plotly(g, layout, title):
    """Takes a igraph graph object and plots it in plotly acccording to given layout"""
    Xe, Ye = [], []
    nr_vertices = len(g.vs)
    labels = g.vs.get_attribute_values('_nx_name')

    lay = g.layout(layout)
    position = {k: lay[k] for k in range(nr_vertices)}
    L = len(position)
    Y = [lay[k][1] for k in range(nr_vertices)]
    M = max(Y)

    E  = [e.tuple for e in g.es]
    Xn = [position[k][0] for k in range(L)]
    Yn = [2*M-position[k][1] for k in range(L)]

    for edge in E:
        Xe += [position[edge[0]][0], position[edge[1]][0], None]
        Ye+=[2*M-position[edge[0]][1],2*M-position[edge[1]][1], None]

    fig = go.Figure(
        layout=go.Layout(title=f'{title}')
    )
    fig.add_trace(go.Scatter(x=Xe,
                    y=Ye,
                    mode='lines',
                    line=dict(color='rgb(210,210,210)', width=3),
                    hoverinfo='none'
                    ))
    fig.add_trace(go.Scatter(x=Xn,
                    y=Yn,
                    mode='markers',
                    name='bla',
                    marker=dict(symbol='circle-dot',
                                    size=6,
                                    color='#6175c1',    #'#DB4551',
                                    line=dict(color='rgb(50,50,50)', width=1)
                                    ),
                    text=labels,
                    hoverinfo='text',
                    opacity=0.8
                    ))
    fig.update_layout(showlegend=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig

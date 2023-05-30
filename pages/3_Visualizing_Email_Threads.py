import streamlit as st
import pickle
import networkx as nx
import igraph as ig
import pandas as pd
from util import *

@st.cache_resource
def get_data():
    with open('./data/threads_by_email.p', 'rb') as f:
        email_threads = pickle.load(f)

    metadata = pd.read_csv('./data/metadata_for_threads.csv')
    return email_threads, metadata
email_threads, metadata = get_data()

st.title("Visualizing Email Threads")
st.write("""
    Once we understood the make up of our data, we selected just the emails and sought to reconstruct
    the connections between them. Below you can see how individual email conversations develop over time.
    **Hover over nodes in the network graph to see details.**
    """)

with st.expander('Read more'):
        st.markdown("""
            Caleb Pekowsky took on the difficult task on reconstructing email threads. After many attempts,
            he was able to create the structure below by using the pagination at the bottom of each page
            in the FOIA response. Each email thread began with a pagination of 1 and continued until the pagination 
            reset to 1 and the thread ended. Although helpful, this process only returned a list of images which 
            were related, so we used a Named Entity Recognition model (see *Person to Person Networks*) to split 
            the images into emails and extract relevant metadata from each. 
            """)

n = st.number_input('Choose a minimum thread length to display', min_value=1, max_value=max([len(l) for l in email_threads])-1, value=5)

longThreads = []
for ethread in email_threads:
    if len(ethread) > n:
        longThreads.append(ethread)

G = nx.DiGraph()
for ethread in longThreads:
    for previous, current in zip(ethread, ethread[1:]):
        G.add_edge(previous,current, attr=ethread )
pos = nx.drawing.layout.spring_layout(G)
nx.set_node_attributes(G, pos, 'pos')

g = ig.Graph.from_networkx(G)
fig = plot_igraph_plotly(g, layout='rt_circular', title=f'Thread Length over {n}')
st.plotly_chart(fig)

st.markdown('<small>This tool is for educational purposes ONLY!</small>', unsafe_allow_html=True)
st.markdown('<small>Assembled by Peter Nadel | TTS Research Technology</small>', unsafe_allow_html=True)
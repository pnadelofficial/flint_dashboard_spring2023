import streamlit as st
import pickle
import plotly.graph_objects as go
import networkx as nx
from util import *

st.set_page_config(page_title="Flint Graph Streamlit App")

st.title("Flint Node of Interest Subgraph")
st.write(
    """
    Because the FOIA response rendered the emails as images, we lost all ability to extract metadata fields,
    like the sender and the recipient. On top of this, errors from Optical Character Recognition (OCR) made
    rules-based systems impractical. To confront this problem, a Named Entity Recognition (NER) model, which 
    was able to extract these details, was trained. Afterwards, Kirin arranged this data
    into a Person to Person network. Below, **you can choose from a pre-defined list of 10 names in a dropdown 
    menu and click a button to display the corresponding subgraph**, or **you can search for a name by 
    manually entering it in the below text box which has a dropdown of possible names that dynamically 
    updates as you type.**
    """
    )
with st.expander('Read more'):
    st.markdown("""
        To train the NER model, Kirin Godhwani and Peter Nadel manually tagged almost 700 emails, highlighting
        the From, To, Cc, Sent (the date an email was sent), Subject and Attachment lines in the OCR text of 
        the emails. With this training data, Kirin used [spaCy](https://spacy.io/)'s command line interface
        to train the model. This process employs the [tok2vec](https://spacy.io/api/tok2vec) algorithm to generate 
        token-level embeddings which can be used to predict whether or not a token is a part of a type of entity.
        In our case, the entity types were each metadata field that we wanted to extract from an email. Once we
        had a model working well, Kirin was able to create a network representation of the Flint emails dataset.
        In this network, each node is a person and an edge exists between two nodes if those two people ever 
        exchanged emails. 
        """)

@st.cache_resource
def get_data():
    graph1 = pickle.load(open('./data/saved_graph4_19.p','rb'))
    C = [graph1.subgraph(c) for c in nx.connected_components(graph1)]
    G = C[0]
    return G
G = get_data()

def display_subgraph(node_name):
    G_sub = G.subgraph(nx.single_source_shortest_path_length(G, node_name, cutoff=1))
    return plot_network(G_sub, title=f'Subgraph containing node {node_name}')

st.markdown("<hr style='border:1px solid black'>", unsafe_allow_html=True)
st.subheader("Choose from 10 Pre-Defined Names")

example_names = [
    "Moore, Kristin", "Kelenske, Chris (MSP)", "McShane, Hilda", 
    "Russo, Mark (MSP)", "Leix, Ron (MSP)", "MSP-EOC-MDEQ", 
    "Kuzera, Michelle (MSP)", "Lasher, Tony P.", "Morris, David (MSP)", "Eickholt, Jay (MSP)"
    ]

selected_name = st.selectbox("Select an example name from 10 choices: ", example_names)
if st.button("Display Graph for Selected Name"):
   chart = display_subgraph(selected_name)
   st.plotly_chart(chart)

st.markdown("<hr style='border:1px solid black'>", unsafe_allow_html=True)
st.subheader("Manually Enter a Name of your Choosing")
possible_names = list(G.nodes)

name = st.selectbox('Manually type a name and choose from a dynamic dropdown containing all possible names: ', possible_names)

if name != '':
    chart = display_subgraph(name)
    st.plotly_chart(chart)

st.markdown('<small>This tool is for educational purposes ONLY!</small>', unsafe_allow_html=True)
st.markdown('<small>Assembled by Peter Nadel | TTS Research Technology</small>', unsafe_allow_html=True)

import streamlit as st
import pandas as pd
from datasets import load_dataset
import ast
from txtai.embeddings import Embeddings

@st.cache_resource
def get_data():
    embeddings = Embeddings()
    embeddings.load('./data/fl_index') # can change

    fl_sents = pd.read_csv('./data/fl_sents.csv')
    dataset = load_dataset("pnadel/michgovparsed8_16")
    df = dataset['train'].to_pandas()
    for col in ['From', 'Sent', 'To', 'Cc', 'Subject', 'Attachment']:
        df[col] = df[col].apply(ast.literal_eval)
    fl_ref = df.copy()
    del df
    return embeddings, fl_sents, fl_ref
embeddings, fl_sents, fl_ref = get_data()

def display_text(tup, context=1):
    selection = fl_ref.iloc[fl_sents.org_idx[tup[0]]]
    st.markdown(f"<small style='text-align: right;'>From: <b>{selection.From[0][0] if len(selection.From)>0 else 'No sender found'}</b></small>",unsafe_allow_html=True)
    st.markdown(f"<small style='text-align: right;'>To: <b>{selection.To[0][0] if len(selection.To)>0 else 'No recipient found'}</b></small>",unsafe_allow_html=True)
    st.markdown(f"<small style='text-align: right;'>Cc: <b>{selection.Cc[0][0] if len(selection.Cc)>0 else 'No Cc found'}</b></small>",unsafe_allow_html=True)
    st.markdown(f"<small style='text-align: right;'>Date: <b>{selection.Sent[0][0] if len(selection.Sent)>0 else 'No date found'}</b></small>",unsafe_allow_html=True)
    st.markdown(f"<small style='text-align: right;'>Subject: <b>{selection.Subject[0][0] if len(selection.Subject)>0 else 'No subject found'}</b></small>",unsafe_allow_html=True)
    st.markdown(f"<small style='text-align: right;'>Similarity score: <b>{round(tup[1], 3)}</b></small>",unsafe_allow_html=True)

    res = fl_sents.sents[tup[0]]
    res = f"<span style='background-color:#fdd835'>{res}</span>"

    before, after = [], []
    for i in range(context+1):
        if i != 0:
            if fl_sents.org_idx[tup[0]-i] == fl_sents.org_idx[tup[0]]:
                before.append(fl_sents.sents[tup[0]-i])
            if fl_sents.org_idx[tup[0]+i] == fl_sents.org_idx[tup[0]]:
                after.append(fl_sents.sents[tup[0]+i] )
    
    before = '\n'.join(before)
    after  = '\n'.join(after )
    to_display = '\n'.join([before,res,after]).replace('$', '\$').replace('`', '\`')

    st.markdown(to_display,unsafe_allow_html=True)
    st.markdown("<hr style='width: 75%;margin: auto;'>",unsafe_allow_html=True)

st.title('Semantic Search')
st.write("""
    Semantic search provides a vital tool for researcher working with large bodies of text. Each document is encoded
    using pretrained text-based transformer models, producing a vector which represents the linguistic features of that
    document. A search query is the encoded in the same way and then the search query vector is compared to each document 
    embedding using cosine similarity. Documents with the highest score are presented as the results of the search. Below,
    **input any search and see what documents from the Flint emails dataset come up.** 
""")
with st.expander('Read more'):
    st.markdown("""
        Ethan Nanavati approached the task of semantic search for the Flint emails dataset. Though the basic principle
        of semantic search are straightfoward, the details can be quite complicated. For instance, maintaining an
        index of document embeddings so that embeddings do not need to be calculated each time a new query is entered 
        is not trivial. Ethan employed [txtai](https://neuml.github.io/txtai/) to handle many of these issues, but still
        needed to determine what pretrianed model would suit the Flint data the best. After much testing, Ethan decided
        on [msmarco-bert-base-dot-v5](https://www.sbert.net/examples/training/ms_marco/README.html) to embed the Flint
        documents. This sentence transformer embeds texts based on similarity scores so it fits the use case of semantic
        search well. 
        """)

query = st.text_input('Search any query')
results = st.number_input(value=5, label='Choose the amount of results you want to see')
context = st.number_input(value=1, label='Choose a context size')

uids = embeddings.search(query, results)

if query != '':
    for tup in uids:
        display_text(tup, context)

st.markdown('<small>This tool is for educational purposes ONLY!</small>', unsafe_allow_html=True)
st.markdown('<small>Assembled by Peter Nadel | TTS Research Technology</small>', unsafe_allow_html=True)
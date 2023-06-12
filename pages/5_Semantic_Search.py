import streamlit as st
import pandas as pd
import numpy as np
from txtai.embeddings import Embeddings
from ast import literal_eval
from streamlit_image_select import image_select

@st.cache_resource
def get_data():
    embeddings = Embeddings()
    embeddings.load('./data/fl_index6122023') # can change

    # fl_sents = pd.read_csv('./data/fl_sents.csv')
    # dataset = load_dataset("pnadel/michgovparsed8_16")
    # df = dataset['train'].to_pandas()
    # for col in ['From', 'Sent', 'To', 'Cc', 'Subject', 'Attachment']:
    #     df[col] = df[col].apply(ast.literal_eval)
    # fl_ref = df.copy()
    # del df

    metadata = pd.read_csv('./data/michgov6122023.csv').dropna(subset='Body').reset_index(drop=True)
    for col in ['To', 'Cc']:
        metadata[col] = metadata[col].apply(literal_eval)

    dit_logits = pd.read_csv('./data/dit_logits_embedded.csv').drop(['Unnamed: 0'], axis=1)

    return embeddings, metadata, dit_logits
embeddings, metadata, dit_logits = get_data()

def lookup_image(page):
    return dit_logits.aws_path.loc[dit_logits.aws_path.str.contains(page.split('.')[0])].iloc[0]

def display_text(tup):
    selection = metadata.iloc[tup[0]]
    st.markdown(f"<small style='text-align: right;'>From: <b>{selection.From if selection.From is not np.nan else 'No sender found'}</b></small>",unsafe_allow_html=True)
    st.markdown(f"<small style='text-align: right;'>To: <b>{selection.To[0] if len(selection.To)>0 else 'No recipient found'}</b></small>",unsafe_allow_html=True)
    st.markdown(f"<small style='text-align: right;'>Cc: <b>{selection.Cc[0] if len(selection.Cc)>0 else 'No Cc found'}</b></small>",unsafe_allow_html=True)
    st.markdown(f"<small style='text-align: right;'>Date: <b>{selection.Sent if selection.Sent is not np.nan else 'No date found'}</b></small>",unsafe_allow_html=True)
    st.markdown(f"<small style='text-align: right;'>Subject: <b>{selection.Subject if selection.Subject is not np.nan else 'No subject found'}</b></small>",unsafe_allow_html=True)
    st.markdown(f"<small style='text-align: right;'>Similarity score: <b>{round(tup[1], 3)}</b></small>",unsafe_allow_html=True)

    thread_id = selection.thread_index
    pages_to_show = metadata.iloc[metadata.groupby('thread_index').groups[thread_id]].image_lookup.unique()
    images_to_show = [lookup_image(page) for page in pages_to_show]
    
    if len(images_to_show) > 2:
        img_select = image_select("See the images in this thread", images_to_show, captions=[cap.split('/')[-1] for cap in images_to_show])
        if img_select:
                st.image(img_select, width=420)
    else:
        st.image(images_to_show[0], caption=images_to_show[0].split('/')[-1])

    st.markdown("<hr style='width: 75%;margin: auto;'>",unsafe_allow_html=True)

st.title('Semantic Search')
st.write("""
    Semantic search provides a vital tool for researchers working with large bodies of text. Each document is encoded
    using pretrained text-based transformer models, producing a vector which represents the linguistic features of that
    document. A search query is then encoded in the same way and then the search query vector is compared to each document 
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

uids = embeddings.search(query, results)

if query != '':
    for tup in uids:
        display_text(tup)

st.markdown('<small>This tool is for educational purposes ONLY!</small>', unsafe_allow_html=True)
st.markdown('<small>Assembled by Peter Nadel | TTS Research Technology</small>', unsafe_allow_html=True)

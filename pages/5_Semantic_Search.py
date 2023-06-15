import streamlit as st
import pandas as pd
import numpy as np
from txtai.embeddings import Embeddings
from thefuzz import fuzz
from ast import literal_eval
from streamlit_image_select import image_select
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
import re 

@st.cache_resource
def get_data():
    embeddings = Embeddings()
    embeddings.load('./data/fl_index6132023') 

    # sentence_encoder = AutoModel.from_pretrained("biu-nlp/abstract-sim-sentence")
    # query_encoder = AutoModel.from_pretrained("biu-nlp/abstract-sim-query")
    # tokenizer = AutoTokenizer.from_pretrained("biu-nlp/abstract-sim-sentence")

    metadata = pd.read_csv('./data/michgov6122023.csv').dropna(subset='Body').reset_index(drop=True)
    for col in ['To', 'Cc']:
        metadata[col] = metadata[col].apply(literal_eval)
    
    sents = pd.read_csv('./data/fl_sents6132023.csv')

    dit_logits = pd.read_csv('./data/dit_logits_embedded.csv').drop(['Unnamed: 0'], axis=1)

    return embeddings, metadata, sents, dit_logits
embeddings, metadata, sents, dit_logits = get_data()

def lookup_image(page):
    return dit_logits.aws_path.loc[dit_logits.aws_path.str.contains(page.split('.')[0])].iloc[0]

def remove_duplicates(strings):
    unique_strings = []
    for string in strings:
        is_dup = False
        for unique in unique_strings:
            fuzz_score = fuzz.ratio(string[0], unique[0])
            if fuzz_score >= 70:
                is_dup = True
                break
        if not is_dup:
            unique_strings.append(string)
    return unique_strings

def encode_batch(model, tokenizer, sentences: List[str], device: str):
    input_ids = tokenizer(sentences, padding=True, max_length=512, truncation=True, return_tensors="pt",
                          add_special_tokens=True).to(device)
    features = model(**input_ids)[0]
    features =  torch.sum(features[:,1:,:] * input_ids["attention_mask"][:,1:].unsqueeze(-1), dim=1) / torch.clamp(torch.sum(input_ids["attention_mask"][:,1:], dim=1, keepdims=True), min=1e-9)
    return features

# def rerank(unique_strings):
#     sent_embeddings = encode_batch(
#         sentence_encoder,
#         tokenizer,
#         [tup[0] for tup in unique_strings],
#         'cpu'
#     ).detach().cpu().numpy()
#     query_embedding = encode_batch(
#         query_encoder,
#         tokenizer,
#         [f'<query>: {query}'],
#         'cpu'
#     ).detach().cpu().numpy()
#     sims = cosine_similarity(query_embedding, sent_embeddings)[0]
#     sentences_sims = list(zip([tup for tup in unique_strings], sims))
#     sentences_sims.sort(key=lambda x: x[1], reverse=True)
#     return sentences_sims

def escape_markdown(text):
    '''Removes characters which have specific meanings in markdown'''
    MD_SPECIAL_CHARS = "\`*_{}#+$"
    for char in MD_SPECIAL_CHARS:
        text = text.replace(char, '').replace('\t', '')
    return text

def find_and_highlight(substring, string):
    escape_substring = substring.replace('.', '\.').replace('(', '\(').replace(')', '\)')
    match  = re.search(escape_substring, string)
    start  = match.start()
    end    = match.end() - 1
    before = string[:start]
    after  = string[:end]
    to_highlight = string[start:end]
    return escape_markdown(before) + f"<span style='background-color:#fdd835'>{to_highlight}</span>" + escape_markdown(after)

def display_text(tup):
    selection = metadata.iloc[sents.iloc[tup[0]].org_index]
    st.markdown(f"<small style='text-align: right;'>From: <b>{selection.From if selection.From is not np.nan else 'No sender found'}</b></small>",unsafe_allow_html=True)
    st.markdown(f"<small style='text-align: right;'>To: <b>{selection.To[0] if len(selection.To)>0 else 'No recipient found'}</b></small>",unsafe_allow_html=True)
    st.markdown(f"<small style='text-align: right;'>Cc: <b>{selection.Cc[0] if len(selection.Cc)>0 else 'No Cc found'}</b></small>",unsafe_allow_html=True)
    st.markdown(f"<small style='text-align: right;'>Date: <b>{selection.Sent if selection.Sent is not np.nan else 'No date found'}</b></small>",unsafe_allow_html=True)
    st.markdown(f"<small style='text-align: right;'>Subject: <b>{selection.Subject if selection.Subject is not np.nan else 'No subject found'}</b></small>",unsafe_allow_html=True)
    st.markdown(f"<small style='text-align: right;'>Similarity score: <b>{round(tup[1], 3)}</b></small>",unsafe_allow_html=True)

    #st.markdown(find_and_highlight(tup[2], selection.Body),unsafe_allow_html=True)

    # need to figure out image/thread indexing
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

uids = embeddings.search(query, results+100)
strings = [(sents.iloc[uid[0]].Body, uid[0], uid[1]) for uid in uids]
unique_strings = remove_duplicates(strings)
# sentences_sims = rerank(unique_strings)
# new_uids =[(tup[0][1], tup[1], tup[0][0]) for tup in sentences_sims][:results]
new_uids = [(u[1], u[2]) for u in unique_strings][:results]

if query != '':
    for tup in new_uids:
        display_text(tup)

st.markdown('<small>This tool is for educational purposes ONLY!</small>', unsafe_allow_html=True)
st.markdown('<small>Assembled by Peter Nadel | TTS Research Technology</small>', unsafe_allow_html=True)

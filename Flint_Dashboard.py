import streamlit as st 
from util import get_data
get_data()

st.title("Flint Dashboard")
st.write("Lorem Ipsum")

st.markdown('<small>This tool is for educational purposes ONLY!</small>', unsafe_allow_html=True)
st.markdown('<small>Assembled by Peter Nadel | TTS Research Technology</small>', unsafe_allow_html=True)
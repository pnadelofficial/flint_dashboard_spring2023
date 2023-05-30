import streamlit as st
from transformers import pipeline
from PIL import Image
from io import BytesIO

@st.cache_resource
def get_data():
    classifier = pipeline("image-classification", model="Kevin-M-Smith/vit_900_900_5000") # can change model here
    return classifier
classifier = get_data()

st.title("Custom Image Classifier")
st.write("""
    After a first pass with Microsoft's DiT (Dataset Explorer), we fine-tuned several models, including DiT,
    to recognize six Flint-specific document classes for which we manually tagged about 4000 examples. Below,
    you can **upload any image from your computer and run it thorugh our fine-tuned classifier**. 
""")
with st.expander('Read more'):
        st.markdown("""
            Though the dataset explorer helps researchers what types of data they are working with, it does not
            solve the problem of having too many document classes. Instead, in the case of the Flint emails dataset,
            it helped us determine what document classes we should focus on for fine-tuning. Specifically, Ken 
            Stephens determined what relevant document classes and manually tagged about 4000 images. With this 
            labelled data, Ken then trained several models on two base models DiT and [ViT](https://huggingface.co/google/vit-base-patch16-224).
            On this page is [the most successful model](https://huggingface.co/Kevin-M-Smith/vit_900_900_5000).
            """)

uploaded_file = st.file_uploader("Upload an image")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    img = Image.open(BytesIO(bytes_data))

    cl = classifier(img)[0]['label']
    st.write(f'Our model predicts that this image is a: **{cl.title()}**')
    st.image(img)

st.markdown('<small>This tool is for educational purposes ONLY!</small>', unsafe_allow_html=True)
st.markdown('<small>Assembled by Peter Nadel | TTS Research Technology</small>', unsafe_allow_html=True)
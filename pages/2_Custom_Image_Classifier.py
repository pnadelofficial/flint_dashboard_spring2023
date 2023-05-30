import streamlit as st
from transformers import pipeline
from PIL import Image
from io import BytesIO
from streamlit_image_select import image_select
import glob

@st.cache_resource
def get_data():
    classifier = pipeline("image-classification", model="Kevin-M-Smith/vit_900_900_5000") # can change model here
    return classifier
classifier = get_data()

st.title("Custom Image Classifier")
st.write("""
    After a first pass with Microsoft's DiT (Dataset Explorer), we fine-tuned several models, including DiT,
    to recognize six Flint-specific document classes for which we manually tagged over 5000 examples. Below,
    you can either **use one of our example images** or **upload any image from your computer and run it 
    thorugh our fine-tuned classifier**. 
""")
with st.expander('Read more'):
        st.markdown("""
            Though the dataset explorer helps researchers what types of data they are working with, it does not
            solve the problem of having too many document classes. Instead, in the case of the Flint emails dataset,
            it helped us determine what document classes we should focus on for fine-tuning. Specifically, Ken 
            Stephens determined what relevant document classes and manually tagged about 4000 images. With this 
            labelled data, Ken then trained several models on two base models DiT and [ViT](https://huggingface.co/google/vit-base-patch16-224).
            On this page is [the most successful model](https://huggingface.co/Kevin-M-Smith/vit_900_900_5000).
            See below for some explanatory visuals.
            """)
        st.image(Image.open('conmat_flint.png'),caption='Finetuned ViT model Confusion Matrix')
        st.image(Image.open('dist.png').resize((1200, 500)),caption='Distribution of document classes')
        

tab1, tab2 = st.tabs(["Use an example Image", "Upload your own image"])

with tab1:
    img_select = image_select("Use an example image", glob.glob('example_images/*.jpg'))
    if img_select:
        cl = classifier(img_select)[0]['label']
        st.write(f'Our model predicts that this image is a: **{cl.title()}**')
        st.image(img_select)

with tab2: 
    uploaded_file = st.file_uploader("Upload your own image")
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        img = Image.open(BytesIO(bytes_data))

        cl = classifier(img)[0]['label']
        st.write(f'Our model predicts that this image is a: **{cl.title()}**')
        st.image(img)

st.markdown('<small>This tool is for educational purposes ONLY!</small>', unsafe_allow_html=True)
st.markdown('<small>Assembled by Peter Nadel | TTS Research Technology</small>', unsafe_allow_html=True)
import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
import requests
from io import BytesIO
from streamlit_plotly_events import plotly_events
import plotly.io as pio
pio.templates.default = "plotly"

st.title('Dataset Explorer')
st.write("""
            Below, you can navigate through [a document classification model](https://huggingface.co/docs/transformers/model_doc/dit)'s 
            predictions on the Flint emails dataset and see where the model does well and where it fails.\
            **Click on any point in the chart below to see the image associated with it.**
""")
with st.expander('Read more'):
        st.markdown("""
            Researchers often lack tools for organizing and managing large datasets. Especially in the case of images, 
            these datasets can contain a wide variety of documents, some of which are well-suited for downstream analysis 
            and others not. The first step of working with the Flint emails dataset was to quickly ascertain what types
            of documents it even contained. Below, you can use an app developed by Gabe Mogollon to navigate through
            the images that make up the Flint emails dataset.\
            Gabe applied [Microsoft's Document Image Transformer (DiT)](https://huggingface.co/docs/transformers/model_doc/dit)
            to the images of the Flint email dataset. This model is trained to predict what type of document an image is. Gabe 
            then used [UMAP](https://umap-learn.readthedocs.io/en/latest/) dimensionality reduction and 
            [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/) clustering to identify where the model was not confident 
            in its predictions. We mark these documents with the tag "variable."\
            """)


@st.cache_resource
def get_data():
    dit_logits = pd.read_csv('./data/dit_logits_embedded.csv').drop(['Unnamed: 0'], axis=1)
    return dit_logits

dit_logits = get_data()

fig = px.scatter(
    dit_logits,
    x='umap_embedding_x', 
    y='umap_embedding_y', 
    color='best_guess_variable', 
    color_discrete_sequence=[
                                'red',
                                'blue',
                                'black',
                                'orange',
                                'purple',
                                'brown',
                                'pink',
                                'gray',
                                'cyan',
                                'magenta',
                                'lime',
                                'teal',
                                'olive',
                                'navy',
                                'maroon',
                                'gold',
                                'green'
                            ],
    hover_name=dit_logits['filename'].apply(lambda x: x.split('/')[-1])
    )
fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0))
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)

clicked = plotly_events(fig, click_event=True)

if len(clicked) > 0:
    point_dict = clicked[-1]
    point_in_df = dit_logits.loc[(dit_logits.umap_embedding_x == point_dict['x']) & (dit_logits.umap_embedding_y == point_dict['y'])].iloc[0]
    st.write(f'DiT guess: **{point_in_df.best_guess_variable.title()}**')
    aws_path = point_in_df.aws_path
    with st.container():
        st.markdown(
            """
        <style>
            div[data-testid="stImage"] {
                border: 2px solid black;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )
        res = requests.get(aws_path)
        img = Image.open(BytesIO(res.content))
        st.image(img, caption=f"{point_in_df.filename.split('/')[-1]}")

st.markdown('<small>This tool is for educational purposes ONLY!</small>', unsafe_allow_html=True)
st.markdown('<small>Assembled by Peter Nadel | TTS Research Technology</small>', unsafe_allow_html=True)

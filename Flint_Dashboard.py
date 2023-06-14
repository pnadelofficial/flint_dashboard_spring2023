import streamlit as st 
from util import get_data
from streamlit_image_select import image_select
import pandas as pd
get_data()

@st.cache_data
def get_images():
    dit_logits = pd.read_csv('./data/dit_logits_embedded.csv').drop(['Unnamed: 0'], axis=1)
    aws_paths = dit_logits.aws_path
    st.session_state['image_sample'] = list(aws_paths.sample(8))
    return aws_paths 
aws_paths = get_images()

st.title("The Network Language Toolkit Visual Dashboard: Showcasing Student Work")
st.write("""
        Welcome! This dashboard showcases the work of five research assistants who spent the
        Spring of 2023 developing research tools to better understand unstructured textual corpora.       
        """)

st.subheader("""
         The Dataset   
         """)
st.write("""
        For this project, we are investigating a large repository of emails concerning the Flint water
        crisis made available through the Freedom of Information Act (FoIA). These emails, sent between 
        members of then-Governor Rick Snyder's executive office, the Michigan State Police (MSP), and 
        members of the Environmental Protection Agency (EPA), constitute a dynamic social network, which 
        grows in complexity over time. Using the Flint emails as an example, this semester, we began to 
        create a series of tools for researchers in the social sciences and humanities to better grasp 
        the breadth and depth of their own datasets.
        """)
st.subheader("""
            The makeup of the Flint datasets
            """)
st.write("""
        The dataset is made up of over 128,000 images grouped in 55 FoIA dumps which were released between 
        January and June of 2016. This equates to 55 GB of JPEG2000 images. These images are scans of 
        printed email threads, attachments and handwritten documents. The original realeases can be found
        on archive.org, here: [Michigan Government/State Police](https://archive.org/details/snyder_flint_emails/Staff_1/),
        [EPA](https://archive.org/details/epa-flint-documents/page/n3/mode/2up). Though these data are 
        publically available, they remain practically inaccessible given their size and ____. In order to shrink 
        this gap between what is available and what is accessible we propose the tools enumerated in this dashboard. 
        Below you can find an image gallery of examples from the dataset. Click on 'Resample images' to see another set of eight examples.
        """)

if st.button('Resample images'):
        st.session_state['image_sample'] = list(aws_paths.sample(8))
img_select = image_select("Select an example image to see it in more detail", st.session_state['image_sample'], captions=[cap.split('/')[-1] for cap in st.session_state['image_sample']])
if img_select:
        st.image(img_select, width=420)

st.subheader("""
            Challenges in using the data
            """)
st.write("""
        There are several challenges in using these data. The staff responsible for preparing the emails
        for the FoIA releases needed to capture each email, redact it for any priviledged information, and then
        save them in a form which could not be altered. As a result, the emails were converted from their 
        original digital format into a static image which must be heavily processed to return to them their
        original metadata. For instance, as explained further in *Person to Person Networks*, we extracted 
        from each email a sender, a recipient, a date, any CC'd individuals, a subject and the name of any
        attachments. This metadata, an expectation for any downstream analysis, is unavailble without signficant 
        effort. Thus, we sought to assemble a series of tools which researchers could choose from to fit their
        dataset and research questions. 
        """)

st.markdown('<small>This tool is for educational purposes ONLY!</small>', unsafe_allow_html=True)
st.markdown('<small>Assembled by Peter Nadel | TTS Research Technology</small>', unsafe_allow_html=True)
import streamlit as st

st.set_page_config(page_title="Fake News Detector", layout="wide")

st.title("Video test(Work in progress)")
uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mpeg4"])

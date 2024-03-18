import streamlit as st
import tensorflow
from transformers import pipeline
import model_loader
import asyncio

st.set_page_config(page_title="Fake News Detector", layout="wide")


async def load_models_async():
    return model_loader.load_models()


# @st.cache_resource
# def fetch_model():
#     MODEL = "jy46604790/Fake-News-Bert-Detect"
#     return MODEL


# MODEL = fetch_model()
# TEXT_CLASSIFIER = pipeline("text-classification", model=MODEL, tokenizer=MODEL)
async def main():
    TEXT_CLASSIFIER, SUMMARIZATION_MODEL, CAPTIONING_MODEL = await load_models_async()
    st.title("Text test(Work in progress)")
    text = st.text_input("Enter the news article's text")
    if text:
        result = TEXT_CLASSIFIER(text)
        if result[0]["label"] == "LABEL_1":
            st.write("The article is real")
            score = result[0]["score"]
            st.write(f"The score is: {score}")
        elif result[0]["label"] == "LABEL_0":
            st.write("The article is fake")
            score = result[0]["score"]
            st.write(f"The score is: {score}")


if __name__ == "__main__":
    asyncio.run(main())

# model_loader.py

from transformers import pipeline
from tensorflow.keras.models import load_model

# from tensorflow.keras.models import load_model
import google.generativeai as ga


def load_models():
    # MODEL = "jy46604790/Fake-News-Bert-Detect"
    # TEXT_CLASSIFIER = pipeline("text-classification", model=MODEL, tokenizer=MODEL)
    TEXT_CLASSIFIER = load_model(r"D:\Final Project\pages\fake_news_bert.h5")
    GOOGLE_API_KEY = "AIzaSyDecPEPHA2dNPF3E2W0a77u5Yvr_7YQtrg"
    ga.configure(api_key=GOOGLE_API_KEY)
    CAPTIONING_MODEL = ga.GenerativeModel(model_name="gemini-pro-vision")

    return TEXT_CLASSIFIER, CAPTIONING_MODEL

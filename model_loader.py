# model_loader.py

from transformers import pipeline
import google.generativeai as ga


def load_models():
    MODEL = "jy46604790/Fake-News-Bert-Detect"
    TEXT_CLASSIFIER = pipeline("text-classification", model=MODEL, tokenizer=MODEL)

    GOOGLE_API_KEY = "AIzaSyDecPEPHA2dNPF3E2W0a77u5Yvr_7YQtrg"
    ga.configure(api_key=GOOGLE_API_KEY)

    SUMMARIZATION_MODEL = ga.GenerativeModel(model_name="gemini-pro")
    CAPTIONING_MODEL = ga.GenerativeModel(model_name="gemini-pro-vision")
    return TEXT_CLASSIFIER, SUMMARIZATION_MODEL, CAPTIONING_MODEL

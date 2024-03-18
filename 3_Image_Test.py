import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import PIL
from PIL import Image
import google.generativeai as ga

import asyncio
import model_loader

st.set_page_config(page_title="Fake News Detector", layout="wide")


async def load_models_async():
    return model_loader.load_models()


async def main():
    CAPTIONING_MODEL = await load_models_async()

    @st.cache_resource
    def load_the_model():
        model = load_model("old_trained_model.h5")
        return model

    model = load_the_model()

    def predict(img):
        img = img.resize((256, 256), Image.LANCZOS)
        processed_image = np.array(img) / 255.0
        prediction = model.predict(np.expand_dims(processed_image, axis=0))
        class_label = "real" if prediction[0][0] > 0.5 else "fake"
        return class_label

    # @st.cache_data
    # def configure_api_key():
    #     GOOGLE_API_KEY = (
    #         "AIzaSyB2ml0oownht1RfN69k8O3msz8S547WgJ8"  # Replace with your actual API key
    #     )
    #     ga.configure(api_key=GOOGLE_API_KEY)

    CAPTIONING_MODEL = ga.GenerativeModel("gemini-pro-vision")

    st.title("Image test(Work in progress)")
    uploaded_file = st.file_uploader(
        "Choose a media file...", type=["jpg", "png", "jpeg", "mp4"]
    )
    if uploaded_file is not None:
        img = PIL.Image.open(uploaded_file)
        prediction = predict(img)
        st.write(f"Predicted class: {prediction}")
        caption = CAPTIONING_MODEL.generate_content(
            ["Generate a short caption for the image", img], stream=True
        )

        caption.resolve()
        st.image(img)
        st.header("Caption:")
        st.write(caption.text)


if __name__ == "__main__":
    asyncio.run(main())

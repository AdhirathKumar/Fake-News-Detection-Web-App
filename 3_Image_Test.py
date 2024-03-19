import streamlit as st
import numpy as np

# from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# import PIL

# from PIL import Image
import google.generativeai as ga

import asyncio
import model_loader

st.set_page_config(page_title="Fake News Detector", layout="wide")


async def load_models_async():
    return model_loader.load_models()


async def main():
    TEXT_CLASSIFIER, SUMMARIZATION_MODEL, CAPTIONING_MODEL, DEEPFAKE_CLASSIFIER = (
        await load_models_async()
    )

    # Function to load and preprocess image

    def preprocess_image(image):
        image = load_img(image, target_size=(256, 256))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image /= 255.0  # Normalize pixel values
        return image

    # Function to predict class of the image
    def predict_image_class(image_path, classifier):
        image = preprocess_image(image_path)
        prediction = classifier.predict(image)
        predicted_class = "Real" if prediction > 0.5 else "Deepfake"
        return predicted_class

    st.title("Image Deepfake Detection")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = load_img(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Predict the class of the image
        predicted_class = predict_image_class(uploaded_file, DEEPFAKE_CLASSIFIER)
        st.write("Prediction:", predicted_class)


if __name__ == "__main__":
    asyncio.run(main())

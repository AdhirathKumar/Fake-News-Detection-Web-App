import streamlit as st
import pandas as pd
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import model_loader
import asyncio


async def load_models_async():
    return model_loader.load_models()


async def main():
    TEXT_CLASSIFIER, CAPTIONING_MODEL = await load_models_async()

    st.set_page_config(page_title="Fake News Detector", layout="wide")
    st.title("Text test(Work in progress)")
    TEXT = st.text_input("Enter the news article's text")
    if TEXT:
        fake = pd.read_csv(r"D:\Final Project\pages\Fake.csv", encoding="utf-8")
        # # text = " ".join(fake["text"].tolist())
        real = pd.read_csv(r"D:\Final Project\pages\True.csv", encoding="utf-8")
        # # text = " ".join(real["text"].tolist())
        unknown_publishers = []
        for index, row in enumerate(real.text.values):
            try:
                record = row.split("-", maxsplit=1)
                assert len(record[0]) < 120
            except:
                unknown_publishers.append(index)
        real = real.drop(8970, axis=0)
        publisher = []
        tmp_text = []
        for index, row in enumerate(real.text.values):
            if index in unknown_publishers:
                tmp_text.append(row)
                publisher.append("Unknown")
            else:
                record = row.split("-", maxsplit=1)
                publisher.append(record[0].strip())
                tmp_text.append(record[1].strip())
        real["publisher"] = publisher
        real["text"] = tmp_text
        empty_fake_index = [
            index
            for index, text in enumerate(fake.text.tolist())
            if str(text).strip() == ""
        ]
        # fake.iloc[empty_fake_index]
        real["text"] = real["title"] + " " + real["text"]
        fake["text"] = fake["title"] + " " + fake["text"]
        real["text"] = real["text"].apply(lambda x: str(x).lower())
        fake["text"] = fake["text"].apply(lambda x: str(x).lower())
        real["class"] = 1
        fake["class"] = 0
        real = real[["text", "class"]]
        fake = fake[["text", "class"]]
        data = real._append(fake, ignore_index=True)
        # data.sample(5)

        def remove_special_chars(x):
            x = re.sub(r"[^\w ]+", "", x)
            x = " ".join(x.split())
            return x

        data["text"].apply(lambda x: remove_special_chars(x))
        x = [d.split() for d in data["text"].tolist()]
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(x)
        x = tokenizer.texts_to_sequences(x)
        maxlen = 1000
        m = [str(TEXT)]
        m = tokenizer.texts_to_sequences(m)
        m = pad_sequences(m, maxlen=maxlen)
        prediction = TEXT_CLASSIFIER.predict(m) >= 0.5
        st.write(prediction[0][0])


if __name__ == "__main__":
    asyncio.run(main())

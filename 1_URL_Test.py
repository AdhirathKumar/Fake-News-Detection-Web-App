import streamlit as st
from newspaper import Config, Article
import textwrap
from IPython.display import Markdown

import PIL
import urllib.request
from trusted import check_iftrusted
import asyncio
import model_loader

from reverse import google_search, filter_social_media_urls, calculate_sts_score

# Configure NewspaperUSER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36"

config = Config()
config.browser_user_agent = USER_AGENT
config.request_timeout = 10


# def to_markdown(text):
#     text = text.replace("•", "  *")
#     return Markdown(textwrap.indent(text, "> ", predicate=lambda _: True))


async def load_models_async():
    return model_loader.load_models()


async def main():
    try:
        TEXT_CLASSIFIER, SUMMARIZATION_MODEL, CAPTIONING_MODEL = (
            await load_models_async()
        )

        st.set_page_config(page_title="Fake News Detector", layout="wide")
        st.title("URL test (Work in progress)")
        URL = st.text_input("Enter the URL of the news article.")

        if URL:
            article = Article(URL, config=config)
            article.download()
            article.parse()
            article.nlp()
            CURRENT_ARTICLE_PUBLISH_DATE = article.publish_date
            CURRENT_ARTICLE_TITLE = article.title
            CURRENT_ARTICLE_TEXT = article.text

            if CURRENT_ARTICLE_TITLE:
                st.write(f"Title : {CURRENT_ARTICLE_TITLE}")
            else:
                st.write("Title : Unavailable")
            if CURRENT_ARTICLE_PUBLISH_DATE:
                st.write(f"Publish date : {CURRENT_ARTICLE_PUBLISH_DATE}")
            else:
                st.write("Publish date : Unavailable")
            if article.summary:
                st.header("Summary")
                st.write(article.summary)
            else:
                st.write("Article summary unavailable")

            CURRENT_ARTICLE_TOP_IMAGE_URL = article.top_image
            if CURRENT_ARTICLE_TOP_IMAGE_URL:
                urllib.request.urlretrieve(
                    CURRENT_ARTICLE_TOP_IMAGE_URL, "top_image.jpg"
                )
                CURRENT_ARTICLE_TOP_IMAGE = PIL.Image.open("top_image.jpg")
                CURRENT_ARTICLE_TOP_IMAGE_CAPTION = CAPTIONING_MODEL.generate_content(
                    [
                        "Generate a short and precise caption for the given image input",
                        CURRENT_ARTICLE_TOP_IMAGE,
                    ],
                )
                CURRENT_ARTICLE_TOP_IMAGE_CAPTION.resolve()
                st.image(CURRENT_ARTICLE_TOP_IMAGE)
                st.header("Caption:")
                st.write(CURRENT_ARTICLE_TOP_IMAGE_CAPTION.text)
            else:
                st.write("Article top image unavailable")
            if CURRENT_ARTICLE_TEXT:
                result = TEXT_CLASSIFIER(CURRENT_ARTICLE_TEXT)
                label = result[0]["label"]
                score = result[0]["score"]
                if label == "LABEL_1":
                    st.header(f"Huggingface bert model text classifier analysis:")
                    st.write(f"The article is real with a score of {score:.4f}")
                elif label == "LABEL_0":
                    st.header(f"Huggingface bert model text classifier analysis:")
                    st.write(f"The article is fake with a score of {score:.4f}")
            else:
                st.write("Article content unavailable")
            domain_name, article_class = check_iftrusted(URL)
            search_results = None
            if article_class:
                st.header(f"Trusted sources analysis:")
                st.write(
                    f"The article is likely real, it has been covered by {domain_name}"
                )

            search_results = google_search(URL)
            st.write(search_results)
            if search_results:
                st.write(
                    "The article has appeared before. Retrieving information from matching articles:"
                )
                search_results = filter_social_media_urls(search_results)

                for result in search_results:
                    # Initialize Article object with matching article URL
                    matching_article = Article(result)
                    matching_article.download()
                    matching_article.parse()
                    matching_article_text = matching_article.text
                    matching_article_title = matching_article.title

                    # Calculate STS score between the entered article and matching article
                    text_sts_score = calculate_sts_score(
                        CURRENT_ARTICLE_TEXT, matching_article_text
                    )

                    st.write("Matching Article URL:", result)
                    st.write("STS Score:", text_sts_score)

                    title_sts_score = calculate_sts_score(
                        CURRENT_ARTICLE_TITLE, matching_article_title
                    )

                    if text_sts_score > 0.75 and title_sts_score > 0.75:
                        st.write("The article is similar to the entered article.")
                    else:
                        st.write("The article is not similar to the entered article.")

                    st.write("-----------------------------------")

            else:
                st.write("The article has not appeared before on the internet.")
                st.header(f"Trusted sources analysis:")
                st.write(f"The article is likely fake")
        else:
            st.error("Enter a valid URL")
    except Exception as e:
        st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(main())

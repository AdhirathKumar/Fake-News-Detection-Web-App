import streamlit as st
from newspaper import Article
from googlesearch import search
from similarity import SIMILARITY_CHECK
from trusted import check_iftrusted


# Function to calculate age from published date
def calculate_age(published_date, current_date):
    age = current_date - published_date
    years = age.days // 365
    months = (age.days % 365) // 30
    return years, months


# Function to filter social media URLs
def filter_social_media_urls(urls):
    filtered_urls = []
    for url in urls:
        if "twitter.com" not in url and "instagram.com" not in url:
            filtered_urls.append(url)
    return filtered_urls


def remove_extra_spaces(text):
    words = text.split()
    return " ".join(words)


# Set up Streamlit app
st.title("Article Similarity Checker")

# Input field for the user to enter the URL
article_url = st.text_input("Enter the URL of the article")

# Button to trigger the search
if st.button("Search"):
    # Check if the URL is entered
    if article_url:
        # Download and parse the entered article
        entered_article = Article(article_url)
        entered_article.download()
        entered_article.parse()
        entered_article.nlp()
        entered_article_text = entered_article.text
        summary = entered_article.summary
        current_article_title = entered_article.title

        # Perform a Google search using the article title
        search_results = search(current_article_title)
        search_results = filter_social_media_urls(search_results)
        # Display the search results
        for i, result in enumerate(search_results):
            st.write(f"Result {i+1}: {result}")
            domain, trusted = check_iftrusted(result)
            st.write(f"Domain: {domain}, Trusted: {'Yes' if trusted else 'No'}")
            searched_article = Article(result)
            searched_article.download()
            searched_article.parse()
            searched_article.nlp()
            searched_article_text = searched_article.text
            cur_summary = searched_article.summary
            searched_article_title = searched_article.title
            class_text, confidence_text = SIMILARITY_CHECK(
                entered_article_text, searched_article_text
            )
            class_title, confidence_title = SIMILARITY_CHECK(
                current_article_title, searched_article_title
            )
            st.write("Text class and score:", class_text, confidence_text)
            st.write("Title class and score:", class_title, confidence_title)
            if (class_text == "True" and confidence_text > 0.6) or (
                class_title == "True" and confidence_title > 0.6
            ):
                st.write("This is a similar article to the searched article.")
            st.write("--------------------------------------------------------")
    else:
        st.warning("Please enter a valid URL.")

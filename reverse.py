# Import necessary libraries
from newspaper import Article
from googlesearch import search
from angle_emb import AnglE
from sklearn.metrics.pairwise import cosine_similarity
from trusted import check_iftrusted


# Function to filter out social media URLs
def filter_social_media_urls(urls):
    filtered_urls = []
    for url in urls:
        if "twitter.com" not in url and "instagram.com" not in url:
            filtered_urls.append(url)
    return filtered_urls


# Function to perform a Google search based on the article URL
def google_search(article_url):
    search_query = f"related:{article_url}"
    search_results = search(search_query, num_results=10)
    return search_results


# Function to calculate Semantic Text Similarity (STS) score between two texts
def calculate_sts_score(text1, text2):
    angle = AnglE.from_pretrained(
        "WhereIsAI/UAE-Large-V1", pooling_strategy="cls"
    ).cuda()
    vec1 = angle.encode(text1, to_numpy=True)
    vec2 = angle.encode(text2, to_numpy=True)
    similarity = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
    return similarity


# Example URL of the news article
# article_url = "https://indianexpress.com/article/opinion/columns/c-raja-mohan-donald-trump-christian-nationalism-opportunity-bjp-9183839/"

# # Download and parse the entered article
# entered_article = Article(article_url)
# entered_article.download()
# entered_article.parse()
# entered_article_text = entered_article.text
# title = entered_article.title

# Check if the article is from a trusted source
# if check_iftrusted(article_url):
#     pass
# else:
#     # Perform a reverse search if the article has appeared before
#     search_results = google_search(article_url)

#     if search_results:
#         print(
#             "The article has appeared before. Retrieving information from matching articles:"
#         )
#         search_results = filter_social_media_urls(search_results)

#         for result in search_results:
#             # Initialize Article object with matching article URL
#             matching_article = Article(result)
#             matching_article.download()
#             matching_article.parse()
#             matching_article_text = matching_article.text
#             current_title = matching_article.title

#             # Calculate STS score between the entered article and matching article
#             text_sts_score = calculate_sts_score(
#                 entered_article_text, matching_article_text
#             )

#             print("Matching Article URL:", result)
#             print("STS Score:", text_sts_score)

#             title_sts_score = calculate_sts_score(title, current_title)

#             if text_sts_score > 0.75 and title_sts_score > 0.75:
#                 print("The article is similar to the entered article.")
#             else:
#                 print("The article is not similar to the entered article.")

#             print("-----------------------------------")

#     else:
#         print("The article has not appeared before on the internet.")

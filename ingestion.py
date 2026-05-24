from newsapi import NewsApiClient
from newspaper import Article
from .config import NEWSAPI_KEY

newsapi = NewsApiClient(api_key=NEWSAPI_KEY)


# 20+ countries
countries = [
    "India",
    "USA",
    "United Kingdom",
    "France",
    "Germany",
    "Japan",
    "China",
    "Russia",
    "Brazil",
    "Canada",
    "Australia",
    "Italy",
    "Spain",
    "South Korea",
    "South Africa",
    "Mexico",
    "UAE",
    "Singapore",
    "Turkey",
    "Netherlands"
]


# Multiple news topics
topics = [
    "sports",
    "politics",
    "crime",
    "business",
    "technology",
    "health",
    "science",
    "entertainment"
]


def fetch_top_headlines():

    all_articles = []

    for country in countries:

        for topic in topics:

            try:
                # Fetch multiple pages
                for page in range(1, 3):

                    response = newsapi.get_everything(
                        q=f"{country} {topic}",
                        language="en",
                        sort_by="publishedAt",
                        page_size=100,
                        page=page
                    )

                    articles = response.get("articles", [])

                    all_articles.extend(articles)

            except Exception as e:
                print(f"Error fetching {country} {topic}: {e}")

    # Remove duplicate articles
    unique_articles = []
    seen_titles = set()

    for article in all_articles:

        title = article.get("title")

        if title and title not in seen_titles:
            seen_titles.add(title)
            unique_articles.append(article)

    return unique_articles


def extract_article_text(url: str):

    try:
        article = Article(url)

        article.download()
        article.parse()

        return article.text

    except Exception as e:
        print(f"Error extracting article: {e}")
        return ""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

import aiohttp
import asyncio

from collections import defaultdict

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

import os

NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY")
GNEWSAPI_KEY = os.environ.get("GNEWSAPI_KEY")
NEWSDATAIOAPI_KEY = os.environ.get("NEWSDATAIOAPI_KEY")

app = FastAPI(title="Global News Summarizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

country_codes = {
    "India": "in",
    "USA": "us",
    "United Kingdom": "gb",
    "France": "fr",
    "Germany": "de",
    "Japan": "jp",
    "China": "cn",
    "Russia": "ru",
    "Brazil": "br",
    "Canada": "ca",
    "Australia": "au",
    "Italy": "it",
    "Spain": "es",
    "South Korea": "kr",
    "South Africa": "za",
    "Mexico": "mx",
    "UAE": "ae",
    "Singapore": "sg",
    "Turkey": "tr",
    "Netherlands": "nl"
}

topic_queries = {
    "sports": ["sports", "football", "cricket", "olympics", "fifa"],
    "technology": ["technology", "AI", "software", "startup", "gadgets"],
    "politics": ["politics", "government", "election", "parliament"],
    "business": ["business", "stock market", "finance", "economy"],
    "crime": ["crime", "police", "investigation", "court"],
    "entertainment": ["movies", "music", "celebrity", "netflix", "hollywood"],
}

@app.get("/")
def root():
    return {"message": "Global News Backend Running"}

def textrank_summary(text, max_len=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= max_len:
        return text
    words = word_tokenize(text.lower())
    freq = defaultdict(int)
    for w in words:
        if w.isalpha():
            freq[w] += 1
    scores = {}
    for sent in sentences:
        score = 0
        for w in word_tokenize(sent.lower()):
            score += freq.get(w, 0)
        scores[sent] = score
    ranked = sorted(scores, key=scores.get, reverse=True)
    return " ".join(ranked[:max_len])

async def fetch_newsapi_page(session, params):
    async with session.get(
        "https://newsapi.org/v2/top-headlines",
        params=params,
        timeout=aiohttp.ClientTimeout(total=10)
    ) as response:
        return await response.json()

async def fetch_newsapi(session, country, topic):
    articles = []
    queries = topic_queries.get(topic, [topic])
    try:
        tasks = []
        for q in queries:
            for page in range(1, 4):
                params = {
                    "q": q,
                    "country": country,
                    "pageSize": 20,
                    "page": page,
                    "apiKey": NEWSAPI_KEY
                }
                tasks.append(fetch_newsapi_page(session, params))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                print("NewsAPI Page Error:", result)
            else:
                articles.extend(result.get("articles", []))
    except Exception as e:
        print("NewsAPI Error:", e)
    return articles

async def fetch_gnews_query(session, url):
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
        return await response.json()

async def fetch_gnews(session, country, topic):
    articles = []
    queries = topic_queries.get(topic, [topic])
    try:
        tasks = []
        for q in queries:
            url = (
                f"https://gnews.io/api/v4/search?"
                f"q={q}&country={country}&lang=en&max=20&apikey={GNEWSAPI_KEY}"
            )
            tasks.append(fetch_gnews_query(session, url))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                print("GNews Query Error:", result)
            else:
                articles.extend(result.get("articles", []))
    except Exception as e:
        print("GNews Error:", e)
    return articles

async def fetch_newsdata_query(session, url):
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
        return await response.json()

async def fetch_newsdata(session, country, topic):
    articles = []
    queries = topic_queries.get(topic, [topic])
    try:
        tasks = []
        for q in queries:
            url = (
                f"https://newsdata.io/api/1/news?"
                f"apikey={NEWSDATAIOAPI_KEY}&q={q}&country={country}&language=en"
            )
            tasks.append(fetch_newsdata_query(session, url))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                print("NewsData Query Error:", result)
            else:
                for item in result.get("results", []):
                    articles.append({
                        "title": item.get("title"),
                        "description": item.get("description"),
                        "url": item.get("link"),
                        "publishedAt": item.get("pubDate"),
                        "urlToImage": item.get("image_url"),
                        "source": {"name": item.get("source_id")}
                    })
    except Exception as e:
        print("NewsData.io Error:", e)
    return articles

async def translate_summary(summary_text, target_lang):
    try:
        if target_lang == "en":
            return summary_text
        def do_translate():
            from deep_translator import MyMemoryTranslator
            translated = MyMemoryTranslator(
                source='en-GB',
                target=target_lang
            ).translate(summary_text[:500])
            return translated
        result = await asyncio.to_thread(do_translate)
        return result if result else summary_text
    except Exception as e:
        print("Translation Error:", e)
        return summary_text

@app.get("/summaries")
async def get_summary(request: Request):
    country_name = request.query_params.get("country", "India")
    topic = request.query_params.get("topic", "sports")
    language = request.query_params.get("language", "en")
    country_code = country_codes.get(country_name, "in")

    async with aiohttp.ClientSession() as session:
        newsapi_articles, gnews_articles, newsdata_articles = await asyncio.gather(
            fetch_newsapi(session, country_code, topic),
            fetch_gnews(session, country_code, topic),
            fetch_newsdata(session, country_code, topic),
            return_exceptions=True
        )

    if isinstance(newsapi_articles, Exception):
        newsapi_articles = []
    if isinstance(gnews_articles, Exception):
        gnews_articles = []
    if isinstance(newsdata_articles, Exception):
        newsdata_articles = []

    all_articles = newsapi_articles + gnews_articles + newsdata_articles

    unique_articles = []
    seen_titles = set()
    for article in all_articles:
        title = article.get("title")
        if not title:
            continue
        normalized = title.lower().strip()
        if normalized not in seen_titles:
            seen_titles.add(normalized)
            unique_articles.append(article)

    async def process_article(article):
        try:
            title = article.get("title")
            content = article.get("description") or ""
            if not content:
                return None
            summary = textrank_summary(content)
            translated_summary = await translate_summary(summary, language)
            return {
                "country": country_name,
                "topic": topic,
                "language": language,
                "title": title,
                "summary": translated_summary,
                "source": article.get("source", {}).get("name", "Unknown"),
                "publishedAt": article.get("publishedAt"),
                "url": article.get("url"),
                "image": article.get("urlToImage", "https://images.unsplash.com/photo-1504711434969-e33886168f5c")
            }
        except Exception as e:
            print("Article Error:", e)
            return None

    summaries_results = await asyncio.gather(*[process_article(a) for a in unique_articles])
    summaries = [s for s in summaries_results if s is not None]
    summaries.sort(key=lambda x: x.get("publishedAt", ""), reverse=True)

    return {
        "country": country_name,
        "topic": topic,
        "language": language,
        "total_articles": len(summaries),
        "summaries": summaries
    }

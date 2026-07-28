# -*- coding: utf-8 -*-
"""
nlp_news.py
===========
NLP News Engine: Fetches stock news RSS feed and performs VADER sentiment analysis
and headline relevance calculation.
"""

import requests
import xml.etree.ElementTree as ET

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
except ImportError:
    analyzer = None


def analyze_news_nlp(ticker: str) -> dict:
    """
    Fetches news, extracts headlines, and performs NLP sentiment analysis.
    Returns composite sentiment (-1.0 to 1.0) and article relevance score (0.0 to 1.0).
    """
    clean_symbol = ticker.split(":")[-1].replace("-INDEX", "").replace("-EQ", "")
    url = f"https://news.google.com/rss/search?q={clean_symbol}+stock&hl=en-US&gl=US&ceid=US:en"

    try:
        response = requests.get(url, timeout=5)
        if response.status_code != 200:
            return {"sentiment_score": 0.0, "relevance_score": 0.0, "headline_count": 0}

        root = ET.fromstring(response.content)
        items = root.findall('./channel/item')

        if not items:
            return {"sentiment_score": 0.0, "relevance_score": 0.0, "headline_count": 0}

        sentiments = []
        relevant_matches = 0

        for item in items[:10]:
            title = item.find('title').text if item.find('title') is not None else ""

            # Relevance check
            if clean_symbol.upper() in title.upper() or "BANK" in title.upper() or "STOCK" in title.upper():
                relevant_matches += 1

            # VADER sentiment score
            if analyzer:
                vs = analyzer.polarity_scores(title)
                sentiments.append(vs['compound'])

        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        relevance_ratio = relevant_matches / len(items[:10]) if items else 0.0

        return {
            "sentiment_score": round(avg_sentiment, 3),
            "relevance_score": round(relevance_ratio, 3),
            "headline_count": len(items),
        }
    except Exception as e:
        print(f"News fetch notice: {e}")
        return {"sentiment_score": 0.0, "relevance_score": 0.0, "headline_count": 0}

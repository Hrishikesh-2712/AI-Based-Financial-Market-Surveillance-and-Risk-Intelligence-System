# -*- coding: utf-8 -*-
"""
data_loader.py
===============
Reads nlp_news_engine/snapshots/latest.json FRESH on every rebuild (see
vector_store.get_or_build) -- whatever run_every_5min.py / backend/pipeline.py
most recently wrote to disk is what the chatbot answers from.

Also injects a fixed "glossary" of documents describing this project's
own scoring system (CARS, TCN, Isolation Forest, D_info) so the chatbot
can answer meta-questions like "what does CARS score mean" or "why is
this flagged CRITICAL" even though that phrasing never literally
appears in the news JSON.
"""

import json

from langchain_core.documents import Document

from . import config

GLOSSARY_TEXT = [
    (
        "This system detects Bank NIFTY anomalies using two models in "
        "parallel: a TCN (Temporal Convolutional Network) Autoencoder, "
        "which flags a candle as anomalous when it cannot reconstruct "
        "the recent 64-candle price/volume pattern well (high "
        "reconstruction error), and an Isolation Forest, which flags a "
        "candle as anomalous when its feature vector is easy to isolate "
        "from normal trading. A candle can be flagged by either model "
        "alone ('either') or by both at once ('both', shown in red on "
        "the chart, orange for 'either')."
    ),
    (
        "The features both models look at, computed by feature_extract.py, "
        "are: Log_Returns, MACD_Hist, RSI_14 (14-period RSI), BB_Width "
        "(Bollinger Band width), ATR_Normalized (normalized Average True "
        "Range), Supertrend_Dir (Supertrend direction), Volume_ZScore, "
        "and OBV_Pct_Change (On-Balance Volume percent change)."
    ),
    (
        "CARS (Composite Anomaly Risk Score, 0-100) is this project's "
        "final risk number, calculated in risk_engine/risk_engine.py. "
        "It combines a technical risk score (40% weight) with an "
        "anomaly-adjusted risk (60% weight). The anomaly-adjusted risk "
        "is the Isolation Forest's intensity multiplied by an "
        "'unexplained' penalty: the less relevant news there is to "
        "explain an anomaly, the bigger that penalty gets."
    ),
    (
        "D_info (information disconnection index, 0 to 1) measures how "
        "well news explains a detected anomaly: it is 1 minus (news "
        "relevance times the absolute value of news sentiment). A high "
        "D_info (above 0.7) means little or no relevant news was found, "
        "so the system labels the anomaly 'CRITICAL: Unexplained Market "
        "Anomaly' -- this is the pattern most consistent with trading "
        "ahead of public information. A low D_info means relevant news "
        "was found and the move is labelled 'MODERATE: News-Catalyzed "
        "Market Movement'. If no anomaly was flagged at all, the risk "
        "classification is 'LOW: Normal Trading Activity'."
    ),
    (
        "composite_score is the relevance ranking (0 to 1) the NLP news "
        "module assigns to a news article. It blends four factors: how "
        "recent the article is relative to the reference time (35% "
        "weight), how strong the sentiment is regardless of direction "
        "(30%), how confidently the article matches a Bank NIFTY "
        "constituent company (15%), and that company's free-float "
        "index_weight in Bank NIFTY (20%). A higher composite_score "
        "means the article is more likely to explain a given anomaly."
    ),
    (
        "sentiment_signed ranges from -1 (very negative news) to +1 "
        "(very positive news), with 0 being neutral. It reflects the "
        "tone of the article toward the mentioned company, not the "
        "market's actual price reaction."
    ),
    (
        "index_weight is how much a constituent bank contributes to "
        "the Bank NIFTY index, based on NSE's free-float market-cap "
        "weighting. Larger banks like HDFC Bank and ICICI Bank have a "
        "bigger index_weight, so news about them is treated as more "
        "likely to move Bank NIFTY overall than news about a smaller "
        "constituent such as Bandhan Bank or AU Small Finance Bank."
    ),
    (
        "minutes_before_reference is how many minutes before the "
        "reference timestamp (the time this news batch was generated, "
        "or a detected anomaly) an article was published. A small "
        "value means the news is very recent relative to that moment."
    ),
]


def _load_glossary_documents():
    return [
        Document(page_content=text, metadata={"type": "glossary"})
        for text in GLOSSARY_TEXT
    ]


def _news_item_to_document(item: dict) -> Document:
    text = (
        f"Headline: {item.get('headline', '')}\n"
        f"Company: {item.get('company', '')} ({item.get('symbol', '')})\n"
        f"Category: {item.get('category', '')}\n"
        f"Source: {item.get('source', '')}\n"
        f"Published at: {item.get('published_at', '')}\n"
        f"Minutes before reference: {item.get('minutes_before_reference', '')}\n"
        f"Sentiment (signed, -1 to +1): {item.get('sentiment_signed', '')}\n"
        f"Index weight of company: {item.get('index_weight', '')}\n"
        f"Composite relevance score: {item.get('composite_score', '')}"
    )
    return Document(
        page_content=text,
        metadata={
            "type": "news",
            "symbol": item.get("symbol", ""),
            "company": item.get("company", ""),
            "composite_score": item.get("composite_score", 0),
            "published_at": item.get("published_at", ""),
            "link": item.get("link", ""),
        },
    )


def load_documents():
    """
    Returns (documents, meta) where meta has generated_at/count for
    display purposes, read fresh from disk every call.
    """
    if not config.NEWS_JSON_PATH.exists():
        raise FileNotFoundError(
            f"No news snapshot found at {config.NEWS_JSON_PATH}. "
            f"Run nlp_news_engine/run_every_5min.py (or let the backend's "
            f"pipeline run once) to generate it."
        )

    with open(config.NEWS_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    news_items = data.get("top_news", [])
    documents = [_news_item_to_document(item) for item in news_items]
    documents.extend(_load_glossary_documents())

    meta = {
        "generated_at": data.get("generated_at", "unknown"),
        "count": data.get("count", len(news_items)),
    }
    return documents, meta

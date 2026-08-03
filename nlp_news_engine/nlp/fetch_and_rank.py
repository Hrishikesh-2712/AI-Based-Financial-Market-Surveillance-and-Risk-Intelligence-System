"""
Fetch BankNifty-relevant news, score it, and rank it by impact.

Sources:
  - Google News RSS, searched once per Bank NIFTY constituent, once per
    direct index term (Bank Nifty / NIFTY Bank) and once per macro/regulatory
    term (RBI, inflation, USDINR, etc).
  - Moneycontrol RSS (optional, see config.USE_MONEYCONTROL) -- general
    business/markets feeds, kept only if a headline mentions one of our
    companies or macro terms.

Pipeline: fetch -> drop stock-tip listicles -> score (recency + sentiment +
index weight) -> deduplicate near-identical headlines -> rank -> keep top N.

Usage:
    from datetime import datetime
    from nlp.fetch_and_rank import get_ranked_news

    ranked = get_ranked_news(datetime.now())
    for item in ranked:
        print(item["composite_score"], item["category"], item["headline"])
"""

from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from urllib.parse import quote
from zoneinfo import ZoneInfo

import feedparser
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from nlp.config import (
    BANK_COMPANIES,
    BANKNIFTY_FREE_FLOAT_WEIGHTS,
    BANKNIFTY_TERMS,
    BANKNIFTY_INDEX_WEIGHT,
    MACRO_TERMS,
    MACRO_INDEX_WEIGHT,
    CATEGORY_KEYWORDS,
    DEFAULT_CATEGORY,
    PROMOTIONAL_PHRASES,
    DEFAULT_LOOKBACK_MINUTES,
    WEIGHT_RECENCY,
    WEIGHT_SENTIMENT,
    WEIGHT_INDEX,
    DUPLICATE_SIMILARITY_THRESHOLD,
    EMBEDDING_MODEL_NAME,
    SENTIMENT_MODEL_NAME,
    TOP_N_RESULTS,
    GOOGLE_NEWS_RSS_URL,
    USE_MONEYCONTROL,
    MONEYCONTROL_FEEDS,
)

# Loaded once per process -- first call downloads weights (~500MB total),
# later calls reuse the cached models.
_embedder = None
_sentiment_pipe = None

# RSS published_parsed is always UTC; display timestamps in IST to match the
# market/session timezone used by the rest of the system.
IST_TZ = ZoneInfo("Asia/Kolkata")


def _to_utc(dt: datetime) -> datetime:
    """Normalise a reference timestamp to timezone-aware UTC.

    Naive datetimes (e.g. datetime.now()) are assumed to be local wall time
    and get the machine's local timezone attached before conversion, so the
    comparison against UTC RSS timestamps is always correct regardless of the
    host's timezone.
    """
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt.astimezone(timezone.utc)


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedder


def _get_sentiment_pipe():
    global _sentiment_pipe
    if _sentiment_pipe is None:
        _sentiment_pipe = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_NAME)
    return _sentiment_pipe


class NewsFetchError(Exception):
    """Raised when an RSS fetch fails outright."""


def is_promotional(headline: str) -> bool:
    """True if this looks like a stock-tip/listicle article, not real news."""
    h = headline.lower()
    return any(phrase in h for phrase in PROMOTIONAL_PHRASES)


def categorize(headline: str) -> str:
    h = headline.lower()
    for category, keywords in CATEGORY_KEYWORDS:
        if any(kw in h for kw in keywords):
            return category
    return DEFAULT_CATEGORY


def mentions_bank_topic(headline: str) -> bool:
    """True if a headline is actually about one of our companies/macro terms.

    Needed for feeds like Moneycontrol's general business RSS, which isn't
    pre-filtered by company the way a per-company Google News search is.
    """
    h = headline.lower()
    names = (
        list(BANK_COMPANIES.values())
        + list(BANKNIFTY_TERMS.values())
        + list(MACRO_TERMS.values())
        + ["bank", "banknifty", "nifty"]
    )
    return any(name.lower() in h for name in names)


def score_sentiments(headlines: List[str]) -> List[float]:
    """Signed sentiment in [-1, 1] for a batch of headlines, using FinBERT
    (tuned on financial text, so e.g. 'raised $200M via bonds' reads as
    neutral/positive rather than generic negative language). Batching is
    much faster than scoring one headline at a time.
    """
    if not headlines:
        return []
    results = _get_sentiment_pipe()(headlines, truncation=True)
    signed = []
    for r in results:
        label = r["label"].lower()
        if label == "positive":
            signed.append(r["score"])
        elif label == "negative":
            signed.append(-r["score"])
        else:
            signed.append(0.0)
    return signed


def _parse_pubdate(entry) -> Optional[datetime]:
    t = getattr(entry, "published_parsed", None)
    # published_parsed is always UTC (the +0000 in the RSS <pubDate>).
    return datetime(*t[:6], tzinfo=timezone.utc) if t else None


def _fetch_feed(url: str) -> List[Dict[str, Any]]:
    """Parse any RSS feed URL into a plain list of article dicts."""
    try:
        feed = feedparser.parse(url)
    except Exception as e:
        raise NewsFetchError(f"Failed to fetch/parse RSS '{url}': {e}") from e
    if feed.bozo and not feed.entries:
        raise NewsFetchError(f"Malformed RSS feed '{url}': {feed.bozo_exception}")

    articles = []
    for entry in feed.entries:
        published = _parse_pubdate(entry)
        if published is None:
            continue
        source = entry.get("source", {}).get("title", "Unknown") if hasattr(entry, "get") else "Unknown"
        articles.append({
            "headline": entry.title,
            "link": entry.link,
            "published": published,
            "source": source,
        })
    return articles


def fetch_query_news(search_term: str, limit: int = 30) -> List[Dict[str, Any]]:
    """Fetch recent Google News articles for a search term.

    Single-word terms are quoted for exact matching; multi-word phrases are
    left unquoted so Google matches all words (better recall for names like
    "HDFC Bank").
    """
    query = quote(f'"{search_term}"' if " " not in search_term else search_term)
    url = f"{GOOGLE_NEWS_RSS_URL}?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    return _fetch_feed(url)[:limit]


def fetch_moneycontrol_news() -> List[Dict[str, Any]]:
    """Pull all configured Moneycontrol feeds, keeping only headlines that
    mention one of our tracked companies or macro terms.
    """
    articles = []
    for url in MONEYCONTROL_FEEDS:
        try:
            articles.extend(_fetch_feed(url))
        except NewsFetchError:
            continue  # one dead feed shouldn't break the whole run
    return [a for a in articles if mentions_bank_topic(a["headline"])]


def _build_entries(reference_timestamp: datetime, lookback_minutes: int) -> List[Dict[str, Any]]:
    """Collect raw (unscored) articles for every company, macro term, and
    optionally Moneycontrol, tagging each with its symbol/company/weight.
    `reference_timestamp` is expected to be timezone-aware UTC.
    """
    window_start = reference_timestamp - timedelta(minutes=lookback_minutes)
    entries = []

    def add(articles, symbol, company_name, weight):
        for article in articles:
            if is_promotional(article["headline"]):
                continue
            if not (window_start <= article["published"] <= reference_timestamp):
                continue
            entries.append({**article, "symbol": symbol, "company": company_name, "index_weight": weight})

    for symbol, company_name in BANK_COMPANIES.items():
        try:
            articles = fetch_query_news(company_name)
        except NewsFetchError:
            continue
        add(articles, symbol, company_name, BANKNIFTY_FREE_FLOAT_WEIGHTS.get(symbol, 0))

    for index_key, search_term in BANKNIFTY_TERMS.items():
        try:
            articles = fetch_query_news(search_term)
        except NewsFetchError:
            continue
        add(articles, index_key, search_term, BANKNIFTY_INDEX_WEIGHT)

    for macro_key, search_term in MACRO_TERMS.items():
        try:
            articles = fetch_query_news(search_term)
        except NewsFetchError:
            continue
        add(articles, macro_key, search_term, MACRO_INDEX_WEIGHT)

    if USE_MONEYCONTROL:
        add(fetch_moneycontrol_news(), "MONEYCONTROL", "Moneycontrol", MACRO_INDEX_WEIGHT)

    return entries


def _score_entries(entries: List[Dict[str, Any]], reference_timestamp: datetime,
                    lookback_minutes: int) -> List[Dict[str, Any]]:
    sentiments = score_sentiments([e["headline"] for e in entries])

    scored = []
    for entry, sentiment_signed in zip(entries, sentiments):
        minutes_before = (reference_timestamp - entry["published"]).total_seconds() / 60
        recency_score = max(0, 1 - (minutes_before / lookback_minutes))
        sentiment_score = abs(sentiment_signed)
        composite_score = (
            WEIGHT_RECENCY * recency_score
            + WEIGHT_SENTIMENT * sentiment_score
            + WEIGHT_INDEX * entry["index_weight"]
        )
        scored.append({
            "headline": entry["headline"],
            "company": entry["company"],
            "symbol": entry["symbol"],
            "category": categorize(entry["headline"]),
            "source": entry["source"],
            "link": entry["link"],
            "published_at": entry["published"].astimezone(IST_TZ).strftime("%Y-%m-%d %H:%M"),
            "minutes_before_reference": round(minutes_before, 1),
            "sentiment_signed": round(sentiment_signed, 3),
            "index_weight": entry["index_weight"],
            "composite_score": round(composite_score, 4),
        })
    return scored


def _deduplicate(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse near-identical headlines (same event, different outlets)
    using sentence-embedding similarity, keeping the highest-scored copy.
    """
    if len(items) <= 1:
        return items

    ranked_by_score = sorted(items, key=lambda x: x["composite_score"], reverse=True)
    embeddings = _get_embedder().encode(
        [item["headline"] for item in ranked_by_score], normalize_embeddings=True
    )

    kept_indices: List[int] = []
    for i in range(len(ranked_by_score)):
        is_dup = any(
            float(np.dot(embeddings[i], embeddings[j])) >= DUPLICATE_SIMILARITY_THRESHOLD
            for j in kept_indices
        )
        if not is_dup:
            kept_indices.append(i)

    return [ranked_by_score[i] for i in kept_indices]


def add_impact_scores(ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add an `impact_score` (0-100) to each item, scaled relative to the
    strongest/weakest composite_score *within this list*. The top story in
    the list always scores 100; equal scores all get 100.
    """
    if not ranked:
        return ranked
    scores = [item["composite_score"] for item in ranked]
    lo, hi = min(scores), max(scores)
    for item in ranked:
        item["impact_score"] = 100.0 if hi == lo else round(
            100 * (item["composite_score"] - lo) / (hi - lo), 1
        )
    return ranked


def rank_news(reference_timestamp: datetime,
              lookback_minutes: int = DEFAULT_LOOKBACK_MINUTES,
              top_n: int = TOP_N_RESULTS) -> List[Dict[str, Any]]:
    """Fetch, score, deduplicate, and return the top_n ranked news events.

    The reference timestamp may be naive (interpreted as the host's local
    time) or timezone-aware; everything is normalised to UTC internally so
    recency and the lookback window are always computed in absolute time.
    """
    reference_utc = _to_utc(reference_timestamp)
    entries = _build_entries(reference_utc, lookback_minutes)
    scored = _score_entries(entries, reference_utc, lookback_minutes)
    deduped = _deduplicate(scored)
    deduped.sort(key=lambda x: x["composite_score"], reverse=True)
    return deduped[:top_n]


def get_ranked_news(reference_timestamp: datetime,
                     lookback_minutes: int = DEFAULT_LOOKBACK_MINUTES,
                     top_n: int = TOP_N_RESULTS) -> List[Dict[str, Any]]:
    """Convenience wrapper."""
    return rank_news(reference_timestamp, lookback_minutes=lookback_minutes, top_n=top_n)

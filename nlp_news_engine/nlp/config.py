"""
Configuration for the NLP news ranking module (Google News RSS version).
No API key needed.
"""

# Bank NIFTY constituent company names (used as Google News search terms).
# Keep in sync with data/data.py BANK_NIFTY_CONSTITUENTS.
BANK_COMPANIES = {
    "HDFCBANK": "HDFC Bank",
    "ICICIBANK": "ICICI Bank",
    "SBIN": "State Bank of India",
    "KOTAKBANK": "Kotak Mahindra Bank",
    "AXISBANK": "Axis Bank",
    "INDUSINDBK": "IndusInd Bank",
    "AUBANK": "AU Small Finance Bank",
    "BANKBARODA": "Bank of Baroda",
    "PNB": "Punjab National Bank",
    "FEDERALBNK": "Federal Bank",
    "IDFCFIRSTB": "IDFC First Bank",
    "BANDHANBNK": "Bandhan Bank",
}

# Approximate Bank NIFTY free-float index weights (update periodically)
BANKNIFTY_FREE_FLOAT_WEIGHTS = {
    "HDFCBANK": 0.26,
    "ICICIBANK": 0.23,
    "SBIN": 0.20,
    "KOTAKBANK": 0.08,
    "AXISBANK": 0.08,
    "INDUSINDBK": 0.02,
    "AUBANK": 0.02,
    "BANKBARODA": 0.03,
    "PNB": 0.03,
    "FEDERALBNK": 0.02,
    "IDFCFIRSTB": 0.02,
    "BANDHANBNK": 0.01,
}

# Direct index-level search terms -- these are the most directly relevant to
# the traded Bank Nifty index itself (as opposed to a single constituent), so
# they are queried on their own and given a high flat weight.
BANKNIFTY_TERMS = {
    "BANKNIFTY": "Bank Nifty",
    "NIFTY_BANK": "NIFTY Bank",
}

# Index-level news gets a weight slightly above a top constituent, since a move
# in the index moves the whole traded contract.
BANKNIFTY_INDEX_WEIGHT = 0.25

# Macro/regulatory search terms -- these move BankNifty as a whole, not just
# one constituent, so they get their own (higher-weighted) search queries.
# symbol is None since these aren't tied to one stock's index weight.
MACRO_TERMS = {
    "RBI": "RBI monetary policy",
    "SEBI": "SEBI banking",
    "REPO_RATE": "RBI repo rate",
    "INFLATION": "India inflation CPI",
    "GDP": "India GDP growth",
    "BOND_YIELD": "India bond yield",
    "USDINR": "USD INR rupee",
    "CRUDE_OIL": "crude oil price India",
    "FII_DII": "FII DII flows India",
}
# Macro news gets a flat weight comparable to a large-cap constituent,
# since a repo rate move affects the whole index, not one bank.
MACRO_INDEX_WEIGHT = 0.20

# Keyword -> category, checked in order (first match wins). Used to group
# the final ranked list for readability.
CATEGORY_KEYWORDS = [
    ("Monetary policy", ["rbi", "repo rate", "monetary policy", "mpc", "crr"]),
    ("Regulatory", ["sebi", "regulation", "compliance", "probe", "penalty"]),
    ("Bank earnings", ["profit", "q1", "q2", "q3", "q4", "results", "earnings", "net income"]),
    ("Corporate actions", ["dividend", "bond", "raises", "fundraise", "ipo", "stake", "merger", "acquisition"]),
    ("Macro", ["inflation", "gdp", "cpi", "bond yield", "crude oil", "fii", "dii"]),
    ("Currency", ["usdinr", "rupee", "dollar"]),
]
DEFAULT_CATEGORY = "General"

# Headlines matching any of these phrases are stock-tip/listicle content,
# not real news events -- excluded entirely before scoring.
PROMOTIONAL_PHRASES = [
    "stocks to buy", "stock to buy", "shares to buy",
    "top picks", "buy now", "should you buy",
    "multibagger", "stocks to watch", "best stocks",
    "penny stocks", "stock recommendation", "top 5 stocks",
    "top 10 stocks", "stocks to add", "portfolio picks",
    "long term buy", "buy for target", "share price target",
    "diwali picks", "muhurat picks",
]

DEFAULT_LOOKBACK_MINUTES = 1440

# Composite score weights
WEIGHT_RECENCY = 0.40
WEIGHT_SENTIMENT = 0.35
WEIGHT_INDEX = 0.25

# Headlines with a semantic (embedding) similarity above this cosine value
# are treated as duplicates of the same underlying event; only the
# highest-scored one is kept. 0.75 catches paraphrases across outlets
# without merging genuinely different stories about the same company.
DUPLICATE_SIMILARITY_THRESHOLD = 0.75

# Sentence-transformers model for semantic dedup -- small, fast, CPU-friendly.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# FinBERT -- sentiment model trained specifically on financial text.
SENTIMENT_MODEL_NAME = "ProsusAI/finbert"

TOP_N_RESULTS = 100

# How many top stories go in the daily report, and how many of those are
# shown in the final CSV.
REPORT_POOL_SIZE = 10
REPORT_TOP_N = 5

GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"

# --- Optional: Moneycontrol as an extra source -----------------------------
# Moneycontrol publishes plain RSS feeds (no API key needed), e.g. its
# markets/business sections. Set USE_MONEYCONTROL = True to pull these in
# alongside Google News. Each entry is fetched as-is (not per-company), then
# filtered by the same BANK_COMPANIES / MACRO_TERMS keywords during scoring.
USE_MONEYCONTROL = False
MONEYCONTROL_FEEDS = [
    "https://www.moneycontrol.com/rss/latestnews.xml",
    "https://www.moneycontrol.com/rss/business.xml",
    "https://www.moneycontrol.com/rss/economy.xml",
]

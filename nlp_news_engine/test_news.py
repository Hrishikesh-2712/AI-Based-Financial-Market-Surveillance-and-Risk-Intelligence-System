"""
Quick terminal test for the news ranking module (Google News RSS version).

Run:
    python test_news.py                       # uses current time
    python test_news.py "2026-07-24 11:20"    # uses a specific timestamp
"""

import sys
from datetime import datetime

from nlp.fetch_and_rank import get_ranked_news, NewsFetchError

if len(sys.argv) > 1:
    ts = datetime.strptime(sys.argv[1], "%Y-%m-%d %H:%M")
else:
    ts = datetime.now()

print(f"Fetching and ranking news around {ts}...\n")

try:
    ranked = get_ranked_news(ts)
except NewsFetchError as e:
    print(f"Fetch failed: {e}")
    sys.exit(1)

if not ranked:
    print("No relevant articles found in this window. Try a wider lookback "
          "(edit DEFAULT_LOOKBACK_MINUTES in config.py) or a timestamp on a "
          "day with known bank news.")
else:
    top_n = len(ranked)
    print(f"Found {len(ranked)} relevant articles.\n")
    for i, item in enumerate(ranked[:top_n], 1):
        print(f"{i}. [{item['composite_score']}] {item['headline']}")
        print(f"   {item['company']} | {item['minutes_before_reference']} min before | "
              f"sentiment {item['sentiment_signed']} | source: {item['source']}")
        print()

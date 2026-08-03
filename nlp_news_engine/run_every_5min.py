"""
Runs the news collection + ranking pipeline every 5 minutes and saves the
result as a JSON snapshot (snapshots/latest.json plus a timestamped copy)
for the backend pipeline and frontend to consume.

Run:
    python run_every_5min.py

Stop with Ctrl+C.
"""

import time
from datetime import datetime

from nlp.fetch_and_rank import get_ranked_news
from nlp.snapshot import save_snapshot

POLL_INTERVAL_SECONDS = 300  # 5 minutes


def run_once():
    now = datetime.now()
    print(f"\n[{now:%Y-%m-%d %H:%M:%S}] Fetching and ranking news...")

    try:
        ranked = get_ranked_news(now)
    except Exception as e:
        print(f"  Fetch/ranking failed: {type(e).__name__}: {e}")
        return

    snapshot = save_snapshot(ranked, now)
    print(f"  Found {len(ranked)} ranked events. Saved to snapshots/latest.json")
    for i, item in enumerate(ranked[:5], 1):
        print(f"   {i}. [{item['composite_score']}] ({item['category']}) {item['headline'][:70]}")


def main():
    print(f"Starting news polling loop -- every {POLL_INTERVAL_SECONDS // 60} minutes.")
    print("Press Ctrl+C to stop.\n")
    try:
        while True:
            run_once()
            time.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

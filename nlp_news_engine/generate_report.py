"""
Run once, no input needed: fetch BankNifty news, rank the top 10 by impact,
score each 0-100 relative to that pool, and write the top 5 to a CSV.

Also writes the same JSON snapshot as run_every_5min.py (unchanged format,
with one added field: impact_score) so existing downstream readers of
snapshots/*.json keep working.

Run:
    python generate_report.py
"""

import csv
from datetime import datetime
from pathlib import Path

from nlp.config import REPORT_POOL_SIZE, REPORT_TOP_N
from nlp.fetch_and_rank import get_ranked_news, add_impact_scores
from nlp.snapshot import save_snapshot

BASE_DIR = Path(__file__).parent

CSV_COLUMNS = [
    "rank", "impact_score", "composite_score", "headline", "company",
    "category", "sentiment_signed", "source", "published_at", "link",
]


def build_report(now: datetime):
    """Fetch, rank top 10, score impact, return (top10_pool, top5_for_csv)."""
    pool = get_ranked_news(now, top_n=REPORT_POOL_SIZE)
    pool = add_impact_scores(pool)  # scores are relative to this top-10 pool
    top5 = pool[:REPORT_TOP_N]
    return pool, top5


def write_csv(now: datetime, top5) -> Path:
    csv_path = BASE_DIR / f"top5_news_{now.strftime('%Y-%m-%d_%H-%M')}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for i, item in enumerate(top5, 1):
            writer.writerow({"rank": i, **{k: item[k] for k in CSV_COLUMNS if k != "rank"}})
    return csv_path


def main():
    now = datetime.now()
    print(f"[{now:%Y-%m-%d %H:%M:%S}] Fetching and ranking BankNifty news...")
    try:
        pool, top5 = build_report(now)
    except Exception as e:
        print(f"Fetch/ranking failed: {type(e).__name__}: {e}")
        return

    if not pool:
        print("No relevant articles found in the lookback window.")
        return

    save_snapshot(pool, now)
    csv_path = write_csv(now, top5)

    print(f"\nTop {len(top5)} of {len(pool)} ranked stories -> {csv_path.name}\n")
    for i, item in enumerate(top5, 1):
        print(f"{i}. [impact {item['impact_score']}] ({item['category']}) {item['headline'][:80]}")


if __name__ == "__main__":
    main()

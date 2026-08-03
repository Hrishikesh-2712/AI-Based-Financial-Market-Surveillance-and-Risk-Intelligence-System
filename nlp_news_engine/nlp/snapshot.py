"""
Single place that persists ranked-news snapshots, so the poller
(run_every_5min.py), the report generator (generate_report.py) and the
backend pipeline (backend/pipeline.py) all write the same format to the
same files instead of duplicating the logic.

Format:
    {"generated_at": "YYYY-MM-DD HH:MM:SS", "count": N, "top_news": [...]}
"""

import json
from datetime import datetime
from pathlib import Path

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "snapshots"


def save_snapshot(ranked: list, now: datetime | None = None) -> dict:
    """Write `ranked` to snapshots/latest.json and a timestamped file."""
    now = now or datetime.now()
    SNAPSHOT_DIR.mkdir(exist_ok=True)
    snapshot = {
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "count": len(ranked),
        "top_news": ranked,
    }
    for path in (
        SNAPSHOT_DIR / "latest.json",
        SNAPSHOT_DIR / f"{now.strftime('%Y-%m-%d_%H-%M')}.json",
    ):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
    return snapshot

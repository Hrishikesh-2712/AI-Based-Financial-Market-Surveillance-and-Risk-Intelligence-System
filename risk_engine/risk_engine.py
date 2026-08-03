# -*- coding: utf-8 -*-
"""
risk_engine.py
==============
Composite risk score = weighted ensemble of Rule Engine + Isolation Forest
+ TCN Autoencoder, each normalized by PERCENTILE RANK against a baseline
distribution (not fixed linear scaling), plus a bidirectional, timing- and
direction-aware news adjustment.

Why percentile rank instead of linear scaling:
    Isolation Forest's decision_function and TCN's reconstruction MSE don't
    have fixed, comparable ranges -- what counts as "high" depends on your
    data's own distribution. Percentile rank against a rolling baseline
    (e.g. the last 20-30 trading days) adapts to that automatically, rather
    than assuming raw scores always fall in some convenient hardcoded range.
    This follows the same principle used in academic capital-markets
    surveillance work, e.g. suspicion scores built from percentile rank of
    anomaly distance rather than raw magnitude (see arXiv:2607.04184).

Why TCN's severity is now used numerically (not just a yes/no flag):
    The previous version only used tcn_anomaly as a boolean trigger and
    discarded the actual reconstruction error magnitude. Isolation Forest
    and TCN are independent detector types (point anomalies vs. sequence
    anomalies) and should each contribute their own weighted, graded score
    to the ensemble -- consistent with how multi-detector composite scores
    are built in the anomaly-fusion literature (e.g. AMRS, arXiv:2512.16103,
    combines independently normalized component signals via weighted sum).

Why the news adjustment is bidirectional and timing-aware:
    News should be able to REDUCE risk (genuinely explains the anomaly, and
    was published before it), not just add a smaller penalty for being
    "less unexplained." News published AFTER the anomaly is treated as the
    strongest suspicion signal, since unusual activity preceding public
    information is the pattern most consistent with trading ahead of news
    (directly analogous to the SEBI-Jane Street framing this project is
    built around).

No single published formula exists for this exact combination (Rule Engine
+ IF + TCN + timing-aware news adjustment, applied to Bank NIFTY options
surveillance) -- this design follows established general patterns
(percentile normalization, weighted multi-detector fusion) from the papers
cited above, combined with a domain-specific news-timing rule. Say so
plainly in your report rather than citing an invented formula name.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensemble weights -- reasonable starting points, NOT empirically tuned.
# Sum must equal 1.0. Revisit these against known event days (expiry,
# budget, RBI policy) if you have time before submission.
# ---------------------------------------------------------------------------
WEIGHT_RULE = 0.25
WEIGHT_IF = 0.35
WEIGHT_TCN = 0.40

# News adjustment deltas (0-100 scale), applied additively to the base score.
NEWS_ADJ_NO_NEWS = 5          # unexplained, but not strongly suspicious alone
NEWS_ADJ_AFTER_ANOMALY = 20    # anomaly PRECEDED the news -> info-leak risk
NEWS_ADJ_EXPLAINED = -15        # news before anomaly, direction matches
NEWS_ADJ_CONTRADICTS = 10        # news before anomaly, direction disagrees
NEWS_SENTIMENT_NEUTRAL_BAND = 0.05  # |sentiment| below this treated as neutral

RISK_LOW_MAX = 40
RISK_MEDIUM_MAX = 70


# ---------------------------------------------------------------------------
# Percentile-rank normalization
# ---------------------------------------------------------------------------

def _percentile_rank(value: float, baseline: Optional[Sequence[float]]) -> Optional[float]:
    """% of baseline values strictly below `value`. Returns None if no usable baseline."""
    if baseline is None:
        return None
    arr = np.asarray(baseline, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 10:  # too few points for a meaningful percentile
        return None
    return float((arr < value).sum()) / len(arr) * 100.0


def _if_risk_score(if_score_raw: float, if_baseline: Optional[Sequence[float]],
                    fallback_linear: bool = True) -> float:
    """0-100, higher = more anomalous. Lower/more-negative if_score_raw = more anomalous."""
    pr = _percentile_rank(if_score_raw, if_baseline)
    if pr is not None:
        return float(np.clip(100.0 - pr, 0, 100))
    if fallback_linear:
        # Degraded fallback when no baseline is available: sklearn's
        # decision_function is typically centered near 0, roughly
        # [-0.5, 0.5] for IsolationForest with default contamination.
        return float(np.clip((0.5 - if_score_raw) * 100, 0, 100))
    return 50.0  # neutral if we truly have nothing to go on


def _tcn_risk_score(tcn_error_raw: float, tcn_baseline: Optional[Sequence[float]],
                     tcn_threshold: Optional[float] = None) -> float:
    """0-100, higher = more anomalous. Higher tcn_error_raw = more anomalous."""
    pr = _percentile_rank(tcn_error_raw, tcn_baseline)
    if pr is not None:
        return float(np.clip(pr, 0, 100))
    if tcn_threshold and tcn_threshold > 0:
        # Degraded fallback: scale relative to the model's own saved
        # threshold.json value (error == threshold -> 50, 2x -> 100).
        return float(np.clip((tcn_error_raw / tcn_threshold) * 50, 0, 100))
    return 50.0


# ---------------------------------------------------------------------------
# News adjustment: bidirectional, timing- and direction-aware
# ---------------------------------------------------------------------------

def _parse_news_time(published_at) -> Optional[datetime]:
    if not published_at:
        return None
    try:
        ts = pd.to_datetime(published_at)
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)  # compare naive-to-naive; see note below
        return ts.to_pydatetime()
    except (ValueError, TypeError):
        return None


def _news_adjustment(nlp_results: dict, anomaly_timestamp: datetime,
                      anomaly_direction: float) -> tuple:
    """
    Returns (delta, reason). anomaly_timestamp should be naive local time
    (strip tz before calling) to match parsed news timestamps.
    anomaly_direction: signed value (e.g. Log_Returns) -- positive = bullish
    move, negative = bearish move, used only for its sign.
    """
    top_news = (nlp_results or {}).get("top_news", [])
    if not top_news:
        return NEWS_ADJ_NO_NEWS, "No relevant news found -- unexplained, but not conclusive alone."

    top = top_news[0]  # already ranked by composite_score
    news_time = _parse_news_time(top.get("published_at"))
    sentiment = float(top.get("sentiment_signed", 0.0) or 0.0)
    headline = top.get("headline", "")[:80]

    if news_time is not None and anomaly_timestamp is not None and news_time > anomaly_timestamp:
        return (
            NEWS_ADJ_AFTER_ANOMALY,
            f"Top news ('{headline}') was published AFTER this anomaly -- "
            f"pattern consistent with activity preceding public information.",
        )

    if abs(sentiment) < NEWS_SENTIMENT_NEUTRAL_BAND or abs(anomaly_direction) < 1e-9:
        return NEWS_ADJ_NO_NEWS, f"News found ('{headline}') but sentiment/direction too weak to classify."

    direction_match = (sentiment > 0 and anomaly_direction > 0) or (sentiment < 0 and anomaly_direction < 0)
    if direction_match:
        return NEWS_ADJ_EXPLAINED, f"News ('{headline}') published before anomaly, sentiment matches direction -- likely explained."
    return NEWS_ADJ_CONTRADICTS, f"News ('{headline}') published before anomaly but CONTRADICTS its direction -- still flagged."


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def calculate_composite_risk_score(
    rule_score: float,
    if_score_raw: float,
    tcn_error_raw: float,
    anomaly_timestamp: datetime,
    anomaly_direction: float,
    nlp_results: dict,
    if_baseline: Optional[Sequence[float]] = None,
    tcn_baseline: Optional[Sequence[float]] = None,
    tcn_threshold: Optional[float] = None,
) -> dict:
    """
    rule_score: 0-100, from the rule engine / tech heuristic.
    if_score_raw: raw IsolationForest.decision_function() output for this row.
    tcn_error_raw: raw TCN reconstruction MSE for this row.
    anomaly_timestamp: naive local datetime of this row (strip tz first).
    anomaly_direction: signed value (e.g. raw Log_Returns) for this row.
    nlp_results: output of pipeline._load_nlp_analysis() (has "top_news").
    if_baseline / tcn_baseline: optional historical arrays of if_score /
        tcn_error values (e.g. last 20-30 trading days) for percentile
        normalization. Falls back to a degraded linear scaling if omitted --
        see the module docstring for why a baseline is strongly preferred.
    tcn_threshold: the model's saved threshold.json value, used only by the
        fallback scaling when tcn_baseline is not provided.
    """
    rule_component = float(np.clip(rule_score, 0, 100))
    if_component = _if_risk_score(if_score_raw, if_baseline)
    tcn_component = _tcn_risk_score(tcn_error_raw, tcn_baseline, tcn_threshold)

    base_risk = (
        WEIGHT_RULE * rule_component
        + WEIGHT_IF * if_component
        + WEIGHT_TCN * tcn_component
    )

    news_delta, news_reason = _news_adjustment(nlp_results, anomaly_timestamp, anomaly_direction)
    final_score = round(float(np.clip(base_risk + news_delta, 0, 100)), 2)

    if final_score >= RISK_MEDIUM_MAX:
        label = "HIGH RISK"
    elif final_score >= RISK_LOW_MAX:
        label = "MEDIUM RISK"
    else:
        label = "LOW RISK"

    if final_score >= RISK_MEDIUM_MAX and news_delta >= NEWS_ADJ_AFTER_ANOMALY:
        label = "CRITICAL: Unexplained / Possible Information Leakage"

    return {
        "risk_score": final_score,
        "risk_classification": label,
        "components": {
            "rule_score": round(rule_component, 2),
            "if_risk_score": round(if_component, 2),
            "tcn_risk_score": round(tcn_component, 2),
            "base_risk_before_news": round(base_risk, 2),
            "news_adjustment": news_delta,
        },
        "news_reason": news_reason,
    }

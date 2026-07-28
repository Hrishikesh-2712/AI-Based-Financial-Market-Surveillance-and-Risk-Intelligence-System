def calculate_research_risk_score(
    if_raw_score: float, 
    is_anomaly: bool, 
    tech_risk_score: float, 
    nlp_results: dict
) -> dict:
    """
    Implements Composite Anomaly Risk Score (CARS) formula.
    
    :param if_raw_score: Raw decision function score from Isolation Forest (e.g. -0.25 to 0.25)
    :param is_anomaly: True if Isolation Forest flagged -1
    :param tech_risk_score: 0-100 score from your technical indicators
    :param nlp_results: Output dictionary from analyze_news_nlp()
    """
    relevance = nlp_results.get("relevance_score", 0.0)
    sentiment = nlp_results.get("sentiment_score", 0.0)
    
    # 1. Normalize Isolation Forest score to [0, 100] scale
    # Raw IF decision scores: negative = anomaly, positive = normal
    if_intensity = max(0.0, min(100.0, (0.5 - if_raw_score) * 100)) if is_anomaly else 0.0

    # 2. Calculate Information Disconnection Index (D_info)
    # Measures how unexplained the market move is by public news
    d_info = 1.0 - (relevance * abs(sentiment))
    
    # 3. Parameters
    lambda_penalty = 0.5   # Penalty scaling factor for unexplained anomalies
    w1, w2 = 0.4, 0.6      # Weights for Tech Score vs Anomaly Engine
    
    # 4. CARS Formula Execution
    unexplained_multiplier = 1.0 + (lambda_penalty * d_info)
    anomaly_adjusted_risk = if_intensity * unexplained_multiplier
    
    cars_score = (w1 * tech_risk_score) + (w2 * anomaly_adjusted_risk)
    final_score = round(min(100.0, max(0.0, cars_score)), 2)

    # Classification Labeling
    if is_anomaly and d_info > 0.7:
        label = "CRITICAL: Unexplained Market Anomaly (Low/No News Support)"
    elif is_anomaly and d_info <= 0.7:
        label = "MODERATE: News-Catalyzed Market Movement"
    else:
        label = "LOW: Normal Trading Activity"

    return {
        "cars_risk_score": final_score,
        "information_disconnection_index": round(d_info, 3),
        "risk_classification": label
    }

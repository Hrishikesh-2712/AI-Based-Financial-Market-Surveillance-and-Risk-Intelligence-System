# AI-Based Market Surveillance & Risk Intelligence System — Bank Nifty

An end-to-end market surveillance platform for the Bank Nifty index that combines a hybrid deep-learning/classical anomaly detector (TCN Autoencoder + Isolation Forest), an NLP news-ranking engine, a composite risk-scoring module (CARS — Composite Anomaly Risk Score), and a Retrieval-Augmented Generation (RAG) chatbot, all served through a FastAPI backend and a Streamlit dashboard.

---

## 1. Overview

The system continuously pulls live 5-minute Bank Nifty candle data from the FYERS broker API, engineers technical-indicator features, scores each candle with a hybrid anomaly-detection model, cross-references the result against ranked financial news, and produces a composite risk score. Results are exposed through a FastAPI backend and visualized on an interactive Streamlit dashboard, which also includes an AI chatbot for querying anomalies and news context.

**Core pipeline:**

```
FYERS live data → Feature Engineering → TCN + Isolation Forest inference
        → NLP news ranking → CARS Composite Risk Score → Streamlit UI
```

---

## 2. Key Features

- **Live data ingestion** from the FYERS API with automatic fallback to a local CSV cache when the API/token is unavailable.
- **Hybrid anomaly detection** combining a TCN (Temporal Convolutional Network) autoencoder for sequence anomalies and an Isolation Forest for point anomalies.
- **NLP news engine** that fetches, ranks, and scores financial news relevance/sentiment on a recurring schedule.
- **CARS risk engine** — a percentile-ranked, news-timing-aware composite risk score combining rule-based, Isolation Forest, and TCN signals.
- **FastAPI backend** exposing REST endpoints for snapshots, risk data, candle data, and news.
- **Streamlit dashboard** with a cinematic landing page and a live candlestick/anomaly/risk/news dashboard.
- **RAG chatbot** ("Ask about anomalies & news") built with LangChain + FAISS, backed by Google Gemini or a fully local Ollama model.

---

## 3. Prerequisites

| Requirement | Notes |
|---|---|
| Python | **3.12.6** (see `.python-version`) |
| pip | Latest version recommended |
| FYERS trading account + API app | Required for live data (`data/auth.py`, `data/data.py`) |
| Google Gemini API key **or** Ollama installed locally | Required only for the RAG chatbot |
| Internet access | Required for live FYERS data and NLP news fetching |

---

## 4. Project Structure

```
.
├── backend/                 # FastAPI application
│   ├── main.py               # API entrypoint (uvicorn target)
│   └── pipeline.py           # End-to-end live surveillance pipeline
├── chatbot/                  # RAG chatbot (LangChain + FAISS)
│   ├── config.py
│   ├── data_loader.py
│   ├── vector_store.py
│   ├── llm_provider.py
│   └── qa_chain.py
├── data/                     # Data access & FYERS authentication
│   ├── auth.py                # FYERS OAuth login / token refresh
│   └── data.py                 # Live/cached data fetch utilities
├── feature_extract/          # Technical indicator feature engineering
├── model/                    # TCN + Isolation Forest training & inference
├── frontend/                 # Streamlit UI
│   ├── index.py                # Combined landing page + dashboard entrypoint
│   ├── app.py                   # Dashboard-only Streamlit page
│   ├── style.py
│   └── landing.html
├── nlp_news_engine/           # News fetching, ranking, and snapshotting
├── risk_engine/               # CARS composite risk score
├── saved_models/               # Pretrained model artifacts (.pkl / .pth)
├── config.py                    # Shared constants and paths
└── requirements.txt
```

---

## 5. Installation

**1. Extract the project and navigate into it:**
```bash
cd AI-Based-Financial-Market-Surveillance-and-Risk-Intelligence-System
```

**2. Create and activate a virtual environment (Python 3.12.6):**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install the core dependencies:**
```bash
pip install fyers-apiv3==3.1.14
```
```bash
pip install -r requirements.txt
```
> **Note on Dependency Compatibility:**  
> You may encounter version conflicts involving **Fyers API** and **LangChain** due to differences in their dependency requirements. These conflicts are related to package-version compatibility and do not necessarily affect the application's functionality. If encountered, review the reported dependencies and proceed with the installation as described below.

**4. Install the NLP news engine dependencies** (some are commented out in the root `requirements.txt`; the engine needs them uncommented/installed):
```bash
pip install -r nlp_news_engine/requirements.txt
```

**5. Install the RAG chatbot dependencies:**
```bash
pip install -r chatbot/requirements-chatbot.txt
```

> Pretrained model artifacts (`isolation_forest.pkl`, `tcn_autoencoder.pth`, `scaler.pkl`, `threshold.json`) are already provided in `saved_models/`, so retraining is **not required** to run the system. Retraining scripts (`model/isolation_forest.py`, `model/tcn.py`, `feature_extract/feature_extract.py`) are available if you wish to retrain on new data.

---

## 6. Configuration (`.env`)

The project uses environment variables loaded via `python-dotenv`. Two `.env` files are used:

### a) Project root `.env`
Used for the FYERS broker credentials **and** the chatbot's LLM provider settings:
(We have included our own api keys.)

```env
# FYERS API credentials (required for live data + authentication)
FYERS_APP_ID=your_fyers_app_id
FYERS_SECRET_KEY=your_fyers_secret_key
REDIRECT_URI=your_fyers_redirect_uri
FYERS_PIN=your_4_digit_trading_pin      # optional, enables silent token refresh

# RAG Chatbot LLM provider (choose one)
LLM_PROVIDER=gemini                      # "gemini" or "ollama"
GOOGLE_API_KEY=your_google_api_key       # required if LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-1.5-flash            # optional override
OLLAMA_MODEL=llama3.2                    # optional, used if LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434   # optional, used if LLM_PROVIDER=ollama
```


### b) Optional runtime override
```env
SURVEILLANCE_API_URL=http://127.0.0.1:5000   # used by the Streamlit UI to reach the backend
```

---

## 7. Execution Commands

Run the following commands **in order**, each in its own terminal (with the virtual environment activated), from the project root.

**Step 1 — Authenticate with FYERS** (one-time, or whenever the saved token expires):
```bash
python data/auth.py
```
This opens a browser window for FYERS login and saves `access_token.txt` / `refresh_token.txt` under `data/`.

**Step 2 — Start the FastAPI backend:**
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 5000 --reload
```
The backend runs a background scheduler that refreshes the surveillance pipeline every 5 minutes and serves REST endpoints for the dashboard.

**Step 3 — Launch the Streamlit dashboard:**
```bash
streamlit run frontend/index.py
```
This serves the combined landing page and live dashboard (candlestick chart, risk/anomaly panel, news panel, and RAG chat panel). Alternatively, `streamlit run frontend/app.py` launches the dashboard directly without the landing page.

---

## 8. API Endpoints (FastAPI backend)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Model artifact availability and system status |
| GET | `/api/snapshot` | Full pipeline output (cached, 5-minute TTL) |
| GET | `/api/risk` | Model inference + CARS risk summary |
| GET | `/api/candles` | Candle data with per-bar anomaly markers |
| GET | `/api/news` | NLP-ranked news feeding the risk engine |
| POST | `/api/refresh` | Force an immediate pipeline re-run |

Interactive API documentation is available at `http://127.0.0.1:5000/docs` once the backend is running.

---

## 9. Troubleshooting

- **`Missing FYERS_APP_ID / FYERS_SECRET_KEY / REDIRECT_URI` on `python data/auth.py`** — Ensure the root `.env` (not just `data/.env`) contains the FYERS credentials; `data/auth.py` reads from the project root.
- **Backend falls back to `csv-cache`** — Indicates the FYERS token is missing/expired or the API is unreachable; re-run `python data/auth.py`.
- **Chat panel shows an inline error instead of answering** — Confirm `LLM_PROVIDER`, `GOOGLE_API_KEY` (or a running Ollama instance), and that `nlp_news_engine/snapshots/latest.json` exists (generated automatically by the backend, or manually via `python nlp_news_engine/run_every_5min.py`).
- **`ModuleNotFoundError` for NLP or chatbot packages** — Confirm steps 4 and 5 in the Installation section were run in addition to the root `requirements.txt`.

---

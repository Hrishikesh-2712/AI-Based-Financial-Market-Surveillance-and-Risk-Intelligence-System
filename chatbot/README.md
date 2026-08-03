# RAG Chatbot — integration guide

This is the "💬 Ask about anomalies & news" chatbot for the RAG-slot that
was already marked "coming soon" in `frontend/app.py`. It's a
LangChain + FAISS RAG pipeline: retrieves the most relevant ranked news
items (and a fixed glossary of this project's own scoring fields) for a
question, then answers using Gemini or a fully-local Ollama model.

## What it reads

`nlp_news_engine/snapshots/latest.json` — the same file already produced
by `nlp_news_engine/run_every_5min.py` / `backend/pipeline.py`. No new
data source, no schema changes needed.

## Files to drop in

```
your-project/
├── chatbot/                  <- copy this whole folder in as-is
│   ├── __init__.py
│   ├── config.py
│   ├── data_loader.py
│   ├── vector_store.py
│   ├── llm_provider.py
│   └── qa_chain.py
├── frontend/
│   ├── app.py                <- replace with the one in this zip
│   └── style.py               <- replace with the one in this zip
└── .env.example                <- copy to project root
```

`frontend/app.py` and `frontend/style.py` already have the integration
applied (the old `RAG Chatbot — coming soon` placeholder is now a real
`st.chat_input` panel wired to `chatbot/qa_chain.ask()`). If you've made
your own edits to those two files since the version I read, diff them
instead of overwriting — the only changes are:
- `style.py`: removed the unused `.rag-placeholder` CSS class.
- `app.py`: added a `chat_panel()` function, and swapped the placeholder
  `st.markdown(...)` line for a call to it.

## Setup

1. `cp .env.example .env` (project root — **not** `data/.env`, which is
   the separate Fyers file) and fill in:
   ```
   LLM_PROVIDER=gemini
   GOOGLE_API_KEY=your_key_here      # https://aistudio.google.com/app/apikey
   ```
   or run fully offline with `LLM_PROVIDER=ollama` (install Ollama, then
   `ollama pull llama3.2`).

2. Add the chatbot's dependencies to `requirements.txt` — see
   `chatbot/requirements-chatbot.txt` for the exact list — then:
   ```
   pip install -r requirements.txt
   ```
   First run downloads a small (~80MB) embedding model
   (`sentence-transformers/all-MiniLM-L6-v2`) once, then it's cached.

3. Make sure `nlp_news_engine/snapshots/latest.json` exists (run
   `python nlp_news_engine/run_every_5min.py` once, or just start the
   backend — it calls the NLP module itself).

4. `streamlit run frontend/app.py` and scroll down to the chat panel.

## What's different from a naive RAG setup

- **Project-specific glossary**: `data_loader.py`'s `GLOSSARY_TEXT` was
  rewritten to describe *this* project's actual scoring system — CARS
  formula weights, D_info / "unexplained anomaly" threshold, the exact
  8 feature columns, TCN vs Isolation Forest — pulled directly from
  `risk_engine/risk_engine.py` and `config.py`, not generic placeholder
  text. Ask it "what does CARS score mean" or "why would something be
  CRITICAL" and it'll answer correctly for your system specifically.
- **Caching**: the original reference version rebuilt the FAISS index
  and re-embedded every document on *every single question*. This
  version (`vector_store.get_or_build`) only rebuilds when
  `latest.json`'s mtime actually changes (i.e. once per 5-minute news
  refresh), and caches the LLM client + RetrievalQA chain across
  questions too. Same freshness guarantee, far less repeated work.
- **Graceful failure**: if no API key/Ollama is set up, or no news
  snapshot exists yet, the chat panel shows a clear inline message
  instead of crashing the whole dashboard — the rest of the panels
  (chart, risk, news) keep working regardless.

## Security note

If you copied a `.env` from an earlier prototype into this project,
double check it doesn't contain a real `GOOGLE_API_KEY` before
committing/sharing the repo — rotate it at
https://aistudio.google.com/app/apikey if it's already been exposed
anywhere (chat, GitHub, etc).

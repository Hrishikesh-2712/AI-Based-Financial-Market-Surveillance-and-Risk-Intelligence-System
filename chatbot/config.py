# -*- coding: utf-8 -*-
"""
config.py
=========
Central config for the GenAI Q&A chatbot. Switch LLM_PROVIDER between
"gemini" and "ollama" without touching any other file.

Reads from environment variables, via a .env file at the project root
(NOT data/.env -- that one holds the Fyers broker credentials and is
loaded separately by data/auth.py / data/data.py).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# "gemini" (needs GOOGLE_API_KEY) or "ollama" (fully local, no key needed)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Always re-read this file fresh -- it's overwritten every 5 minutes by
# nlp_news_engine/run_every_5min.py (see nlp_news_engine/nlp/snapshot.py).
NEWS_JSON_PATH = PROJECT_ROOT / "nlp_news_engine" / "snapshots" / "latest.json"

RETRIEVER_TOP_K = 6

# The vector store is rebuilt only when latest.json's mtime changes
# (see vector_store.get_or_build), not on every single question -- the
# original MVP rebuilt the FAISS index + re-embedded ~90 docs on every
# ask() call, which is wasted work once a session is already warm.

# -*- coding: utf-8 -*-
"""
vector_store.py
================
Builds an in-memory FAISS index from the news + glossary documents.

Improvement over a naive "rebuild every question" approach: the index
is cached in-process and only rebuilt when nlp_news_engine/snapshots/
latest.json's mtime actually changes (it's overwritten every 5 minutes
by run_every_5min.py). Re-embedding ~90 short documents on CPU only
takes a couple of seconds, but there's no reason to pay that cost on
every single question when the underlying news hasn't changed.
"""

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from . import config
from .data_loader import load_documents

_embeddings = None
_cache = {"mtime": None, "vectordb": None, "meta": None}


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)
    return _embeddings


def _news_mtime():
    try:
        return config.NEWS_JSON_PATH.stat().st_mtime
    except FileNotFoundError:
        return None


def build_vector_store(documents):
    embeddings = get_embeddings()
    return FAISS.from_documents(documents=documents, embedding=embeddings)


def get_or_build():
    """Returns (vectordb, meta), rebuilding only when latest.json changed."""
    current_mtime = _news_mtime()
    if current_mtime is None:
        raise FileNotFoundError(
            f"No news snapshot found at {config.NEWS_JSON_PATH}. "
            f"Run nlp_news_engine/run_every_5min.py first."
        )

    if _cache["vectordb"] is None or _cache["mtime"] != current_mtime:
        documents, meta = load_documents()
        _cache["vectordb"] = build_vector_store(documents)
        _cache["meta"] = meta
        _cache["mtime"] = current_mtime

    return _cache["vectordb"], _cache["meta"]

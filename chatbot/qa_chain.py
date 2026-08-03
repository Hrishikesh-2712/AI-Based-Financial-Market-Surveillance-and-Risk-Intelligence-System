# -*- coding: utf-8 -*-
"""
qa_chain.py
============
Builds a retrieval-augmented QA chain over the news + glossary documents,
and exposes a single ask(question) function for the Streamlit UI to call.

The vector store is rebuilt only when the news snapshot changes (see
vector_store.get_or_build); the LLM client and RetrievalQA chain are
built once per process and reused across questions.
"""

try:
    # langchain < 1.0
    from langchain.chains import RetrievalQA
except ModuleNotFoundError:
    # langchain >= 1.0 moved legacy chains here
    from langchain_classic.chains import RetrievalQA

try:
    from langchain.prompts import PromptTemplate
except ModuleNotFoundError:
    from langchain_core.prompts import PromptTemplate

from . import config
from .llm_provider import get_llm
from .vector_store import get_or_build

PROMPT_TEMPLATE = """You are an assistant embedded in the "AI-Based Financial Market \
Surveillance and Risk Intelligence System" -- a Bank NIFTY anomaly detection \
project combining a TCN Autoencoder, an Isolation Forest, an NLP news-ranking \
module, and a CARS (Composite Anomaly Risk Score) risk engine.

Answer questions about recent Bank NIFTY / banking-sector news, and about \
what the system's own scores and fields mean (CARS, D_info, composite_score, \
sentiment_signed, index_weight, TCN, Isolation Forest), using ONLY the \
context below.

Explain things in plain, clear English suitable for a non-technical \
evaluator. If the context doesn't contain the answer, say so honestly \
instead of guessing -- do not invent numbers, headlines, or company names.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

_llm = None
_chain_cache = {"vectordb_id": None, "chain": None}


def _get_llm_cached():
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


def _get_chain(vectordb):
    # Rebuild the RetrievalQA wrapper only if the underlying vector store
    # instance actually changed (i.e. the news snapshot was refreshed).
    if _chain_cache["chain"] is None or _chain_cache["vectordb_id"] != id(vectordb):
        retriever = vectordb.as_retriever(search_kwargs={"k": config.RETRIEVER_TOP_K})
        prompt = PromptTemplate(
            template=PROMPT_TEMPLATE, input_variables=["context", "question"]
        )
        chain = RetrievalQA.from_chain_type(
            llm=_get_llm_cached(),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        _chain_cache["chain"] = chain
        _chain_cache["vectordb_id"] = id(vectordb)

    return _chain_cache["chain"]


def ask(question: str) -> dict:
    """Returns {'answer': str, 'sources': [str, ...], 'meta': {...}}"""
    vectordb, meta = get_or_build()
    chain = _get_chain(vectordb)
    result = chain.invoke({"query": question})

    sources = []
    for doc in result.get("source_documents", []):
        if doc.metadata.get("type") == "news":
            sources.append(
                f"{doc.metadata.get('company', '?')} "
                f"(score {doc.metadata.get('composite_score', '?')}, "
                f"{doc.metadata.get('published_at', '?')})"
            )

    return {
        "answer": result.get("result", ""),
        "sources": sources,
        "meta": meta,
    }

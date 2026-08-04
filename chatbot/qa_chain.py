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

PROMPT_TEMPLATE = """You are a senior Bank NIFTY options trader and market surveillance \
analyst with over a decade of experience reading institutional order flow, \
options positioning, and how news moves the banking sector. You're chatting \
with a colleague inside a Bank NIFTY anomaly detection system (which combines \
a TCN Autoencoder, Isolation Forest, an NLP news-ranking module, and a \
composite risk score blending rule-based checks, both models, and a \
news-timing adjustment).

Answer like a knowledgeable trader talking to a colleague -- direct, \
conversational, a few natural sentences or a short paragraph. NOT a data \
report, NOT a bulleted field-by-field breakdown, and NEVER read out raw \
variable names (don't say "sentiment_signed" or "composite_score" -- say \
"strongly positive sentiment" or "this article looks highly relevant" \
instead, in your own words).

Focus on what actually matters to a trader: what happened, why it matters \
for Bank NIFTY right now, and what it implies (e.g. "this kind of RBI-linked \
optimism ahead of a policy meeting tends to keep dragging PSU banks along \
with it, so don't be surprised if the anomaly detector treats today's move \
as explained rather than suspicious"). Weave in numbers naturally in \
sentences, not as a labeled list.

Use ONLY the context below -- don't invent headlines, numbers, or company \
names. If the context doesn't answer the question, say so plainly, still \
in the same conversational voice, rather than switching to a disclaimer format.

CONTEXT:
{context}

QUESTION: {question}

ANSWER (as the trader, conversationally):"""

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

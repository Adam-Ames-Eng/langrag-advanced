"""
A FastAPI service that answers questions about the sample Nimbus docs,
grounded in retrieval — the same RAG pattern as ScopeSense, in the
Python/LangChain stack most Upwork AI-agent job posts ask for.

What changed vs. the original demo:
  - Hybrid retrieval: keyword search (BM25) + vector search, merged
    with an EnsembleRetriever, so exact names/codes match even when
    the embedding model misses them semantically.
  - Optional cross-encoder reranking of the merged candidates, on by
    default, one env var to turn off (RERANK_ENABLED=false) if a
    machine can't afford the extra model.
  - Multi-turn conversation: the client can send prior turns; a
    "condense" step rewrites a follow-up question ("what about the
    Team plan?") into a standalone question before retrieval runs.
  - A streaming endpoint (/ask/stream) for token-by-token output.
  - Config pulled from environment variables instead of hardcoded
    values, so it's deployable without editing source.

Run:
    python ingest.py            # once, to build the vector store
    uvicorn app:app --reload

Then:
    curl -X POST http://localhost:8000/ask \
        -H "Content-Type: application/json" \
        -d '{"question": "How many automations can I have on the Team plan?"}'
"""

import logging
import os
import pathlib
from typing import Iterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("langrag-demo")

# ---------------------------------------------------------------------------
# Config (all overridable via environment variables)
# ---------------------------------------------------------------------------
PERSIST_DIR = pathlib.Path(
    os.getenv("CHROMA_DIR", str(pathlib.Path(__file__).parent / "chroma_db"))
)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:7b-instruct")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

VECTOR_K = int(os.getenv("VECTOR_K", "6"))       # candidates pulled from vector search
BM25_K = int(os.getenv("BM25_K", "6"))           # candidates pulled from keyword search
FINAL_K = int(os.getenv("FINAL_K", "3"))         # how many go into the LLM's context

RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

SYSTEM_PROMPT = """You answer questions about Nimbus using ONLY the context below.
If the answer isn't in the context, say so plainly — never guess or use
outside knowledge. Be concise. When you use a fact from the context,
you don't need to cite it inline; sources are returned separately.

Context:
{context}"""

CONDENSE_PROMPT = """Given the conversation so far and a new follow-up question,
rewrite the follow-up as a standalone question that makes sense without the
conversation. If the follow-up is already standalone, return it unchanged.
Return ONLY the rewritten question, nothing else.

Conversation so far:
{chat_history}

Follow-up question: {question}
Standalone question:"""

app = FastAPI(title="Nimbus Docs RAG Demo")

# ---------------------------------------------------------------------------
# Models & retrievers — built once at startup, not per-request
# ---------------------------------------------------------------------------
_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
_vectorstore = Chroma(
    persist_directory=str(PERSIST_DIR),
    embedding_function=_embeddings,
    collection_name="nimbus_docs",
)
_vector_retriever = _vectorstore.as_retriever(search_kwargs={"k": VECTOR_K})

# BM25 needs the raw documents up front (it isn't a persisted index like
# Chroma), so we pull everything currently in the collection once at
# startup. Fine for a demo-sized corpus; for a large corpus you'd persist
# a BM25 index separately instead of rebuilding it from Chroma's dump.
_all_docs_raw = _vectorstore.get(include=["documents", "metadatas"])
_all_documents = [
    Document(page_content=text, metadata=meta or {})
    for text, meta in zip(_all_docs_raw["documents"], _all_docs_raw["metadatas"])
]

if _all_documents:
    _bm25_retriever = BM25Retriever.from_documents(_all_documents)
    _bm25_retriever.k = BM25_K
    _retriever = EnsembleRetriever(
        retrievers=[_bm25_retriever, _vector_retriever],
        weights=[0.4, 0.6],
    )
else:
    # Empty store (e.g. ingest.py hasn't run yet) — fall back to vector-only
    # so the app still boots instead of crashing on an empty BM25 index.
    logger.warning("Vector store is empty — run ingest.py first. Falling back to vector-only retrieval.")
    _retriever = _vector_retriever

_reranker = None
if RERANK_ENABLED:
    try:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANK_MODEL)
        logger.info("Reranker loaded: %s", RERANK_MODEL)
    except Exception as exc:  # missing package, no internet to download weights, etc.
        logger.warning("Reranking disabled — could not load %s (%s)", RERANK_MODEL, exc)
        _reranker = None

_llm = ChatOllama(model=CHAT_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)

_prompt = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)
_generation_chain = _prompt | _llm | StrOutputParser()

_condense_prompt = ChatPromptTemplate.from_template(CONDENSE_PROMPT)
_condense_chain = _condense_prompt | _llm | StrOutputParser()


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------
def _rerank(question: str, docs: list[Document], top_n: int) -> list[Document]:
    """Re-score candidate docs against the question with a cross-encoder and
    keep the top_n. Falls back to the original order if no reranker loaded."""
    if not _reranker or not docs:
        return docs[:top_n]
    pairs = [(question, doc.page_content) for doc in docs]
    scores = _reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda pair: pair[0], reverse=True)
    return [doc for _, doc in ranked[:top_n]]


def _format_history(chat_history: list["ChatTurn"]) -> str:
    if not chat_history:
        return "(no previous turns)"
    return "\n".join(f"User: {turn.question}\nAssistant: {turn.answer}" for turn in chat_history)


def _resolve_question(question: str, chat_history: list["ChatTurn"]) -> str:
    """Rewrite a follow-up question into a standalone one when there's history."""
    if not chat_history:
        return question
    return _condense_chain.invoke(
        {"chat_history": _format_history(chat_history), "question": question}
    ).strip()


def _retrieve_and_format(standalone_question: str) -> tuple[str, list[str]]:
    retrieved_docs = _retriever.invoke(standalone_question)
    retrieved_docs = _rerank(standalone_question, retrieved_docs, FINAL_K)
    context = "\n\n".join(
        f"[{doc.metadata.get('source', 'unknown')}]: {doc.page_content}"
        for doc in retrieved_docs
    )
    sources = sorted({doc.metadata.get("source", "unknown") for doc in retrieved_docs})
    return context, sources


# ---------------------------------------------------------------------------
# API schema
# ---------------------------------------------------------------------------
class ChatTurn(BaseModel):
    question: str
    answer: str


class AskRequest(BaseModel):
    question: str
    chat_history: list[ChatTurn] = []


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    standalone_question: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "documents_indexed": len(_all_documents),
        "reranking": _reranker is not None,
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    standalone_question = _resolve_question(req.question, req.chat_history)
    context, sources = _retrieve_and_format(standalone_question)
    answer = _generation_chain.invoke({"context": context, "question": standalone_question})
    return AskResponse(answer=answer, sources=sources, standalone_question=standalone_question)


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    standalone_question = _resolve_question(req.question, req.chat_history)
    context, sources = _retrieve_and_format(standalone_question)

    def token_stream() -> Iterator[str]:
        for chunk in _generation_chain.stream({"context": context, "question": standalone_question}):
            yield chunk
        # Sources arrive as a final line the client can split off, so the
        # streamed text itself never mixes citation bookkeeping into prose.
        yield f"\n\n[sources: {', '.join(sources)}]"

    return StreamingResponse(token_stream(), media_type="text/plain")

"""
Builds (or rebuilds) the Chroma vector store from files in data/.

What changed vs. the original demo:
  - Loads .txt/.md, .pdf, and .html files (dispatched by extension)
    instead of assuming plain text — so dropping a client's real docs
    into data/ mostly just works, per the README's "adapt this for a
    real contract" note.
  - Chunk size/overlap and the data directory are env-configurable.
  - Rebuilds the collection from scratch each run (simplest correct
    behavior for a demo-sized corpus) rather than silently duplicating
    chunks if you run it twice.

Run once (or again, any time data/ changes):
    python ingest.py
"""

import logging
import os
import pathlib

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    BSHTMLLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ingest")

DATA_DIR = pathlib.Path(os.getenv("DATA_DIR", "data"))
PERSIST_DIR = pathlib.Path(os.getenv("CHROMA_DIR", "chroma_db"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

LOADERS_BY_SUFFIX = {
    ".txt": TextLoader,
    ".md": TextLoader,
    ".pdf": PyPDFLoader,
    ".html": BSHTMLLoader,
    ".htm": BSHTMLLoader,
}


def load_documents(data_dir: pathlib.Path) -> list:
    documents = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        loader_cls = LOADERS_BY_SUFFIX.get(path.suffix.lower())
        if loader_cls is None:
            logger.info("Skipping unsupported file type: %s", path.name)
            continue
        loader = loader_cls(str(path)) if loader_cls is not TextLoader else loader_cls(str(path), encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            # Normalize to a plain filename so app.py's citations stay
            # readable regardless of which loader produced the metadata.
            doc.metadata["source"] = path.name
        documents.extend(docs)
        logger.info("Loaded %s (%d section(s))", path.name, len(docs))
    return documents


def main() -> None:
    if not DATA_DIR.exists():
        raise SystemExit(f"Data directory not found: {DATA_DIR}")

    raw_documents = load_documents(DATA_DIR)
    if not raw_documents:
        raise SystemExit(f"No supported documents found in {DATA_DIR}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(raw_documents)
    logger.info("Split %d document(s) into %d chunk(s)", len(raw_documents), len(chunks))

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    # Rebuild from scratch so re-running ingest.py after editing data/
    # doesn't leave stale or duplicate chunks behind.
    if PERSIST_DIR.exists():
        import shutil

        shutil.rmtree(PERSIST_DIR)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
        collection_name="nimbus_docs",
    )
    logger.info("Vector store built at %s", PERSIST_DIR)


if __name__ == "__main__":
    main()

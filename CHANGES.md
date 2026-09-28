# What's in this update

I could only fetch `app.py` and `README.md` from the repo directly — GitHub
blocked re-fetching `ingest.py`, `query.py`, `test_app.py`, and
`app_streamlit.py` on a second request. So those four files here are
**complete rewrites** consistent with `app.py`'s behavior and the README,
not edits of your originals. Please diff/review before overwriting,
especially `test_app.py` if you had assertions tied to the old single-shot
`/ask` response shape (it's unchanged, just extended with
`chat_history` and `standalone_question`).

`app_streamlit.py` has now been updated too, from the version you shared
directly (so this one *is* a real edit, not a guess). Note it's the
Gemini-based variant — a separate, independent demo from the Ollama-based
`app.py`/`ingest.py` pair, with its own in-memory vector store built from
`data/*.md` on each cold start.

## Changed / added files
- `app.py` — hybrid retrieval (BM25 + vector via `EnsembleRetriever`),
  optional cross-encoder reranking, multi-turn `chat_history` with
  question condensing, `/ask/stream` for token streaming, env-based config.
- `ingest.py` — loads `.txt`/`.md`/`.pdf`/`.html`, env-configurable
  chunk size/overlap, rebuilds the store cleanly instead of duplicating
  chunks on re-run.
- `query.py` — terminal chat loop, now keeps history across turns.
- `eval.py` + `eval_questions.json` — retrieval hit-rate check against a
  small labeled question set (edit the JSON to match your `data/` files).
- `test_app.py` — mocks the LLM/retriever so tests run without Ollama.
- `requirements.txt` — adds `rank_bm25`, `sentence-transformers`, `pypdf`,
  `beautifulsoup4`, `pytest`, `httpx`.
- `Dockerfile` — packages the FastAPI service (Ollama still runs
  separately — see comments in the file).
- `.github/workflows/ci.yml` — runs `pytest` on every push/PR.
- `app_streamlit.py` — same hybrid retrieval + optional reranking (a
  sidebar checkbox now, since a Gemini API key is already user-supplied
  there) + multi-turn question condensing + streamed answers via
  `st.write_stream`, and now loads `.pdf`/`.html` files too, not just `.md`.

## Before running
1. `pip install -r requirements.txt`
2. `python ingest.py` (rebuilds the vector store with the new loaders)
3. `pytest -v` (should pass without Ollama running, since the model calls
   are mocked)
4. `uvicorn app:app --reload`, then try `/ask`, `/ask/stream`, and sending
   a second request with `chat_history` filled in.

## If reranking is more weight than you want
`sentence-transformers` downloads a small model on first run. If that's
unwanted on a given machine, set `RERANK_ENABLED=false` — `app.py` falls
back to the ensemble retriever's order with no code changes needed.

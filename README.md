# langrag-advanced

> 🚀 **Live Interactive Demo:** [Try the deployed app here](https://langrag-advanced.streamlit.app/)

A grounded RAG (Retrieval-Augmented Generation) demo that answers questions
about a sample SaaS product ("Nimbus") using only its own documentation —
never guessing, always citing sources. Built to show the retrieval and
generation patterns that show up in real client work: hybrid search,
reranking, multi-turn conversation, streaming, and automated evaluation.

Two independent, fully working variants are included:

| | Backend | Setup cost | Good for |
|---|---|---|---|
| `app.py` + `ingest.py` | Local models via [Ollama](https://ollama.com) | Install Ollama, pull two models | Fully local/offline, no API key, no per-request cost |
| `app_streamlit.py` | [Google Gemini](https://aistudio.google.com/) | Just a free API key | Fastest to demo, no local model downloads |

Both share the same RAG techniques below — pick whichever fits the client's
constraints.

## What this demonstrates

- **Hybrid retrieval** — keyword search (BM25) and vector search merged
  with an `EnsembleRetriever`, so exact names, plan tiers, and codes match
  even when pure semantic search misses them.
- **Reranking** — a small cross-encoder re-scores the merged candidates by
  relevance before they reach the model (toggleable — see below).
- **Multi-turn conversation** — a follow-up question ("and the Starter
  plan?") is rewritten into a standalone question using chat history
  before retrieval runs.
- **Streaming answers** — token-by-token output via FastAPI's
  `StreamingResponse` (`/ask/stream`) or Streamlit's `st.write_stream`.
- **Multi-format ingestion** — `.txt`, `.md`, `.pdf`, and `.html` files in
  `data/` are all picked up automatically.
- **Automated evaluation** — `eval.py` checks retrieval hit-rate against a
  small labeled question set, instead of eyeballing answers.
- **Grounded answers, honestly** — the system prompt requires the model to
  say when something isn't in the docs rather than guess.

## Project structure

```
app.py                  FastAPI backend (Ollama)
ingest.py               Builds the vector store from data/ (Ollama variant)
query.py                Terminal chat loop (Ollama variant)
app_streamlit.py        Streamlit UI + backend, all-in-one (Gemini variant)
eval.py                 Retrieval hit-rate evaluation
eval_questions.json     Labeled question set used by eval.py
test_app.py             Unit tests (LLM/retriever mocked — no Ollama needed)
data/                   Sample Nimbus docs (pricing.pdf, security-and-sso.pdf,
                         automations-and-integrations.pdf)
requirements.txt
Dockerfile              Packages the FastAPI service (Ollama variant)
.github/workflows/ci.yml
```

## Setup — Ollama variant (`app.py`)

1. Install [Ollama](https://ollama.com) and pull the two models used here:
   ```
   ollama pull bge-m3
   ollama pull qwen2.5:7b-instruct
   ollama serve
   ```
2. Create a virtual environment and install dependencies:
   ```
   python -m venv .venv
   source .venv/bin/activate   # .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
3. Build the vector store from `data/`:
   ```
   python ingest.py
   ```
4. Run the API:
   ```
   uvicorn app:app --reload
   ```
   Then `POST /ask` with `{"question": "..."}`, or `POST /ask/stream` for a
   streamed response. `chat_history` is optional on both — pass prior
   `{"question": ..., "answer": ...}` turns for follow-up questions.

   Or use the terminal instead of the API:
   ```
   python query.py
   ```

## Setup — Gemini/Streamlit variant (`app_streamlit.py`)

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/).
2. Same virtual environment / `pip install -r requirements.txt` as above.
3. Run:
   ```
   streamlit run app_streamlit.py
   ```
4. Paste the API key into the sidebar. Reranking can be toggled there too.

## Evaluation

```
python eval.py
```
Runs each question in `eval_questions.json` through the same retrieval path
as the app and reports whether the expected source document came back.
Replace the sample questions with ones matched to your own `data/` files.

## Tests

```
pytest -v
```
The LLM and retriever calls are mocked, so this runs without Ollama or a
populated vector store — safe for CI (see `.github/workflows/ci.yml`).

## Configuration

Both variants read config from environment variables instead of hardcoded
values — e.g. `OLLAMA_BASE_URL`, `CHAT_MODEL`, `EMBEDDING_MODEL`, `FINAL_K`,
`RERANK_ENABLED` for `app.py`/`ingest.py`. Reasonable defaults are set if
you don't override them.

## Docker

```
docker build -t langrag-demo-advanced .
docker run -p 8000:8000 langrag-demo-advanced
```
Packages the FastAPI service only — Ollama runs separately (see comments in
`Dockerfile` for pointing the container at a host-side Ollama instance).

## Known limitations

- `ingest.py` rebuilds the vector store from scratch on every run rather
  than incrementally updating it — fine for demo-sized corpora, worth
  changing for large ones.
- Conversation history is used to rewrite follow-up questions, but isn't
  fed into the final answer generation step itself — so the model can't
  reference exactly what it said earlier, only what the documents say.
- The Streamlit variant rebuilds its in-memory vector store on cold start
  (cached per session), rather than persisting it like the Ollama variant.

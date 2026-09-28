# Packages the FastAPI service only. Ollama is a separate process — run it
# on the host (or as its own container) and point OLLAMA_BASE_URL at it,
# e.g. http://host.docker.internal:11434 on Docker Desktop.

FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OLLAMA_BASE_URL=http://host.docker.internal:11434
ENV CHROMA_DIR=/app/chroma_db

EXPOSE 8000

# The vector store (chroma_db/) is expected to already exist — build it
# with `python ingest.py` before building the image, or mount it as a
# volume, rather than baking network calls to Ollama into the image build.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

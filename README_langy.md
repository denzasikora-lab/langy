# Langy

Langy is a small RAG playground that grows in three steps:

- `rag_simple.py` - simple RAG with Chroma, JSON parsing, no chunk overlap, no LangSmith.
- `rag_middle.py` - planned FAISS version with chunk overlap, Pydantic outputs, and LangSmith.
- `rag_senior.py` - planned Qdrant version with a more production-oriented pipeline.

## Current Status

Only `rag_simple.py` is implemented now. The middle and senior files are intentionally skeletons.

## Project Layout

```text
langy/
  main.py
  rag_simple.py
  rag_middle.py
  rag_senior.py
  README_langy.md
  requirements.txt
  .env.example
```

## Setup

Install the latest package versions:

```bash
python3 -m pip install -U -r requirements.txt
```

Create `.env` from `.env.example` and set:

```text
OLLAMA_HOST=https://ollama.com
OLLAMA_API_KEY=your_ollama_api_key
OLLAMA_MODEL=gemma4:31b-cloud
USER_AGENT=langy-rag-simple/0.1
```

## Run

```bash
python3 main.py "What are the types of agent memory?"
```

Rebuild the local Chroma index:

```bash
python3 main.py "What are the types of agent memory?" --rebuild-index
```

## How `rag_simple.py` Works

1. Downloads a few source articles.
2. Extracts clean text with BeautifulSoup.
3. Splits text into chunks without overlap.
4. Embeds chunks with `sentence-transformers/all-MiniLM-L6-v2`.
5. Stores vectors in local Chroma under `.chroma/rag_simple`.
6. Embeds the user question.
7. Compares the question vector with document vectors.
8. Sends the best chunks to Ollama Cloud.
9. Expects JSON back from the model.


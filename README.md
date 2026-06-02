# Langy

Langy is a small RAG playground that grows in three steps:

- `rag_junior.py` - junior RAG with Chroma, JSON parsing, no chunk overlap, no LangSmith.
- `rag_middle.py` - planned FAISS version with chunk overlap, Pydantic outputs, and LangSmith.
- `rag_senior.py` - planned Qdrant version with a more production-oriented pipeline.

## Current Status

Only `rag_junior.py` is implemented now. The middle and senior files are intentionally skeletons.

## Project Layout

```text
langy/
  main.py
  rag_junior.py
  rag_middle.py
  rag_senior.py
  README.md
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
USER_AGENT=langy-rag-junior/0.1
```

## Run

`main.py` selects the RAG level by number:

- `1` - junior, Chroma, JSON, no overlap
- `2` - middle, planned FAISS + Pydantic + LangSmith
- `3` - senior, planned Qdrant

For now the default is level `1`:

```bash
python3 main.py
```

To switch levels, call `run_langy(1)`, `run_langy(2)`, or `run_langy(3)`.
The script entrypoint currently uses `DEFAULT_LEVEL`:

```python
DEFAULT_LEVEL = 1
```

## How `rag_junior.py` Works

1. Downloads a few source articles.
2. Extracts clean text with BeautifulSoup.
3. Splits text into chunks without overlap.
4. Embeds chunks with `sentence-transformers/all-MiniLM-L6-v2`.
5. Stores vectors in local Chroma under `.chroma/rag_junior`.
6. Embeds the user question.
7. Compares the question vector with document vectors.
8. Sends the best chunks to Ollama Cloud.
9. Expects JSON back from the model.

# Langy

Langy is a small RAG playground that grows in three steps:

- `rag_junior.py` - level `1`, InMemoryVectorStore, JSON parsing, no chunk overlap, and `init_chat_model(..., model_provider="ollama")`.
- `rag_middle.py` - level `21`, Chroma + MMR; level `22`, FAISS + MMR. Both use LangGraph, overlap, Pydantic outputs, LangSmith, and OpenAI via `init_chat_model`.
- `rag_senior.py` - level `31`, planned Qdrant; level `32`, planned Weaviate. Both will use embedding ensembling.

## Current Status

`rag_junior.py` and `rag_middle.py` are implemented now. The senior file is still a skeleton.

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
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini
LANGSMITH_API_KEY=your_langsmith_api_key
LANGY_MIDDLE_LANGSMITH_PROJECT=langy-rag-middle
USER_AGENT=langy-rag-junior/0.1
```

## Run

`main.py` selects the RAG level by number:

- `1` - junior, InMemoryVectorStore, JSON, no overlap
- `21` - middle, Chroma + MMR, `k=3`, `fetch_k=10`
- `22` - middle, FAISS + MMR, `k=3`, `fetch_k=10`
- `31` - senior, planned Qdrant
- `32` - senior, planned Weaviate

For now the default is level `1`:

```bash
python3 main.py
```

To switch levels, call `run_langy(1)`, `run_langy(21)`, `run_langy(22)`, `run_langy(31)`, or `run_langy(32)`.
The script entrypoint currently uses `DEFAULT_LEVEL`:

```python
DEFAULT_LEVEL = 1
```

## How `rag_junior.py` Works

1. Downloads a few source articles.
2. Extracts clean text with BeautifulSoup.
3. Splits text into chunks without overlap.
4. Embeds chunks with `sentence-transformers/all-MiniLM-L6-v2`.
5. Stores vectors in memory only.
6. Embeds the user question.
7. Compares the question vector with document vectors.
8. Sends the best chunks to Ollama Cloud.
9. Expects JSON back from the model.

## How `rag_middle.py` Works

1. Downloads the same source articles.
2. Splits text into chunks with overlap.
3. Embeds chunks with `BAAI/bge-m3`.
4. Uses MMR, which is a search mode, not a metric: it fetches 10 candidates and selects 3 diverse chunks.
5. Level `21` persists vectors in Chroma.
6. Level `22` persists vectors in FAISS.
7. Uses LangGraph to orchestrate retrieval and answer generation.
8. Uses `with_structured_output(...)` with a Pydantic schema.
9. Calls OpenAI models through `init_chat_model(...)`.
10. Sends traces to LangSmith when `LANGSMITH_API_KEY` is configured.

## Diagrams

Each level saves Mermaid source and PNG files under `diagrams/`.

# Langy

Langy is a small RAG playground that grows in three steps:

- `rag_junior.py` - level `1`, InMemoryVectorStore, default splitter settings, JSON parsing, and `init_chat_model(..., model_provider="ollama")`.
- `rag_middle.py` - level `21`, Chroma + MMR; level `22`, FAISS + MMR. Both use LangGraph, overlap, Pydantic outputs, LangSmith, and OpenAI via `init_chat_model`.
- `rag_senior.py` - level `31`, Qdrant-oriented async senior pipeline; level `32`, Weaviate-oriented async senior pipeline. Both use PII anonymization, HyDE, hybrid retrieval, rerankers, graders, fallback, and embedding ensembling.

## Current Status

`rag_junior.py`, `rag_middle.py`, and `rag_senior.py` are implemented now.

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

- `1` - junior, InMemoryVectorStore, JSON, default splitter settings
- `21` - middle, Chroma + MMR, `k=3`, `fetch_k=10`, `lambda_mult=0.5`
- `22` - middle, FAISS + MMR, `k=3`, `fetch_k=10`, `lambda_mult=0.5`
- `31` - senior, Qdrant-oriented async hybrid RAG
- `32` - senior, Weaviate-oriented async hybrid RAG

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
3. Splits text with the default LangChain splitter settings.
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
4. Uses MMR, which is a search mode, not a metric: it fetches 10 candidates and selects 3 diverse chunks with `lambda_mult=0.5`.
5. Level `21` persists vectors in Chroma.
6. Level `22` persists vectors in FAISS.
7. Uses LangGraph to orchestrate retrieval and answer generation.
8. Uses `with_structured_output(...)` with a Pydantic schema.
9. Calls OpenAI models through `init_chat_model(...)`.
10. Sends traces to LangSmith when `LANGSMITH_API_KEY` is configured.

## How `rag_senior.py` Works

1. Anonymizes PII with regex filters and a lightweight NER-like person detector.
2. Uses HyDE: the LLM writes a hypothetical document that should answer the question.
3. Builds ensemble embeddings from `BAAI/bge-m3` and `sentence-transformers/all-MiniLM-L6-v2`.
4. Retrieves with semantic cosine search `top_k=20` and BM25 keyword search `top_k=20`.
5. Merges semantic and BM25 candidates into a hybrid list.
6. Reranks to `top_k=5`.
7. Grades documents with yes/no relevance.
8. Falls back to web search if too few documents pass grading.
9. Builds a controlled context for a model with a context window greater than 64k.
10. Generates the answer, checks hallucinations, and optionally self-corrects.

## Diagrams

Each level saves PNG diagrams under `diagrams/`. Mermaid `.mmd` files are not generated for now.

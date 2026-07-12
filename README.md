# Langy

Langy is a small RAG playground that grows in four steps:

- `rag_junior.py` - level `1`, InMemoryVectorStore, default splitter settings, LLM-as-a-judge reranker, JSON parsing, and Ollama via `init_chat_model`.
- `rag_middle.py` - level `21`, Chroma + MMR; level `22`, FAISS + MMR. Both use LangGraph router, simple tool calling, vectorstore/web-search branches, rerank eval metrics, Pydantic outputs, LangSmith, and OpenAI via `init_chat_model`.
- `rag_senior.py` - level `31`, Weaviate-oriented async senior graph; level `32`, Qdrant-oriented async senior graph with optional PostgreSQL/pgvector sync. Both use optional MCP tools, PII anonymization, HyDE, hybrid retrieval, BGE reranking, graders, fallback, hallucination checks, loops, and embedding ensembling.
- `rag_staff.py` - level `41`, staff RAG over Qdrant with LightRAG-style graph retrieval, LlamaIndex-style hierarchical ingestion, tool-map routing, context expansion, BGE+Qwen reranking, and optional CatBoostRanker.

## Current Status

`rag_junior.py`, `rag_middle.py`, `rag_senior.py`, and `rag_staff.py` are implemented now.

## Project Layout

```text
langy/
  main.py
  rag_junior.py
  rag_middle.py
  rag_senior.py
  rag_staff.py
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
LANGY_MCP_CONNECTIONS_JSON=
QDRANT_URL=
QDRANT_API_KEY=
QDRANT_COLLECTION=langy_staff_41
POSTGRES_CONNECTION_STRING=
LANGY_PGVECTOR_COLLECTION=langy_senior_32
USER_AGENT=langy-rag-junior/0.1
```

## Run

`main.py` selects the RAG level by number:

- `1` - junior, InMemoryVectorStore, retrieve 20 candidates, rerank to 3, JSON
- `21` - middle, Chroma + MMR, `fetch_k=20`, `k=10`, rerank `top_k=3`, `lambda_mult=0.5`
- `22` - middle, FAISS + MMR, `fetch_k=20`, `k=10`, rerank `top_k=3`, `lambda_mult=0.5`
- `31` - senior, Weaviate-oriented async LangGraph RAG
- `32` - senior, Qdrant-oriented async LangGraph RAG with optional PostgreSQL/pgvector sync
- `41` - staff, Qdrant-oriented LightRAG/LlamaIndex-style graph RAG

For now the default is level `1`:

```bash
python3 main.py
```

To switch levels, call `run_langy(1)`, `run_langy(21)`, `run_langy(22)`, `run_langy(31)`, `run_langy(32)`, or `run_langy(41)`.
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
8. Uses LLM-as-a-judge to rerank the retrieved chunks.
9. Sends the best chunks to Ollama Cloud.
10. Expects JSON back from the model.

## How `rag_middle.py` Works

1. Downloads the same source articles.
2. Splits text into chunks with overlap.
3. Embeds chunks with `BAAI/bge-m3`.
4. Uses an LLM tool-calling router to choose `vectorstore_search` or `web_search`.
5. Level `21` persists vectors in Chroma.
6. Level `22` persists vectors in FAISS.
7. Uses MMR for vectorstore retrieval: fetches 20 candidates and selects 10 diverse candidates with `lambda_mult=0.5`.
8. Reranks candidates using retrieval-vs-lexical weights selected against a tiny eval set, keeping the best 3.
9. Reports `Recall@K`, `MRR`, `MAP`, and `nDCG` when the question is in the eval set.
10. Uses `with_structured_output(...)` with a Pydantic schema.
11. Calls OpenAI models through `init_chat_model(...)`.
12. Sends traces to LangSmith when `LANGSMITH_API_KEY` is configured.

## How `rag_senior.py` Works

1. Boots optional MCP tools from `LANGY_MCP_CONNECTIONS_JSON`.
2. Anonymizes PII with regex filters and a lightweight NER-like person detector.
3. Uses HyDE: the LLM writes a hypothetical document that should answer the question.
4. Builds ensemble embeddings from `BAAI/bge-m3` and `sentence-transformers/all-MiniLM-L6-v2`.
5. Retrieves with semantic cosine search `top_k=20` and BM25 keyword search `top_k=20`.
6. Merges semantic and BM25 candidates into a hybrid list.
7. Reranks to `top_k=3` with `BAAI/bge-reranker-base`.
8. Grades documents with yes/no relevance.
9. Falls back to web search if too few documents pass grading.
10. Builds a controlled context for a model with a context window greater than 64k.
11. Generates the answer and checks hallucinations.
12. If the answer is not grounded, self-corrects and loops back to hallucination grading.

## How `rag_staff.py` Works

1. Routes the request through a structured Pydantic tool-map: vectorstore, web, summarise, translate, human pause, or txt2sql.
2. Uses HyDE before retrieval so semantic search runs against a cleaner hypothetical target document.
3. Loads html/json/markdown/code/pdf documents.
4. Builds parent-child hierarchical chunks: large parent chunks preserve context; small child chunks improve retrieval precision.
5. Vectorizes chunk text together with metadata labels and source type.
6. Builds a LightRAG-style entity graph over chunks.
7. Uploads vectors to Qdrant when `QDRANT_URL` is configured, otherwise uses the local fallback index.
8. Retrieves with cosine top_k=20 and BM25 top_k=20, then hybrid-merges candidates.
9. Expands context through neighboring chunks only when cosine continuity is high enough.
10. Repairs broken next/previous navigation by checking nearby chunks in the same parent.
11. Reranks with BGE and Qwen3 rerankers, merges both rankings, and keeps the best 3.
12. Uses CatBoostRanker only when enough labeled ranking data exists in `staff_ltr_training.json`.

## Diagrams

Each level saves PNG diagrams under `diagrams/`. Mermaid `.mmd` files are not generated for now.

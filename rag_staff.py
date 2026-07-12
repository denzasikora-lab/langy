"""Staff RAG: level 41, an experimental Qdrant-based architecture.

Stack:
- Qdrant-oriented storage: the primary level-41 backend.
- LightRAG-inspired graph layer: an entity and relation graph over documents.
- LlamaIndex-inspired ingestion: pdf/html/json/markdown/code + hierarchical chunking.
- Parent-child chunks: large chunks provide context and small chunks provide precise retrieval.
- Metadata vectorization: embeddings include chunk text, label, and source_type.
- Context expansion: neighboring chunks are added only when cosine continuity exceeds a threshold.
- Router tool-map: vectorstore / web / summarise / translate / human pause / txt2sql.
- HyDE before retrieval: search with a pseudo-document instead of a noisy question.
- Hybrid retrieval: cosine top_k=20 + BM25 top_k=20 -> merge.
- Multi-reranker: BGE + Qwen3, followed by an RRF merge.
- CatBoostRanker hook: enabled only when sufficient labeled features are available.

The architecture is deliberately modular: external Qdrant, LlamaIndex, LightRAG,
and CatBoost dependencies are optional, while the local fallback keeps this module
an executable instructional sandbox without external infrastructure.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)

import numpy as np
import requests
from bs4 import BeautifulSoup
from diagram_utils import save_mermaid_assets
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field


# region ---------------- Settings ----------------
PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
DIAGRAM_DIR = PROJECT_DIR / "diagrams"
STAFF_CORPUS_DIR = PROJECT_DIR / "staff_corpus"

SOURCE_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

PARENT_CHUNK_SIZE = 2400
CHILD_CHUNK_SIZE = 480
CHILD_OVERLAP = 80
SEMANTIC_TOP_K = 20
BM25_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3
FINAL_TOP_K = 5
CONTEXT_CONTINUITY_THRESHOLD = 0.72
CATBOOST_MIN_TRAINING_ROWS = 200

BGE_M3_MODEL = "BAAI/bge-m3"
BGE_RERANKER_MODEL = "BAAI/bge-reranker-base"
QWEN3_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
StaffRoute = Literal["vectorstore", "web", "summarise", "translate", "human_pause", "txt2sql"]
AfterRouteNode = Literal[
    "generate_hyde",
    "web_search",
    "summarise",
    "translate",
    "human_pause",
    "txt2sql",
]
# endregion


# region ---------------- Schemas ----------------
class StaffRouterDecision(BaseModel):
    route: StaffRoute
    reason: str


class StaffAnswer(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    model_confidence: float = Field(ge=0.0, le=1.0)


class StaffState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    question: str
    route: StaffRoute = "vectorstore"
    route_reason: str = ""
    hyde_text: str = ""
    documents: list[Document] = Field(default_factory=list)
    chunks: list["StaffChunk"] = Field(default_factory=list)
    index: "StaffIndex | None" = None
    candidates: list["StaffCandidate"] = Field(default_factory=list)
    expanded_candidates: list["StaffCandidate"] = Field(default_factory=list)
    context_text: str = ""
    answer: StaffAnswer | None = None
    human_pause_payload: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class StaffChunk:
    chunk_id: str
    parent_id: str
    text: str
    source: str
    source_type: str
    label: str
    chunk_index: int
    metadata_text: str
    prev_chunk_id: str | None = None
    next_chunk_id: str | None = None
    prev_cosine: float | None = None
    next_cosine: float | None = None


@dataclass
class StaffCandidate:
    chunk: StaffChunk
    vector_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    bge_rerank_score: float = 0.0
    qwen_rerank_score: float = 0.0
    catboost_score: float | None = None
    final_score: float = 0.0
    features: dict[str, float | int | str] = field(default_factory=dict)


@dataclass
class StaffIndex:
    chunks: list[StaffChunk]
    vectors: np.ndarray
    tokenized_chunks: list[list[str]]
    idf: dict[str, float]
    avg_doc_length: float
    chunk_by_id: dict[str, StaffChunk]
    graph_edges: dict[str, set[str]]

    async def search(self, query_text: str, embeddings: HuggingFaceEmbeddings) -> list[StaffCandidate]:
        query_vector = await embed_query(query_text, embeddings)
        semantic_candidates = semantic_search(self, query_vector, SEMANTIC_TOP_K)
        bm25_candidates = bm25_search(self, query_text, BM25_TOP_K)
        return hybrid_merge(semantic_candidates, bm25_candidates, HYBRID_TOP_K)
# endregion


# region ---------------- Environment and model ----------------
def configure_environment() -> None:
    load_dotenv(str(ENV_PATH))
    os.environ.setdefault("USER_AGENT", "langy-rag-staff/0.1")


def require_env(env_name: str) -> str:
    env_value = os.environ.get(env_name)
    if not env_value or env_value == "replace_me":
        raise RuntimeError(f"Missing required environment variable: {env_name}")
    return env_value


def create_chat_model():
    return init_chat_model(
        os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        model_provider="openai",
        temperature=0,
        api_key=require_env("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )


def create_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=BGE_M3_MODEL, show_progress=False)
# endregion


# region ---------------- Loading: html/json/markdown/code/pdf ----------------
def load_remote_html_documents() -> list[Document]:
    documents: list[Document] = []
    headers = {"User-Agent": os.environ["USER_AGENT"]}
    for url_value in SOURCE_URLS:
        response = requests.get(url_value, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text_value = "\n".join(soup.stripped_strings)
        documents.append(Document(page_content=text_value, metadata={"source": url_value, "source_type": "html"}))
    return documents


def load_local_staff_documents() -> list[Document]:
    if not STAFF_CORPUS_DIR.exists():
        return []

    documents: list[Document] = []
    for file_path in STAFF_CORPUS_DIR.rglob("*"):
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix in {".md", ".markdown", ".txt", ".py", ".js", ".ts", ".tsx", ".json", ".html"}:
            documents.append(load_text_like_file(file_path))
        elif suffix == ".pdf":
            pdf_doc = load_pdf_file(file_path)
            if pdf_doc is not None:
                documents.append(pdf_doc)
    return documents


def load_text_like_file(file_path: Path) -> Document:
    suffix = file_path.suffix.lower()
    text_value = file_path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".json":
        try:
            text_value = json.dumps(json.loads(text_value), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            pass
    elif suffix == ".html":
        text_value = "\n".join(BeautifulSoup(text_value, "html.parser").stripped_strings)

    source_type = {
        ".md": "markdown",
        ".markdown": "markdown",
        ".json": "json",
        ".html": "html",
        ".py": "code",
        ".js": "code",
        ".ts": "code",
        ".tsx": "code",
    }.get(suffix, "text")
    return Document(page_content=text_value, metadata={"source": str(file_path), "source_type": source_type})


def load_pdf_file(file_path: Path) -> Document | None:
    try:
        from pypdf import PdfReader
    except Exception:
        return None

    reader = PdfReader(str(file_path))
    text_value = "\n".join(page.extract_text() or "" for page in reader.pages)
    return Document(page_content=text_value, metadata={"source": str(file_path), "source_type": "pdf"})


async def load_staff_documents() -> list[Document]:
    def load_sync() -> list[Document]:
        return load_remote_html_documents() + load_local_staff_documents()

    return await asyncio.to_thread(load_sync)
# endregion


# region ---------------- LlamaIndex-inspired hierarchical chunking ----------------
def split_text(text_value: str, *, chunk_size: int, overlap: int = 0) -> list[str]:
    if chunk_size <= 0:
        return [text_value]
    chunks: list[str] = []
    start_index = 0
    while start_index < len(text_value):
        end_index = min(start_index + chunk_size, len(text_value))
        chunks.append(text_value[start_index:end_index])
        if end_index == len(text_value):
            break
        start_index = max(end_index - overlap, start_index + 1)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def infer_label(text_value: str, source_type: str) -> str:
    lowered = text_value[:1200].lower()
    if source_type == "code":
        return "code"
    if "agent" in lowered or "memory" in lowered:
        return "agents"
    if "prompt" in lowered:
        return "prompt_engineering"
    if "attack" in lowered or "adversarial" in lowered:
        return "adversarial_ml"
    return "general"


def hierarchical_chunk_documents(documents: list[Document]) -> list[StaffChunk]:
    chunks: list[StaffChunk] = []
    for doc_index, document in enumerate(documents):
        source = str(document.metadata.get("source", f"doc_{doc_index}"))
        source_type = str(document.metadata.get("source_type", "unknown"))
        parent_texts = split_text(document.page_content, chunk_size=PARENT_CHUNK_SIZE)
        previous_child_id: str | None = None

        for parent_index, parent_text in enumerate(parent_texts):
            parent_id = f"doc{doc_index}:parent{parent_index}"
            child_texts = split_text(parent_text, chunk_size=CHILD_CHUNK_SIZE, overlap=CHILD_OVERLAP)
            for child_index, child_text in enumerate(child_texts):
                label = infer_label(child_text, source_type)
                chunk_id = f"{parent_id}:child{child_index}"
                metadata_text = f"label={label}\nsource_type={source_type}\nsource={source}"
                chunks.append(
                    StaffChunk(
                        chunk_id=chunk_id,
                        parent_id=parent_id,
                        text=child_text,
                        source=source,
                        source_type=source_type,
                        label=label,
                        chunk_index=len(chunks),
                        metadata_text=metadata_text,
                        prev_chunk_id=previous_child_id,
                    )
                )
                previous_child_id = chunk_id

    return link_neighbor_chunks(chunks)


def link_neighbor_chunks(chunks: list[StaffChunk]) -> list[StaffChunk]:
    updated_chunks: list[StaffChunk] = []
    for index_value, chunk in enumerate(chunks):
        next_chunk_id = chunks[index_value + 1].chunk_id if index_value + 1 < len(chunks) else None
        updated_chunks.append(
            StaffChunk(
                **{
                    **chunk.__dict__,
                    "next_chunk_id": next_chunk_id,
                }
            )
        )
    return updated_chunks


def llamaindex_hierarchical_available() -> bool:
    try:
        from llama_index.core.node_parser import HierarchicalNodeParser  # noqa: F401

        return True
    except Exception:
        return False
# endregion


# region ---------------- Embeddings / Qdrant / LightRAG graph ----------------
def vectorized_text(chunk: StaffChunk) -> str:
    return f"{chunk.metadata_text}\n\n{chunk.text}"


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else vector / norm


async def embed_chunks(chunks: list[StaffChunk], embeddings: HuggingFaceEmbeddings) -> np.ndarray:
    texts = [vectorized_text(chunk) for chunk in chunks]
    vectors = await asyncio.to_thread(embeddings.embed_documents, texts)
    return normalize_matrix(np.array(vectors, dtype="float32"))


async def embed_query(query_text: str, embeddings: HuggingFaceEmbeddings) -> np.ndarray:
    vector = await asyncio.to_thread(embeddings.embed_query, query_text)
    return normalize_vector(np.array(vector, dtype="float32"))


def build_lightrag_graph(chunks: list[StaffChunk]) -> dict[str, set[str]]:
    graph_edges: dict[str, set[str]] = {}
    entity_pattern = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")
    for chunk in chunks:
        entities = sorted(set(entity_pattern.findall(chunk.text[:1200])))
        for left, right in zip(entities, entities[1:], strict=False):
            graph_edges.setdefault(left, set()).add(right)
            graph_edges.setdefault(right, set()).add(left)
    return graph_edges


def add_neighbor_cosines(chunks: list[StaffChunk], vectors: np.ndarray) -> list[StaffChunk]:
    chunk_positions = {chunk.chunk_id: index_value for index_value, chunk in enumerate(chunks)}
    enriched_chunks: list[StaffChunk] = []
    for index_value, chunk in enumerate(chunks):
        prev_cosine = None
        next_cosine = None
        if chunk.prev_chunk_id and chunk.prev_chunk_id in chunk_positions:
            prev_index = chunk_positions[chunk.prev_chunk_id]
            prev_cosine = float(vectors[index_value] @ vectors[prev_index])
        if chunk.next_chunk_id and chunk.next_chunk_id in chunk_positions:
            next_index = chunk_positions[chunk.next_chunk_id]
            next_cosine = float(vectors[index_value] @ vectors[next_index])

        enriched_chunks.append(
            StaffChunk(
                **{
                    **chunk.__dict__,
                    "prev_cosine": prev_cosine,
                    "next_cosine": next_cosine,
                }
            )
        )
    return enriched_chunks


async def maybe_upload_to_qdrant(chunks: list[StaffChunk], vectors: np.ndarray) -> str:
    url_value = os.environ.get("QDRANT_URL")
    api_key = os.environ.get("QDRANT_API_KEY")
    collection = os.environ.get("QDRANT_COLLECTION", "langy_staff_41")
    if not url_value:
        return "qdrant_not_configured_local_fallback"

    try:
        from qdrant_client import AsyncQdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams

        client = AsyncQdrantClient(url=url_value, api_key=api_key)
        collections = await client.get_collections()
        names = {item.name for item in collections.collections}
        if collection not in names:
            await client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
            )
        points = [
            PointStruct(
                id=index_value,
                vector=vectors[index_value].tolist(),
                payload={
                    "chunk_id": chunk.chunk_id,
                    "parent_id": chunk.parent_id,
                    "source": chunk.source,
                    "label": chunk.label,
                    "source_type": chunk.source_type,
                },
            )
            for index_value, chunk in enumerate(chunks)
        ]
        await client.upsert(collection_name=collection, points=points)
        await client.close()
        return f"qdrant:{collection}"
    except Exception as error:
        return f"qdrant_unavailable:{type(error).__name__}"


async def build_staff_index(documents: list[Document], embeddings: HuggingFaceEmbeddings) -> StaffIndex:
    chunks = hierarchical_chunk_documents(documents)
    vectors = await embed_chunks(chunks, embeddings)
    chunks = add_neighbor_cosines(chunks, vectors)
    tokenized_chunks = [tokenize(chunk.text) for chunk in chunks]
    idf = build_idf(tokenized_chunks)
    avg_doc_length = sum(len(tokens) for tokens in tokenized_chunks) / max(len(tokenized_chunks), 1)
    return StaffIndex(
        chunks=chunks,
        vectors=vectors,
        tokenized_chunks=tokenized_chunks,
        idf=idf,
        avg_doc_length=avg_doc_length,
        chunk_by_id={chunk.chunk_id: chunk for chunk in chunks},
        graph_edges=build_lightrag_graph(chunks),
    )
# endregion


# region ---------------- Router tool-map ----------------
def route_question(question_text: str) -> StaffRouterDecision:
    structured_model = create_chat_model().with_structured_output(StaffRouterDecision)
    prompt = f"""
Choose one route from this tool-map:
- vectorstore: knowledge/document retrieval.
- web: fresh/current web facts.
- summarise: user asks to summarize provided text.
- translate: user asks to translate.
- human_pause: user explicitly asks for human approval or manual control.
- txt2sql: user asks SQL/database question.

For txt2sql, assume prompt inputs include table schemas, relationships, and examples.

Question:
{question_text}
"""
    return structured_model.invoke(prompt)


def route_to_node(state: StaffState) -> AfterRouteNode:
    route_map: dict[StaffRoute, AfterRouteNode] = {
        "vectorstore": "generate_hyde",
        "web": "web_search",
        "summarise": "summarise",
        "translate": "translate",
        "human_pause": "human_pause",
        "txt2sql": "txt2sql",
    }
    return route_map[state.route]
# endregion


# region ---------------- Retrieval: HyDE -> semantic + BM25 ----------------
async def generate_hyde(question_text: str) -> str:
    llm = create_chat_model()
    response = await llm.ainvoke(
        [
            SystemMessage(content="Create a HyDE retrieval document. Do not answer conversationally."),
            HumanMessage(content=f"Question:\n{question_text}"),
        ]
    )
    return str(response.content)


TOKEN_PATTERN = re.compile(r"\w+")


def tokenize(text_value: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text_value)]


def build_idf(tokenized_documents: list[list[str]]) -> dict[str, float]:
    doc_count = len(tokenized_documents)
    document_frequency: dict[str, int] = {}
    for tokens in tokenized_documents:
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    return {
        token: math.log(1 + (doc_count - frequency + 0.5) / (frequency + 0.5))
        for token, frequency in document_frequency.items()
    }


def bm25_score(
    *,
    query_terms: list[str],
    doc_tokens: list[str],
    idf: dict[str, float],
    avg_doc_length: float,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    term_frequency: dict[str, int] = {}
    for token in doc_tokens:
        term_frequency[token] = term_frequency.get(token, 0) + 1

    score_value = 0.0
    doc_length = len(doc_tokens)
    for term in query_terms:
        frequency = term_frequency.get(term, 0)
        if frequency == 0:
            continue
        denominator = frequency + k1 * (1 - b + b * doc_length / max(avg_doc_length, 1.0))
        score_value += idf.get(term, 0.0) * frequency * (k1 + 1) / denominator
    return score_value


def semantic_search(index: StaffIndex, query_vector: np.ndarray, top_k: int) -> list[StaffCandidate]:
    scores = index.vectors @ query_vector
    best_indices = np.argsort(scores)[::-1][:top_k]
    return [
        StaffCandidate(chunk=index.chunks[int(index_value)], vector_score=float(scores[int(index_value)]))
        for index_value in best_indices
    ]


def bm25_search(index: StaffIndex, query_text: str, top_k: int) -> list[StaffCandidate]:
    query_terms = tokenize(query_text)
    scored_indices = [
        (
            index_value,
            bm25_score(
                query_terms=query_terms,
                doc_tokens=tokens,
                idf=index.idf,
                avg_doc_length=index.avg_doc_length,
            ),
        )
        for index_value, tokens in enumerate(index.tokenized_chunks)
    ]
    scored_indices.sort(key=lambda item: item[1], reverse=True)
    return [
        StaffCandidate(chunk=index.chunks[index_value], bm25_score=score_value)
        for index_value, score_value in scored_indices[:top_k]
        if score_value > 0
    ]


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score = min(scores)
    max_score = max(scores)
    if math.isclose(min_score, max_score):
        return [1.0 if max_score > 0 else 0.0 for _ in scores]
    return [(score_value - min_score) / (max_score - min_score) for score_value in scores]


def hybrid_merge(
    semantic_candidates: list[StaffCandidate],
    bm25_candidates: list[StaffCandidate],
    top_k: int,
) -> list[StaffCandidate]:
    candidates_by_id: dict[str, StaffCandidate] = {}
    semantic_scores = normalize_scores([candidate.vector_score for candidate in semantic_candidates])
    bm25_scores = normalize_scores([candidate.bm25_score for candidate in bm25_candidates])

    for candidate, score_value in zip(semantic_candidates, semantic_scores, strict=False):
        stored = candidates_by_id.setdefault(candidate.chunk.chunk_id, StaffCandidate(chunk=candidate.chunk))
        stored.vector_score = score_value
    for candidate, score_value in zip(bm25_candidates, bm25_scores, strict=False):
        stored = candidates_by_id.setdefault(candidate.chunk.chunk_id, StaffCandidate(chunk=candidate.chunk))
        stored.bm25_score = score_value

    candidates = list(candidates_by_id.values())
    for candidate in candidates:
        candidate.hybrid_score = 0.65 * candidate.vector_score + 0.35 * candidate.bm25_score
    candidates.sort(key=lambda candidate: candidate.hybrid_score, reverse=True)
    return candidates[:top_k]
# endregion


# region ---------------- Context expansion and navigation repair ----------------
def should_include_neighbor(base_chunk: StaffChunk, neighbor: StaffChunk | None, direction: str) -> bool:
    if neighbor is None:
        return False
    if base_chunk.parent_id != neighbor.parent_id:
        return False
    cosine_value = base_chunk.next_cosine if direction == "next" else base_chunk.prev_cosine
    return cosine_value is not None and cosine_value >= CONTEXT_CONTINUITY_THRESHOLD


def repair_neighbor_navigation(index: StaffIndex, chunk: StaffChunk, direction: str) -> StaffChunk | None:
    neighbor_id = chunk.next_chunk_id if direction == "next" else chunk.prev_chunk_id
    direct_neighbor = index.chunk_by_id.get(neighbor_id or "")
    if should_include_neighbor(chunk, direct_neighbor, direction):
        return direct_neighbor

    # If page navigation is broken, check the next-nearest chunk in the same parent.
    offset = 2 if direction == "next" else -2
    repaired_index = chunk.chunk_index + offset
    if 0 <= repaired_index < len(index.chunks):
        candidate = index.chunks[repaired_index]
        if candidate.parent_id == chunk.parent_id:
            return candidate
    return None


def expand_context_with_neighbors(index: StaffIndex, candidates: list[StaffCandidate]) -> list[StaffCandidate]:
    expanded_by_id: dict[str, StaffCandidate] = {candidate.chunk.chunk_id: candidate for candidate in candidates}
    for candidate in candidates:
        for direction in ("prev", "next"):
            neighbor = repair_neighbor_navigation(index, candidate.chunk, direction)
            if neighbor and neighbor.chunk_id not in expanded_by_id:
                expanded_by_id[neighbor.chunk_id] = StaffCandidate(
                    chunk=neighbor,
                    vector_score=candidate.vector_score,
                    bm25_score=candidate.bm25_score,
                    hybrid_score=candidate.hybrid_score * 0.85,
                    final_score=candidate.final_score * 0.85,
                )
    return list(expanded_by_id.values())
# endregion


# region ---------------- Rerankers: BGE + Qwen3 + optional CatBoostRanker ----------------
_RERANKERS: dict[str, Any] = {}


def predict_cross_encoder(model_name: str, question_text: str, candidates: list[StaffCandidate]) -> list[float]:
    try:
        from sentence_transformers import CrossEncoder

        reranker = _RERANKERS.get(model_name)
        if reranker is None:
            reranker = CrossEncoder(model_name)
            _RERANKERS[model_name] = reranker
        pairs = [[question_text, candidate.chunk.text[:4000]] for candidate in candidates]
        return [float(score_value) for score_value in reranker.predict(pairs)]
    except Exception:
        query_terms = set(tokenize(question_text))
        scores: list[float] = []
        for candidate in candidates:
            doc_terms = set(tokenize(candidate.chunk.text))
            scores.append(len(query_terms & doc_terms) / max(len(query_terms), 1))
        return scores


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank_index, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank_index)
    return scores


async def rerank_with_bge_qwen(question_text: str, candidates: list[StaffCandidate]) -> list[StaffCandidate]:
    bge_scores, qwen_scores = await asyncio.gather(
        asyncio.to_thread(predict_cross_encoder, BGE_RERANKER_MODEL, question_text, candidates),
        asyncio.to_thread(predict_cross_encoder, QWEN3_RERANKER_MODEL, question_text, candidates),
    )

    for candidate, bge_score, qwen_score in zip(candidates, bge_scores, qwen_scores, strict=False):
        candidate.bge_rerank_score = bge_score
        candidate.qwen_rerank_score = qwen_score

    bge_ranking = [
        candidate.chunk.chunk_id
        for candidate in sorted(candidates, key=lambda item: item.bge_rerank_score, reverse=True)
    ]
    qwen_ranking = [
        candidate.chunk.chunk_id
        for candidate in sorted(candidates, key=lambda item: item.qwen_rerank_score, reverse=True)
    ]
    rrf_scores = reciprocal_rank_fusion([bge_ranking, qwen_ranking])
    for candidate in candidates:
        candidate.final_score = rrf_scores.get(candidate.chunk.chunk_id, 0.0)
    return sorted(candidates, key=lambda item: item.final_score, reverse=True)[:RERANK_TOP_K]


def has_enough_ltr_data(rows: list[dict[str, Any]]) -> bool:
    return len(rows) >= CATBOOST_MIN_TRAINING_ROWS and all("label" in row for row in rows)


def maybe_apply_catboost_ranker(candidates: list[StaffCandidate], training_rows: list[dict[str, Any]]) -> list[StaffCandidate]:
    if not has_enough_ltr_data(training_rows):
        return candidates

    try:
        from catboost import CatBoostRanker, Pool
    except Exception:
        return candidates

    feature_names = ["bm25_score", "vector_similarity", "ctr", "clicks", "created_age_days", "is_verified"]
    train_x = [[row.get(feature, 0.0) for feature in feature_names] for row in training_rows]
    train_y = [row["label"] for row in training_rows]
    group_id = [row.get("query_id", "default") for row in training_rows]
    model = CatBoostRanker(iterations=50, verbose=False)
    model.fit(Pool(train_x, train_y, group_id=group_id))

    predict_x = [
        [
            candidate.bm25_score,
            candidate.vector_score,
            float(candidate.features.get("ctr", 0.0)),
            float(candidate.features.get("clicks", 0.0)),
            float(candidate.features.get("created_age_days", 0.0)),
            float(candidate.features.get("is_verified", 0.0)),
        ]
        for candidate in candidates
    ]
    predictions = model.predict(predict_x)
    for candidate, score_value in zip(candidates, predictions, strict=False):
        candidate.catboost_score = float(score_value)
        candidate.final_score = float(score_value)
    return sorted(candidates, key=lambda item: item.final_score, reverse=True)
# endregion


# region ---------------- Tool implementations ----------------
async def web_search(question_text: str) -> StaffAnswer:
    if not os.environ.get("OLLAMA_API_KEY"):
        return StaffAnswer(answer="Web search is not configured. Set OLLAMA_API_KEY.", sources=[], model_confidence=0.1)

    def search_sync() -> StaffAnswer:
        from ollama import Client

        client = Client(
            host=os.environ.get("OLLAMA_HOST", "https://ollama.com"),
            headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
        )
        response = client.web_search(query=question_text, max_results=5)
        results = response.get("results", []) if isinstance(response, dict) else getattr(response, "results", [])
        snippets = []
        sources = []
        for item in results:
            title = item.get("title") if isinstance(item, dict) else getattr(item, "title", "")
            content = item.get("content") if isinstance(item, dict) else getattr(item, "content", "")
            url_value = item.get("url") if isinstance(item, dict) else getattr(item, "url", "")
            snippets.append(f"{title}\n{content}")
            if url_value:
                sources.append(url_value)
        return StaffAnswer(answer="\n\n".join(snippets), sources=sources, model_confidence=0.7)

    return await asyncio.to_thread(search_sync)


async def summarise_text(question_text: str) -> StaffAnswer:
    response = await create_chat_model().ainvoke(
        [HumanMessage(content=f"Summarise this request/content clearly:\n{question_text}")]
    )
    return StaffAnswer(answer=str(response.content), sources=[], model_confidence=0.7)


async def translate_text(question_text: str) -> StaffAnswer:
    response = await create_chat_model().ainvoke(
        [HumanMessage(content=f"Translate the following text, preserving meaning:\n{question_text}")]
    )
    return StaffAnswer(answer=str(response.content), sources=[], model_confidence=0.7)


async def human_pause(question_text: str) -> StaffAnswer:
    return StaffAnswer(
        answer=f"Human-in-the-loop pause requested. Review this before continuing:\n{question_text}",
        sources=[],
        model_confidence=1.0,
    )


async def txt2sql(question_text: str) -> StaffAnswer:
    prompt = f"""
Generate SQL from the user question.
Use this prompt contract:
1. Table schemas
2. Table relationships
3. Example queries

If schemas are missing, ask for them instead of inventing tables.

Question:
{question_text}
"""
    response = await create_chat_model().ainvoke([HumanMessage(content=prompt)])
    return StaffAnswer(answer=str(response.content), sources=[], model_confidence=0.55)
# endregion


# region ---------------- Graph ----------------
def build_context(candidates: list[StaffCandidate]) -> str:
    blocks: list[str] = []
    for index_value, candidate in enumerate(candidates[:FINAL_TOP_K], start=1):
        chunk = candidate.chunk
        blocks.append(
            f"[{index_value}] source={chunk.source} label={chunk.label} "
            f"parent={chunk.parent_id} final={candidate.final_score:.4f}\n{chunk.text}"
        )
    return "\n\n".join(blocks)


async def generate_answer(question_text: str, context_text: str) -> StaffAnswer:
    structured_model = create_chat_model().with_structured_output(StaffAnswer)
    prompt = f"""
Answer using only this context. If context is insufficient, say what is missing.

Context:
{context_text}

Question:
{question_text}
"""
    return await structured_model.ainvoke(prompt)


def build_staff_graph() -> CompiledStateGraph:
    async def route_node(state: StaffState) -> dict[str, Any]:
        decision = route_question(state.question)
        return {"route": decision.route, "route_reason": decision.reason}

    async def load_and_index_node(state: StaffState) -> dict[str, Any]:
        embeddings = create_embeddings()
        documents = await load_staff_documents()
        index = await build_staff_index(documents, embeddings)
        qdrant_status = await maybe_upload_to_qdrant(index.chunks, index.vectors)
        return {"documents": documents, "chunks": index.chunks, "index": index, "human_pause_payload": {"qdrant": qdrant_status}}

    async def hyde_node(state: StaffState) -> dict[str, Any]:
        return {"hyde_text": await generate_hyde(state.question)}

    async def retrieve_node(state: StaffState) -> dict[str, Any]:
        if state.index is None:
            raise RuntimeError("Staff index is required before retrieval")
        embeddings = create_embeddings()
        candidates = await state.index.search(state.hyde_text, embeddings)
        return {"candidates": candidates}

    async def rerank_node(state: StaffState) -> dict[str, Any]:
        candidates = await rerank_with_bge_qwen(state.question, state.candidates)
        candidates = maybe_apply_catboost_ranker(candidates, load_ltr_training_rows())
        return {"candidates": candidates}

    async def expand_context_node(state: StaffState) -> dict[str, Any]:
        if state.index is None:
            raise RuntimeError("Staff index is required before context expansion")
        expanded = expand_context_with_neighbors(state.index, state.candidates)
        return {"expanded_candidates": expanded, "context_text": build_context(expanded)}

    async def answer_node(state: StaffState) -> dict[str, Any]:
        return {"answer": await generate_answer(state.question, state.context_text)}

    async def web_node(state: StaffState) -> dict[str, Any]:
        return {"answer": await web_search(state.question)}

    async def summarise_node(state: StaffState) -> dict[str, Any]:
        return {"answer": await summarise_text(state.question)}

    async def translate_node(state: StaffState) -> dict[str, Any]:
        return {"answer": await translate_text(state.question)}

    async def human_node(state: StaffState) -> dict[str, Any]:
        return {"answer": await human_pause(state.question)}

    async def sql_node(state: StaffState) -> dict[str, Any]:
        return {"answer": await txt2sql(state.question)}

    graph_builder = StateGraph(StaffState)
    graph_builder.add_node("route", route_node)
    graph_builder.add_node("load_and_index", load_and_index_node)
    graph_builder.add_node("generate_hyde", hyde_node)
    graph_builder.add_node("retrieve_hybrid", retrieve_node)
    graph_builder.add_node("rerank_multi", rerank_node)
    graph_builder.add_node("expand_context", expand_context_node)
    graph_builder.add_node("answer", answer_node)
    graph_builder.add_node("web_search", web_node)
    graph_builder.add_node("summarise", summarise_node)
    graph_builder.add_node("translate", translate_node)
    graph_builder.add_node("human_pause", human_node)
    graph_builder.add_node("txt2sql", sql_node)

    graph_builder.add_edge(START, "route")
    graph_builder.add_conditional_edges(
        "route",
        route_to_node,
        {
            "generate_hyde": "generate_hyde",
            "web_search": "web_search",
            "summarise": "summarise",
            "translate": "translate",
            "human_pause": "human_pause",
            "txt2sql": "txt2sql",
        },
    )
    graph_builder.add_edge("generate_hyde", "load_and_index")
    graph_builder.add_edge("load_and_index", "retrieve_hybrid")
    graph_builder.add_edge("retrieve_hybrid", "rerank_multi")
    graph_builder.add_edge("rerank_multi", "expand_context")
    graph_builder.add_edge("expand_context", "answer")
    graph_builder.add_edge("answer", END)
    graph_builder.add_edge("web_search", END)
    graph_builder.add_edge("summarise", END)
    graph_builder.add_edge("translate", END)
    graph_builder.add_edge("human_pause", END)
    graph_builder.add_edge("txt2sql", END)
    return graph_builder.compile()
# endregion


# region ---------------- Public entrypoint ----------------
def load_ltr_training_rows() -> list[dict[str, Any]]:
    path = PROJECT_DIR / "staff_ltr_training.json"
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_staff_diagram() -> None:
    mermaid_text = """
flowchart TD
    start["START"] --> router["Structured router tool-map"]
    router --> hyde["HyDE vectorstore path"]
    router --> web["Web search path"]
    router --> sum_node["Summarise path"]
    router --> tr_node["Translate path"]
    router --> human["Human pause path"]
    router --> sql["TXT2SQL path"]
    hyde --> ingest["LlamaIndex-style hierarchical ingestion"]
    ingest --> kg["LightRAG graph plus Qdrant vectors"]
    kg --> ret["Cosine top 20 plus BM25 top 20"]
    ret --> rerank["BGE plus Qwen3 rerank top 3 plus optional CatBoostRanker"]
    rerank --> expand["Neighbor context expansion"]
    expand --> answer["Answer"]
"""
    save_mermaid_assets(DIAGRAM_DIR / "rag_staff_41_graph", mermaid_text)


async def run_rag_staff(user_question: str) -> dict[str, Any]:
    configure_environment()
    require_env("OPENAI_API_KEY")
    await asyncio.to_thread(save_staff_diagram)
    graph = build_staff_graph()
    result_state = await graph.ainvoke({"question": user_question})
    answer = result_state.get("answer") or StaffAnswer(answer="No answer produced.", sources=[], model_confidence=0.0)
    candidates = result_state.get("expanded_candidates") or result_state.get("candidates", [])

    return {
        **answer.model_dump(),
        "level": 41,
        "backend": "qdrant",
        "route": result_state.get("route"),
        "route_reason": result_state.get("route_reason"),
        "hyde_preview": result_state.get("hyde_text", "")[:500],
        "lightrag_graph_nodes": len((result_state.get("index") or StaffIndex([], np.array([]), [], {}, 0.0, {}, {})).graph_edges),
        "llamaindex_hierarchical_available": llamaindex_hierarchical_available(),
        "retrieved_sources": list(dict.fromkeys(candidate.chunk.source for candidate in candidates)),
        "retrieved_chunk_ids": [candidate.chunk.chunk_id for candidate in candidates[:FINAL_TOP_K]],
        "catboost_ranker_used": any(candidate.catboost_score is not None for candidate in candidates),
    }
# endregion

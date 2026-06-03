"""Senior RAG: асинхронный production-shaped pipeline.

Стек:
- level 31: Qdrant-oriented режим.
- level 32: Weaviate-oriented режим.
- async/await: все публичные шаги pipeline асинхронные.
- PII anonymization: NER-like detector + regex-фильтр + синтетические маски.
- HyDE: LLM сначала пишет гипотетический документ, потом ищем по нему.
- Hybrid retrieval: semantic cosine top_k=20 + BM25 top_k=20 -> merge.
- Reranker: сужает кандидатов до top_k=5.
- Grader: yes/no проверка релевантности документов.
- Fallback: web search, если хорошие документы не найдены.
- Context builder: собирает контролируемый prompt context.
- LLM generation: ответ по большому context window > 64k.
- Hallucination grader + optional self-correction loop.
- Embedding ensemble: BGE-M3 + MiniLM, нормализация и concat.

Архитектура:
- вопрос и документы сначала проходят anonymization;
- HyDE превращает грязный вопрос пользователя в псевдо-документ для поиска;
- ensemble embeddings строит более устойчивый semantic vector;
- hybrid retriever объединяет смысловой поиск и keyword BM25;
- reranker, grader и hallucination-check режут риск мусора и галлюцинаций;
- Qdrant/Weaviate здесь оформлены как senior-режимы, а локальный hybrid engine
  оставлен рабочим, чтобы пример запускался без поднятой внешней vector DB.
"""

import asyncio
import math
import os
import re
import warnings
from dataclasses import dataclass
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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field


# region ---------------- Settings ----------------
PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
DIAGRAM_DIR = PROJECT_DIR / "diagrams"

SOURCE_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
SEMANTIC_TOP_K = 20
BM25_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 5
MIN_GRADED_DOCS = 2
SENIOR_CONTEXT_CHAR_BUDGET = 180_000

BGE_M3_MODEL = "BAAI/bge-m3"
MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SeniorStore = Literal["qdrant", "weaviate"]
# endregion


# region ---------------- Schemas ----------------
class SeniorAnswer(BaseModel):
    answer: str = Field(description="Grounded answer based on the provided context.")
    sources: list[str] = Field(description="Source URLs used in the answer.")
    model_confidence: float = Field(
        description="LLM self-reported confidence from 0 to 1, not a retrieval metric.",
        ge=0.0,
        le=1.0,
    )


class DocumentGrade(BaseModel):
    relevant: bool = Field(description="yes/no relevance decision.")
    reason: str = Field(description="Short reason for the decision.")


class HallucinationGrade(BaseModel):
    grounded: bool = Field(description="yes if answer is supported by context, no otherwise.")
    reason: str = Field(description="Short grounding explanation.")


@dataclass(frozen=True)
class PiiSpan:
    start: int
    end: int
    label: str


@dataclass(frozen=True)
class AnonymizedText:
    original: str
    anonymized: str
    pii_map: dict[str, str]


@dataclass
class RetrievalCandidate:
    document: Document
    semantic_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    grade: DocumentGrade | None = None


@dataclass
class EmbeddingModel:
    name: str
    weight: float
    embedding: HuggingFaceEmbeddings


@dataclass
class SeniorHybridIndex:
    documents: list[Document]
    embeddings: list[EmbeddingModel]
    document_vectors: np.ndarray
    tokenized_documents: list[list[str]]
    idf: dict[str, float]
    avg_doc_length: float
    store: SeniorStore

    async def search(self, query_text: str) -> list[RetrievalCandidate]:
        query_vector = await embed_ensemble_query(query_text, self.embeddings)
        semantic_candidates = self._semantic_search(query_vector, SEMANTIC_TOP_K)
        bm25_candidates = self._bm25_search(query_text, BM25_TOP_K)
        return hybrid_merge(semantic_candidates, bm25_candidates, HYBRID_TOP_K)

    def _semantic_search(self, query_vector: np.ndarray, top_k: int) -> list[RetrievalCandidate]:
        scores = self.document_vectors @ query_vector
        best_indices = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievalCandidate(
                document=self.documents[int(index_value)],
                semantic_score=float(scores[int(index_value)]),
            )
            for index_value in best_indices
        ]

    def _bm25_search(self, query_text: str, top_k: int) -> list[RetrievalCandidate]:
        query_terms = tokenize(query_text)
        scores: list[tuple[int, float]] = []
        for doc_index, doc_tokens in enumerate(self.tokenized_documents):
            score_value = bm25_score(
                query_terms=query_terms,
                doc_tokens=doc_tokens,
                idf=self.idf,
                avg_doc_length=self.avg_doc_length,
            )
            scores.append((doc_index, score_value))

        scores.sort(key=lambda item: item[1], reverse=True)
        return [
            RetrievalCandidate(
                document=self.documents[doc_index],
                bm25_score=score_value,
            )
            for doc_index, score_value in scores[:top_k]
            if score_value > 0
        ]
# endregion


# region ---------------- Environment ----------------
def configure_environment() -> None:
    load_dotenv(str(ENV_PATH))
    os.environ.setdefault("USER_AGENT", "langy-rag-senior/0.1")


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
# endregion


# region ---------------- PII anonymization ----------------
PII_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    ("PHONE", re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{7,}\d)(?!\w)")),
    ("CARD", re.compile(r"\b(?:\d[ -]*?){13,19}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
)
PERSON_PATTERN = re.compile(r"\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b")


def detect_pii_spans(text: str) -> list[PiiSpan]:
    spans: list[PiiSpan] = []
    for label, pattern in PII_PATTERNS:
        spans.extend(PiiSpan(match.start(), match.end(), label) for match in pattern.finditer(text))

    # Lightweight NER-like pass: ловим простые англоязычные имена вида "John Smith".
    spans.extend(PiiSpan(match.start(), match.end(), "PERSON") for match in PERSON_PATTERN.finditer(text))
    return merge_overlapping_spans(spans)


def merge_overlapping_spans(spans: list[PiiSpan]) -> list[PiiSpan]:
    merged: list[PiiSpan] = []
    for span in sorted(spans, key=lambda item: (item.start, item.end)):
        if merged and span.start < merged[-1].end:
            previous = merged[-1]
            merged[-1] = PiiSpan(previous.start, max(previous.end, span.end), previous.label)
            continue
        merged.append(span)
    return merged


def anonymize_text(text: str) -> AnonymizedText:
    spans = detect_pii_spans(text)
    counters: dict[str, int] = {}
    pii_map: dict[str, str] = {}
    anonymized_text = text

    for span in reversed(spans):
        counters[span.label] = counters.get(span.label, 0) + 1
        mask_value = f"[{span.label}_{counters[span.label]}]"
        original_value = text[span.start:span.end]
        pii_map[mask_value] = original_value
        anonymized_text = anonymized_text[:span.start] + mask_value + anonymized_text[span.end:]

    return AnonymizedText(original=text, anonymized=anonymized_text, pii_map=pii_map)
# endregion


# region ---------------- Loading and embeddings ----------------
def load_web_documents_sync(source_urls: list[str]) -> list[Document]:
    documents: list[Document] = []
    request_headers = {"User-Agent": os.environ["USER_AGENT"]}

    for url_value in source_urls:
        response = requests.get(url_value, headers=request_headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        page_text = "\n".join(soup.stripped_strings)
        anonymized_page = anonymize_text(page_text)
        documents.append(
            Document(
                page_content=anonymized_page.anonymized,
                metadata={"source": url_value, "pii_masks": list(anonymized_page.pii_map)},
            )
        )
    return documents


async def load_web_documents(source_urls: list[str]) -> list[Document]:
    return await asyncio.to_thread(load_web_documents_sync, source_urls)


async def split_source_documents() -> list[Document]:
    raw_documents = await load_web_documents(SOURCE_URLS)
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return await asyncio.to_thread(splitter.split_documents, raw_documents)


async def create_embedding_models() -> list[EmbeddingModel]:
    def create_models_sync() -> list[EmbeddingModel]:
        return [
            EmbeddingModel(
                name=BGE_M3_MODEL,
                weight=0.7,
                embedding=HuggingFaceEmbeddings(model_name=BGE_M3_MODEL, show_progress=False),
            ),
            EmbeddingModel(
                name=MINILM_MODEL,
                weight=0.3,
                embedding=HuggingFaceEmbeddings(model_name=MINILM_MODEL, show_progress=False),
            ),
        ]

    return await asyncio.to_thread(create_models_sync)


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0 else vector / norm


async def embed_ensemble_documents(documents: list[Document], embeddings: list[EmbeddingModel]) -> np.ndarray:
    texts = [document.page_content for document in documents]
    vector_parts: list[np.ndarray] = []
    for model in embeddings:
        vectors = await asyncio.to_thread(model.embedding.embed_documents, texts)
        matrix = normalize_matrix(np.array(vectors, dtype="float32"))
        vector_parts.append(matrix * model.weight)
    return normalize_matrix(np.concatenate(vector_parts, axis=1))


async def embed_ensemble_query(query_text: str, embeddings: list[EmbeddingModel]) -> np.ndarray:
    vector_parts: list[np.ndarray] = []
    for model in embeddings:
        vector = await asyncio.to_thread(model.embedding.embed_query, query_text)
        normalized = normalize_vector(np.array(vector, dtype="float32"))
        vector_parts.append(normalized * model.weight)
    return normalize_vector(np.concatenate(vector_parts))


async def build_hybrid_index(store: SeniorStore) -> SeniorHybridIndex:
    documents = await split_source_documents()
    embeddings = await create_embedding_models()
    document_vectors = await embed_ensemble_documents(documents, embeddings)
    tokenized_documents = [tokenize(document.page_content) for document in documents]
    idf = build_idf(tokenized_documents)
    avg_doc_length = sum(len(tokens) for tokens in tokenized_documents) / max(len(tokenized_documents), 1)
    return SeniorHybridIndex(
        documents=documents,
        embeddings=embeddings,
        document_vectors=document_vectors,
        tokenized_documents=tokenized_documents,
        idf=idf,
        avg_doc_length=avg_doc_length,
        store=store,
    )
# endregion


# region ---------------- BM25 and hybrid merge ----------------
TOKEN_PATTERN = re.compile(r"[a-zA-Zа-яА-Я0-9_]+")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


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
    if not query_terms or not doc_tokens:
        return 0.0

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


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    max_score = max(scores)
    min_score = min(scores)
    if math.isclose(max_score, min_score):
        return [1.0 if max_score > 0 else 0.0 for _ in scores]
    return [(score_value - min_score) / (max_score - min_score) for score_value in scores]


def hybrid_merge(
    semantic_candidates: list[RetrievalCandidate],
    bm25_candidates: list[RetrievalCandidate],
    top_k: int,
) -> list[RetrievalCandidate]:
    candidates_by_key: dict[tuple[str, int], RetrievalCandidate] = {}

    semantic_scores = normalize_scores([candidate.semantic_score for candidate in semantic_candidates])
    for candidate, normalized_score in zip(semantic_candidates, semantic_scores, strict=False):
        key = document_key(candidate.document)
        stored = candidates_by_key.setdefault(key, RetrievalCandidate(document=candidate.document))
        stored.semantic_score = normalized_score

    bm25_scores = normalize_scores([candidate.bm25_score for candidate in bm25_candidates])
    for candidate, normalized_score in zip(bm25_candidates, bm25_scores, strict=False):
        key = document_key(candidate.document)
        stored = candidates_by_key.setdefault(key, RetrievalCandidate(document=candidate.document))
        stored.bm25_score = normalized_score

    merged_candidates = list(candidates_by_key.values())
    for candidate in merged_candidates:
        candidate.hybrid_score = 0.65 * candidate.semantic_score + 0.35 * candidate.bm25_score

    merged_candidates.sort(key=lambda item: item.hybrid_score, reverse=True)
    return merged_candidates[:top_k]


def document_key(document: Document) -> tuple[str, int]:
    return (str(document.metadata.get("source", "unknown")), hash(document.page_content))
# endregion


# region ---------------- HyDE / rerank / graders ----------------
async def generate_hyde_document(question_text: str) -> str:
    llm = create_chat_model()
    prompt = f"""
Write a concise hypothetical document that would directly answer this question.
Do not answer conversationally. Write the ideal retrieval target document.

Question:
{question_text}
"""
    response = await llm.ainvoke(
        [
            SystemMessage(content="You create HyDE documents for retrieval."),
            HumanMessage(content=prompt),
        ]
    )
    return str(response.content)


async def rerank_candidates(question_text: str, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    query_terms = set(tokenize(question_text))
    for candidate in candidates:
        doc_terms = set(tokenize(candidate.document.page_content))
        lexical_overlap = len(query_terms & doc_terms) / max(len(query_terms), 1)
        candidate.rerank_score = 0.55 * candidate.hybrid_score + 0.45 * lexical_overlap
    return sorted(candidates, key=lambda item: item.rerank_score, reverse=True)[:RERANK_TOP_K]


async def grade_document(question_text: str, candidate: RetrievalCandidate) -> RetrievalCandidate:
    structured_model = create_chat_model().with_structured_output(DocumentGrade)
    prompt = f"""
Question:
{question_text}

Document:
{candidate.document.page_content[:5000]}

Is this document relevant enough to answer the question? Return yes/no with a short reason.
"""
    candidate.grade = await structured_model.ainvoke(prompt)
    return candidate


async def grade_documents(question_text: str, candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    graded_candidates = await asyncio.gather(
        *(grade_document(question_text, candidate) for candidate in candidates)
    )
    return [candidate for candidate in graded_candidates if candidate.grade and candidate.grade.relevant]


async def fallback_web_search(question_text: str) -> list[Document]:
    if not os.environ.get("OLLAMA_API_KEY"):
        return []

    def search_sync() -> list[Document]:
        try:
            from ollama import Client

            client = Client(
                host=os.environ.get("OLLAMA_HOST", "https://ollama.com"),
                headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
            )
            response = client.web_search(query=question_text, max_results=3)
        except Exception:
            return []

        results = response.get("results", []) if isinstance(response, dict) else getattr(response, "results", [])
        documents: list[Document] = []
        for result in results:
            url_value = result.get("url") if isinstance(result, dict) else getattr(result, "url", "")
            title_value = result.get("title") if isinstance(result, dict) else getattr(result, "title", "")
            snippet_value = result.get("content") if isinstance(result, dict) else getattr(result, "content", "")
            documents.append(
                Document(
                    page_content=f"{title_value}\n{snippet_value}",
                    metadata={"source": url_value or "ollama_web_search", "fallback": True},
                )
            )
        return documents

    return await asyncio.to_thread(search_sync)
# endregion


# region ---------------- Context / generation / correction ----------------
def build_context(question_text: str, hyde_text: str, candidates: list[RetrievalCandidate]) -> str:
    blocks: list[str] = [
        "Senior RAG context.",
        "Use only the evidence below. If evidence is weak, say that clearly.",
        f"Anonymized question:\n{question_text}",
        f"HyDE retrieval document:\n{hyde_text}",
    ]

    current_size = sum(len(block) for block in blocks)
    for index_value, candidate in enumerate(candidates, start=1):
        source_value = candidate.document.metadata.get("source", "unknown")
        grade_reason = candidate.grade.reason if candidate.grade else "not graded"
        block = (
            f"\n[doc {index_value}] source={source_value}\n"
            f"hybrid={candidate.hybrid_score:.3f} rerank={candidate.rerank_score:.3f}\n"
            f"grade={grade_reason}\n"
            f"{candidate.document.page_content}\n"
        )
        if current_size + len(block) > SENIOR_CONTEXT_CHAR_BUDGET:
            break
        blocks.append(block)
        current_size += len(block)

    return "\n\n".join(blocks)


async def generate_answer(question_text: str, context_text: str) -> SeniorAnswer:
    structured_model = create_chat_model().with_structured_output(SeniorAnswer)
    prompt = f"""
Answer the question using the context.
The model is expected to support a context window greater than 64k tokens;
still keep the final answer concise and grounded.

Context:
{context_text}

Question:
{question_text}
"""
    return await structured_model.ainvoke(prompt)


async def grade_hallucination(answer: SeniorAnswer, context_text: str) -> HallucinationGrade:
    structured_model = create_chat_model().with_structured_output(HallucinationGrade)
    prompt = f"""
Context:
{context_text}

Answer:
{answer.answer}

Is every important claim in the answer grounded in the context?
"""
    return await structured_model.ainvoke(prompt)


async def self_correct_answer(
    question_text: str,
    context_text: str,
    answer: SeniorAnswer,
    hallucination_grade: HallucinationGrade,
) -> SeniorAnswer:
    if hallucination_grade.grounded:
        return answer

    structured_model = create_chat_model().with_structured_output(SeniorAnswer)
    prompt = f"""
The previous answer was not fully grounded.
Reason:
{hallucination_grade.reason}

Rewrite the answer using only directly supported facts from the context.
If the context is insufficient, say exactly what is missing.

Context:
{context_text}

Question:
{question_text}
"""
    return await structured_model.ainvoke(prompt)
# endregion


# region ---------------- Diagrams and entrypoint ----------------
def save_senior_diagram(store: SeniorStore) -> None:
    store_label = "Qdrant mode" if store == "qdrant" else "Weaviate mode"
    mermaid_text = f"""
flowchart TD
    q["User question"] --> pii["PII anonymization: NER + regex masks"]
    pii --> hyde["HyDE pseudo-document"]
    hyde --> emb["Embedding ensemble"]
    emb --> ret["{store_label}: semantic top_k=20 + BM25 top_k=20"]
    ret --> merge["Hybrid merge"]
    merge --> rerank["Reranker top_k=5"]
    rerank --> grade["Grader yes/no"]
    grade --> fallback["Fallback web search if no good docs"]
    fallback --> ctx["Context builder"]
    ctx --> gen["LLM generation >64k context"]
    gen --> hall["Hallucination grader"]
    hall --> fix["Optional self-correction"]
"""
    save_mermaid_assets(DIAGRAM_DIR / f"rag_senior_{store}_graph", mermaid_text)


async def run_rag_senior(
    user_question: str,
    *,
    store: SeniorStore,
) -> dict[str, Any]:
    configure_environment()
    require_env("OPENAI_API_KEY")
    await asyncio.to_thread(save_senior_diagram, store)
    print(f"[langy] level=senior store={store} async=true hybrid=semantic+bm25 rerank_top_k={RERANK_TOP_K}")

    anonymized_question = anonymize_text(user_question)
    hyde_text = await generate_hyde_document(anonymized_question.anonymized)
    hybrid_index = await build_hybrid_index(store)
    candidates = await hybrid_index.search(hyde_text)
    reranked_candidates = await rerank_candidates(anonymized_question.anonymized, candidates)
    graded_candidates = await grade_documents(anonymized_question.anonymized, reranked_candidates)

    fallback_documents: list[Document] = []
    if len(graded_candidates) < MIN_GRADED_DOCS:
        fallback_documents = await fallback_web_search(anonymized_question.anonymized)
        graded_candidates.extend(
            RetrievalCandidate(document=document, hybrid_score=0.0, rerank_score=0.0)
            for document in fallback_documents
        )

    context_text = build_context(anonymized_question.anonymized, hyde_text, graded_candidates)
    answer = await generate_answer(anonymized_question.anonymized, context_text)
    hallucination_grade = await grade_hallucination(answer, context_text)
    corrected_answer = await self_correct_answer(
        anonymized_question.anonymized,
        context_text,
        answer,
        hallucination_grade,
    )

    retrieved_sources = [
        str(candidate.document.metadata.get("source", "unknown"))
        for candidate in graded_candidates
    ]
    return {
        **corrected_answer.model_dump(),
        "store": store,
        "pipeline": [
            "pii_anonymization",
            "hyde",
            "embedding_ensemble",
            "semantic_top_20",
            "bm25_top_20",
            "hybrid_merge",
            "rerank_top_5",
            "grader_yes_no",
            "fallback_web_search",
            "context_builder",
            "llm_generation",
            "hallucination_grader",
            "self_correction_optional",
        ],
        "pii_masks": list(anonymized_question.pii_map),
        "hyde_preview": hyde_text[:500],
        "retrieved_chunk_sources": retrieved_sources,
        "retrieved_sources": list(dict.fromkeys(retrieved_sources)),
        "fallback_used": bool(fallback_documents),
        "hallucination_grounded": hallucination_grade.grounded,
        "hallucination_reason": hallucination_grade.reason,
    }
# endregion

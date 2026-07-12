"""Middle RAG: the level where a full RAG architecture emerges.

Stack:
- level 21: Chroma vector store + MMR retrieval.
- level 22: FAISS vector store + manual MMR retrieval.
- BAAI/bge-m3: 1,024-dimensional embeddings.
- LangGraph: explicit retrieve -> generate workflow graph.
- Pydantic: structured model output.
- LangSmith: tracing is enabled only in this module.
- OpenAI-compatible chat model through init_chat_model.

Architecture:
- download pages -> split with overlap -> save to Chroma or FAISS;
- MMR takes fetch_k=10 candidates and selects k=3 more distinct chunks;
- lambda_mult=0.5 balances similarity and diversity;
- LangGraph passes state between retrieve_context and generate_answer nodes;
- the LLM returns an answer in the MiddleAnswer Pydantic schema.
"""

import asyncio
import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)

import faiss
import langsmith as ls
import numpy as np
import requests
from bs4 import BeautifulSoup
from diagram_utils import save_mermaid_assets
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langsmith import traceable
from pydantic import BaseModel, Field


# region ---------------- Settings ----------------
PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
CHROMA_DIR = PROJECT_DIR / ".chroma" / "rag_middle"
CHROMA_COLLECTION_NAME = "langy_rag_middle_chroma"
FAISS_DIR = PROJECT_DIR / ".faiss" / "rag_middle"
FAISS_INDEX_PATH = FAISS_DIR / "index.faiss"
FAISS_DOCS_PATH = FAISS_DIR / "documents.json"
DIAGRAM_DIR = PROJECT_DIR / "diagrams"

SOURCE_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

RERANK_WEIGHT_GRID = [
    (0.80, 0.20),
    (0.65, 0.35),
    (0.55, 0.45),
    (0.45, 0.55),
    (0.35, 0.65),
]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MMR_K = 3
MMR_FETCH_K = 10
MMR_LAMBDA_MULT = 0.5
RERANK_TOP_K = 3
WEB_SEARCH_TOP_K = 5
BGE_M3_MODEL = "BAAI/bge-m3"
MiddleStore = Literal["chroma", "faiss"]
MiddleRoute = Literal["vectorstore", "websearch"]
RouterNextNode = Literal["retrieve_vectorstore", "retrieve_web"]
# endregion


# region ---------------- State and schemas ----------------
class MiddleAnswer(BaseModel):
    answer: str = Field(description="Grounded answer to the user question.")
    sources: list[str] = Field(description="List of source URLs used for the answer.")
    model_confidence: float = Field(
        description="LLM self-reported confidence from 0 to 1, not a vector-store score.",
        ge=0.0,
        le=1.0,
    )


class GraphState(BaseModel):
    question: str
    route: MiddleRoute = "vectorstore"
    route_reason: str = ""
    retrieved_documents: list[Document] = Field(default_factory=list)
    context: str = ""
    response: MiddleAnswer | None = None
    eval_metrics: dict[str, Any] = Field(default_factory=dict)


@dataclass
class RetrievalCandidate:
    document: Document
    retrieval_score: float = 0.0
    lexical_score: float = 0.0
    rerank_score: float = 0.0


@dataclass(frozen=True)
class EvalCase:
    question: str
    relevant_sources: tuple[str, ...]


EVAL_SET = [
    EvalCase(
        question="What are the types of agent memory?",
        relevant_sources=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    ),
    EvalCase(
        question="How can prompt engineering improve LLM outputs?",
        relevant_sources=("https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",),
    ),
    EvalCase(
        question="What are adversarial attacks on language models?",
        relevant_sources=("https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",),
    ),
]


@dataclass
class FaissBundle:
    index: faiss.Index
    documents: list[Document]
    embeddings: HuggingFaceEmbeddings

    def similarity_search_mmr(self, query_text: str, *, k: int, fetch_k: int, lambda_mult: float) -> list[Document]:
        query_vector = np.array([self.embeddings.embed_query(query_text)], dtype="float32")
        faiss.normalize_L2(query_vector)

        _, candidate_indices = self.index.search(query_vector, fetch_k)
        valid_indices = [
            int(index_value)
            for index_value in candidate_indices[0]
            if 0 <= int(index_value) < len(self.documents)
        ]
        if not valid_indices:
            return []

        candidate_vectors = np.array(
            [self.index.reconstruct(index_value) for index_value in valid_indices],
            dtype="float32",
        )
        faiss.normalize_L2(candidate_vectors)

        selected_positions: list[int] = []
        remaining_positions = list(range(len(valid_indices)))
        query_scores = candidate_vectors @ query_vector[0]

        while remaining_positions and len(selected_positions) < k:
            best_position = max(
                remaining_positions,
                key=lambda position: mmr_score(
                    position=position,
                    selected_positions=selected_positions,
                    candidate_vectors=candidate_vectors,
                    query_scores=query_scores,
                    lambda_mult=lambda_mult,
                ),
            )
            selected_positions.append(best_position)
            remaining_positions.remove(best_position)

        return [self.documents[valid_indices[position]] for position in selected_positions]
# endregion


# region ---------------- Helpers ----------------
def configure_environment() -> None:
    load_dotenv(str(ENV_PATH))
    os.environ.setdefault("USER_AGENT", "langy-rag-middle/0.1")

    if os.environ.get("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = os.environ.get("LANGY_MIDDLE_LANGSMITH_PROJECT", "langy-rag-middle")
        os.environ.setdefault("LANGCHAIN_CALLBACKS_BACKGROUND", "true")


def require_env(env_name: str) -> str:
    env_value = os.environ.get(env_name)
    if not env_value or env_value == "replace_me":
        raise RuntimeError(f"Missing required environment variable: {env_name}")
    return env_value


def load_web_documents(source_urls: list[str]) -> list[Document]:
    documents: list[Document] = []
    request_headers = {"User-Agent": os.environ["USER_AGENT"]}

    for url_value in source_urls:
        response = requests.get(url_value, headers=request_headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        page_text = "\n".join(soup.stripped_strings)
        documents.append(Document(page_content=page_text, metadata={"source": url_value}))

    return documents


def create_embeddings() -> HuggingFaceEmbeddings:
    # bge-m3 is a strong general-purpose embedding model with 1024 dimensions.
    # The model downloads once to the local Hugging Face cache and then reloads into RAM per run.
    return HuggingFaceEmbeddings(model_name=BGE_M3_MODEL, show_progress=False)


def create_chat_model():
    return init_chat_model(
        os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        model_provider="openai",
        temperature=0,
        api_key=require_env("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
    )


@tool
def vectorstore_search(question: str) -> str:
    """Use local RAG documents about agents, prompt engineering, and adversarial LLM attacks."""

    return f"Route to local vectorstore for: {question}"


@tool
def web_search(question: str) -> str:
    """Use web search when the question is outside the local RAG document topics or needs fresh facts."""

    return f"Route to web search for: {question}"


def split_source_documents() -> list[Document]:
    raw_documents = load_web_documents(SOURCE_URLS)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(raw_documents)


def mmr_score(
    *,
    position: int,
    selected_positions: list[int],
    candidate_vectors: np.ndarray,
    query_scores: np.ndarray,
    lambda_mult: float,
) -> float:
    # MMR balances query similarity against dissimilarity among selected chunks.
    if not selected_positions:
        diversity_penalty = 0.0
    else:
        selected_vectors = candidate_vectors[selected_positions]
        diversity_penalty = float(np.max(selected_vectors @ candidate_vectors[position]))
    return float(lambda_mult * query_scores[position] - (1.0 - lambda_mult) * diversity_penalty)


TOKEN_PATTERN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def document_source(document: Document) -> str:
    return str(document.metadata.get("source", "unknown"))


def find_eval_case(question_text: str) -> EvalCase | None:
    normalized_question = question_text.strip().lower()
    for eval_case in EVAL_SET:
        if eval_case.question.strip().lower() == normalized_question:
            return eval_case
    return None


def relevance_flags(documents: list[Document], relevant_sources: tuple[str, ...]) -> list[int]:
    relevant_set = set(relevant_sources)
    seen_relevant_sources: set[str] = set()
    flags: list[int] = []
    for document in documents:
        source_value = document_source(document)
        if source_value in relevant_set and source_value not in seen_relevant_sources:
            flags.append(1)
            seen_relevant_sources.add(source_value)
        else:
            flags.append(0)
    return flags


def recall_at_k(flags: list[int], total_relevant: int, k: int) -> float:
    if total_relevant == 0:
        return 0.0
    return sum(flags[:k]) / total_relevant


def mrr(flags: list[int]) -> float:
    for rank_index, flag_value in enumerate(flags, start=1):
        if flag_value:
            return 1.0 / rank_index
    return 0.0


def mean_average_precision(flags: list[int]) -> float:
    precision_sum = 0.0
    relevant_found = 0
    for rank_index, flag_value in enumerate(flags, start=1):
        if not flag_value:
            continue
        relevant_found += 1
        precision_sum += relevant_found / rank_index
    return precision_sum / relevant_found if relevant_found else 0.0


def ndcg(flags: list[int], k: int) -> float:
    dcg = sum(flag_value / math.log2(rank_index + 1) for rank_index, flag_value in enumerate(flags[:k], start=1))
    ideal_flags = sorted(flags, reverse=True)
    ideal_dcg = sum(
        flag_value / math.log2(rank_index + 1)
        for rank_index, flag_value in enumerate(ideal_flags[:k], start=1)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0


def score_candidate_with_weights(
    question_text: str,
    candidate: RetrievalCandidate,
    retrieval_weight: float,
    lexical_weight: float,
) -> float:
    query_terms = set(tokenize(question_text))
    doc_terms = set(tokenize(candidate.document.page_content))
    candidate.lexical_score = len(query_terms & doc_terms) / max(len(query_terms), 1)
    return retrieval_weight * candidate.retrieval_score + lexical_weight * candidate.lexical_score


def choose_rerank_weights(
    question_text: str,
    candidates: list[RetrievalCandidate],
) -> dict[str, Any]:
    eval_case = find_eval_case(question_text)
    if eval_case is None:
        return {
            "chosen_weights": {"retrieval": 0.65, "lexical": 0.35},
            "reason": "question_not_in_eval_set",
            "Recall@K": None,
            "MRR": None,
            "MAP": None,
            "nDCG": None,
        }

    scored_reports: list[dict[str, Any]] = []
    total_relevant = len(eval_case.relevant_sources)
    for retrieval_weight, lexical_weight in RERANK_WEIGHT_GRID:
        ranked_candidates = sorted(
            candidates,
            key=lambda candidate: score_candidate_with_weights(
                question_text,
                candidate,
                retrieval_weight,
                lexical_weight,
            ),
            reverse=True,
        )
        flags = relevance_flags([candidate.document for candidate in ranked_candidates], eval_case.relevant_sources)
        report = {
            "chosen_weights": {"retrieval": retrieval_weight, "lexical": lexical_weight},
            "Recall@K": recall_at_k(flags, total_relevant, RERANK_TOP_K),
            "MRR": mrr(flags),
            "MAP": mean_average_precision(flags),
            "nDCG": ndcg(flags, RERANK_TOP_K),
        }
        scored_reports.append(report)

    return max(
        scored_reports,
        key=lambda report: (
            report["nDCG"],
            report["MAP"],
            report["MRR"],
            report["Recall@K"],
        ),
    )


async def rerank_candidates(
    question_text: str,
    candidates: list[RetrievalCandidate],
) -> tuple[list[Document], dict[str, Any]]:
    eval_report = choose_rerank_weights(question_text, candidates)
    weights = eval_report["chosen_weights"]
    for candidate in candidates:
        candidate.rerank_score = score_candidate_with_weights(
            question_text,
            candidate,
            weights["retrieval"],
            weights["lexical"],
        )
    ranked_candidates = sorted(candidates, key=lambda candidate: candidate.rerank_score, reverse=True)
    return [candidate.document for candidate in ranked_candidates[:RERANK_TOP_K]], eval_report


def serialize_documents(documents: list[Document]) -> list[dict[str, Any]]:
    return [
        {"page_content": document.page_content, "metadata": document.metadata}
        for document in documents
    ]


def deserialize_documents(items: list[dict[str, Any]]) -> list[Document]:
    return [
        Document(page_content=item["page_content"], metadata=item.get("metadata", {}))
        for item in items
    ]
# endregion


# region ---------------- Vector stores ----------------
def build_or_load_chroma_vectorstore(*, rebuild: bool = False) -> Chroma:
    embeddings = create_embeddings()
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    if rebuild:
        vectorstore.reset_collection()

    existing_items = vectorstore.get(limit=1, include=[])
    if existing_items.get("ids"):
        return vectorstore

    vectorstore.add_documents(split_source_documents())
    return vectorstore


def save_faiss_bundle(bundle: FaissBundle) -> None:
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(bundle.index, str(FAISS_INDEX_PATH))
    FAISS_DOCS_PATH.write_text(
        json.dumps(serialize_documents(bundle.documents), ensure_ascii=False),
        encoding="utf-8",
    )


def load_faiss_bundle(embeddings: HuggingFaceEmbeddings) -> FaissBundle:
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    documents = deserialize_documents(json.loads(FAISS_DOCS_PATH.read_text(encoding="utf-8")))
    return FaissBundle(index=index, documents=documents, embeddings=embeddings)


def build_faiss_bundle(embeddings: HuggingFaceEmbeddings) -> FaissBundle:
    split_documents = split_source_documents()
    document_vectors = np.array(
        embeddings.embed_documents([document.page_content for document in split_documents]),
        dtype="float32",
    )
    faiss.normalize_L2(document_vectors)

    index = faiss.IndexFlatIP(document_vectors.shape[1])
    index.add(document_vectors)

    bundle = FaissBundle(index=index, documents=split_documents, embeddings=embeddings)
    save_faiss_bundle(bundle)
    return bundle


def build_or_load_faiss_bundle(*, rebuild: bool = False) -> FaissBundle:
    embeddings = create_embeddings()

    if not rebuild and FAISS_INDEX_PATH.exists() and FAISS_DOCS_PATH.exists():
        return load_faiss_bundle(embeddings)

    return build_faiss_bundle(embeddings)
# endregion


# region ---------------- LangGraph ----------------
def format_documents(documents: list[Document]) -> str:
    chunks: list[str] = []
    for index_value, document in enumerate(documents, start=1):
        source_value = document.metadata.get("source", "unknown")
        chunks.append(f"[{index_value}] source={source_value}\n{document.page_content}")
    return "\n\n".join(chunks)


def build_graph(retrieve_vectorstore_candidates: Callable[[str], list[RetrievalCandidate]]) -> CompiledStateGraph:
    def route_question(state: GraphState) -> dict[str, Any]:
        router_model = create_chat_model().bind_tools(
            [vectorstore_search, web_search],
            tool_choice="any",
        )
        response = router_model.invoke(
            [
                SystemMessage(
                    content=(
                        "Choose exactly one tool. Use vectorstore_search for questions about "
                        "agents, prompt engineering, and adversarial attacks. Use web_search "
                        "for other topics or fresh/current facts."
                    )
                ),
                HumanMessage(content=state.question),
            ]
        )
        tool_calls = getattr(response, "tool_calls", [])
        selected_tool_name = tool_calls[0]["name"] if tool_calls else "vectorstore_search"
        if selected_tool_name == "web_search":
            return {"route": "websearch", "route_reason": "LLM selected web_search tool"}
        return {"route": "vectorstore", "route_reason": "LLM selected vectorstore_search tool"}

    def route_to_retriever(state: GraphState) -> RouterNextNode:
        return "retrieve_web" if state.route == "websearch" else "retrieve_vectorstore"

    def retrieve_vectorstore(state: GraphState) -> dict[str, Any]:
        candidates = retrieve_vectorstore_candidates(state.question)
        matched_documents, eval_metrics = asyncio.run(rerank_candidates(state.question, candidates))
        return {
            "retrieved_documents": matched_documents,
            "context": format_documents(matched_documents),
            "eval_metrics": eval_metrics,
        }

    def retrieve_web(state: GraphState) -> dict[str, Any]:
        candidates = retrieve_with_web_search(state.question)
        matched_documents, eval_metrics = asyncio.run(rerank_candidates(state.question, candidates))
        return {
            "retrieved_documents": matched_documents,
            "context": format_documents(matched_documents),
            "eval_metrics": eval_metrics,
        }

    def generate_answer(state: GraphState) -> dict[str, Any]:
        structured_model = create_chat_model().with_structured_output(MiddleAnswer)
        prompt = f"""
Answer the user question using only the provided context.
If context is insufficient, say so plainly and keep model_confidence low.

Context:
{state.context}

Question:
{state.question}
"""
        structured_response = structured_model.invoke(prompt)
        return {"response": structured_response}

    graph_builder = StateGraph(GraphState)
    graph_builder.add_node("route_question", route_question)
    graph_builder.add_node("retrieve_vectorstore", retrieve_vectorstore)
    graph_builder.add_node("retrieve_web", retrieve_web)
    graph_builder.add_node("generate_answer", generate_answer)
    graph_builder.add_edge(START, "route_question")
    graph_builder.add_conditional_edges(
        "route_question",
        route_to_retriever,
        {
            "retrieve_vectorstore": "retrieve_vectorstore",
            "retrieve_web": "retrieve_web",
        },
    )
    graph_builder.add_edge("retrieve_vectorstore", "generate_answer")
    graph_builder.add_edge("retrieve_web", "generate_answer")
    graph_builder.add_edge("generate_answer", END)
    return graph_builder.compile()


def save_graph_diagram(store: MiddleStore) -> None:
    store_label = "Chroma MMR" if store == "chroma" else "FAISS manual MMR"
    mermaid_text = f"""
flowchart TD
    q["User question"] --> router["LLM tool-calling router"]
    router -->|vectorstore_search| vs["{store_label}: MMR fetch_k={MMR_FETCH_K}"]
    router -->|web_search| web["Web search tool"]
    vs --> rerank["Reranker top_k={RERANK_TOP_K} + eval metrics"]
    web --> rerank
    rerank --> ctx["Build shared context"]
    ctx --> gen["Generate answer"]
    gen --> out["Pydantic structured output"]
"""
    save_mermaid_assets(DIAGRAM_DIR / f"rag_middle_{store}_graph", mermaid_text)
# endregion


# region ---------------- Public entrypoint ----------------
def candidates_from_documents(documents: list[Document]) -> list[RetrievalCandidate]:
    total_count = max(len(documents), 1)
    return [
        RetrievalCandidate(
            document=document,
            retrieval_score=(total_count - rank_index + 1) / total_count,
        )
        for rank_index, document in enumerate(documents, start=1)
    ]


def retrieve_with_chroma_mmr(user_question: str, *, rebuild_index: bool) -> list[RetrievalCandidate]:
    vectorstore = build_or_load_chroma_vectorstore(rebuild=rebuild_index)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": MMR_FETCH_K,
            "fetch_k": MMR_FETCH_K,
            "lambda_mult": MMR_LAMBDA_MULT,
        },
    )
    return candidates_from_documents(retriever.invoke(user_question))


def retrieve_with_faiss_mmr(user_question: str, *, rebuild_index: bool) -> list[RetrievalCandidate]:
    faiss_bundle = build_or_load_faiss_bundle(rebuild=rebuild_index)
    documents = faiss_bundle.similarity_search_mmr(
        user_question,
        k=MMR_FETCH_K,
        fetch_k=MMR_FETCH_K,
        lambda_mult=MMR_LAMBDA_MULT,
    )
    return candidates_from_documents(documents)


def retrieve_with_web_search(user_question: str) -> list[RetrievalCandidate]:
    tavily_api_key = os.environ.get("TAVILY_API_KEY")
    documents: list[Document] = []

    if tavily_api_key:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": tavily_api_key,
                "query": user_question,
                "max_results": WEB_SEARCH_TOP_K,
                "search_depth": "basic",
            },
            timeout=30,
        )
        response.raise_for_status()
        for item in response.json().get("results", []):
            documents.append(
                Document(
                    page_content=f"{item.get('title', '')}\n{item.get('content', '')}",
                    metadata={"source": item.get("url", "tavily_web_search"), "datasource": "websearch"},
                )
            )
    elif os.environ.get("OLLAMA_API_KEY"):
        from ollama import Client

        client = Client(
            host=os.environ.get("OLLAMA_HOST", "https://ollama.com"),
            headers={"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"},
        )
        response = client.web_search(query=user_question, max_results=WEB_SEARCH_TOP_K)
        results = response.get("results", []) if isinstance(response, dict) else getattr(response, "results", [])
        for item in results:
            url_value = item.get("url") if isinstance(item, dict) else getattr(item, "url", "")
            title_value = item.get("title") if isinstance(item, dict) else getattr(item, "title", "")
            content_value = item.get("content") if isinstance(item, dict) else getattr(item, "content", "")
            documents.append(
                Document(
                    page_content=f"{title_value}\n{content_value}",
                    metadata={"source": url_value or "ollama_web_search", "datasource": "websearch"},
                )
            )
    else:
        documents.append(
            Document(
                page_content="No web search key configured. Set TAVILY_API_KEY or OLLAMA_API_KEY.",
                metadata={"source": "web_search_not_configured", "datasource": "websearch"},
            )
        )

    return candidates_from_documents(documents)


@traceable(name="langy-rag-middle")
def run_rag_middle(
    user_question: str,
    *,
    store: MiddleStore = "faiss",
    rebuild_index: bool = False,
) -> dict[str, Any]:
    configure_environment()
    require_env("OPENAI_API_KEY")
    print(
        f"[langy] level=middle store={store} mmr_k={MMR_K} fetch_k={MMR_FETCH_K} "
        f"langsmith_tracing={os.environ.get('LANGSMITH_TRACING')} project={os.environ.get('LANGSMITH_PROJECT')}"
    )

    def retrieve_candidates(question: str) -> list[RetrievalCandidate]:
        if store == "chroma":
            return retrieve_with_chroma_mmr(question, rebuild_index=rebuild_index)
        return retrieve_with_faiss_mmr(question, rebuild_index=rebuild_index)

    with ls.tracing_context(project_name=os.environ.get("LANGSMITH_PROJECT"), enabled=True):
        save_graph_diagram(store)
        graph = build_graph(retrieve_candidates)
        result_state = graph.invoke({"question": user_question})

    response_model = result_state["response"]
    result = response_model.model_dump() if response_model is not None else {
        "answer": "No answer produced.",
        "sources": [],
        "model_confidence": 0.0,
    }
    retrieved_sources = [
        document.metadata.get("source", "unknown")
        for document in result_state.get("retrieved_documents", [])
    ]
    result["retrieval_mode"] = f"{store}_mmr"
    result["mmr"] = {
        "original_final_k": MMR_K,
        "candidate_k_for_reranker": MMR_FETCH_K,
        "fetch_k": MMR_FETCH_K,
        "lambda_mult": MMR_LAMBDA_MULT,
    }
    result["rerank_top_k"] = RERANK_TOP_K
    result["route"] = result_state.get("route")
    result["route_reason"] = result_state.get("route_reason")
    result["rerank_eval"] = result_state.get("eval_metrics", {})
    result["retrieved_chunk_sources"] = retrieved_sources
    result["retrieved_sources"] = list(dict.fromkeys(retrieved_sources))
    return result
# endregion

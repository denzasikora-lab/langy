"""Middle RAG: уровень, где появляется настоящая RAG-архитектура.

Стек:
- level 21: Chroma vector store + MMR retrieval.
- level 22: FAISS vector store + ручной MMR retrieval.
- BAAI/bge-m3: embeddings на 1024 измерения.
- LangGraph: явный граф шагов retrieve -> generate.
- Pydantic: структурированный ответ модели.
- LangSmith: tracing включается только в этом файле.
- OpenAI-compatible chat model через init_chat_model.

Архитектура:
- скачали страницы -> нарезали с overlap -> сохранили в Chroma/FAISS;
- MMR берет fetch_k=10 кандидатов и выбирает k=3 более разных чанка;
- lambda_mult=0.5 держит баланс между similarity и diversity;
- LangGraph передает состояние между узлами retrieve_context и generate_answer;
- LLM возвращает ответ в Pydantic-схему MiddleAnswer.
"""

import json
import os
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

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MMR_K = 3
MMR_FETCH_K = 10
MMR_LAMBDA_MULT = 0.5
BGE_M3_MODEL = "BAAI/bge-m3"
MiddleStore = Literal["chroma", "faiss"]
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
    retrieved_documents: list[Document] = Field(default_factory=list)
    context: str = ""
    response: MiddleAnswer | None = None


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
    # MMR - это режим поиска: балансируем близость к вопросу и непохожесть выбранных чанков.
    if not selected_positions:
        diversity_penalty = 0.0
    else:
        selected_vectors = candidate_vectors[selected_positions]
        diversity_penalty = float(np.max(selected_vectors @ candidate_vectors[position]))
    return float(lambda_mult * query_scores[position] - (1.0 - lambda_mult) * diversity_penalty)


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


def build_graph(retrieve_documents: Callable[[str], list[Document]]) -> CompiledStateGraph:
    def retrieve_context(state: GraphState) -> dict[str, Any]:
        matched_documents = retrieve_documents(state.question)
        context_text = format_documents(matched_documents)
        return {"retrieved_documents": matched_documents, "context": context_text}

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
    graph_builder.add_node("retrieve_context", retrieve_context)
    graph_builder.add_node("generate_answer", generate_answer)
    graph_builder.add_edge(START, "retrieve_context")
    graph_builder.add_edge("retrieve_context", "generate_answer")
    graph_builder.add_edge("generate_answer", END)
    return graph_builder.compile()


def save_graph_diagram(store: MiddleStore) -> None:
    store_label = "Chroma MMR" if store == "chroma" else "FAISS manual MMR"
    mermaid_text = f"""
flowchart TD
    q["User question"] --> r["{store_label}: fetch_k={MMR_FETCH_K}"]
    r --> m["MMR selects k={MMR_K} diverse chunks"]
    m --> c["Build context"]
    c --> g["LangGraph generate_answer"]
    g --> p["Pydantic structured output"]
"""
    save_mermaid_assets(DIAGRAM_DIR / f"rag_middle_{store}_graph", mermaid_text)
# endregion


# region ---------------- Public entrypoint ----------------
def retrieve_with_chroma_mmr(user_question: str, *, rebuild_index: bool) -> list[Document]:
    vectorstore = build_or_load_chroma_vectorstore(rebuild=rebuild_index)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": MMR_K,
            "fetch_k": MMR_FETCH_K,
            "lambda_mult": MMR_LAMBDA_MULT,
        },
    )
    return retriever.invoke(user_question)


def retrieve_with_faiss_mmr(user_question: str, *, rebuild_index: bool) -> list[Document]:
    faiss_bundle = build_or_load_faiss_bundle(rebuild=rebuild_index)
    return faiss_bundle.similarity_search_mmr(
        user_question,
        k=MMR_K,
        fetch_k=MMR_FETCH_K,
        lambda_mult=MMR_LAMBDA_MULT,
    )


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

    def retrieve_documents(question: str) -> list[Document]:
        if store == "chroma":
            return retrieve_with_chroma_mmr(question, rebuild_index=rebuild_index)
        return retrieve_with_faiss_mmr(question, rebuild_index=rebuild_index)

    with ls.tracing_context(project_name=os.environ.get("LANGSMITH_PROJECT"), enabled=True):
        save_graph_diagram(store)
        graph = build_graph(retrieve_documents)
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
    result["mmr"] = {"k": MMR_K, "fetch_k": MMR_FETCH_K, "lambda_mult": MMR_LAMBDA_MULT}
    result["retrieved_chunk_sources"] = retrieved_sources
    result["retrieved_sources"] = list(dict.fromkeys(retrieved_sources))
    return result
# endregion

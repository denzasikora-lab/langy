import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
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
FAISS_DIR = PROJECT_DIR / ".faiss" / "rag_middle"
FAISS_INDEX_PATH = FAISS_DIR / "index.faiss"
FAISS_DOCS_PATH = FAISS_DIR / "documents.json"

SOURCE_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 4
BGE_M3_MODEL = "BAAI/bge-m3"
# endregion


# region ---------------- State and schemas ----------------
class MiddleAnswer(BaseModel):
    answer: str = Field(description="Grounded answer to the user question.")
    sources: list[str] = Field(description="List of source URLs used for the answer.")
    confidence: float = Field(description="Confidence score from 0 to 1.", ge=0.0, le=1.0)


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

    def similarity_search(self, query_text: str, k: int) -> list[Document]:
        query_vector = np.array([self.embeddings.embed_query(query_text)], dtype="float32")
        faiss.normalize_L2(query_vector)
        _, matched_indices = self.index.search(query_vector, k)

        matched_documents: list[Document] = []
        for doc_index in matched_indices[0]:
            if 0 <= doc_index < len(self.documents):
                matched_documents.append(self.documents[doc_index])
        return matched_documents
# endregion


# region ---------------- Helpers ----------------
def configure_environment() -> None:
    load_dotenv(str(ENV_PATH))
    os.environ.setdefault("USER_AGENT", "langy-rag-middle/0.1")

    if os.environ.get("LANGSMITH_API_KEY"):
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ.setdefault("LANGSMITH_PROJECT", "langy-rag-middle")
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


# region ---------------- FAISS ----------------
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
    raw_documents = load_web_documents(SOURCE_URLS)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    split_documents = splitter.split_documents(raw_documents)

    document_vectors = np.array(embeddings.embed_documents([doc.page_content for doc in split_documents]), dtype="float32")
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


def build_graph(faiss_bundle: FaissBundle) -> CompiledStateGraph:
    def retrieve_context(state: GraphState) -> dict[str, Any]:
        matched_documents = faiss_bundle.similarity_search(state.question, TOP_K)
        context_text = format_documents(matched_documents)
        return {"retrieved_documents": matched_documents, "context": context_text}

    def generate_answer(state: GraphState) -> dict[str, Any]:
        structured_model = create_chat_model().with_structured_output(MiddleAnswer)
        prompt = f"""
Answer the user question using only the provided context.
If context is insufficient, say so plainly and keep confidence low.

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
# endregion


# region ---------------- Public entrypoint ----------------
@traceable(name="langy-rag-middle")
def run_rag_middle(user_question: str, *, rebuild_index: bool = False) -> dict[str, Any]:
    configure_environment()
    require_env("OPENAI_API_KEY")

    faiss_bundle = build_or_load_faiss_bundle(rebuild=rebuild_index)
    graph = build_graph(faiss_bundle)
    result_state = graph.invoke({"question": user_question})

    response_model = result_state["response"]
    result = response_model.model_dump() if response_model is not None else {
        "answer": "No answer produced.",
        "sources": [],
        "confidence": 0.0,
    }
    result["retrieved_sources"] = [
        document.metadata.get("source", "unknown")
        for document in result_state.get("retrieved_documents", [])
    ]
    return result
# endregion

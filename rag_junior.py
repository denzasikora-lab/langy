"""Junior RAG: the simplest learning level.

Stack:
- InMemoryVectorStore: vectors live only in process memory.
- sentence-transformers/all-MiniLM-L6-v2: local embeddings.
- Ollama via init_chat_model: answer generation.
- JSON without Pydantic: simple manual response validation.

Architecture:
- download pages -> split with default splitter settings, no fine tuning;
- the user question is also converted into a vector;
- the retriever compares the question vector with chunk vectors;
- the LLM-as-a-judge reranker sorts retrieved chunks by usefulness;
- the best chunks go to the LLM as context.
"""

import json
import os
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
    category=UserWarning,
)

import langsmith as ls
import requests
from bs4 import BeautifulSoup
from diagram_utils import save_mermaid_assets
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# region ---------------- Settings ----------------
PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
GRAPH_STEM = PROJECT_DIR / "diagrams" / "rag_junior_graph"

SOURCE_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# endregion


# region ---------------- Helpers ----------------
def configure_environment() -> None:
    load_dotenv(str(ENV_PATH))
    os.environ.setdefault("USER_AGENT", "langy-rag-junior/0.1")
    os.environ["LANGSMITH_TRACING"] = "false"


def require_env(env_name: str) -> str:
    env_value = os.environ.get(env_name)
    if not env_value or env_value == "replace_me":
        raise RuntimeError(f"Missing required environment variable: {env_name}")
    return env_value


def parse_json_response(response_content: Any) -> dict[str, Any]:
    response_text = response_content if isinstance(response_content, str) else str(response_content)
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        json_start_index = response_text.find("{")
        json_end_index = response_text.rfind("}") + 1
        if 0 <= json_start_index < json_end_index:
            try:
                return json.loads(response_text[json_start_index:json_end_index])
            except json.JSONDecodeError:
                pass
    return {"answer": response_text, "sources": []}


def save_junior_diagram() -> None:
    mermaid_text = """
flowchart TD
    q["User question"] --> r["InMemoryVectorStore retriever"]
    r --> c["Default top-k chunks"]
    c --> judge["LLM-as-a-judge reranker"]
    judge --> m["Ollama via init_chat_model"]
    m --> json["JSON parsing"]
"""
    save_mermaid_assets(GRAPH_STEM, mermaid_text)
# endregion


# region ---------------- Indexing ----------------
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
    # The model downloads once to the Hugging Face cache.
    # Each new Python run reloads weights from cache/disk into memory.
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, show_progress=False)


def build_vectorstore() -> VectorStore:
    raw_documents = load_web_documents(SOURCE_URLS)
    # Junior level: keep default chunk_size/chunk_overlap so topics stay separate.
    splitter = RecursiveCharacterTextSplitter()
    split_documents = splitter.split_documents(raw_documents)
    embeddings = create_embeddings()
    return InMemoryVectorStore.from_documents(documents=split_documents, embedding=embeddings)
# endregion


# region ---------------- RAG ----------------
def create_chat_model():
    ollama_headers = {"Authorization": f"Bearer {require_env('OLLAMA_API_KEY')}"}
    return init_chat_model(
        os.environ.get("OLLAMA_MODEL", "gemma4:31b-cloud"),
        model_provider="ollama",
        temperature=0,
        format="json",
        base_url=os.environ.get("OLLAMA_HOST", "https://ollama.com"),
        client_kwargs={"headers": ollama_headers},
    )


def format_documents(documents: list[Document]) -> str:
    chunks: list[str] = []
    for index_value, document in enumerate(documents, start=1):
        source_value = document.metadata.get("source", "unknown")
        chunks.append(f"[{index_value}] source={source_value}\n{document.page_content}")
    return "\n\n".join(chunks)


def rerank_with_llm_judge(
    *,
    llm,
    user_question: str,
    documents: list[Document],
) -> list[Document]:
    scored_documents: list[tuple[float, int, Document]] = []
    for index_value, document in enumerate(documents, start=1):
        judge_prompt = f"""
Question:
{user_question}

Chunk:
{document.page_content[:3000]}

Return JSON only:
{{"score": 0.0, "reason": "..."}}

Score means how useful this chunk is for answering the question.
"""
        response = llm.invoke(
            [
                SystemMessage(content="You are a strict retrieval judge."),
                HumanMessage(content=judge_prompt),
            ]
        )
        score_payload = parse_json_response(response.content)
        try:
            score_value = float(score_payload.get("score", 0.0))
        except (TypeError, ValueError):
            score_value = 0.0
        scored_documents.append((score_value, index_value, document))

    scored_documents.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    return [document for _, _, document in scored_documents]


def run_rag_junior(user_question: str) -> dict[str, Any]:
    configure_environment()
    save_junior_diagram()
    print("[langy] level=1 junior store=in_memory langsmith_tracing=false")

    with ls.tracing_context(enabled=False):
        vectorstore = build_vectorstore()

        # k is intentionally omitted: LangChain retrievers default to k=4.
        retriever = vectorstore.as_retriever()
        retrieved_documents = retriever.invoke(user_question)

        llm = create_chat_model()
        reranked_documents = rerank_with_llm_judge(
            llm=llm,
            user_question=user_question,
            documents=retrieved_documents,
        )

        prompt = f"""
Answer the user question using only the context.
Return JSON only:
{{"answer": "...", "sources": ["..."]}}

Context:
{format_documents(reranked_documents)}

Question:
{user_question}
"""
        response = llm.invoke(
            [
                SystemMessage(content="You are a precise RAG assistant."),
                HumanMessage(content=prompt),
            ]
        )

    result = parse_json_response(response.content)
    retrieved_sources = [document.metadata.get("source", "unknown") for document in reranked_documents]
    result["reranker"] = "llm_as_a_judge"
    result["retrieved_chunk_sources"] = retrieved_sources
    result["retrieved_sources"] = list(dict.fromkeys(retrieved_sources))
    return result
# endregion

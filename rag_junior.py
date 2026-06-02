import json
import os
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStore
# from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# region ---------------- Settings ----------------
PROJECT_DIR = Path(__file__).resolve().parent
ENV_PATH = PROJECT_DIR / ".env"
CHROMA_DIR = PROJECT_DIR / ".chroma" / "rag_junior"
COLLECTION_NAME = "langy_rag_junior"

SOURCE_URLS = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

CHUNK_SIZE = 1000
TOP_K = 3
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# endregion


# region ---------------- Helpers ----------------
def configure_environment() -> None:
    load_dotenv(str(ENV_PATH))
    os.environ.setdefault("USER_AGENT", "langy-rag-junior/0.1")


def require_env(env_name: str) -> str:
    env_value = os.environ.get(env_name)
    if not env_value or env_value == "replace_me":
        raise RuntimeError(f"Missing required environment variable: {env_name}")
    return env_value


def parse_json_response(response_content: Any) -> dict[str, Any]:
    """Junior level: use json.loads + a small fallback instead of Pydantic."""

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
    # The model is downloaded once to the local Hugging Face cache.
    # Each Python run still loads the cached weights from disk into memory.
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, show_progress=False)


def build_or_load_vectorstore(*, rebuild: bool = False) -> VectorStore:
    embeddings = create_embeddings()
    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    if rebuild:
        vectorstore.reset_collection()

    existing_items = vectorstore.get(limit=1, include=[])
    if existing_items.get("ids"):
        return vectorstore

    raw_documents = load_web_documents(SOURCE_URLS)

    # Junior level: no overlap. Overlap appears later in rag_middle.py.
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=0)
    split_documents = splitter.split_documents(raw_documents)

    # Chroma stores document vectors on disk in CHROMA_DIR.
    vectorstore.add_documents(split_documents)

    # Easy switch for an in-memory demo:
    # memory_vectorstore = InMemoryVectorStore.from_documents(
    #     documents=split_documents,
    #     embedding=embeddings,
    # )
    # return memory_vectorstore
    return vectorstore
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


def run_rag_junior(user_question: str, *, rebuild_index: bool = False) -> dict[str, Any]:
    configure_environment()
    vectorstore = build_or_load_vectorstore(rebuild=rebuild_index)

    # The retriever embeds the user question and compares it with vectors stored in Chroma.
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    retrieved_documents = retriever.invoke(user_question)

    prompt = f"""
Answer the user question using only the context.
Return JSON only:
{{"answer": "...", "sources": ["..."]}}

Context:
{format_documents(retrieved_documents)}

Question:
{user_question}
"""
    llm = create_chat_model()
    response = llm.invoke(
        [
            SystemMessage(content="You are a precise RAG assistant."),
            HumanMessage(content=prompt),
        ]
    )

    result = parse_json_response(response.content)
    result["retrieved_sources"] = [
        document.metadata.get("source", "unknown") for document in retrieved_documents
    ]
    return result
# endregion

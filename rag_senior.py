"""Senior RAG: каркас будущего production-подхода.

Стек:
- level 31: будущий Qdrant retrieval.
- level 32: будущий Weaviate retrieval.
- embedding ensembling: несколько embedding-моделей перед индексированием/поиском.
- reranking, evaluation, observability: будущие production-слои.

Архитектура:
- несколько embedding-моделей строят разные векторные представления;
- векторы нормализуются и объединяются;
- поиск идет через Qdrant или Weaviate;
- дальше планируются reranker, фильтры, оценка качества и мониторинг.
"""

from pathlib import Path
from typing import Literal

from diagram_utils import save_mermaid_assets


PROJECT_DIR = Path(__file__).resolve().parent
DIAGRAM_DIR = PROJECT_DIR / "diagrams"
SeniorStore = Literal["qdrant", "weaviate"]


def build_ensemble_embedding_plan() -> list[str]:
    """План для будущего ансамблирования embeddings.

    Идея: прогнать текст через несколько embedders, нормализовать векторы,
    затем concat или weighted-average перед записью в Qdrant/Weaviate.
    """

    return [
        "BAAI/bge-m3",
        "sentence-transformers/all-MiniLM-L6-v2",
        "future-domain-specific-embedder",
    ]


def save_senior_diagram(store: SeniorStore) -> None:
    store_label = "Qdrant" if store == "qdrant" else "Weaviate"
    mermaid_text = f"""
flowchart TD
    q["User question"] --> e["Embedding ensemble"]
    e --> v["{store_label} vector search"]
    v --> r["Reranker / filters"]
    r --> a["Answer synthesis"]
    a --> ev["Evaluation + observability"]
"""
    save_mermaid_assets(DIAGRAM_DIR / f"rag_senior_{store}_graph", mermaid_text)


def run_rag_senior(*, store: SeniorStore) -> None:
    save_senior_diagram(store)
    embedders = ", ".join(build_ensemble_embedding_plan())
    raise NotImplementedError(f"rag_senior.py {store} is planned. Embedding ensemble: {embedders}")

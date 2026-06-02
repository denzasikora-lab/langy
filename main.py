import json
from typing import Any

from rag_junior import run_rag_junior
from rag_middle import run_rag_middle
from rag_senior import run_rag_senior


DEFAULT_QUESTION = "What are the types of agent memory?"
DEFAULT_LEVEL = 1


def run_langy(level: int, question: str = DEFAULT_QUESTION) -> dict[str, Any] | None:
    if level == 1:
        return run_rag_junior(question)
    if level == 2:
        run_rag_middle()
        return None
    if level == 3:
        run_rag_senior()
        return None
    raise ValueError("level must be 1 (junior), 2 (middle), or 3 (senior)")


def main() -> None:
    result = run_langy(DEFAULT_LEVEL)
    if result is not None:
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

import argparse
import json

from rag_simple import run_rag_simple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Langy RAG playground")
    parser.add_argument("question", nargs="?", default="What are the types of agent memory?")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild the local Chroma index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_rag_simple(args.question, rebuild_index=args.rebuild_index)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


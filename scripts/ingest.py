"""Ingest documents into the vector store."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running as `python scripts/ingest.py` without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger

from rag.ingestion.pipeline import ingest_path
from rag.retrieval.vector_store import VectorStore


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw", help="File or directory")
    parser.add_argument("--collection", default=None, help="Override Qdrant collection name")
    parser.add_argument("--recreate", action="store_true", help="Drop & recreate the collection")
    args = parser.parse_args()

    store = VectorStore(collection=args.collection) if args.collection else VectorStore()
    if args.recreate:
        store.ensure_collection(recreate=True)
    else:
        store.ensure_collection()

    summary = ingest_path(Path(args.input), store=store)
    logger.info("Done")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

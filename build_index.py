#!/usr/bin/env python
"""
build_index.py
--------------
Build the searchable index from the documents folder. Run this once before
starting the app, and again whenever the documents change.

    python build_index.py

What it does:
    1. loads every document in documents/            (ingestion.py)
    2. splits them into overlapping chunks           (ingestion.py)
    3. embeds each chunk with the OpenAI API         (vectorstore.py)
    4. saves a FAISS index and a manifest into index/

This is the only step that costs money, and it is a few cents for a corpus this
size. The keyword (BM25) index is built in memory when the app starts, so it
needs nothing here.
"""

from __future__ import annotations

import logging
import sys

import config
import ingestion
import vectorstore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build-index")


def main() -> int:
    """Build and save the index. Returns a process exit code."""
    print("\n" + "=" * 70)
    print("  BUILDING THE KNOWLEDGE INDEX")
    print("=" * 70)

    if not config.OPENAI_API_KEY:
        log.error("No OpenAI API key found.")
        log.error("Create a file named  .env  in this folder containing one line:")
        log.error("    OPENAI_API_KEY=sk-your-key-here")
        return 1

    # --- 1 and 2: load and chunk ------------------------------------------
    try:
        chunks, problems = ingestion.build_corpus()
    except ingestion.IngestionError as error:
        log.error("%s", error)
        return 1

    for problem in problems:
        log.warning("Skipped: %s", problem)

    sources: dict[str, int] = {}
    for chunk in chunks:
        name = chunk.metadata.get("source", "unknown")
        sources[name] = sources.get(name, 0) + 1

    print(f"\n  Documents found : {len(sources)}")
    for name, count in sorted(sources.items()):
        print(f"      {name:<36} {count:>3} chunks")
    print(f"  Total chunks    : {len(chunks)}")
    print(f"  Chunk size      : {config.CHUNK_SIZE} chars, "
          f"{config.CHUNK_OVERLAP} overlap")

    # --- 3 and 4: embed and save ------------------------------------------
    print(f"\n  Embedding with {config.EMBEDDING_MODEL} ...")
    try:
        store = vectorstore.build_index(chunks)
        directory = vectorstore.save_index(store, chunks)
    except vectorstore.VectorStoreError as error:
        log.error("%s", error)
        return 1

    print(f"  Index saved to  : {directory}")
    print("\n  Done. Start the application with:  streamlit run app.py\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

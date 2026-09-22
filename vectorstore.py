"""
vectorstore.py
--------------
Stage 2 of the pipeline: embed the chunks and store them in a local FAISS index.

FAISS was chosen over Chroma because it needs no server and no background
process - the whole index is two files on disk, which suits a local application
and makes it easy to rebuild from scratch.

The index is NOT committed to the repository. It is built once with

    python build_index.py

and rebuilt whenever the documents change.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document

import config

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"


class VectorStoreError(RuntimeError):
    """Raised when the index cannot be built, saved or loaded."""


def build_embeddings():
    """Create the OpenAI embedding model.

    Raises:
        VectorStoreError: if no API key is configured. Embedding is the one step
            that cannot be done offline, so this is checked early and clearly.
    """
    if not config.OPENAI_API_KEY:
        raise VectorStoreError(
            "No OpenAI API key found. Create a file named .env next to app.py "
            "containing: OPENAI_API_KEY=sk-your-key-here"
        )
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        api_key=config.OPENAI_API_KEY,
    )


def build_index(chunks: list[Document]):
    """Embed every chunk and build a FAISS index from them.

    Args:
        chunks: The output of `ingestion.chunk_documents`.

    Returns:
        The populated FAISS vector store.

    Raises:
        VectorStoreError: if there is nothing to index, or embedding fails.
    """
    if not chunks:
        raise VectorStoreError("There are no chunks to index.")

    from langchain_community.vectorstores import FAISS

    embeddings = build_embeddings()
    log.info("Embedding %d chunk(s) with %s ...", len(chunks), config.EMBEDDING_MODEL)
    try:
        store = FAISS.from_documents(chunks, embeddings)
    except Exception as error:  # noqa: BLE001 - normalised for the caller
        raise VectorStoreError(f"The index could not be built: {error}") from error

    log.info("Index built.")
    return store


def save_index(store, chunks: list[Document], directory: Path | None = None) -> Path:
    """Write the index to disk together with a small manifest.

    The manifest records what went into the index - which files, how many
    chunks, when it was built - so the UI can show it and so a stale index is
    obvious.
    """
    directory = directory or config.INDEX_DIR
    directory.mkdir(parents=True, exist_ok=True)

    try:
        store.save_local(str(directory))
    except Exception as error:  # noqa: BLE001
        raise VectorStoreError(f"The index could not be saved: {error}") from error

    sources: dict[str, int] = {}
    for chunk in chunks:
        name = chunk.metadata.get("source", "unknown")
        sources[name] = sources.get(name, 0) + 1

    manifest = {
        "built_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "embedding_model": config.EMBEDDING_MODEL,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "chunk_count": len(chunks),
        "document_count": len(sources),
        "sources": sources,
    }
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return directory


def index_exists(directory: Path | None = None) -> bool:
    """Return True when a built index is present on disk."""
    directory = directory or config.INDEX_DIR
    return (directory / "index.faiss").exists() or (directory / MANIFEST_NAME).exists()


def load_index(directory: Path | None = None):
    """Load the FAISS index from disk.

    Raises:
        VectorStoreError: if the index is missing or cannot be read.
    """
    directory = directory or config.INDEX_DIR
    if not index_exists(directory):
        raise VectorStoreError(
            f"No index found in {directory}. Build one first with: python build_index.py"
        )

    from langchain_community.vectorstores import FAISS

    embeddings = build_embeddings()
    try:
        # The index is a local file this application wrote itself, so
        # deserialisation is safe here.
        return FAISS.load_local(
            str(directory), embeddings, allow_dangerous_deserialization=True
        )
    except Exception as error:  # noqa: BLE001
        raise VectorStoreError(
            f"The index could not be loaded: {error}. Try rebuilding it with "
            "python build_index.py"
        ) from error


def load_manifest(directory: Path | None = None) -> dict:
    """Return the index manifest, or an empty dict if there is none."""
    directory = directory or config.INDEX_DIR
    path = directory / MANIFEST_NAME
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}

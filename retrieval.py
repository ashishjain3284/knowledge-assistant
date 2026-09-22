"""
retrieval.py
------------
Stage 3 of the pipeline: hybrid search.

                    ┌── vector search (FAISS + embeddings) ──┐
    question ───────┤                                        ├── RRF ── candidates
                    └── keyword search (BM25)  ──────────────┘

Why both? They fail in opposite ways.

  * Vector search understands meaning. Ask "can I take my laptop abroad?" and it
    finds the remote-working clause even though it never uses the word laptop.
    But it is weak on exact tokens: a question about "TKT-1001" or "0.45 per
    kilometre" can miss, because the number carries little semantic signal.
  * BM25 is exact. It nails rare words, codes and figures. But it finds nothing
    when the user's words differ from the document's.

Fusing the two covers both cases. BM25 is implemented here in about forty lines
rather than pulled from a library, because the scoring is the part worth
showing.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field

from langchain_core.documents import Document

import config

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #
_WORD = re.compile(r"[a-z0-9][a-z0-9.\-]*")
_STOPWORDS = frozenset(
    """a an and any are as at be by can do does for from get give has have how i
    if in into is it its may me my not of on or our shall should so than that the
    their them then there these they this to was we what when where which who why
    will with would you your""".split()
)


def tokenise(text: str) -> list[str]:
    """Lowercase, split into words and drop stop words.

    Numbers and hyphenated tokens are kept whole ("0.45", "carry-forward"),
    because they are exactly the terms BM25 is good at matching.
    """
    return [
        token.strip(".")
        for token in _WORD.findall(text.lower())
        if token not in _STOPWORDS and len(token.strip(".")) > 1
    ]


# --------------------------------------------------------------------------- #
# BM25
# --------------------------------------------------------------------------- #
class BM25Index:
    """A small BM25 (Okapi) keyword index over the chunk corpus.

    BM25 scores a document for a query as the sum, over the query terms, of

        idf(term) * (f * (k1 + 1)) / (f + k1 * (1 - b + b * len / avg_len))

    where ``f`` is how often the term appears in the document. The ``k1`` term
    makes the score saturate, so a word appearing ten times is not worth ten
    times a word appearing once. The ``b`` term normalises for length, so a long
    document does not win simply by containing more words.
    """

    def __init__(self, chunks: list[Document], k1: float = 1.5, b: float = 0.75) -> None:
        """Build the index.

        Args:
            chunks: The corpus, one Document per chunk.
            k1: Term-frequency saturation. Higher rewards repetition more.
            b: Length normalisation, 0 = none, 1 = full.
        """
        self.chunks = chunks
        self.k1 = k1
        self.b = b

        self.tokenised = [tokenise(f"{c.metadata.get('title', '')} {c.page_content}")
                          for c in chunks]
        self.lengths = [len(t) for t in self.tokenised]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0

        self.frequencies = [Counter(t) for t in self.tokenised]

        document_frequency: Counter[str] = Counter()
        for tokens in self.tokenised:
            document_frequency.update(set(tokens))

        total = max(len(chunks), 1)
        self.idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    def search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Return the ``k`` highest-scoring chunks for the query."""
        terms = tokenise(query)
        if not terms or not self.chunks:
            return []

        scored: list[tuple[float, int]] = []
        for index, frequencies in enumerate(self.frequencies):
            length = self.lengths[index] or 1
            score = 0.0
            for term in terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                idf = self.idf.get(term, 0.0)
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * length / (self.avg_length or 1)
                )
                score += idf * (frequency * (self.k1 + 1)) / denominator
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [(self.chunks[i], round(s, 4)) for s, i in scored[:k]]


# --------------------------------------------------------------------------- #
# Candidates and fusion
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    """One retrieved chunk, with a record of how it was found.

    Keeping the per-retriever ranks is what lets the UI show *why* a chunk was
    considered, rather than presenting retrieval as a black box.
    """

    document: Document
    vector_rank: int | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """The stable identifier assigned during chunking."""
        return self.document.metadata.get("chunk_id", "")

    @property
    def source(self) -> str:
        """The file name this chunk came from."""
        return self.document.metadata.get("source", "unknown")

    def found_by(self) -> str:
        """A short human-readable description of which retrievers found this."""
        parts = []
        if self.vector_rank is not None:
            parts.append(f"vector #{self.vector_rank}")
        if self.bm25_rank is not None:
            parts.append(f"BM25 #{self.bm25_rank}")
        return " + ".join(parts) or "unknown"


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], k: int = None
) -> dict[str, float]:
    """Fuse several ranked lists into one score per chunk.

    Reciprocal Rank Fusion gives each chunk ``1 / (k + rank)`` from every list it
    appears in, and adds those up. It needs no score normalisation, which is the
    whole point: a cosine similarity and a BM25 score are not comparable
    numbers, but their *ranks* are.

    A chunk that both retrievers rank highly wins; a chunk that only one of them
    found can still get through on the strength of a top position.

    Args:
        ranked_lists: One list of Documents per retriever, best first.
        k: The RRF damping constant (default from config).

    Returns:
        A mapping of chunk_id to fused score.
    """
    k = k if k is not None else config.RRF_K
    scores: dict[str, float] = {}
    for documents in ranked_lists:
        for rank, document in enumerate(documents, start=1):
            chunk_id = document.metadata.get("chunk_id", "")
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    """Runs vector search and BM25 side by side and fuses the results."""

    def __init__(self, vector_store, bm25: BM25Index) -> None:
        """Wire the two retrievers together.

        Args:
            vector_store: A FAISS store exposing ``similarity_search_with_score``.
            bm25: The keyword index built over the same chunks.
        """
        self.vector_store = vector_store
        self.bm25 = bm25

    def _vector_search(self, query: str, k: int) -> list[tuple[Document, float]]:
        """Run semantic search, returning an empty list if the store fails."""
        try:
            return self.vector_store.similarity_search_with_score(query, k=k)
        except Exception as error:  # noqa: BLE001 - degrade to keyword-only
            log.error("Vector search failed, continuing with BM25 only: %s", error)
            return []

    def retrieve(self, query: str, limit: int | None = None) -> list[Candidate]:
        """Return fused candidates for a query, best first.

        If one retriever fails or returns nothing, the other still produces
        results - the system degrades rather than breaking.
        """
        limit = limit or config.HYBRID_CANDIDATES

        vector_hits = self._vector_search(query, config.VECTOR_TOP_K)
        bm25_hits = self.bm25.search(query, config.BM25_TOP_K)

        candidates: dict[str, Candidate] = {}

        for rank, (document, distance) in enumerate(vector_hits, start=1):
            chunk_id = document.metadata.get("chunk_id", f"v{rank}")
            candidates[chunk_id] = Candidate(
                document=document,
                vector_rank=rank,
                extra={"vector_distance": round(float(distance), 4)},
            )

        for rank, (document, score) in enumerate(bm25_hits, start=1):
            chunk_id = document.metadata.get("chunk_id", f"b{rank}")
            if chunk_id in candidates:
                candidates[chunk_id].bm25_rank = rank
                candidates[chunk_id].bm25_score = score
            else:
                candidates[chunk_id] = Candidate(
                    document=document, bm25_rank=rank, bm25_score=score
                )

        fused = reciprocal_rank_fusion(
            [[d for d, _ in vector_hits], [d for d, _ in bm25_hits]]
        )
        for chunk_id, candidate in candidates.items():
            candidate.rrf_score = round(fused.get(chunk_id, 0.0), 6)

        ordered = sorted(
            candidates.values(), key=lambda c: (-c.rrf_score, c.chunk_id)
        )
        log.info(
            "Hybrid retrieval: %d vector + %d BM25 -> %d unique, keeping %d",
            len(vector_hits), len(bm25_hits), len(ordered), min(limit, len(ordered)),
        )
        return ordered[:limit]

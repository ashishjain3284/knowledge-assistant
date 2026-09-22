"""
reranker.py
-----------
Stage 4 of the pipeline: reranking.

    query → retrieval → candidates → RERANKING → final context → LLM

Retrieval optimises for recall: cast a wide net so the right passage is
somewhere in the top ten. Reranking optimises for precision: read those ten
properly and keep only the few that actually answer the question.

The difference matters because context is not free. Feeding ten passages to the
LLM when two are relevant makes the answer worse, not better - the model has to
decide what to ignore, and it sometimes gets that wrong.

This implementation scores the candidates with the LLM itself. A cross-encoder
such as ms-marco-MiniLM would be the classic choice and is faster per query, but
it pulls in PyTorch and a model download; using the model already configured
keeps the project installable with one pip command. The interface below would
not change if you swapped one in.
"""

from __future__ import annotations

import json
import logging
import re

import config
from prompts import RERANK_PROMPT
from retrieval import Candidate

log = logging.getLogger(__name__)

#: How much of each passage the reranker sees. Enough to judge relevance
#: without paying for the whole chunk.
PASSAGE_PREVIEW_CHARS = 700


def _format_passages(candidates: list[Candidate]) -> str:
    """Render the candidates as a numbered list for the scoring prompt."""
    blocks = []
    for number, candidate in enumerate(candidates, start=1):
        meta = candidate.document.metadata
        location = meta.get("section") or (
            f"page {meta['page']}" if meta.get("page") else ""
        )
        header = f"[{number}] {meta.get('source', 'unknown')}"
        if location:
            header += f" - {location}"
        text = candidate.document.page_content[:PASSAGE_PREVIEW_CHARS].strip()
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


def _parse_scores(raw: str, expected: int) -> dict[int, float]:
    """Read the model's JSON scores, tolerating the usual formatting noise.

    Models sometimes wrap JSON in a code fence or add a sentence before it, so
    the array is extracted with a regex rather than parsed strictly.
    """
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise ValueError("no JSON array found in the reranker response")

    parsed = json.loads(match.group(0))
    scores: dict[int, float] = {}
    for item in parsed:
        try:
            identifier = int(item["id"])
            score = float(item["score"])
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= identifier <= expected:
            scores[identifier] = max(0.0, min(10.0, score))
    if not scores:
        raise ValueError("the reranker returned no usable scores")
    return scores


def rerank(
    llm,
    question: str,
    candidates: list[Candidate],
    top_n: int | None = None,
    minimum_score: float | None = None,
) -> tuple[list[Candidate], dict]:
    """Score the candidates against the question and keep the best.

    Args:
        llm: Anything with ``.invoke(prompt) -> message`` - the chat model.
        question: The standalone question.
        candidates: Output of the hybrid retriever.
        top_n: How many to keep (default from config).
        minimum_score: Candidates scoring below this are dropped entirely.

    Returns:
        A ``(kept, info)`` tuple. ``info`` records whether reranking actually
        ran, so the UI can be honest about it.

    Never raises: if the model is unavailable or returns nonsense, the original
    hybrid order is kept and the failure is reported in ``info``.
    """
    top_n = top_n or config.RERANK_TOP_N
    minimum_score = minimum_score if minimum_score is not None else config.MIN_RELEVANCE_SCORE

    if not candidates:
        return [], {"applied": False, "reason": "no candidates to rerank"}

    prompt = RERANK_PROMPT.format(
        question=question, passages=_format_passages(candidates)
    )

    try:
        response = llm.invoke(prompt)
        raw = getattr(response, "content", str(response))
        scores = _parse_scores(raw, expected=len(candidates))
    except Exception as error:  # noqa: BLE001 - reranking is an optimisation
        log.warning("Reranking failed, falling back to the hybrid order: %s", error)
        fallback = candidates[:top_n]
        return fallback, {
            "applied": False,
            "reason": f"reranker unavailable ({type(error).__name__}), kept hybrid order",
            "scored": 0,
        }

    for number, candidate in enumerate(candidates, start=1):
        candidate.rerank_score = scores.get(number)

    scored = [c for c in candidates if c.rerank_score is not None]
    scored.sort(key=lambda c: (-(c.rerank_score or 0), c.chunk_id))

    relevant = [c for c in scored if (c.rerank_score or 0) >= minimum_score]
    kept = relevant[:top_n]

    info = {
        "applied": True,
        "scored": len(scored),
        "kept": len(kept),
        "dropped_as_irrelevant": len(scored) - len(relevant),
        "minimum_score": minimum_score,
        "best_score": scored[0].rerank_score if scored else None,
    }
    log.info(
        "Reranked %d candidate(s): kept %d, dropped %d below the threshold",
        len(scored), len(kept), info["dropped_as_irrelevant"],
    )
    return kept, info

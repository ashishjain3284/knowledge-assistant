"""
rag_chain.py
------------
The pipeline that ties everything together.

    question
       ↓  condense against the conversation history        (memory.py)
    standalone question
       ↓  vector search + BM25, fused with RRF             (retrieval.py)
    ~10 candidates
       ↓  rerank and drop anything irrelevant              (reranker.py)
    up to 4 passages
       ↓  is anything relevant left?  ── no ──▶ "not found", no LLM call
       ↓ yes
    build context + history
       ↓  generate                                          (prompts.py)
    grounded answer + citations

Two design points worth noting.

First, the "not found" path never calls the answering LLM. If reranking finds
nothing above the relevance threshold, the application says so directly. You
cannot hallucinate an answer you never asked for.

Second, every answer comes back with a full trace - the rewritten question, what
each retriever found, the rerank scores, the passages actually used. The UI
shows it. A RAG system that cannot show its working is impossible to debug and
hard to trust.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import config
from memory import ConversationMemory, condense_question
from prompts import ANSWER_PROMPT, NOT_FOUND_MESSAGE
from reranker import rerank
from retrieval import BM25Index, Candidate, HybridRetriever

log = logging.getLogger(__name__)


class AssistantError(RuntimeError):
    """Raised when a question cannot be answered at all. The UI shows this."""


@dataclass
class Source:
    """One document cited in an answer."""

    source: str
    title: str
    section: str = ""
    page: int | None = None
    score: float | None = None
    snippet: str = ""

    def label(self) -> str:
        """A short human-readable location, for example 'page 2' or a heading."""
        if self.section:
            return self.section
        if self.page:
            return f"page {self.page}"
        return ""


@dataclass
class Answer:
    """Everything one question produced."""

    text: str
    sources: list[Source] = field(default_factory=list)
    found: bool = True
    trace: dict[str, Any] = field(default_factory=dict)


def build_llm():
    """Create the chat model used for condensing, reranking and answering.

    Raises:
        AssistantError: if no API key is configured.
    """
    if not config.OPENAI_API_KEY:
        raise AssistantError(
            "No OpenAI API key found. Create a file named .env next to app.py "
            "containing: OPENAI_API_KEY=sk-your-key-here"
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=config.CHAT_MODEL,
        api_key=config.OPENAI_API_KEY,
        temperature=config.TEMPERATURE,
    )


def _call_with_retry(llm, prompt: str) -> str:
    """Invoke the model, retrying briefly on a transient failure.

    Raises:
        AssistantError: when every attempt fails.
    """
    last_error: Exception | None = None
    for attempt in range(1, config.LLM_MAX_ATTEMPTS + 1):
        try:
            response = llm.invoke(prompt)
            return getattr(response, "content", str(response))
        except Exception as error:  # noqa: BLE001 - normalised below
            last_error = error
            log.warning(
                "LLM call failed (attempt %d of %d): %s",
                attempt, config.LLM_MAX_ATTEMPTS, error,
            )
            if attempt < config.LLM_MAX_ATTEMPTS:
                time.sleep(2**attempt)  # 2s, then 4s
    raise AssistantError(f"Could not reach the language model: {last_error}")


def build_context(candidates: list[Candidate], max_chars: int | None = None) -> str:
    """Format the chosen passages for the answer prompt.

    Each passage is labelled with its file name, because that label is what the
    model is instructed to cite. The total is capped so an unusually long set of
    chunks cannot blow up the request.
    """
    max_chars = max_chars or config.MAX_CONTEXT_CHARS
    blocks: list[str] = []
    used = 0

    for candidate in candidates:
        meta = candidate.document.metadata
        location = meta.get("section") or (
            f"page {meta['page']}" if meta.get("page") else ""
        )
        header = f"--- {meta.get('source', 'unknown')}"
        if location:
            header += f" ({location})"
        header += " ---"

        block = f"{header}\n{candidate.document.page_content.strip()}"
        if used + len(block) > max_chars and blocks:
            log.info("Context cap reached; using %d of %d passages",
                     len(blocks), len(candidates))
            break
        blocks.append(block)
        used += len(block)

    return "\n\n".join(blocks)


def build_sources(candidates: list[Candidate]) -> list[Source]:
    """Turn the used passages into a de-duplicated citation list.

    One entry per document, keeping the best-scoring passage from each - an
    answer drawn from three chunks of the same policy should cite it once.
    """
    best: dict[str, Source] = {}
    for candidate in candidates:
        meta = candidate.document.metadata
        name = meta.get("source", "unknown")
        score = candidate.rerank_score
        existing = best.get(name)
        if existing is not None and (existing.score or 0) >= (score or 0):
            continue
        snippet = " ".join(candidate.document.page_content.split())[:280]
        best[name] = Source(
            source=name,
            title=meta.get("title", name),
            section=meta.get("section", ""),
            page=meta.get("page"),
            score=score,
            snippet=snippet + ("..." if len(snippet) == 280 else ""),
        )
    return sorted(best.values(), key=lambda s: -(s.score or 0))


class KnowledgeAssistant:
    """The full RAG application: retrieval, reranking, memory and generation."""

    def __init__(self, vector_store, chunks: list, llm=None,
                 memory: ConversationMemory | None = None) -> None:
        """Assemble the assistant.

        Args:
            vector_store: A loaded FAISS index.
            chunks: The same chunks the index was built from, used for BM25.
            llm: The chat model; built from config when omitted.
            memory: Conversation memory; a fresh one when omitted.
        """
        self.llm = llm or build_llm()
        self.bm25 = BM25Index(chunks)
        self.retriever = HybridRetriever(vector_store, self.bm25)
        self.memory = memory or ConversationMemory()
        self.chunk_count = len(chunks)

    # ----------------------------------------------------------------- API --
    def ask(self, question: str) -> Answer:
        """Answer one question and record it in the conversation memory.

        Raises:
            AssistantError: if the answering model cannot be reached.
        """
        question = (question or "").strip()
        if not question:
            raise AssistantError("Please type a question.")

        started = time.perf_counter()
        trace: dict[str, Any] = {"original_question": question}

        # --- 1. condense the question against the history ------------------
        search_question, rewritten = condense_question(self.llm, self.memory, question)
        trace["search_question"] = search_question
        trace["question_rewritten"] = rewritten

        # --- 2. hybrid retrieval -------------------------------------------
        candidates = self.retriever.retrieve(search_question)
        trace["candidates"] = [
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "found_by": c.found_by(),
                "rrf_score": c.rrf_score,
                "bm25_score": c.bm25_score,
            }
            for c in candidates
        ]

        if not candidates:
            return self._not_found(question, trace, started,
                                   reason="retrieval returned nothing")

        # --- 3. rerank ------------------------------------------------------
        kept, rerank_info = rerank(self.llm, search_question, candidates)
        trace["rerank"] = rerank_info
        trace["used"] = [
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "section": c.document.metadata.get("section", ""),
                "page": c.document.metadata.get("page"),
                "rerank_score": c.rerank_score,
            }
            for c in kept
        ]

        # --- 4. the honesty gate --------------------------------------------
        # Nothing cleared the relevance bar, so we answer "not found" without
        # ever asking the model to write an answer.
        if not kept:
            return self._not_found(
                question, trace, started,
                reason="no passage scored above the relevance threshold",
            )

        # --- 5. generate -----------------------------------------------------
        context = build_context(kept)
        prompt = ANSWER_PROMPT.format(
            context=context,
            history_block=self.memory.history_block(),
            question=question,
        )
        trace["context_chars"] = len(context)

        answer_text = _call_with_retry(self.llm, prompt).strip()
        sources = build_sources(kept)

        # The model was told to say this when the extracts fall short; respect
        # it rather than presenting citations for an answer it did not give.
        not_found = "could not find this in the company documents" in answer_text.lower()
        if not_found:
            sources = []

        trace["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        self.memory.add(question, answer_text, [s.source for s in sources])

        return Answer(text=answer_text, sources=sources, found=not not_found, trace=trace)

    def reset(self) -> None:
        """Clear the conversation memory."""
        self.memory.clear()

    # ----------------------------------------------------------- internals --
    def _not_found(self, question: str, trace: dict, started: float, reason: str) -> Answer:
        """Return the standard 'not in the documents' answer, with no LLM call."""
        log.info("Answering 'not found': %s", reason)
        trace["not_found_reason"] = reason
        trace["elapsed_seconds"] = round(time.perf_counter() - started, 2)
        self.memory.add(question, NOT_FOUND_MESSAGE, [])
        return Answer(text=NOT_FOUND_MESSAGE, sources=[], found=False, trace=trace)

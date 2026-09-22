"""
test_basic.py
-------------
Tests for the parts of the pipeline we wrote ourselves.

    pytest -v

None of these tests calls OpenAI. They cover ingestion, chunking, BM25, the
fusion maths, the reranker's parsing and fallback behaviour, conversation
memory, and the context and citation builders - which is everything except the
two steps that are an API call.

That is deliberate: it means the suite is free, runs in about a second, and can
run on every push in GitHub Actions without a secret.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.documents import Document

import config
import ingestion
import memory as memory_module
import rag_chain
import retrieval
from reranker import _parse_scores, rerank
from retrieval import BM25Index, Candidate, reciprocal_rank_fusion, tokenise


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def make_chunk(text: str, source: str = "Doc.pdf", index: int = 0, **meta) -> Document:
    """Build a chunk with the metadata the pipeline expects."""
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "title": source.replace(".pdf", "").replace("_", " "),
            "format": "pdf",
            "chunk_id": f"{source}#{index}",
            "chunk_index": index,
            **meta,
        },
    )


class FakeLLM:
    """A stand-in chat model. Returns a canned reply, or raises."""

    def __init__(self, reply: str = "", error: Exception | None = None):
        self.reply = reply
        self.error = error
        self.prompts: list[str] = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return type("Message", (), {"content": self.reply})()


@pytest.fixture(scope="module")
def corpus():
    """The real document corpus, loaded once for the whole module."""
    chunks, problems = ingestion.build_corpus()
    assert not problems, f"documents failed to load: {problems}"
    return chunks


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
def test_all_documents_are_found():
    assert len(ingestion.find_documents()) == 8


def test_at_least_three_formats_are_supported():
    formats = {path.suffix for path in ingestion.find_documents()}
    assert {".pdf", ".docx", ".md"}.issubset(formats)


def test_unsupported_files_are_ignored(tmp_path):
    (tmp_path / "policy.md").write_text("# Policy\n\nSome text.", encoding="utf-8")
    (tmp_path / "notes.csv").write_text("a,b\n1,2", encoding="utf-8")
    (tmp_path / ".hidden.md").write_text("hidden", encoding="utf-8")
    found = ingestion.find_documents(tmp_path)
    assert [p.name for p in found] == ["policy.md"]


def test_a_missing_folder_raises_a_clear_error(tmp_path):
    with pytest.raises(ingestion.IngestionError):
        ingestion.find_documents(tmp_path / "nope")


def test_an_empty_folder_raises_a_clear_error(tmp_path):
    with pytest.raises(ingestion.IngestionError):
        ingestion.find_documents(tmp_path)


def test_clean_text_normalises_whitespace():
    assert ingestion.clean_text("a\r\n\r\n\r\n\r\nb   \n") == "a\n\nb"


def test_section_headings_are_detected():
    assert ingestion.detect_section("3. Carry-forward of unused leave\n\ntext") == \
        "3. Carry-forward of unused leave"
    assert ingestion.detect_section("## Gifts and hospitality\n\ntext") == \
        "Gifts and hospitality"
    assert ingestion.detect_section("just a sentence of prose") == ""


def test_readable_title():
    from pathlib import Path

    assert ingestion.readable_title(Path("Leave_Policy.pdf")) == "Leave Policy"


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
def test_every_chunk_carries_citation_metadata(corpus):
    for chunk in corpus:
        assert chunk.metadata["source"]
        assert chunk.metadata["chunk_id"]
        assert chunk.metadata["format"] in {"pdf", "docx", "md", "txt"}


def test_chunk_ids_are_unique(corpus):
    ids = [c.metadata["chunk_id"] for c in corpus]
    assert len(ids) == len(set(ids))


def test_chunks_respect_the_configured_size(corpus):
    # The splitter can overshoot slightly on an unbreakable run of text, but
    # nothing should be wildly over the limit.
    assert all(len(c.page_content) <= config.CHUNK_SIZE * 1.5 for c in corpus)


def test_pdf_chunks_know_their_page(corpus):
    pdf_chunks = [c for c in corpus if c.metadata["format"] == "pdf"]
    assert any(c.metadata.get("page") for c in pdf_chunks)


def test_key_facts_survive_chunking(corpus):
    """A fact split across a chunk boundary is unanswerable, so check a few."""
    joined = " ".join(" ".join(c.page_content.split()) for c in corpus)
    for fact in [
        "maximum of 5 unused annual leave days",
        "0.45 units per kilometre",
        "25 days of paid annual leave",
    ]:
        assert fact in joined


# --------------------------------------------------------------------------- #
# Tokenisation and BM25
# --------------------------------------------------------------------------- #
def test_tokeniser_keeps_numbers_and_hyphenated_terms():
    tokens = tokenise("The rate is 0.45 per km for carry-forward days")
    assert "0.45" in tokens
    assert "carry-forward" in tokens


def test_tokeniser_drops_stopwords():
    assert "the" not in tokenise("the policy")
    assert "policy" in tokenise("the policy")


def test_bm25_finds_the_document_containing_a_rare_term():
    chunks = [
        make_chunk("Annual leave is 25 days per year.", "Leave.pdf", 0),
        make_chunk("Mileage is reimbursed at 0.45 units per kilometre.", "Travel.pdf", 1),
        make_chunk("The dress code is business casual.", "Handbook.pdf", 2),
    ]
    hits = BM25Index(chunks).search("mileage 0.45 kilometre", k=2)
    assert hits[0][0].metadata["source"] == "Travel.pdf"


def test_bm25_returns_nothing_for_an_unmatched_query():
    chunks = [make_chunk("Annual leave is 25 days.", "Leave.pdf", 0)]
    assert BM25Index(chunks).search("zzzz qqqq", k=3) == []


def test_bm25_handles_an_empty_corpus():
    assert BM25Index([]).search("anything", k=3) == []


def test_bm25_scores_are_positive_and_ordered():
    chunks = [
        make_chunk("leave leave leave policy", "A.pdf", 0),
        make_chunk("leave policy", "B.pdf", 1),
        make_chunk("expenses policy", "C.pdf", 2),
    ]
    hits = BM25Index(chunks).search("leave", k=3)
    assert all(score > 0 for _, score in hits)
    assert [s for _, s in hits] == sorted([s for _, s in hits], reverse=True)


# --------------------------------------------------------------------------- #
# Reciprocal Rank Fusion
# --------------------------------------------------------------------------- #
def test_rrf_rewards_appearing_in_both_lists():
    a, b, c, d = (make_chunk("x", "S.pdf", i) for i in range(4))
    # b is only ever second, but both retrievers found it
    scores = reciprocal_rank_fusion([[a, b, c], [d, b]], k=60)
    assert max(scores, key=scores.get) == b.metadata["chunk_id"]


def test_rrf_still_scores_a_chunk_only_one_retriever_found():
    a, b = make_chunk("x", "S.pdf", 0), make_chunk("y", "S.pdf", 1)
    scores = reciprocal_rank_fusion([[a], [b]], k=60)
    assert scores[a.metadata["chunk_id"]] > 0
    assert scores[b.metadata["chunk_id"]] > 0


def test_rrf_of_nothing_is_empty():
    assert reciprocal_rank_fusion([[], []]) == {}


# --------------------------------------------------------------------------- #
# Hybrid retrieval
# --------------------------------------------------------------------------- #
class StubVectorStore:
    """Returns a fixed ranked list, or raises to simulate an outage."""

    def __init__(self, documents, error=None):
        self.documents = documents
        self.error = error

    def similarity_search_with_score(self, query, k=4):
        if self.error:
            raise self.error
        return [(d, 0.1 * i) for i, d in enumerate(self.documents[:k])]


def test_hybrid_merges_both_retrievers():
    chunks = [
        make_chunk("Annual leave is 25 days per year.", "Leave.pdf", 0),
        make_chunk("Mileage is 0.45 units per kilometre.", "Travel.pdf", 1),
    ]
    hybrid = retrieval.HybridRetriever(StubVectorStore(chunks), BM25Index(chunks))
    candidates = hybrid.retrieve("mileage kilometre")

    assert candidates
    found = [c for c in candidates if c.vector_rank and c.bm25_rank]
    assert found, "expected at least one chunk found by both retrievers"
    assert "vector" in found[0].found_by() and "BM25" in found[0].found_by()


def test_hybrid_degrades_to_keyword_only_when_the_vector_store_fails():
    chunks = [make_chunk("Mileage is 0.45 units per kilometre.", "Travel.pdf", 0)]
    hybrid = retrieval.HybridRetriever(
        StubVectorStore(chunks, error=RuntimeError("index down")), BM25Index(chunks)
    )
    candidates = hybrid.retrieve("mileage")
    assert len(candidates) == 1
    assert candidates[0].vector_rank is None
    assert candidates[0].bm25_rank == 1


def test_hybrid_results_are_ordered_by_fused_score():
    chunks = [make_chunk(f"leave policy clause {i}", "Leave.pdf", i) for i in range(5)]
    hybrid = retrieval.HybridRetriever(StubVectorStore(chunks), BM25Index(chunks))
    candidates = hybrid.retrieve("leave policy")
    scores = [c.rrf_score for c in candidates]
    assert scores == sorted(scores, reverse=True)


# --------------------------------------------------------------------------- #
# Reranking
# --------------------------------------------------------------------------- #
def _candidates(n=4):
    return [Candidate(document=make_chunk(f"passage {i}", "S.pdf", i)) for i in range(n)]


def test_reranker_parses_scores_and_keeps_the_best():
    reply = json.dumps([{"id": 1, "score": 9}, {"id": 2, "score": 8},
                        {"id": 3, "score": 2}, {"id": 4, "score": 1}])
    kept, info = rerank(FakeLLM(reply), "question", _candidates(), top_n=2)
    assert info["applied"] is True
    assert len(kept) == 2
    assert kept[0].rerank_score == 9
    assert info["dropped_as_irrelevant"] == 2


def test_reranker_drops_everything_below_the_threshold():
    reply = json.dumps([{"id": i, "score": 1} for i in range(1, 5)])
    kept, info = rerank(FakeLLM(reply), "question", _candidates())
    assert kept == []
    assert info["applied"] is True


def test_reranker_falls_back_to_the_hybrid_order_on_bad_output():
    kept, info = rerank(FakeLLM("sorry, I can't"), "question", _candidates(), top_n=2)
    assert info["applied"] is False
    assert len(kept) == 2          # degraded, not broken


def test_reranker_falls_back_when_the_model_is_unreachable():
    llm = FakeLLM(error=ConnectionError("down"))
    kept, info = rerank(llm, "question", _candidates(), top_n=3)
    assert info["applied"] is False
    assert len(kept) == 3


def test_reranker_handles_no_candidates():
    kept, info = rerank(FakeLLM("[]"), "question", [])
    assert kept == []
    assert info["applied"] is False


def test_score_parser_tolerates_a_code_fence():
    raw = 'Here you go:\n```json\n[{"id": 1, "score": 7}]\n```'
    assert _parse_scores(raw, expected=1) == {1: 7.0}


def test_score_parser_rejects_output_with_no_json():
    with pytest.raises(ValueError):
        _parse_scores("no json here", expected=2)


def test_score_parser_ignores_out_of_range_ids():
    assert _parse_scores('[{"id": 1, "score": 5}, {"id": 99, "score": 9}]', expected=2) == {1: 5.0}


# --------------------------------------------------------------------------- #
# Conversation memory
# --------------------------------------------------------------------------- #
def test_memory_starts_empty_and_records_turns():
    mem = memory_module.ConversationMemory()
    assert mem.is_empty
    mem.add("q", "a")
    assert not mem.is_empty
    assert mem.turns[0].question == "q"


def test_memory_keeps_only_the_configured_number_of_turns():
    mem = memory_module.ConversationMemory(max_turns=2)
    for i in range(5):
        mem.add(f"q{i}", f"a{i}")
    assert len(mem.turns) == 2
    assert mem.turns[0].question == "q3"


def test_memory_truncates_long_answers_in_the_history():
    mem = memory_module.ConversationMemory()
    mem.add("q", "x" * 1000)
    assert "..." in mem.format_history(answer_chars=50)


def test_memory_clear_empties_everything():
    mem = memory_module.ConversationMemory()
    mem.add("q", "a")
    mem.clear()
    assert mem.is_empty
    assert mem.history_block() == ""


def test_first_question_is_never_rewritten():
    mem = memory_module.ConversationMemory()
    question, changed = memory_module.condense_question(None, mem, "What is the leave policy?")
    assert changed is False
    assert question == "What is the leave policy?"


def test_follow_up_is_rewritten_into_a_standalone_question():
    mem = memory_module.ConversationMemory()
    mem.add("What is the leave policy?", "Employees receive 25 days.")
    llm = FakeLLM("What is the carry-forward policy for annual leave?")
    question, changed = memory_module.condense_question(llm, mem, "What about carry-forward?")
    assert changed is True
    assert "leave" in question.lower()


def test_condensing_falls_back_to_the_original_on_failure():
    mem = memory_module.ConversationMemory()
    mem.add("q", "a")
    llm = FakeLLM(error=ConnectionError("down"))
    question, changed = memory_module.condense_question(llm, mem, "What about that?")
    assert question == "What about that?"
    assert changed is False


def test_condensing_rejects_an_implausibly_long_rewrite():
    mem = memory_module.ConversationMemory()
    mem.add("q", "a")
    question, changed = memory_module.condense_question(
        FakeLLM("x" * 500), mem, "What about that?"
    )
    assert question == "What about that?"
    assert changed is False


# --------------------------------------------------------------------------- #
# Context and citations
# --------------------------------------------------------------------------- #
def test_context_labels_every_passage_with_its_source():
    candidates = [
        Candidate(document=make_chunk("Leave text", "Leave.pdf", 0, section="3. Leave")),
        Candidate(document=make_chunk("Travel text", "Travel.docx", 0, page=2)),
    ]
    context = rag_chain.build_context(candidates)
    assert "--- Leave.pdf (3. Leave) ---" in context
    assert "--- Travel.docx (page 2) ---" in context


def test_context_respects_the_character_cap():
    candidates = [Candidate(document=make_chunk("x" * 400, "S.pdf", i)) for i in range(10)]
    assert len(rag_chain.build_context(candidates, max_chars=900)) < 1400


def test_sources_are_deduplicated_per_document():
    candidates = [
        Candidate(document=make_chunk("a", "Leave.pdf", 0), rerank_score=9),
        Candidate(document=make_chunk("b", "Leave.pdf", 1), rerank_score=6),
        Candidate(document=make_chunk("c", "Travel.docx", 0), rerank_score=7),
    ]
    sources = rag_chain.build_sources(candidates)
    assert [s.source for s in sources] == ["Leave.pdf", "Travel.docx"]
    assert sources[0].score == 9          # kept the better passage of the two


def test_source_label_prefers_a_section_then_a_page():
    assert rag_chain.Source("S.pdf", "S", section="3. Leave").label() == "3. Leave"
    assert rag_chain.Source("S.pdf", "S", page=4).label() == "page 4"
    assert rag_chain.Source("S.pdf", "S").label() == ""


def test_empty_question_is_rejected():
    assistant = object.__new__(rag_chain.KnowledgeAssistant)
    with pytest.raises(rag_chain.AssistantError):
        rag_chain.KnowledgeAssistant.ask(assistant, "   ")

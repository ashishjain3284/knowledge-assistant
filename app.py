"""
app.py
======
THIS IS THE FILE YOU RUN. It is the Streamlit interface and the landing page.

    streamlit run app.py

Build the index first, once:

    python build_index.py

What this file does:
  * shows a landing page explaining what the assistant knows about,
  * runs the chat, keeping the conversation in st.session_state,
  * sends each question through the RAG pipeline in rag_chain.py,
  * shows the source documents behind every answer, and the retrieval trace,
  * handles errors so a failure never leaves a blank screen.

All the retrieval logic lives in the other modules. This file is the interface.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

import config
import ingestion
import prompts
import vectorstore
from rag_chain import AssistantError, KnowledgeAssistant

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("knowledge-assistant")


# --------------------------------------------------------------------------- #
# Friendly guard: this file needs `streamlit run`, not `python`
# --------------------------------------------------------------------------- #
def _running_under_streamlit() -> bool:
    """Return True when this script was started by `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:  # noqa: BLE001 - never block a legitimate run
        return True


if not _running_under_streamlit():
    print("\nThis is a Streamlit application, so it needs to be started with:\n")
    print("    streamlit run app.py\n")
    print("If you have not built the index yet, run this first:\n")
    print("    python build_index.py\n")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# One-time setup, cached so it does not repeat on every interaction
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the knowledge base...")
def load_assistant():
    """Load the index, rebuild the keyword index and construct the assistant.

    Returns:
        A ``(assistant, error_message)`` tuple. The error is returned rather
        than raised so the page can render a helpful message instead of a
        traceback.
    """
    try:
        chunks, problems = ingestion.build_corpus()
    except ingestion.IngestionError as error:
        return None, str(error)

    try:
        store = vectorstore.load_index()
    except vectorstore.VectorStoreError as error:
        return None, str(error)

    try:
        assistant = KnowledgeAssistant(store, chunks)
    except AssistantError as error:
        return None, str(error)

    if problems:
        log.warning("Some documents were skipped: %s", "; ".join(problems))
    return assistant, None


def init_session() -> None:
    """Set up the per-user session state on first load."""
    defaults = {
        "history": [],           # list of {role, content, sources, trace, found}
        "questions_asked": 0,
        "pending_input": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_conversation(assistant) -> None:
    """Clear the chat and the assistant's memory."""
    st.session_state.history = []
    st.session_state.questions_asked = 0
    st.session_state.pending_input = None
    if assistant is not None:
        assistant.reset()


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def render_sources(sources: list) -> None:
    """Show the documents an answer was drawn from."""
    if not sources:
        return

    names = ", ".join(s.source for s in sources)
    st.markdown(f"**Sources:** {names}")

    with st.expander("Show the passages these came from", expanded=False):
        for source in sources:
            location = source.label()
            header = f"**{source.source}**"
            if location:
                header += f"  ·  {location}"
            if source.score is not None:
                header += f"  ·  relevance {source.score:.0f}/10"
            st.markdown(header)
            st.caption(source.snippet)
            st.divider()


def render_trace(trace: dict) -> None:
    """Show how the answer was retrieved - the working, not just the result."""
    if not trace:
        return

    with st.expander("How this answer was retrieved", expanded=False):
        if trace.get("question_rewritten"):
            st.markdown("**Question rewritten for search**")
            st.caption(
                f"You asked: *{trace['original_question']}*  \n"
                f"Searched for: *{trace['search_question']}*"
            )
            st.divider()

        candidates = trace.get("candidates", [])
        if candidates:
            st.markdown(f"**Hybrid retrieval — {len(candidates)} candidates**")
            rows = [
                {
                    "source": c["source"],
                    "found by": c["found_by"],
                    "RRF score": round(c["rrf_score"], 5),
                }
                for c in candidates
            ]
            st.dataframe(rows, use_container_width=True, hide_index=True)

        rerank_info = trace.get("rerank", {})
        if rerank_info:
            if rerank_info.get("applied"):
                st.markdown(
                    f"**Reranking** — scored {rerank_info.get('scored', 0)}, "
                    f"kept {rerank_info.get('kept', 0)}, dropped "
                    f"{rerank_info.get('dropped_as_irrelevant', 0)} below "
                    f"{rerank_info.get('minimum_score')}/10"
                )
            else:
                st.warning(f"Reranking did not run: {rerank_info.get('reason', 'unknown')}")

        used = trace.get("used", [])
        if used:
            st.markdown("**Passages sent to the model**")
            for item in used:
                location = item.get("section") or (
                    f"page {item['page']}" if item.get("page") else ""
                )
                score = item.get("rerank_score")
                line = f"- `{item['source']}`"
                if location:
                    line += f" — {location}"
                if score is not None:
                    line += f"  ({score:.0f}/10)"
                st.markdown(line)

        if trace.get("not_found_reason"):
            st.info(f"No answer was generated because: {trace['not_found_reason']}")

        st.caption(
            f"Context: {trace.get('context_chars', 0)} characters  ·  "
            f"{trace.get('elapsed_seconds', 0)}s"
        )


def render_landing_page(manifest: dict) -> None:
    """The first thing a new user sees, before any question."""
    st.markdown(f"#### Ask a question about {config.ORG_NAME} policy")
    st.write(
        "This assistant answers from the company's own policy documents. It "
        "searches them two different ways, reranks what it finds, and answers "
        "only from the passages it retrieved — citing the document each fact "
        "came from. If the answer is not in the documents, it says so."
    )

    st.markdown("##### What it knows about")
    sources = manifest.get("sources", {})
    if sources:
        columns = st.columns(2)
        for index, (name, count) in enumerate(sorted(sources.items())):
            with columns[index % 2]:
                st.markdown(f"**{ingestion.readable_title(Path(name))}**")
                st.caption(f"{name}  ·  {count} passages")

    st.markdown("##### Try one of these")
    columns = st.columns(2)
    for index, example in enumerate(prompts.EXAMPLE_QUESTIONS):
        with columns[index % 2]:
            if st.button(example, key=f"example-{index}", use_container_width=True):
                st.session_state.pending_input = example
                st.rerun()

    st.caption(
        "Follow-up questions work: ask about leave, then just ask "
        "\"what about carry-forward?\""
    )


def render_sidebar(assistant, manifest: dict) -> None:
    """Index information, conversation stats and the reset control."""
    with st.sidebar:
        st.markdown(f"### {config.APP_TITLE}")
        st.caption(config.ORG_NAME)
        st.divider()

        st.markdown("**This conversation**")
        st.metric("Questions asked", st.session_state.questions_asked)
        if st.button("Clear conversation", use_container_width=True):
            reset_conversation(assistant)
            st.rerun()

        st.divider()
        st.markdown("**Knowledge index**")
        if manifest:
            st.caption(f"Documents: {manifest.get('document_count', '?')}")
            st.caption(f"Passages: {manifest.get('chunk_count', '?')}")
            st.caption(f"Built: {manifest.get('built_at', 'unknown')}")
            st.caption(f"Embeddings: {manifest.get('embedding_model', '?')}")
        else:
            st.caption("No manifest found.")

        st.divider()
        st.markdown("**Retrieval settings**")
        st.caption(f"Vector top-k: {config.VECTOR_TOP_K}")
        st.caption(f"BM25 top-k: {config.BM25_TOP_K}")
        st.caption(f"Candidates after fusion: {config.HYBRID_CANDIDATES}")
        st.caption(f"Kept after reranking: {config.RERANK_TOP_N}")
        st.caption(f"Relevance threshold: {config.MIN_RELEVANCE_SCORE}/10")

        st.divider()
        st.caption(f"Model: {config.CHAT_MODEL}")


def render_setup_help(message: str) -> None:
    """Shown when the index or the key is missing, instead of the chat."""
    st.error(message)
    st.markdown("#### Getting started")
    st.markdown(
        "1. Install the dependencies:\n"
        "   ```\n   pip install -r requirements.txt\n   ```\n"
        "2. Copy `.env.example` to `.env` and add your OpenAI key.\n"
        "3. Build the index (this reads the documents and embeds them):\n"
        "   ```\n   python build_index.py\n   ```\n"
        "4. Reload this page."
    )


# --------------------------------------------------------------------------- #
# Handling one question
# --------------------------------------------------------------------------- #
def handle_question(assistant, question: str) -> None:
    """Send one question through the pipeline and store the answer."""
    st.session_state.history.append(
        {"role": "user", "content": question, "sources": [], "trace": {}, "found": True}
    )
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the policy documents..."):
            try:
                answer = assistant.ask(question)
            except AssistantError as error:
                message = str(error)
                log.error("Assistant error: %s", message)
                st.error(message)
                st.session_state.history.append(
                    {"role": "assistant", "content": f"**Something went wrong.** {message}",
                     "sources": [], "trace": {}, "found": False}
                )
                return

        if not answer.found:
            st.warning(answer.text)
        else:
            st.markdown(answer.text)
        render_sources(answer.sources)
        render_trace(answer.trace)

    st.session_state.questions_asked += 1
    st.session_state.history.append(
        {
            "role": "assistant",
            "content": answer.text,
            "sources": answer.sources,
            "trace": answer.trace,
            "found": answer.found,
        }
    )


# --------------------------------------------------------------------------- #
# The page
# --------------------------------------------------------------------------- #
def main() -> None:
    """Draw the whole page. Streamlit re-runs this on every interaction."""
    st.set_page_config(
        page_title=f"{config.APP_TITLE} | {config.ORG_NAME}",
        page_icon="📚",
        layout="wide",
    )
    init_session()

    st.title(config.APP_TITLE)
    st.caption(
        "Hybrid search over the company's private documents, reranked and "
        "answered with citations."
    )

    # --- the index and the key must both be present ------------------------
    if not config.OPENAI_API_KEY:
        render_setup_help(
            "No OpenAI API key found. Create a file named .env next to app.py "
            "containing OPENAI_API_KEY=sk-your-key-here"
        )
        st.stop()

    if not vectorstore.index_exists():
        render_setup_help(
            "No knowledge index found. It has to be built once before the "
            "assistant can answer anything."
        )
        st.stop()

    assistant, error = load_assistant()
    if assistant is None:
        render_setup_help(error or "The assistant could not be loaded.")
        st.stop()

    manifest = vectorstore.load_manifest()
    render_sidebar(assistant, manifest)

    # --- warn if the index looks out of date -------------------------------
    if manifest and manifest.get("chunk_count") not in (None, assistant.chunk_count):
        st.warning(
            f"The documents have changed since the index was built "
            f"({assistant.chunk_count} passages now, {manifest['chunk_count']} in the "
            "index). Answers may miss recent edits. Rebuild with "
            "`python build_index.py`."
        )

    # --- landing page, or the conversation ---------------------------------
    if not st.session_state.history:
        render_landing_page(manifest)
    else:
        for turn in st.session_state.history:
            with st.chat_message(turn["role"]):
                if turn["role"] == "assistant" and not turn.get("found", True):
                    st.warning(turn["content"])
                else:
                    st.markdown(turn["content"])
                if turn["role"] == "assistant":
                    render_sources(turn.get("sources", []))
                    render_trace(turn.get("trace", {}))

    # --- input --------------------------------------------------------------
    typed = st.chat_input("Ask about leave, expenses, IT, benefits, remote working...")
    pending = st.session_state.pending_input
    st.session_state.pending_input = None

    question = typed or pending
    if question:
        handle_question(assistant, question)


main()

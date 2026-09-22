"""
memory.py
---------
Conversation memory, and the piece that makes it actually work for RAG.

Keeping the chat history is the easy half. The hard half is that retrieval
happens on the *question*, and a follow-up question is often meaningless on its
own:

    User: What is the leave policy?
    AI:   Employees receive 25 days...
    User: What about carry-forward?

Searching for "what about carry-forward" retrieves poorly - the words "leave"
and "annual" are not in it. So before retrieval, the question is rewritten
against the history into a standalone one:

    "What is the carry-forward policy for annual leave?"

That rewritten question is what goes to the retriever. The user never sees it,
but it is shown in the retrieval trace so the behaviour is inspectable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import config
from prompts import CONDENSE_PROMPT

log = logging.getLogger(__name__)


@dataclass
class Turn:
    """One exchange: what the user asked and what the assistant replied."""

    question: str
    answer: str
    sources: list[str] = field(default_factory=list)


class ConversationMemory:
    """Holds the recent conversation and formats it for the prompts.

    Only the last `config.MEMORY_TURNS` exchanges are kept. Old turns are
    dropped rather than summarised: for a policy question-and-answer assistant,
    relevance falls off quickly, and an unbounded history would slowly push the
    retrieved policy text out of the model's attention.
    """

    def __init__(self, max_turns: int | None = None) -> None:
        """Create an empty memory.

        Args:
            max_turns: How many exchanges to keep (default from config).
        """
        self.max_turns = max_turns or config.MEMORY_TURNS
        self.turns: list[Turn] = []

    def add(self, question: str, answer: str, sources: list[str] | None = None) -> None:
        """Record one exchange, discarding the oldest if the buffer is full."""
        self.turns.append(Turn(question=question, answer=answer, sources=sources or []))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def clear(self) -> None:
        """Forget the whole conversation."""
        self.turns.clear()

    @property
    def is_empty(self) -> bool:
        """True when nothing has been said yet."""
        return not self.turns

    def format_history(self, answer_chars: int = 300) -> str:
        """Render the history for a prompt.

        Answers are truncated: the reason for including them is so the model can
        resolve a reference like "that", not so it can re-read its own output.
        """
        if not self.turns:
            return ""
        lines = []
        for turn in self.turns:
            answer = turn.answer.strip().replace("\n", " ")
            if len(answer) > answer_chars:
                answer = answer[:answer_chars].rstrip() + "..."
            lines.append(f"User: {turn.question}\nAssistant: {answer}")
        return "\n\n".join(lines)

    def history_block(self) -> str:
        """The history section injected into the answer prompt, or an empty string."""
        if not self.turns:
            return ""
        return (
            "Earlier in this conversation:\n"
            f"{self.format_history()}\n\n"
            "Use the history only to understand what the question refers to. "
            "Do not treat it as a source of policy facts.\n\n"
        )


def condense_question(llm, memory: ConversationMemory, question: str) -> tuple[str, bool]:
    """Rewrite a follow-up question so it stands on its own.

    Args:
        llm: The chat model.
        memory: The conversation so far.
        question: What the user just typed.

    Returns:
        A ``(question_to_search, was_rewritten)`` tuple.

    Never raises: if the rewrite fails for any reason the original question is
    used, which is the safe fallback - slightly worse retrieval, not an error.
    """
    if memory.is_empty:
        return question, False

    prompt = CONDENSE_PROMPT.format(
        history=memory.format_history(), question=question
    )
    try:
        response = llm.invoke(prompt)
        rewritten = getattr(response, "content", str(response)).strip().strip('"')
    except Exception as error:  # noqa: BLE001 - degrade to the original question
        log.warning("Question condensing failed, using the original: %s", error)
        return question, False

    # Guard against a model that returns an explanation, an empty string, or
    # something implausibly long instead of a question.
    if not rewritten or len(rewritten) > 300:
        return question, False

    changed = rewritten.strip().lower() != question.strip().lower()
    if changed:
        log.info("Condensed question: %r -> %r", question, rewritten)
    return rewritten, changed

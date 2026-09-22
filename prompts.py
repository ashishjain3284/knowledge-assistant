"""
prompts.py
----------
The three prompts the application uses.

They are kept together, away from the pipeline code, because they are the part
most likely to be tuned:

  CONDENSE_PROMPT - rewrites a follow-up question into a standalone one, so
                    retrieval works on "what about carry-forward?"
  RERANK_PROMPT   - scores each retrieved chunk against the question
  ANSWER_PROMPT   - generates the grounded answer, with the anti-hallucination
                    rules
"""

import config

# --------------------------------------------------------------------------- #
# 1. Question condensing - what makes conversational RAG work
# --------------------------------------------------------------------------- #
# A follow-up like "what about carry-forward?" retrieves nothing useful on its
# own, because the words "leave" and "annual" are not in it. Rewriting it
# against the history into "What is the carry-forward policy for annual leave?"
# is what lets the retriever find the right chunk.
CONDENSE_PROMPT = """Rewrite the user's latest question so that it can be understood on its own, \
without the conversation history.

Rules:
- Replace pronouns and vague references ("it", "that", "what about...") with the \
subject they refer to, taken from the history.
- Keep the user's own wording wherever possible. Do not answer the question, \
expand it, or add detail that is not implied.
- If the question is already standalone, return it unchanged.
- Return only the rewritten question, with no preamble and no quotation marks.

Conversation history:
{history}

Latest question: {question}

Standalone question:"""


# --------------------------------------------------------------------------- #
# 2. Reranking - improving the relevance of the final context
# --------------------------------------------------------------------------- #
RERANK_PROMPT = """You are scoring passages from a company policy library for how well they \
answer a specific question.

Score each passage from 0 to 10:
  0-2   unrelated to the question
  3-4   same general topic, but does not contain the answer
  5-7   contains part of the answer, or useful surrounding context
  8-10  directly answers the question

Judge only whether the passage answers THIS question. A well-written passage \
about a different subject still scores low.

Question: {question}

Passages:
{passages}

Return ONLY a JSON array, one object per passage, with no other text:
[{{"id": 1, "score": 7}}, {{"id": 2, "score": 3}}]"""


# --------------------------------------------------------------------------- #
# 3. Answer generation - grounding and hallucination control
# --------------------------------------------------------------------------- #
ANSWER_PROMPT = f"""You are the Employee Knowledge Assistant for {config.ORG_NAME}. \
You answer employees' questions about company policy using ONLY the policy \
extracts you are given.

GROUNDING RULES - these are the point of the system:

1. Every factual statement in your answer must be supported by the extracts \
below. If the extracts do not contain the answer, say so plainly: "I could not \
find this in the company documents." Then suggest who to ask, such as the People \
team or the Service Desk.
2. Never use your own general knowledge about how companies usually work. A \
plausible-sounding number that is not in the extracts is the worst possible \
answer.
3. Never invent or adjust a figure, a deadline, a percentage or an entitlement. \
Quote them exactly as written.
4. If the extracts partly answer the question, answer that part and say clearly \
what you could not find.
5. If two extracts appear to conflict, say so and cite both rather than choosing \
one.

CITATION RULES:

6. Cite the source document in square brackets immediately after the fact it \
supports, using the file name shown with each extract, for example \
[Leave_Policy.pdf].
7. Cite every document you actually used. Do not cite one you did not use.

STYLE:

- Answer the question directly in the first sentence, then give the detail.
- Use a short numbered list for steps or for several separate entitlements.
- Quote exact figures and deadlines; do not round or paraphrase them.
- Keep the answer under about 200 words unless the question needs more.
- Plain professional English. No emojis. Never leave a placeholder in the text.

Policy extracts:
{{context}}

{{history_block}}Question: {{question}}

Answer:"""


#: Shown on the landing page so a new user knows what to try.
EXAMPLE_QUESTIONS = [
    "How many annual leave days do I get, and can I carry any forward?",
    "What is the mileage rate, and how long do I have to claim expenses?",
    "Can I use ChatGPT for work?",
    "How many days a week do I have to be in the office?",
]

#: Text used when the retriever finds nothing above the relevance threshold.
NOT_FOUND_MESSAGE = (
    "I could not find this in the company documents.\n\n"
    "The knowledge base covers leave, the employee handbook, benefits, IT "
    "acceptable use, travel and expenses, remote working, the code of conduct "
    "and the company FAQs. If your question is about something else, or about "
    "your own personal circumstances, the People team will be able to help."
)

"""
config.py
---------
Every setting for the project lives here. No other module reads an environment
variable or hard-codes a path or a tuning number, so this is the only file you
change to re-tune the retrieval pipeline.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --------------------------------------------------------------------------- #
# Organisation (fictional)
# --------------------------------------------------------------------------- #
ORG_NAME = "Nimbus Technologies"
APP_TITLE = "Employee Knowledge Assistant"

# --------------------------------------------------------------------------- #
# Folders
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).parent
DOCUMENTS_DIR = BASE_DIR / "documents"   # the private corpus
INDEX_DIR = BASE_DIR / "index"           # the built FAISS index (git-ignored)

SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".md", ".txt"]

# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Low temperature: the assistant must repeat what the policy says, not improvise.
TEMPERATURE = 0.0

# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
# Policy documents are made of short numbered clauses. Chunks of roughly 900
# characters keep a whole clause together; the 150-character overlap stops an
# answer being split across a boundary.
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150

# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
# Stage 1: each retriever proposes its own candidates.
VECTOR_TOP_K = 8      # semantic / dense results
BM25_TOP_K = 8        # keyword / sparse results

# Stage 2: the two lists are fused with Reciprocal Rank Fusion.
# RRF_K dampens the influence of the very top ranks; 60 is the value from the
# original RRF paper and works well without tuning.
RRF_K = 60
HYBRID_CANDIDATES = 10   # how many fused candidates go forward to reranking

# Stage 3: the reranker scores each candidate against the question and keeps the
# best few. Fewer, better chunks beat more, noisier ones.
RERANK_TOP_N = 4

# A candidate scoring below this (out of 10) is treated as irrelevant. If every
# candidate falls below it, the application answers "not found" WITHOUT calling
# the LLM - this is the main defence against a confident but unsupported answer.
MIN_RELEVANCE_SCORE = 4

# --------------------------------------------------------------------------- #
# Conversation memory
# --------------------------------------------------------------------------- #
# How many previous exchanges are kept and shown to the model. Six turns is
# enough for follow-up questions without bloating the prompt.
MEMORY_TURNS = 6

# --------------------------------------------------------------------------- #
# Answer generation
# --------------------------------------------------------------------------- #
MAX_CONTEXT_CHARS = 6000   # hard cap on the context sent to the LLM
LLM_MAX_ATTEMPTS = 3       # retries on a transient API failure

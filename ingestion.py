"""
ingestion.py
------------
Stage 1 of the pipeline: turn a folder of documents into chunks ready to embed.

    documents/  ->  load  ->  clean  ->  split  ->  list[Document]

Three formats are supported, each with its own loader. Every chunk carries the
metadata the rest of the system needs:

    source     the file name, which is what the answer cites
    title      a readable document title for the UI
    format     pdf | docx | md
    page       the PDF page the text came from, where known
    section    the nearest preceding heading, where one could be found
    chunk_id   a stable identifier used by the reranker and the trace

The metadata is not decoration: citations are only trustworthy if each chunk
knows exactly where it came from.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

log = logging.getLogger(__name__)


class IngestionError(RuntimeError):
    """Raised when the document corpus cannot be read."""


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def find_documents(folder: Path | None = None) -> list[Path]:
    """Return every supported document in the folder, sorted by name.

    Hidden files and unsupported types are skipped rather than raising, so a
    stray .DS_Store or spreadsheet never breaks the build.
    """
    folder = folder or config.DOCUMENTS_DIR
    if not folder.exists():
        raise IngestionError(f"The documents folder does not exist: {folder}")

    files = [
        path
        for path in sorted(folder.iterdir())
        if path.is_file()
        and path.suffix.lower() in config.SUPPORTED_EXTENSIONS
        and not path.name.startswith(".")
    ]
    if not files:
        raise IngestionError(
            f"No supported documents found in {folder}. "
            f"Supported types are: {', '.join(config.SUPPORTED_EXTENSIONS)}"
        )
    return files


def readable_title(path: Path) -> str:
    """Turn 'Leave_Policy.pdf' into 'Leave Policy' for display."""
    return path.stem.replace("_", " ").replace("-", " ").strip()


# --------------------------------------------------------------------------- #
# Format-specific loaders
# --------------------------------------------------------------------------- #
def _load_pdf(path: Path) -> list[tuple[str, dict]]:
    """Read a PDF page by page so that citations can name the page."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as error:  # noqa: BLE001 - one bad page must not fail the file
            log.warning("Skipping unreadable page %d of %s: %s", number, path.name, error)
            continue
        if text.strip():
            pages.append((text, {"page": number}))
    return pages


def _load_docx(path: Path) -> list[tuple[str, dict]]:
    """Read a Word document, keeping headings so sections can be identified."""
    import docx

    document = docx.Document(str(path))
    parts = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        # Mark headings with a hash so the section detector below can find them.
        if paragraph.style.name.lower().startswith(("heading", "title")):
            parts.append(f"\n## {text}\n")
        else:
            parts.append(text)
    return [("\n".join(parts), {})]


def _load_text(path: Path) -> list[tuple[str, dict]]:
    """Read a Markdown or plain text file."""
    return [(path.read_text(encoding="utf-8", errors="replace"), {})]


LOADERS = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".md": _load_text,
    ".txt": _load_text,
}


def load_document(path: Path) -> list[Document]:
    """Load one file into one or more LangChain Documents.

    A PDF produces one Document per page; other formats produce a single one.

    Raises:
        IngestionError: the file could not be read or contained no text.
    """
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise IngestionError(f"No loader for '{path.suffix}' ({path.name})")

    try:
        parts = loader(path)
    except IngestionError:
        raise
    except Exception as error:  # noqa: BLE001 - third-party parsers raise many types
        raise IngestionError(f"Could not read {path.name}: {error}") from error

    documents = []
    for text, extra in parts:
        cleaned = clean_text(text)
        if not cleaned:
            continue
        metadata = {
            "source": path.name,
            "title": readable_title(path),
            "format": path.suffix.lower().lstrip("."),
            **extra,
        }
        documents.append(Document(page_content=cleaned, metadata=metadata))

    if not documents:
        raise IngestionError(
            f"{path.name} contained no readable text. If it is a scanned PDF it "
            "would need OCR, which this application does not do."
        )
    return documents


def load_all_documents(folder: Path | None = None) -> tuple[list[Document], list[str]]:
    """Load every document in the folder.

    Returns:
        A ``(documents, problems)`` tuple. A file that cannot be read is
        reported in ``problems`` rather than aborting the whole build.
    """
    documents: list[Document] = []
    problems: list[str] = []

    for path in find_documents(folder):
        try:
            documents.extend(load_document(path))
        except IngestionError as error:
            log.error("%s", error)
            problems.append(str(error))

    return documents, problems


# --------------------------------------------------------------------------- #
# Cleaning and chunking
# --------------------------------------------------------------------------- #
_MULTI_BLANK = re.compile(r"\n{3,}")
_TRAILING_SPACE = re.compile(r"[ \t]+(?=\n)")

#: A numbered clause heading ("3. Carry-forward of unused leave") or a markdown
#: heading. Used to label each chunk with the section it belongs to.
_HEADING = re.compile(r"^\s*(?:#{1,6}\s+(.+)|(\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{3,80}))\s*$")


def clean_text(text: str) -> str:
    """Normalise whitespace so chunk boundaries are predictable."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", " ").replace("­", "")
    text = _TRAILING_SPACE.sub("", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def detect_section(text: str) -> str:
    """Return the first heading found in a chunk, for citation detail."""
    for line in text.splitlines():
        match = _HEADING.match(line)
        if match:
            return (match.group(1) or match.group(2) or "").strip()
    return ""


def chunk_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks and label each one.

    The separators are ordered so the splitter prefers to break at a blank line,
    then a line break, then a sentence - it only splits mid-sentence as a last
    resort. That keeps a policy clause intact inside a single chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)

    # Give every chunk a stable id and a section label.
    per_source: dict[str, int] = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        index = per_source.get(source, 0)
        per_source[source] = index + 1

        chunk.metadata["chunk_id"] = f"{source}#{index}"
        chunk.metadata["chunk_index"] = index
        section = detect_section(chunk.page_content)
        if section:
            chunk.metadata["section"] = section

    log.info("Split %d document part(s) into %d chunk(s)", len(documents), len(chunks))
    return chunks


def build_corpus(folder: Path | None = None) -> tuple[list[Document], list[str]]:
    """Load and chunk the whole corpus in one call.

    Returns:
        A ``(chunks, problems)`` tuple.
    """
    documents, problems = load_all_documents(folder)
    if not documents:
        raise IngestionError(
            "No documents could be read. " + (" ".join(problems) if problems else "")
        )
    return chunk_documents(documents), problems

"""Corpus ingestion: raw course files -> corpus/processed/chunks.jsonl.

Provenance-first chunking. A citation must be checkable, so every chunk keeps
the source filename and the page its text starts on; chunking therefore works
page-by-page (splitting oversized pages, merging undersized runs) instead of
flattening the document into one stream and losing the page map.

What ingestion refuses to guess at, it reports instead: scanned PDFs with no
text layer are skipped and listed, byte-identical duplicate files are skipped
and listed, and unsupported formats are listed. The summary is the checklist
for deciding what needs OCR or manual handling — silence would hide exactly
the files most likely to be missing from retrieval.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .config import CourseConfig

# Rough chars-per-token, consistent with the estimate used elsewhere.
_CHARS_PER_TOKEN = 4

# Per-PAGE keep threshold, not per-file: slide decks legitimately carry ~100
# characters a page (a title and a bullet next to a figure) and are worth
# keeping, while a scanned page extracts only its printed page number. Below
# this, a page is figure-only or scan noise and is dropped; a file where no
# page clears it has no usable text layer at all.
_MIN_CHARS_PER_PAGE = 80

_SKIP_NAMES = {"README.md", "MANIFEST.md", ".gitkeep", ".DS_Store"}

_WS = re.compile(r"[ \t]+")
_BLANK_LINES = re.compile(r"\n\s*\n+")


def _normalise(text: str) -> str:
    text = _WS.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    return text.strip()


def _slug(name: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", name.lower())).strip("-")


def topic_for(filename: str, config: CourseConfig) -> str | None:
    """First topic (in config order) whose pattern appears in the filename."""
    lowered = filename.lower()
    for topic in config.topics:
        if any(p.lower() in lowered for p in topic.patterns):
            return topic.id
    return None


def _pdf_pages(path: Path) -> list[tuple[int, str]]:
    import logging

    from pypdf import PdfReader

    # Old course PDFs are full of odd Type1 fonts; pypdf grumbles per glyph
    # table. The ingest summary is the signal channel — keep it readable.
    logging.getLogger("pypdf").setLevel(logging.ERROR)

    reader = PdfReader(path)
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((number, _normalise(text)))
    return pages


def _split_oversized(text: str, page: int, max_chars: int, overlap_chars: int):
    """Split one page's text at whitespace boundaries, with overlap."""
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            # Break at the last whitespace in the window, if there is one.
            cut = text.rfind(" ", start + max_chars // 2, end)
            if cut > start:
                end = cut
        pieces.append((page, text[start:end].strip()))
        if end >= len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return pieces


def _chunk_pages(
    pages: list[tuple[int, str]], chunk_tokens: int, overlap_tokens: int
) -> list[tuple[int, str]]:
    """Merge small pages and split big ones toward ~chunk_tokens per chunk.

    Returns (starting_page, text) pairs. Merges never cross a split piece, so
    a chunk's starting page is always the page its first sentence is on.
    """
    target = chunk_tokens * _CHARS_PER_TOKEN
    overlap = overlap_tokens * _CHARS_PER_TOKEN

    pieces: list[tuple[int, str]] = []
    for page, text in pages:
        if not text:
            continue
        if len(text) > target:
            pieces.extend(_split_oversized(text, page, target, overlap))
        else:
            pieces.append((page, text))

    chunks: list[tuple[int, str]] = []
    current_page: int | None = None
    current: list[str] = []
    length = 0
    for page, text in pieces:
        if current and length + len(text) > target:
            chunks.append((current_page, "\n\n".join(current)))
            current, length, current_page = [], 0, None
        if current_page is None:
            current_page = page
        current.append(text)
        length += len(text)
    if current:
        chunks.append((current_page, "\n\n".join(current)))
    return chunks


def ingest(
    *,
    raw_dir: Path,
    out_path: Path,
    config: CourseConfig,
    progress=print,
) -> dict:
    """Build the chunk file and return a summary of what happened to each file."""
    files = sorted(
        p for p in raw_dir.rglob("*")
        if p.is_file() and p.name not in _SKIP_NAMES and not p.name.startswith(".")
    )
    if not files:
        raise FileNotFoundError(f"no corpus files found under {raw_dir}")

    seen_digests: dict[str, str] = {}
    duplicates: list[str] = []
    no_text_layer: list[str] = []
    partial: list[str] = []
    unsupported: list[str] = []
    ingested: list[str] = []
    records: list[dict] = []
    topic_counts: dict[str, int] = {}

    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen_digests:
            duplicates.append(f"{path.name} (same bytes as {seen_digests[digest]})")
            continue
        seen_digests[digest] = path.name

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            pages = _pdf_pages(path)
            kept = [(n, t) for n, t in pages if len(t) >= _MIN_CHARS_PER_PAGE]
            if not kept:
                no_text_layer.append(path.name)
                continue
            if len(kept) * 2 <= len(pages):
                partial.append(f"{path.name} ({len(kept)}/{len(pages)} pages had text)")
            chunked = _chunk_pages(
                kept, config.retrieval.chunk_tokens, config.retrieval.chunk_overlap_tokens
            )
        elif suffix == ".m":
            # MATLAB demos are small worked examples; keep each whole.
            text = _normalise(path.read_text(encoding="utf-8", errors="replace"))
            chunked = [(None, text)] if text else []
        else:
            unsupported.append(path.name)
            continue

        if not chunked:
            no_text_layer.append(path.name)
            continue

        topic = topic_for(path.name, config)
        stem = _slug(path.stem)
        for index, (page, text) in enumerate(chunked, start=1):
            record = {
                "chunk_id": f"{stem}-{index:03d}",
                "topic": topic,
                "source": path.name,
                "page": page,
                "text": text,
            }
            records.append(record)
        ingested.append(path.name)
        key = topic or "(untagged)"
        topic_counts[key] = topic_counts.get(key, 0) + len(chunked)
        progress(f"  {path.name}: {len(chunked)} chunk(s)" + (f"  [{topic}]" if topic else ""))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "files_ingested": len(ingested),
        "chunks": len(records),
        "topic_counts": dict(sorted(topic_counts.items())),
        "duplicates": duplicates,
        "no_text_layer": no_text_layer,
        "partial": partial,
        "unsupported": unsupported,
        "out_path": str(out_path),
    }

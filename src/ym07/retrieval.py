"""Corpus retrieval.

v0 is lexical (TF-IDF cosine, pure stdlib). That is deliberate: it needs no
embedding model, no API calls and no extra dependencies, so C2 is runnable the
day a corpus exists. It is also a legitimate comparison point — "dense retrieval
beats lexical retrieval on this corpus" is a C3 result worth reporting, and you
cannot report it without a lexical arm to compare against.

Swap in a dense retriever later by implementing the same two methods.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_WORD = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass(frozen=True)
class Chunk:
    """One retrievable passage, with the provenance a citation needs."""

    chunk_id: str
    text: str
    source: str
    topic: str | None = None
    page: int | None = None
    # Other files carrying this exact passage, when copies were folded into it
    # by dedupe_chunks. Empty for nearly every chunk.
    also_in: tuple[str, ...] = ()

    def citation(self) -> str:
        return f"{self.source}, p.{self.page}" if self.page else self.source


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class LexicalRetriever:
    """TF-IDF cosine over the processed corpus."""

    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self._doc_tf: list[Counter] = []
        df: Counter = Counter()

        for chunk in chunks:
            tf = Counter(tokenize(chunk.text))
            self._doc_tf.append(tf)
            df.update(tf.keys())

        n = max(len(chunks), 1)
        self._idf = {term: math.log((n + 1) / (count + 1)) + 1.0 for term, count in df.items()}
        self._norms = [self._norm(tf) for tf in self._doc_tf]

    def _norm(self, tf: Counter) -> float:
        return math.sqrt(sum((c * self._idf.get(t, 0.0)) ** 2 for t, c in tf.items())) or 1.0

    def search(self, query: str, top_k: int) -> list[Hit]:
        q_tf = Counter(tokenize(query))
        if not q_tf:
            return []
        q_norm = self._norm(q_tf)

        scored: list[Hit] = []
        for idx, tf in enumerate(self._doc_tf):
            dot = sum(
                q_count * self._idf.get(term, 0.0) * tf.get(term, 0) * self._idf.get(term, 0.0)
                for term, q_count in q_tf.items()
                if term in tf
            )
            if dot > 0:
                scored.append(Hit(self.chunks[idx], dot / (q_norm * self._norms[idx])))

        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self.chunks)


def load_chunks(path: str | Path) -> list[Chunk]:
    """Read corpus/processed/chunks.jsonl, as produced by `ym07 ingest`."""
    path = Path(path)
    if not path.exists():
        return []

    chunks: list[Chunk] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        for required in ("chunk_id", "text", "source"):
            if required not in obj:
                raise ValueError(f"{path}:{line_no}: chunk missing '{required}'")
        chunks.append(
            Chunk(
                chunk_id=obj["chunk_id"],
                text=obj["text"],
                source=obj["source"],
                topic=obj.get("topic"),
                page=obj.get("page"),
            )
        )
    return chunks


def dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Drop chunks whose text repeats an earlier chunk, remembering where.

    19 of the 335 chunks are word-for-word copies: one class test saved under
    two filenames, question pages reprinted in their solutions, and the formula
    sheet attached to paper after paper. Each copy takes a retrieval slot a
    different passage could have had, and the formula sheet is the dense table
    of symbols that lexical scoring ranks highest. The first copy is kept and
    the files carrying the others are recorded in `also_in`, so no provenance
    is lost.

    Only exact copies are removed, after folding case and whitespace. Near
    copies were tried and rejected: a similarity threshold loose enough to
    catch the reprinted formula sheets also merges ML Tutorial 2 into its
    solution page, and whichever of the pair survived, a question page and the
    page that answers it are not the same passage.
    """
    kept: dict[str, Chunk] = {}
    also: dict[str, list[str]] = {}
    for chunk in chunks:
        key = " ".join(chunk.text.lower().split())
        if key not in kept:
            kept[key] = chunk
            also[key] = []
        elif chunk.source != kept[key].source and chunk.source not in also[key]:
            also[key].append(chunk.source)
    return [
        Chunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            source=chunk.source,
            topic=chunk.topic,
            page=chunk.page,
            also_in=tuple(also[key]),
        )
        for key, chunk in kept.items()
    ]


RETRIEVAL_METHODS = ("tfidf", "bm25", "dense", "fused")


def build_retriever(path: str | Path, method: str = "tfidf"):
    """The retriever the course config names, over the deduplicated corpus.

    TF-IDF is built here. The other methods live in retrieval_eval, where they
    were compared before one was chosen, and are imported only when named, so a
    course that keeps TF-IDF needs neither torch nor an embedding model.
    """
    if method not in RETRIEVAL_METHODS:
        raise ValueError(
            f"Unknown retrieval method '{method}'. Known: {', '.join(RETRIEVAL_METHODS)}"
        )
    chunks = dedupe_chunks(load_chunks(path))
    if not chunks:
        return None
    if method == "tfidf":
        return LexicalRetriever(chunks)

    from .retrieval_eval import BM25Retriever, DenseRetriever, FusedRetriever

    if method == "bm25":
        return BM25Retriever(chunks)
    if method == "dense":
        return DenseRetriever(chunks)
    return FusedRetriever(BM25Retriever(chunks), DenseRetriever(chunks))

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


# Questions about how the course runs are answered by one file, the course handout, and
# the words a student uses for them - convenor, DP, class test, credits - barely overlap
# with its text. Naming the handout in the query found it in the top four for 13 of the
# 23 course questions against 7 without, measured before any model answered them.
_ADMIN = re.compile(
    r"\b(convenor|convener|convenes|lecturers?|lectures|teaching assistant|ta|tutor|office|"
    r"consultation|venue|timetable|credits?|prerequisites?|class tests?|"
    r"(must|should|need|needed|have to)[^.?!]{0,30}done before|"
    r"first class test|second class test|exam|dp|textbook|recommended (text|book)|book|"
    # "must I have done before I can take this one" is a held-back wording of the
    # prerequisites question. Bare "done before" also matches "upsampling is done before
    # downsampling", which is a question about the notes, and once this decides whether
    # to retrieve at all, a false positive costs that question its passages.
    r"plagiarism|declaration|medical certificate|sick|illness|miss(ed)? (a|the) test|"
    r"weeks|notional|contact (sessions|time)|how often do we meet|meet each week|"
    r"how many hours|hours in total|study time|email|final mark|course mark|weighting|"
    r"percentage|worth|count(s)? (for|towards))\b", re.I)


def about_the_course(question: str) -> bool:
    """Is this a question about how the course runs, rather than about its content?"""
    return bool(_ADMIN.search(question or ""))


def passages_help(question: str) -> bool:
    """Whether a *search* of the corpus can help with this question.

    It cannot help with a question about how the course runs. One document answers
    those, and `handout_passages` supplies it directly; what a search returns instead is
    seventeen years of class test and exam papers, each printed with its own date and
    time. Asked when Class Test 1 is, the retriever returned one of them and the model
    read the date off the paper: "Class Test 1 is on March 2025, with duration 1 hour",
    cited to a retrieved passage. All three date-and-time questions failed that way.

    One function, used by the browser and the suites alike, so the delivered system and
    the measured one cannot disagree about when to search.
    """
    return not about_the_course(question)


def handout_passages(retriever, handout: str) -> list[Hit]:
    """Every page of the course handout, in page order, as retrieved passages.

    A question about how the course runs is answered from this one document, and it is
    supplied whole rather than searched. Three reasons it can be:

    Ranked search is the thing that was failing. Over the whole corpus it put the right
    handout page in the top four for 14 of 23 such questions, and the facts a student
    asks about sit in tables - "Tests 20" in an assessment grid, "Class Test 1 24th
    March" under a heading - which score badly against the words of the question. Here
    there is no ranking step, so there is nothing left for it to get wrong.

    It fits. The four pages come to about 1,624 tokens, inside the 2,100 the card was
    measured to carry without degrading, so the whole document goes in front of the
    model with room for the answer.

    It reads as a document. The same facts transcribed into a flat `name: value` list
    lost the structure that makes them findable: asked what the class tests count for,
    the model returned their dates instead, and asked whether a declaration must be
    attached it said the handout does not mention one. Both are on the page, under
    headings, in prose and in a table.

    Scored passages, so citations and the token accounting work as they do for any
    other retrieved material. The score is 1.0 because the document was chosen by name,
    not ranked.
    """
    if not retriever or not handout:
        return []
    pages = [c for c in getattr(retriever, "chunks", ()) if c.source == handout]
    pages.sort(key=lambda c: (c.page is None, c.page, c.chunk_id))
    return [Hit(chunk, 1.0) for chunk in pages]


def passages_for(question: str, retriever, *, code: str, top_k: int,
                 handout: str = "", search_course: bool = False,
                 query: str | None = None, about: bool | None = None) -> list[Hit]:
    """The passages a question is answered from. The one decision, made once.

    Three outcomes: the handout for a question about how the course runs, a ranked
    search of the corpus for anything else, and nothing at all when there is no
    retriever. The browser and every suite call this, so what the delivered tutor
    retrieves and what the measurements retrieve cannot drift apart.

    `search_course` forces the ranked search even for a course question, which is the
    arm that shows what supplying the handout is worth. With no handout named and no
    search forced, a course question retrieves nothing and the answer comes from the
    transcribed facts instead - the third arm.

    `query` is the text to search with, when that is not the question itself: the
    browser searches with a screenshot's words folded in and its mathematics rewritten
    as the corpus spells it. Which document to supply is always decided from the
    student's own question, so a transcription cannot change it.

    `about` overrides the word list's verdict on whether this is a question about the
    course. A condition with the handout lookup passes the lookup's wider test
    (ym07.coursefacts); every other condition leaves it None and is unchanged.
    """
    if not retriever:
        return []
    course = about_the_course(question) if about is None else about
    if search_course or not course:
        return retriever.search(course_query(query if query is not None else question,
                                             code), top_k)
    return handout_passages(retriever, handout)


# E33 (protocols/followup-fix.md): a follow-up is searched on the conversation rather than
# on its own words. False restores the search every result before 16 October used.
# Off by E33's registered rule (18 October): with it on, Opus 5's follow-ups that retract
# or carry the not-course-material note fell from 21 to 4 of 54, against a pass mark of 2.
CONVERSATION_QUERY = False


def conversation_query(message: str, history=None) -> str:
    """The text a message is searched with: the student's previous message, then this one.

    A follow-up that changes one number - "what if the coefficient were 0.6 instead of
    0.8?" - names no topic, so searched on its own it found regularisation and multirate
    passages. The citation check then told the student no course material covered the
    answer, and Opus 5 retracted its first reply's citations against passages it had
    never cited: 21 of 54 held-out follow-ups (E30). Only the search text changes; whether
    the message is about the course is still decided from its own words. With no
    conversation before it, the text is the message, so nothing single-turn moves.
    """
    if not CONVERSATION_QUERY:
        return message
    earlier = [text for who, text in (history or []) if str(who).lower().startswith("student")]
    return f"{earlier[-1]}\n{message}" if earlier else message


def course_query(question: str, code: str) -> str:
    """The query to retrieve with: a question about the course names the handout."""
    return f"{code} course handout: {question}" if _ADMIN.search(question or "") else question


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

"""Past papers as question units, each with its memorandum where one can be attached.

The passage corpus holds class tests, exams, problem sets and tutorials as pages. A
student stuck on a problem is better served by the past question of the same kind and
the memorandum that solves it, so this module cuts each paper into its top-level
questions and attaches the matching memorandum segment, under the rules of
protocols/problems.md:

- a paper is cut at its question headers, in whichever of three styles it uses
  (`Question N [m marks]`, `N. (m marks)`, or plain `N.` numbering with PART headings);
  when a file has `Question N` headers only those count, so an older question pasted
  into a newer paper as a figure does not become a unit of its own;
- a memorandum is cut the same way and linked by question number; the link is
  `accepted` only when the memorandum repeats the question's opening words (token
  Jaccard of the first forty tokens at least 0.3), otherwise it is `number only`;
- problem-set solution files do not follow the sets' numbering and three of them are
  scans with no text, so they link at file level only; exams 2011-2024 have no
  memorandum and say so.

Nothing here reads a memorandum for correctness: a memorandum surfaced is a
memorandum's error surfaced, and the protocol says so.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .ingest import _pdf_pages
from .retrieval import Chunk

RAW = Path("corpus/raw")
JACCARD_MIN = 0.3
OPENING_TOKENS = 40

_Q_HEADER = re.compile(r"^\s*Question\s+(\d{1,2})\s*(?::\s*([^\[\n]*?))?\s*(?:\[\s*(\d+)\s*marks?\s*\])?\s*$", re.I | re.M)
_MARKS_HEADER = re.compile(r"^\s*(\d{1,2})\.\s*\(\s*(\d+)\s*marks?\s*\)", re.M)
_PLAIN_HEADER = re.compile(r"^\s*(\d{1,2})\.\s+(?=\S)", re.M)
# A fourth style, seen in the 2016 exam's Part B: "Problem N:" headings.
_PROBLEM_HEADER = re.compile(r"^\s*Problem\s+(\d{1,2})\s*:", re.M)
_SOLUTION = re.compile(r"\bSolutions?\s*:", re.I)
_TOKEN = re.compile(r"[a-z0-9]+")
# Where a paper's attached formula sheet begins: a unit that runs to the end of the file
# would otherwise carry the whole table of transform pairs as its "question".
_SHEET = re.compile(r"^\s*(Fourier transform properties|Discrete-time Fourier transform properties|"
                    r"z-transform properties|Transform pairs|Common (?:transform|DTFT|z-transform) pairs|"
                    r"Formula sheet|Table of standard|Standard (?:Fourier|transform)|Useful (?:formulae|formulas)|"
                    r"Information sheet|Sequences x\[n\], y\[n\])", re.I | re.M)


def _without_sheet(text: str) -> str:
    m = _SHEET.search(text)
    return text[:m.start()].rstrip() if m and m.start() > 40 else text


@dataclass
class Unit:
    unit_id: str
    source: str            # the paper's file name
    kind: str              # class test | exam | problem set | tutorial
    year: int | None
    number: int
    marks: int | None
    page: int
    text: str              # the question as set (memorandum files: the part before "Solution:")
    solution_text: str = ""   # only when the same file carries the solution
    is_memo_file: bool = False
    memo: dict | None = None  # {"file", "unit_id", "page", "link"} when a memorandum is attached

    def as_chunk(self) -> Chunk:
        return Chunk(chunk_id=self.unit_id, source=self.source, page=self.page, text=self.text, topic=None)

    def reference(self) -> str:
        label = f"{self.kind} {self.year}" if self.year else self.kind
        marks = f", {self.marks} marks" if self.marks else ""
        return f"{label}, question {self.number}{marks} ({self.source}, p.{self.page})"


# ----------------------------------------------------------------------------- files

def classify(name: str) -> tuple[str, int | None, bool] | None:
    """(kind, year, is_memo) for a paper-like file, else None."""
    stem = Path(name).stem
    low = stem.lower()
    is_memo = "sol" in low
    m = re.match(r"classtest(?:_2_)?(\d{4})b?", low)
    if m:
        return "class test", int(m.group(1)), is_memo
    m = re.match(r"exam(\d{4})", low)
    if m:
        return "exam", int(m.group(1)), is_memo
    if re.match(r"prob\d{2}", low):
        return "problem set", None, is_memo
    m = re.search(r"class[ _]test[ _]\d[ _](?:of[ _]\d[ _])?(\d{4})|class[ _]test[ _]\d[ _](\d{4})", low)
    if m:
        return "class test", int(m.group(1) or m.group(2)), is_memo
    m = re.search(r"final[ _]exam[ _](\d{4})", low)
    if m:
        return "exam", int(m.group(1)), is_memo
    m = re.search(r"(\d{4}) ml tutorial", low)
    if m:
        return "tutorial", int(m.group(1)), is_memo
    return None


def memo_for(name: str, names: set[str]) -> tuple[str | None, str]:
    """The memorandum file for a paper, and how it is linked: 'question' or 'file'."""
    stem = Path(name).stem
    low = stem.lower()
    if low.startswith("prob"):
        for cand in (f"{stem}sol.pdf", f"{stem}solutions.pdf"):
            if cand in names:
                return cand, "file"
        return None, "none"
    if "ml tutorial" in low and "solution" not in low:
        cand = name.replace("Tutorial", "Tutorial Solution")
        return (cand, "question") if cand in names else (None, "none")
    cand = f"{stem}_sol.pdf"
    if cand in names:
        return cand, "question"
    return None, "none"


def paper_files(raw: Path = RAW) -> list[tuple[Path, str, int | None, bool]]:
    out = []
    for path in sorted(raw.glob("*.pdf")):
        info = classify(path.name)
        if info:
            out.append((path, *info))
    return out


# ------------------------------------------------------------------------ segmentation

def _page_text(pages: list[tuple[int, str]]) -> tuple[str, list[int]]:
    """All pages as one string, with the page number of every character offset."""
    text_parts, owner = [], []
    for number, text in pages:
        text_parts.append(text + "\n")
        owner.extend([number] * (len(text) + 1))
    return "".join(text_parts), owner


def _headers(text: str) -> tuple[str, list[tuple[int, int, int | None, str]]]:
    """The style used and [(offset, number, marks, title)] of the top-level headers.

    Numbers must run 1, 2, 3 ... from the first header found; a header out of sequence
    is noise (a page number, a list inside a question, an older paper pasted as a figure).
    """
    found = [(m.start(), int(m.group(1)), int(m.group(3)) if m.group(3) else None, (m.group(2) or "").strip())
             for m in _Q_HEADER.finditer(text)]
    style = "question"
    if not found:
        found = [(m.start(), int(m.group(1)), int(m.group(2)), "") for m in _MARKS_HEADER.finditer(text)]
        style = "marks"
    if not found:
        found = [(m.start(), int(m.group(1)), None, "") for m in _PLAIN_HEADER.finditer(text)]
        style = "plain"
    if style != "question":
        found = sorted(found + [(m.start(), int(m.group(1)), None, "")
                                for m in _PROBLEM_HEADER.finditer(text)])
    ordered, expect = [], 1
    for h in found:
        if h[1] == expect:
            ordered.append(h)
            expect += 1
        elif h[1] == 1 and expect > 1 and style != "question":
            # a second numbering run (Part B restarting at 1) - keep going from it
            ordered.append(h)
            expect = 2
    return style, ordered


def segment(path: Path, kind: str, year: int | None, is_memo: bool) -> list[Unit]:
    pages = _pdf_pages(path)
    text, owner = _page_text(pages)
    style, headers = _headers(text)
    units: list[Unit] = []
    for i, (start, number, marks, title) in enumerate(headers):
        end = headers[i + 1][0] if i + 1 < len(headers) else len(text)
        body = _without_sheet(text[start:end].strip())
        q_text, sol_text = body, ""
        m = _SOLUTION.search(body)
        if m and (is_memo or kind in ("class test", "exam")):
            q_text, sol_text = body[:m.start()].strip(), body[m.start():].strip()
        units.append(Unit(
            unit_id=f"{path.stem}#q{len(units) + 1}", source=path.name, kind=kind, year=year,
            number=number, marks=marks, page=owner[min(start, len(owner) - 1)],
            text=q_text, solution_text=sol_text, is_memo_file=is_memo))
    return units


# ----------------------------------------------------------------------------- linking

def _tokens(text: str, n: int = OPENING_TOKENS) -> set[str]:
    return set(_TOKEN.findall(text.lower())[:n])


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def _opening(unit: Unit) -> str:
    """The question's words after its header line, so the header's own words do not count."""
    lines = unit.text.splitlines()
    return " ".join(lines[1:]) if len(lines) > 1 else unit.text


def link(papers: dict[str, list[Unit]], memos: dict[str, list[Unit]], names: set[str]) -> None:
    for name, units in papers.items():
        memo_file, how = memo_for(name, names)
        if memo_file is None:
            for u in units:
                u.memo = None
            continue
        if how == "file":
            for u in units:
                u.memo = {"file": memo_file, "unit_id": None, "page": 1, "link": "file only"}
            continue
        by_number: dict[int, Unit] = {}
        for mu in memos.get(memo_file, []):
            by_number.setdefault(mu.number, mu)
        for u in units:
            mu = by_number.get(u.number)
            if mu is None:
                u.memo = {"file": memo_file, "unit_id": None, "page": 1, "link": "file only"}
                continue
            score = jaccard(_tokens(_opening(u)), _tokens(_opening(mu)))
            u.memo = {"file": memo_file, "unit_id": mu.unit_id, "page": mu.page,
                      "link": "accepted" if score >= JACCARD_MIN else "number only",
                      "overlap": round(score, 3)}


def build(raw: Path = RAW) -> list[Unit]:
    """Every question unit of every paper, memoranda attached; memorandum-only files
    (no separate paper) are returned as units too, with their own solution text."""
    files = paper_files(raw)
    names = {p.name for p, *_ in files}
    papers: dict[str, list[Unit]] = {}
    memos: dict[str, list[Unit]] = {}
    for path, kind, year, is_memo in files:
        units = segment(path, kind, year, is_memo)
        (memos if is_memo else papers)[path.name] = units
    link(papers, memos, names)
    out: list[Unit] = []
    for name in sorted(papers):
        out.extend(papers[name])
    # memorandum files with no paper of their own are the paper (the 2025-2026 files)
    paired = {memo_for(n, names)[0] for n in papers}
    for name in sorted(memos):
        if name not in paired:
            for u in memos[name]:
                u.memo = {"file": name, "unit_id": u.unit_id, "page": u.page, "link": "same file"}
                out.append(u)
    return out


# ------------------------------------------------------------------------------ store

def save(units: list[Unit], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(asdict(u), ensure_ascii=False) + "\n" for u in units),
                    encoding="utf-8", newline="\n")


def load(path: Path) -> list[Unit]:
    return [Unit(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ProblemIndex:
    """Fused retrieval over question units, with the retriever the passages use."""

    def __init__(self, units: list[Unit], method: str = "fused"):
        from .lectures import _retriever_over

        self.units = {u.unit_id: u for u in units}
        self.retriever = _retriever_over([u.as_chunk() for u in units], method)

    def __len__(self) -> int:
        return len(self.units)

    def search(self, question: str, top_k: int = 3) -> list[tuple[Unit, float]]:
        hits = self.retriever.search(question, top_k)
        return [(self.units[h.chunk.chunk_id], h.score) for h in hits]

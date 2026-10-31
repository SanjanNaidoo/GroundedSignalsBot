"""What else the platform shows beside an answer: lecture minutes and past questions.

The tutor answers from passages; a student also wants the minute of the lecture that
explains the idea and a past question of the same kind with its memorandum. Both are
found here with the indexes measured in E24 and E25, and both are withheld when the
best match is not close enough, under the thresholds those experiments fixed:

- a lecture window is shown only when the dense similarity of the best window is at
  least 0.72 (protocols/lectures.md, amendment of 12 October, set on the development
  half: it withholds three of four references the lectures cannot supply);
- a past question is shown only when the best unit's dense similarity is at least 0.74
  (protocols/problems.md, amendment of 12 October, set on the twenty registered queries).

A reference is a file name and a second, or a paper, question number and page, never a
quotation the student cannot check. Where the recording or the PDF is on this machine,
the web server serves it under /video and /doc so the reference opens at that second or
that page; the files themselves are never committed.

Nothing here reads a memorandum for correctness, and a past question from a file that
carries its worked answers in the text is flagged, so the student knows the answer is in
what they are about to open.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .lectures import LectureIndex, Window
from .problems import ProblemIndex, Unit, load as load_units

LECTURE_MIN_SIMILARITY = 0.72
PROBLEM_MIN_SIMILARITY = 0.74
TRANSCRIPTS = Path("lectures/transcripts")
UNITS = Path("corpus/processed/problems.jsonl")
VIDEOS = Path("Lecture Videos")
RAW = Path("corpus/raw")
SNIPPET = 240

# Found while labelling E25c: these exam files carry their worked answers interleaved
# with the questions, with no "Solution:" marker to cut on. A unit from them hands the
# student an answer, and the card says so.
PAPERS_WITH_ANSWERS = {"exam2021.pdf", "exam2022.pdf", "exam2023.pdf"}


def _snippet(text: str, n: int = SNIPPET) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= n else flat[: n - 1].rsplit(" ", 1)[0] + "…"


@dataclass
class Resources:
    lectures: list[dict]
    problems: list[dict]
    lecture_note: str
    problem_note: str

    def as_dict(self) -> dict:
        return {"lectures": self.lectures, "problems": self.problems,
                "lecture_note": self.lecture_note, "problem_note": self.problem_note}


class ResourceFinder:
    """Both indexes, built once at startup, searched per question."""

    def __init__(self, *, transcripts: Path = TRANSCRIPTS, units: Path = UNITS,
                 videos: Path = VIDEOS, raw: Path = RAW, method: str = "fused"):
        self.lectures: LectureIndex | None = None
        self.problems: ProblemIndex | None = None
        self.videos = videos
        self.raw = raw
        if transcripts.exists() and any(transcripts.glob("*.json")):
            self.lectures = LectureIndex.from_dir(transcripts, method=method)
        if units.exists():
            loaded = load_units(units)
            if loaded:
                self.problems = ProblemIndex(loaded, method=method)
        self._video_files = {p.name: p for p in videos.rglob("*") if p.suffix.lower() in (".mp4", ".m4v")} if videos.exists() else {}

    # -- what the server can serve -------------------------------------------------
    def video_path(self, name: str) -> Path | None:
        return self._video_files.get(name)

    def doc_path(self, name: str) -> Path | None:
        path = self.raw / name
        return path if path.is_file() and path.suffix.lower() == ".pdf" else None

    def describe(self) -> str:
        parts = []
        parts.append(f"{len(self.lectures)} lecture windows" if self.lectures else "no lecture index")
        parts.append(f"{len(self.problems)} past questions" if self.problems else "no past-question index")
        parts.append(f"{len(self._video_files)} recordings on disk")
        return ", ".join(parts)

    # -- the search ------------------------------------------------------------------
    @staticmethod
    def _dense(retriever):
        return retriever._retrievers[1] if hasattr(retriever, "_retrievers") else retriever

    def find(self, query: str, top_k: int = 3, lectures: bool = True) -> Resources:
        """Lecture minutes and past questions for a question. With `lectures` false the
        lecture index is not searched at all, for a model that does not offer them."""
        lectures, lecture_note = self._lectures(query, top_k) if lectures else ([], "")
        problems, problem_note = self._problems(query, top_k)
        return Resources(lectures=lectures, problems=problems,
                         lecture_note=lecture_note, problem_note=problem_note)

    def _lectures(self, query: str, top_k: int) -> tuple[list[dict], str]:
        if self.lectures is None or not query.strip():
            return [], "no lecture index is loaded"
        best = self._dense(self.lectures.retriever).search(query, 1)
        top = best[0].score if best else 0.0
        if top < LECTURE_MIN_SIMILARITY:
            return [], "no lecture explains this closely enough to point at a minute"
        out = []
        for window, score in self.lectures.search(query, top_k):
            out.append(self._lecture_card(window, score))
        return out, ""

    def _lecture_card(self, w: Window, score: float) -> dict:
        path = self.video_path(w.recording)
        return {
            "recording": w.recording, "series": w.series, "start_s": round(w.start, 1),
            "end_s": round(w.end, 1), "stamp": w.stamp, "reference": w.reference(),
            "snippet": _snippet(w.text), "score": round(score, 5),
            "video": f"/video/{w.recording}#t={int(w.start)}" if path else None,
        }

    def _problems(self, query: str, top_k: int) -> tuple[list[dict], str]:
        if self.problems is None or not query.strip():
            return [], "no past-question index is loaded"
        best = self._dense(self.problems.retriever).search(query, 1)
        top = best[0].score if best else 0.0
        if top < PROBLEM_MIN_SIMILARITY:
            return [], "no past question is close enough to this one to show"
        out = []
        for unit, score in self.problems.search(query, top_k):
            out.append(self._problem_card(unit, score))
        return out, ""

    def _problem_card(self, u: Unit, score: float) -> dict:
        memo = u.memo or {}
        link = memo.get("link", "none")
        memo_card = None
        if link in ("accepted", "same file"):
            memo_card = {"file": memo["file"], "page": memo.get("page", 1), "kind": "memorandum",
                         "doc": f"/doc/{memo['file']}#page={memo.get('page', 1)}" if self.doc_path(memo["file"]) else None}
        elif link == "number only":
            memo_card = {"file": memo["file"], "page": memo.get("page", 1), "kind": "memorandum, question not matched by text",
                         "doc": f"/doc/{memo['file']}#page={memo.get('page', 1)}" if self.doc_path(memo["file"]) else None}
        elif link == "file only":
            memo_card = {"file": memo["file"], "page": 1, "kind": "solutions file, not matched by question",
                         "doc": f"/doc/{memo['file']}" if self.doc_path(memo["file"]) else None}
        return {
            "unit": u.unit_id, "reference": u.reference(), "kind": u.kind, "year": u.year,
            "number": u.number, "marks": u.marks, "source": u.source, "page": u.page,
            "snippet": _snippet(u.text), "score": round(score, 5),
            "doc": f"/doc/{u.source}#page={u.page}" if self.doc_path(u.source) else None,
            "answers_in_text": u.source in PAPERS_WITH_ANSWERS or bool(u.solution_text),
            "memo": memo_card,
        }

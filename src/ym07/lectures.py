"""Lecture transcripts as a searchable index of timestamped windows.

A transcript written by scripts/transcribe_lectures.py is a list of segments, each a
sentence or so with a start and end. A student's question is rarely answered by one
sentence, so the segments are joined into windows of about seventy-five seconds that
overlap by half, and the windows are what retrieval ranks. A window carries the
recording it came from, its series, its start and end, so that a result is always a
file name and a second rather than a passage of text with no address.

The retriever is the same one the course passages use - BM25 and the dense model fused
by rank, from ym07.retrieval_eval - built over the windows instead of the passages.
That is deliberate: protocols/lectures.md asks whether a student can be pointed at the
right minute of the right lecture, and that question should be answered with the method
already measured on passages, not with a second retriever whose own quality would then
be entangled with the answer.

Nothing here reads the recordings. The transcripts are text; the windows are text; a
reference the platform shows is a file name and a time that the student opens in the
course's own recording store.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .retrieval import Chunk

WINDOW_S = 75.0      # about a sentence's worth of lecture either side of the point asked about
STEP_S = 37.5        # half a window, so every moment sits inside two windows
MIN_WORDS = 12       # a window with fewer words is a pause or a tail, not an explanation

_WS = re.compile(r"\s+")


@dataclass(frozen=True)
class Window:
    """One searchable stretch of one recording."""

    window_id: str
    recording: str      # the file name, e.g. L03b.mp4
    series: str         # dsp or ml
    start: float
    end: float
    text: str
    segment_ids: tuple[int, ...] = field(default_factory=tuple)

    @property
    def stamp(self) -> str:
        m, s = divmod(int(self.start), 60)
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def reference(self) -> str:
        """How the platform names this window to a student: the file and the second."""
        return f"{self.recording} at {self.stamp}"

    def as_chunk(self) -> Chunk:
        """The window in the shape the passage retrievers index.

        `source` is the recording, `page` the window's start in whole seconds, so a
        citation produced by the existing code reads as a file and a time.
        """
        return Chunk(chunk_id=self.window_id, source=self.recording,
                     page=int(self.start), text=self.text, topic=None)


def load_transcript(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not data.get("complete"):
        raise ValueError(f"{path.name}: transcript is not marked complete")
    return data


def windows_of(transcript: dict, *, window_s: float = WINDOW_S, step_s: float = STEP_S,
               min_words: int = MIN_WORDS) -> list[Window]:
    """Cut one transcript into overlapping windows.

    A window runs from t to t + window_s and holds every segment that starts inside it.
    Steps advance by step_s until the last segment's end is passed. A segment is never
    split, so a window's real span is the span of the segments it holds, and that is
    what is recorded - the nominal grid only decides membership.
    """
    segments = [s for s in transcript.get("segments", []) if s.get("text")]
    if not segments:
        return []
    recording = transcript["recording"]
    series = transcript.get("series", "")
    last_end = max(s["end"] for s in segments)
    out: list[Window] = []
    t = 0.0
    n = 0
    while t < last_end:
        held = [s for s in segments if t <= s["start"] < t + window_s]
        if held:
            text = _WS.sub(" ", " ".join(s["text"].strip() for s in held)).strip()
            if len(text.split()) >= min_words:
                stem = Path(recording).stem
                out.append(Window(
                    window_id=f"{stem}-w{n:04d}", recording=recording, series=series,
                    start=held[0]["start"], end=held[-1]["end"], text=text,
                    segment_ids=tuple(s["id"] for s in held)))
                n += 1
        t += step_s
    return out


def load_windows(transcripts_dir: Path, **kwargs) -> list[Window]:
    """Every window of every complete transcript in the directory, in file order."""
    windows: list[Window] = []
    for path in sorted(transcripts_dir.glob("*.json")):
        try:
            data = load_transcript(path)
        except (ValueError, json.JSONDecodeError):
            continue
        windows.extend(windows_of(data, **kwargs))
    return windows


def _retriever_over(chunks: list[Chunk], method: str):
    """The same construction as retrieval.build_retriever, over windows instead of a file.

    Kept in step with that function by hand: the point of indexing the windows with the
    passage retriever is that the two cannot differ in method.
    """
    if method == "tfidf":
        from .retrieval import LexicalRetriever

        return LexicalRetriever(chunks)

    from .retrieval_eval import BM25Retriever, DenseRetriever, FusedRetriever

    if method == "bm25":
        return BM25Retriever(chunks)
    if method == "dense":
        return DenseRetriever(chunks)
    if method == "fused":
        return FusedRetriever(BM25Retriever(chunks), DenseRetriever(chunks))
    raise ValueError(f"Unknown retrieval method '{method}'")


class LectureIndex:
    """Fused retrieval over lecture windows, with the retriever the passages use."""

    def __init__(self, windows: list[Window], method: str = "fused"):
        self.windows = {w.window_id: w for w in windows}
        self.retriever = _retriever_over([w.as_chunk() for w in windows], method)

    @classmethod
    def from_dir(cls, transcripts_dir: Path, method: str = "fused", **kwargs) -> "LectureIndex":
        return cls(load_windows(transcripts_dir, **kwargs), method)

    def __len__(self) -> int:
        return len(self.windows)

    def search(self, question: str, top_k: int = 3) -> list[tuple[Window, float]]:
        hits = self.retriever.search(question, top_k)
        return [(self.windows[h.chunk.chunk_id], h.score) for h in hits]

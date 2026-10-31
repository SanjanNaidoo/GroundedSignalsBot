"""Worked solutions: the training material the first fine-tune left out, and the
held-out ruler used to judge whether adding it helped.

The v2 adapter answers in a median of 104 tokens and scores 0.35 on the benchmark's
calculation items, against 0.67 for the same base model untuned. The cause is in the
training set: 903 answers at a median of 60 words, only 10% of which show any
arithmetic. The model learned what an answer looks like, and on a calculation, not
showing the working is not doing the working.

The course has the missing material already - twenty-seven past papers and tutorials
with worked solutions, in its own notation. This module uses them twice, and the split
between the two uses is the part that matters:

- **Training.** Worked question-and-answer pairs that set out each step, to be mixed
  with the existing short expository pairs rather than replacing them. The aim is a
  model that stays brief on a concept question and shows its working on a calculation,
  because the brevity is what makes it three times faster than the frontier arm at no
  API cost, and that is worth keeping.
- **Validation.** A held-out set of calculation questions whose answers are known, so
  the fix can be measured on questions the model was not trained on and that are not
  in the benchmark. The benchmark items are already known to fail; designing against
  them would be fitting to the test.

The split is by *file*, not by chunk. A question and its solution can land in adjacent
chunks of the same paper, so splitting mid-paper would leak a validation answer into
training. It is drawn once from a recorded seed and written to disk, so every later
step reads the same split rather than redrawing it.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

from .digests import content_digest
from .retrieval import Chunk

# Files whose names mark them as carrying worked answers rather than bare questions.
SOLUTION_FILE = re.compile(r"sol|memo|answer", re.I)

VALIDATION_SHARE = 0.35
SPLIT_SEED = 41140

EXTRACT_SYSTEM = """You prepare test questions for checking whether a tutor can do EEE4114F calculations.

Below is a page of worked solutions from an EEE4114F past paper or tutorial, extracted from a PDF, so its layout and formulas may be damaged.

Find questions on that page that ask for a calculation with a definite numerical answer. For each one:
- Write the question so it stands alone, including every given it needs. Repair extraction damage and write mathematics in LaTeX.
- Do not write the solution into the question.
- Skip any question needing a figure, sketch, plot, table or an earlier answer you cannot include.
- Skip any question whose answer is not a number or a short list of numbers.
- List the answer values: the final answer first, then any intermediate values a correct solution must pass through. Give plain decimal numbers, no units and no symbols.

Reply for each question in exactly this form, and nothing else:

### QUESTION
<the question>
### VALUES
<one number per line>

If the page has no suitable question, reply NONE."""

WORKED_SYSTEM = """You write worked solutions for students of EEE4114F (Digital Signal Processing and Machine Learning).

Below is a page of worked solutions from an EEE4114F past paper or tutorial, extracted from a PDF, so its layout and formulas may be damaged.

Write up to 3 question-and-answer pairs a student revising this material might work through. For each:
- The question must stand alone and ask for a calculation.
- The answer must show the working: set out each step of the arithmetic on its own line, evaluate it, and only then state the result. A student should be able to follow every step without doing any arithmetic themselves.
- Every number and formula must come from the page. Add nothing it does not state.
- Write mathematics in LaTeX.
- Skip any question needing a figure, sketch, plot or table.

Reply for each pair in exactly this form, and nothing else:

### QUESTION
<the question>
### ANSWER
<the worked solution>

If the page has no suitable material, reply NONE."""

_QUESTION = re.compile(r"(?m)^\s*###\s*QUESTION\s*$")
_VALUES = re.compile(r"(?m)^\s*###\s*VALUES\s*$")
_ANSWER = re.compile(r"(?m)^\s*###\s*ANSWER\s*$")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class Split:
    """Which solution files train the model and which measure it."""

    train_files: list[str] = field(default_factory=list)
    validate_files: list[str] = field(default_factory=list)
    seed: int = SPLIT_SEED

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "drawn_by": "file, so a question and its answer cannot land on both sides",
            "train_files": sorted(self.train_files),
            "validate_files": sorted(self.validate_files),
        }


def solution_chunks(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    """The corpus chunks that come from files carrying worked answers, by file."""
    by_file: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        if SOLUTION_FILE.search(chunk.source):
            by_file.setdefault(chunk.source, []).append(chunk)
    return by_file


def draw_split(by_file: dict[str, list[Chunk]], *, share: float = VALIDATION_SHARE,
               seed: int = SPLIT_SEED) -> Split:
    """Reserve a share of the solution files for validation. Drawn once, then recorded."""
    files = sorted(by_file)
    rng = random.Random(seed)
    held = sorted(rng.sample(files, max(1, round(len(files) * share))))
    return Split(train_files=[f for f in files if f not in held],
                 validate_files=held, seed=seed)


def parse_extraction(text: str) -> list[dict]:
    """Questions and their answer values, from one reply to EXTRACT_SYSTEM."""
    out: list[dict] = []
    for block in _QUESTION.split(text or "")[1:]:
        if not _VALUES.search(block):
            continue
        question, values_text = _VALUES.split(block, maxsplit=1)
        values = [float(v) for v in _NUMBER.findall(values_text)]
        question = question.strip()
        if question and values:
            out.append({"question": question, "values": values})
    return out


def parse_worked(text: str) -> list[dict]:
    """Question-and-worked-answer pairs, from one reply to WORKED_SYSTEM."""
    out: list[dict] = []
    for block in _QUESTION.split(text or "")[1:]:
        if not _ANSWER.search(block):
            continue
        question, answer = _ANSWER.split(block, maxsplit=1)
        question, answer = question.strip(), answer.strip()
        if question and answer:
            out.append({"question": question, "answer": answer})
    return out


def shows_working(answer: str, minimum: int = 2) -> bool:
    """Does this answer evaluate arithmetic, or only describe it?

    The test is an equals sign with digits on both sides, more than once - the shape
    of a step carried out rather than a formula quoted. It is the same test used to
    measure that only 10% of the original training answers did any arithmetic, so the
    before and after numbers mean the same thing.
    """
    return len(re.findall(r"\d[^=\n]{0,40}=\s*-?\d", answer or "")) >= minimum


def checks_from_values(item_id: str, values: list[float]) -> list[dict]:
    """Turn extracted answer values into rules the existing checker can score.

    Only values distinctive enough to be unlikely by accident become rules. A bare 2
    or 4 appears in almost any worked solution, so small integers are dropped rather
    than scored loosely - a rule that is usually true measures nothing.
    """
    rules = []
    for n, value in enumerate(values, start=1):
        # The extraction prompt puts the final answer first, and the final answer is
        # the point of the question, so it becomes a rule whatever its size. Among the
        # intermediate values, small whole numbers are dropped: a bare 2 or 4 appears
        # in almost any worked solution, and a rule that is usually true measures
        # nothing. Dropping them cost six of fourteen questions on the first pass,
        # which was too strict - the final answer was often exactly such a number.
        if n > 1 and abs(value) < 10 and float(value).is_integer():
            continue
        rules.append({
            "label": f"{'final answer' if n == 1 else f'value {n}'}: {value:g}",
            "kind": "number",
            "value": value,
            "tolerance": 0.01,
        })
    return rules


def write_validation_set(rows: list[dict], path: Path) -> dict:
    """The held-out calculation set, with its rules, as one committed file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = [r for r in rows if r.get("checks")]
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "questions": len(kept),
        "dropped_no_checkable_value": len(rows) - len(kept),
        "rules": sum(len(r["checks"]) for r in kept),
        "digest": content_digest(path.read_bytes()),
    }

"""Blind human grading.

Two things make a grade defensible: the grader cannot see which system produced
an answer, and all conditions for one item are graded together so the standard
does not drift between them. Both are enforced here — responses are pooled per
item, shuffled, and relabelled 'System A', 'System B'.

Grades live in grades/grades.jsonl, keyed by response id and appended as you go,
so grading can be stopped and resumed. That file is expensive human labour and
belongs in version control; run outputs are regenerable and do not.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path

from .conditions import CITED_CONDITIONS
from .items import Item, ItemSet

# How a correctness fraction is reported as a word. The fraction stays the number of
# record; these labels exist because a table of 0.83s is harder to read at a glance
# than "correct", and a reader asked for three buckets rather than a scale.
CORRECT_AT = 0.8
PARTIAL_AT = 0.4

# Faithfulness: is what the answer states actually supported by the course material?
# Separate from correctness, because an answer can make every required point and still
# invent a figure alongside them - and because an answer with no retrieval can only be
# judged against the reference answer and the grader's knowledge of the course.
FAITHFULNESS = {
    "g": "grounded",      # every claim traceable to the material shown or the course
    "p": "partly",        # supported in the main, with unsupported additions
    "u": "unfaithful",    # states something the material does not support, as fact
}


@dataclass
class Grade:
    response_id: str
    item_id: str
    condition: str
    must_include_hits: list[bool]
    must_not_include_hits: list[bool]
    policy_pass: bool | None
    citations_valid: bool | None
    faithfulness: str | None = None
    note: str = ""
    # "hand" when the grader decided it, "csv" when it was read in from a file. A mark
    # made with a model's help is not a hand mark, and the two must not be mixed by
    # accident, so imported grades live in their own file and say where they came from.
    source: str = "hand"

    @property
    def correctness(self) -> float:
        if not self.must_include_hits:
            return 0.0
        return sum(self.must_include_hits) / len(self.must_include_hits)

    @property
    def hard_fail(self) -> bool:
        """Any must_not_include claim present, or an invalid citation.

        Reported separately from correctness. A confident wrong answer is worse
        than a refusal, and averaging the two hides exactly that.
        """
        if any(self.must_not_include_hits):
            return True
        return self.citations_valid is False

    @property
    def label(self) -> str:
        """The correctness fraction as one of three words.

        A hard fail is capped at "partly correct" however many points were made: an
        answer that states something known to be false has not answered correctly,
        and letting it read as "correct" in a summary table would mislead. The two
        measures are still reported separately, as the protocol requires.
        """
        if self.correctness >= CORRECT_AT and not self.hard_fail:
            return "correct"
        if self.correctness >= PARTIAL_AT:
            return "partly correct"
        return "incorrect"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["correctness"] = round(self.correctness, 4)
        d["hard_fail"] = self.hard_fail
        d["label"] = self.label
        return d


def load_grades(path: Path) -> dict[str, Grade]:
    if not path.exists():
        return {}
    grades: dict[str, Grade] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        grades[obj["response_id"]] = Grade(
            response_id=obj["response_id"],
            item_id=obj["item_id"],
            condition=obj["condition"],
            must_include_hits=obj["must_include_hits"],
            must_not_include_hits=obj["must_not_include_hits"],
            policy_pass=obj.get("policy_pass"),
            citations_valid=obj.get("citations_valid"),
            faithfulness=obj.get("faithfulness"),
            note=obj.get("note", ""),
            source=obj.get("source", "hand"),
        )
    return grades


def append_grade(path: Path, grade: Grade) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(grade.to_dict(), ensure_ascii=False) + "\n")


def _ask_yes_no(prompt: str) -> bool:
    while True:
        reply = input(f"{prompt} [y/n] ").strip().lower()
        if reply in ("y", "yes"):
            return True
        if reply in ("n", "no"):
            return False
        print("    Please answer y or n.")


def _ask_choice(prompt: str, choices: dict[str, str]) -> str:
    keys = "/".join(choices)
    while True:
        reply = input(f"{prompt} [{keys}] ").strip().lower()
        if reply in choices:
            return choices[reply]
        for key, value in choices.items():
            if reply == value:
                return value
        print(f"    Please answer one of: {', '.join(f'{k} ({v})' for k, v in choices.items())}")


def _collapse_repeats(text: str, keep: int = 3) -> str:
    """Fold a line repeated many times in a row into a count.

    A model that loops can fill an answer with hundreds of copies of one line —
    C0-oss answers `eee4114f-scope-001` with 671 empty table rows. Printing them
    all scrolls the question, the reference answer and the other systems off the
    screen, so the grader is left judging what they can still see. The fold says
    plainly what happened, which is the thing being graded anyway.
    """
    lines = text.split("\n")
    out: list[str] = []
    index = 0
    while index < len(lines):
        run = 1
        while index + run < len(lines) and lines[index + run] == lines[index]:
            run += 1
        if run > keep * 2:
            out.extend([lines[index]] * keep)
            out.append(f"[... the same line {run - keep} more times ...]")
        else:
            out.extend([lines[index]] * run)
        index += run
    return "\n".join(out)


def _wrap(text: str, width: int = 88, indent: str = "    ") -> str:
    import textwrap

    out = []
    for para in _collapse_repeats(text).split("\n"):
        out.extend(textwrap.wrap(para, width=width) or [""])
    return "\n".join(indent + line for line in out)


def grade_interactively(
    *,
    item_set: ItemSet,
    responses_by_item: dict[str, list[dict]],
    grades_path: Path,
    seed: int | None = None,
    limit: int | None = None,
    only: set[str] | None = None,
) -> int:
    """Walk the grader through every ungraded response. Returns the count graded.

    `only` restricts grading to a drawn sample of response ids - the audit the protocol
    asks for - without changing anything else: the answers are still pooled per item,
    shuffled and unlabelled.
    """
    existing = load_grades(grades_path)
    rng = random.Random(seed)
    items_by_id: dict[str, Item] = {item.id: item for item in item_set}

    def wanted(record: dict) -> bool:
        return record["response_id"] not in existing and (
            only is None or record["response_id"] in only
        )

    pending_items = [
        item for item in item_set
        if any(wanted(r) for r in responses_by_item.get(item.id, []))
    ]
    if limit is not None:
        pending_items = pending_items[:limit]

    if not pending_items:
        print("Nothing to grade — every response already has a grade.")
        return 0

    print(
        f"\n{len(pending_items)} item(s) to grade. Answers are shuffled and unlabelled.\n"
        "Ctrl-C to stop; progress is saved after each response.\n"
    )

    graded = 0
    for position, item in enumerate(pending_items, start=1):
        responses = [
            r for r in responses_by_item.get(item.id, [])
            if r["response_id"] not in existing
        ]
        if not responses:
            continue
        rng.shuffle(responses)

        print("=" * 88)
        print(f"ITEM {position}/{len(pending_items)}  {item.id}  "
              f"[{item.type}, {item.topic}, difficulty {item.difficulty}]")
        print("=" * 88)
        print("\nQUESTION")
        print(_wrap(item.question))
        print("\nREFERENCE ANSWER")
        print(_wrap(item.reference_answer or "(none recorded)"))

        for idx, record in enumerate(responses):
            label = chr(ord("A") + idx)
            print("\n" + "-" * 88)
            print(f"SYSTEM {label}")
            print("-" * 88)
            answer = record["answer"]
            if answer.get("error"):
                print(_wrap(f"[CALL FAILED: {answer['error']}]"))
            else:
                print(_wrap(answer.get("text") or "(empty response)"))

            if record.get("retrieved"):
                print("\n    CITED SOURCES AVAILABLE TO THIS SYSTEM")
                for hit in record["retrieved"]:
                    print(f"      [{hit['tag']}] {hit['source']}")

            print()
            must_include_hits = [
                _ask_yes_no(f"  Makes this point? — {point}")
                for point in item.must_include
            ]
            must_not_include_hits = [
                _ask_yes_no(f"  Contains this failure? — {bad}")
                for bad in item.must_not_include
            ]

            policy_pass = None
            if item.is_policy_item:
                policy_pass = _ask_yes_no(
                    "  Behaved correctly (declined / scaffolded rather than answering)?"
                )

            citations_valid = None
            if record["condition"] in CITED_CONDITIONS and record.get("retrieved"):
                citations_valid = _ask_yes_no(
                    "  Does every citation actually support the claim it is attached to?"
                )

            # Asked of every answer, retrieval or none. A system with no passages is
            # judged against the reference answer and the course: the question is
            # whether it stated anything the material does not support, not whether
            # it had passages to quote.
            faithfulness = _ask_choice(
                "  Is everything it states supported by the course material?",
                FAITHFULNESS,
            )

            note = input("  Note (optional, Enter to skip): ").strip()

            grade = Grade(
                response_id=record["response_id"],
                item_id=item.id,
                condition=record["condition"],
                must_include_hits=must_include_hits,
                must_not_include_hits=must_not_include_hits,
                policy_pass=policy_pass,
                citations_valid=citations_valid,
                faithfulness=faithfulness,
                note=note,
            )
            append_grade(grades_path, grade)
            existing[grade.response_id] = grade
            graded += 1
            print(f"  -> {grade.label}, correctness {grade.correctness:.2f}, "
                  f"{grade.faithfulness}"
                  f"{', HARD FAIL' if grade.hard_fail else ''}")

        # The system labels are deliberately NOT revealed here. Printing them
        # after each item kept the grading of that item blind but leaked the
        # mapping into every item after it: over 47 items a grader learns which
        # system is the terse one and which is the verbose one, and recognises
        # it thereafter. The mapping is already recorded against every
        # response_id in grades.jsonl, so nothing is lost by staying silent.
        print()

    return graded


def pooled_labels(item_set, responses_by_item: dict[str, list[dict]], seed: int) -> dict:
    """(item id, label) -> response record, in the shuffle a given seed produces."""
    rng = random.Random(seed)
    pooled = {}
    for item in item_set:
        responses = list(responses_by_item.get(item.id, []))
        if not responses:
            continue
        rng.shuffle(responses)
        for index, record in enumerate(responses):
            pooled[(item.id, chr(ord("A") + index))] = record
    return pooled


def import_csv(*, path: Path, item_set, responses_by_item: dict[str, list[dict]],
               grades_path: Path, seed: int) -> tuple[int, list[str]]:
    """Read marks from a CSV of item_id,label,point,made,faithfulness,note.

    The shuffle seed has to match the one the file was exported with, or the labels
    point at the wrong answers. Every imported grade records source="csv".
    """
    import csv

    items = {item.id: item for item in item_set}
    pooled = pooled_labels(item_set, responses_by_item, seed)
    decisions: dict[tuple[str, str], dict] = {}
    problems: list[str] = []

    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for line_no, row in enumerate(csv.DictReader(handle), start=2):
            try:
                key = (row["item_id"].strip(), row["label"].strip().upper())
                point = int(row["point"])
            except (KeyError, ValueError, AttributeError):
                problems.append(f"line {line_no}: could not read item_id, label and point")
                continue
            if key not in pooled:
                problems.append(f"line {line_no}: no answer {key[1]} for item {key[0]}")
                continue
            entry = decisions.setdefault(key, {"points": {}, "faithfulness": None, "note": ""})
            entry["points"][point] = (row.get("made") or "").strip().lower().startswith("y")
            faith = (row.get("faithfulness") or "").strip().lower()
            if faith:
                entry["faithfulness"] = {"grounded": "grounded", "partly": "partly",
                                         "unfaithful": "unfaithful"}.get(faith)
            note = (row.get("note") or "").strip()
            if note:
                entry["note"] = note

    graded = 0
    for (item_id, label), entry in decisions.items():
        item = items[item_id]
        record = pooled[(item_id, label)]
        missing = [n for n in range(1, len(item.must_include) + 1) if n not in entry["points"]]
        if missing:
            problems.append(f"{item_id} {label}: no decision for point(s) "
                            f"{', '.join(map(str, missing))}; not imported")
            continue
        grade = Grade(
            response_id=record["response_id"],
            item_id=item_id,
            condition=record["condition"],
            must_include_hits=[entry["points"][n] for n in range(1, len(item.must_include) + 1)],
            # The CSV carries the points and faithfulness only. The failure list, the
            # policy verdict and citation validity stay unjudged rather than being
            # guessed at, and the report says which measures an imported file can carry.
            must_not_include_hits=[False] * len(item.must_not_include),
            policy_pass=None,
            citations_valid=None,
            faithfulness=entry["faithfulness"],
            note=entry["note"],
            source="csv",
        )
        append_grade(grades_path, grade)
        graded += 1
    return graded, problems

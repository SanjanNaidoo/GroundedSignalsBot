"""Benchmark items — the frozen evidence base.

Items are written before the system exists and frozen before tuning starts, so
that later improvements cannot be accused of having been fitted to the test.
`items_digest` is recorded in every run manifest; if it changes between two runs,
those runs are not comparable and the report must say so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .digests import content_digest

ITEM_TYPES = {
    "concept",
    "calculation",
    "derivation",
    "error_finding",
    "scope",
    "integrity",
}

# Types graded on behaviour rather than factual recall. A tutor that scores well
# on facts while doing a student's assignment for them is a failed project, so
# these are scored separately and never averaged into correctness.
POLICY_TYPES = {"scope", "integrity"}


@dataclass(frozen=True)
class Item:
    id: str
    topic: str
    type: str
    question: str
    reference_answer: str
    must_include: tuple[str, ...]
    must_not_include: tuple[str, ...]
    source: str | None
    difficulty: int

    @property
    def is_policy_item(self) -> bool:
        return self.type in POLICY_TYPES


@dataclass(frozen=True)
class ItemSet:
    items: tuple[Item, ...]
    path: Path
    digest: str

    def __iter__(self):
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


def _validate(obj: dict, line_no: int, path: Path, seen: set[str]) -> Item:
    def require(key: str, kind: type):
        if key not in obj:
            raise ValueError(f"{path}:{line_no}: missing required field '{key}'")
        if not isinstance(obj[key], kind):
            raise ValueError(
                f"{path}:{line_no}: field '{key}' must be {kind.__name__}, "
                f"got {type(obj[key]).__name__}"
            )
        return obj[key]

    item_id = require("id", str)
    if item_id in seen:
        raise ValueError(f"{path}:{line_no}: duplicate item id '{item_id}'")
    seen.add(item_id)

    item_type = require("type", str)
    if item_type not in ITEM_TYPES:
        raise ValueError(
            f"{path}:{line_no}: unknown type '{item_type}' "
            f"(expected one of {sorted(ITEM_TYPES)})"
        )

    must_include = tuple(require("must_include", list))
    if not must_include:
        raise ValueError(
            f"{path}:{line_no}: '{item_id}' has an empty must_include; "
            "correctness is scored as a fraction of these, so an item without "
            "them cannot be graded"
        )

    difficulty = int(obj.get("difficulty", 2))
    if difficulty not in (1, 2, 3):
        raise ValueError(f"{path}:{line_no}: difficulty must be 1, 2 or 3")

    return Item(
        id=item_id,
        topic=require("topic", str),
        type=item_type,
        question=require("question", str),
        reference_answer=obj.get("reference_answer", ""),
        must_include=must_include,
        must_not_include=tuple(obj.get("must_not_include") or ()),
        source=obj.get("source"),
        difficulty=difficulty,
    )


def load_items(path: str | Path, known_topics: set[str] | None = None) -> ItemSet:
    """Load and validate a .jsonl item file.

    Validation is strict and fails on the first bad item. A benchmark that
    silently drops malformed items produces results nobody can defend.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No item file at {path}. The benchmark set is the evidence base for "
            "the report — see bench/schema.md for the format."
        )

    raw_bytes = path.read_bytes()
    items: list[Item] = []
    seen: set[str] = set()

    for line_no, line in enumerate(raw_bytes.decode("utf-8").splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON — {exc}") from exc
        items.append(_validate(obj, line_no, path, seen))

    if not items:
        raise ValueError(f"{path}: no items found")

    if known_topics:
        for item in items:
            if item.topic not in known_topics:
                raise ValueError(
                    f"{path}: item '{item.id}' has topic '{item.topic}' which is not "
                    f"in the course config. Topic ids tag both corpus chunks and "
                    f"items — fix one or the other before running."
                )

    return ItemSet(
        items=tuple(items),
        path=path,
        digest=content_digest(raw_bytes),
    )


def coverage(item_set: ItemSet) -> dict[str, dict[str, int]]:
    """Item counts per topic and per type — used by `ym07 items`."""
    by_topic: dict[str, int] = {}
    by_type: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    for item in item_set:
        by_topic[item.topic] = by_topic.get(item.topic, 0) + 1
        by_type[item.type] = by_type.get(item.type, 0) + 1
        key = str(item.difficulty)
        by_difficulty[key] = by_difficulty.get(key, 0) + 1
    return {"topic": by_topic, "type": by_type, "difficulty": by_difficulty}

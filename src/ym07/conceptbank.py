"""A true-or-false concept bank whose answer key is the course notes.

The calculation bank checks arithmetic. This one checks whether the tutor knows what the
course says. Each entry is a pair of statements that differ in one place - "inside the
unit circle" against "outside the unit circle" - with the sentence from the notes that
settles it. The true statement and its false twin are asked separately, so a tutor that
answers "true" to everything scores exactly half.

Two things are checked before any model sees the bank:

- **The key is in the notes.** Every supporting quote must be found word for word in the
  chunk it names, ignoring only what PDF extraction does to text: spacing, line-break
  hyphens and ligatures.
- **The pair is a minimal pair.** The false statement must differ from the true one by a
  single edit, so the only thing being tested is the fact that the edit changes.

Half the pairs, drawn per topic with a fixed seed, are held back for the final test and
are not run on any model before it.
"""

from __future__ import annotations

import difflib
import json
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from .digests import content_digest

SOURCE_DIR = Path("stress/source/concepts")
DEV_BANK = Path("stress/bank/concept-dev.jsonl")
FINAL_BANK = Path("stress/bank/concept-final.jsonl")
SPLIT_SEED = 41140917
PROMPT = "True or false: {statement}\n\nStart your reply with True or False, then explain briefly."


def squash(text: str) -> str:
    """Text as compared for quoting: case, spacing, hyphens and ligatures ignored."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    for curly, plain in (("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'), ("­", "")):
        t = t.replace(curly, plain)
    return re.sub(r"\s+", "", t.replace("-", ""))


def load_pairs(source_dir: Path = SOURCE_DIR) -> list[dict]:
    pairs = []
    for path in sorted(Path(source_dir).glob("*.jsonl")):
        pairs += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pairs


def verify(pairs: list[dict], chunks_path: Path) -> list[str]:
    chunks = {}
    for line in Path(chunks_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            chunks[obj["chunk_id"]] = squash(obj["text"])
    problems = []
    seen = set()
    for pair in pairs:
        if pair["id"] in seen:
            problems.append(f"{pair['id']}: duplicate id")
        seen.add(pair["id"])
        if squash(pair["quote"]) not in chunks.get(pair["chunk_id"], ""):
            problems.append(f"{pair['id']}: quote not found in {pair['chunk_id']}")
        edits = [op for op in difflib.SequenceMatcher(None, pair["true"].split(), pair["false"].split(),
                                                      autojunk=False).get_opcodes() if op[0] != "equal"]
        if len(edits) != 1:
            problems.append(f"{pair['id']}: true and false differ by {len(edits)} edits, not one")
    return problems


def split_pairs(pairs: list[dict], seed: int = SPLIT_SEED) -> tuple[list[dict], list[dict]]:
    """Per topic, half to development and the larger half held back for the final test."""
    rng = random.Random(seed)
    by_topic = defaultdict(list)
    for pair in pairs:
        by_topic[pair["topic"]].append(pair)
    dev, final = [], []
    for topic in sorted(by_topic):
        group = sorted(by_topic[topic], key=lambda p: p["id"])
        rng.shuffle(group)
        cut = len(group) // 2
        dev += group[:cut]
        final += group[cut:]
    return dev, final


def items(pairs: list[dict]) -> list[dict]:
    out = []
    for pair in pairs:
        for truth in ("true", "false"):
            out.append({
                "id": f"{pair['id']}-{truth[0]}",
                "pair": pair["id"],
                "topic": pair["topic"],
                "answer": truth,
                "statement": pair[truth],
                "question": PROMPT.format(statement=pair[truth]),
                "chunk_id": pair["chunk_id"],
                "quote": pair["quote"],
            })
    return out


def write(path: Path, rows: list[dict], **meta) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    path.write_text(body, encoding="utf-8", newline="\n")
    manifest = {**meta, "items": len(rows), "digest": content_digest(body.encode("utf-8"))}
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                  encoding="utf-8", newline="\n")
    return manifest


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

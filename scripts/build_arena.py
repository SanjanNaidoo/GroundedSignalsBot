"""The one set every model is compared on.

Each model in the comparison - the size ladder, the candidate replacements, and the
frontier models answered by hand - has to answer the same questions, or the columns
cannot be read against each other. Two different samples of the calculation bank were
in use before this existed and they overlapped on 22 items, which would have made the
local and frontier columns incomparable.

The sample is balanced on both axes that are known to move the score. Family, because
the eleven families differ in difficulty by a factor of three and an unbalanced draw
moves the headline without anything about the model changing. Phrasing, because the
same question asked three ways already differs by 8 points on the delivered system.
Three of each phrasing from each family: 99 questions, every cell equal.

It is drawn from `fresh`, the split reserved after the calculator and course-facts
changes, and the draw is deterministic given the bank, so the set regenerates exactly.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "stress" / "bank" / "fresh" / "calc-bank.jsonl"
OUT_DIR = ROOT / "stress" / "bank" / "arena"
PER_CELL = 3


def main() -> int:
    items = [json.loads(line) for line in
             SOURCE.read_text(encoding="utf-8").splitlines() if line.strip()]

    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for item in items:
        cells[(item["family"], item["style"])].append(item)

    short = {key: len(group) for key, group in cells.items() if len(group) < PER_CELL}
    if short:
        print(f"not enough items in {len(short)} cells: {short}")
        return 1

    # Bank order within each cell, so the draw is reproducible without a seed.
    chosen: list[dict] = []
    for family in sorted({f for f, _ in cells}):
        for style in sorted({s for _, s in cells}):
            chosen += cells[(family, style)][:PER_CELL]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in chosen)
    (OUT_DIR / "calc-bank.jsonl").write_text(body, encoding="utf-8")

    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:16]
    manifest = {
        "drawn_from": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        "per_family_per_style": PER_CELL,
        "families": sorted({i["family"] for i in chosen}),
        "styles": sorted({i["style"] for i in chosen}),
        "items": len(chosen),
        "digest": digest,
        "purpose": "the shared comparison set for every model, local and frontier",
    }
    (OUT_DIR / "calc-bank.manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"{len(chosen)} items, {PER_CELL} per family per phrasing")
    print(f"  families {len(manifest['families'])}  styles {len(manifest['styles'])}")
    print(f"  digest   {digest}")
    print(f"  wrote    {(OUT_DIR / 'calc-bank.jsonl').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

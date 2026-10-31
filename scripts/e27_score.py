"""E27 against its pass marks: the fresh course set on A-r1-tuned and A-r1-facts.

    python scripts/e27_score.py

Reads the rows `ym07 stress run --suites course-fresh` wrote under
stress/results/course-facts/, and the lookup's own check
(stress/results/course-facts/lookup-heldout.json, from
`python scripts/e27_lookup_check.py heldout`). Prints F1-F4 as registered in
protocols/course-facts.md, the before and after on every kind, and each item either
system got wrong, and writes stress/results/course-facts/e27.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ym07 import coursequeries  # noqa: E402

RESULTS = ROOT / "stress/results/course-facts"
BEFORE, AFTER = "A-r1-tuned", "A-r1-facts"


def rows(condition: str) -> dict[str, dict]:
    path = RESULTS / "course-fresh" / f"{condition}.jsonl"
    if not path.exists():
        return {}
    return {r["id"]: r for r in (json.loads(l) for l in
                                 path.read_text(encoding="utf-8").splitlines() if l.strip())}


def main() -> int:
    items = coursequeries.fresh_items()
    kinds = ("fact", "absent", "content")
    got = {c: rows(c) for c in (BEFORE, AFTER)}
    record: dict = {"items": len(items)}

    print(f"{'kind':<8} {'items':>5} {BEFORE:>12} {AFTER:>12}")
    for kind in kinds:
        ids = [i["id"] for i in items if i["kind"] == kind]
        line = f"{kind:<8} {len(ids):>5}"
        for c in (BEFORE, AFTER):
            done = [got[c][i] for i in ids if i in got[c]]
            right = sum(bool(r.get("correct")) for r in done)
            record.setdefault(c, {})[kind] = {"right": right, "answered": len(done), "of": len(ids)}
            line += f" {f'{right}/{len(done)}' if done else 'not run':>12}"
        print(line)

    after = {kind: v for kind, v in record.get(AFTER, {}).items() if v["answered"]}
    lookup = got[AFTER]
    content_fired = sum(bool(lookup[i["id"]].get("by_lookup")) for i in items
                        if i["kind"] == "content" and i["id"] in lookup)
    heldout = RESULTS / "lookup-heldout.json"
    f4 = None
    if heldout.exists():
        sets = json.loads(heldout.read_text(encoding="utf-8"))["leave_alone"]
        f4 = sum(v["answered_by_lookup"] for k, v in sets.items() if not k.startswith("F3"))
    marks = {
        "F1 fresh facts right (>= 45 of 47)": (after.get("fact", {}).get("right"), 45, ">="),
        "F2 fresh absent right (>= 11 of 12)": (after.get("absent", {}).get("right"), 11, ">="),
        "F3 fresh content the lookup answered (0)": (content_fired if lookup else None, 0, "=="),
        "F4 held-out questions the lookup answered (0)": (f4, 0, "=="),
    }
    print()
    record["marks"] = {}
    for name, (value, bar, op) in marks.items():
        met = None if value is None else (value >= bar if op == ">=" else value == bar)
        record["marks"][name] = {"value": value, "met": met}
        verdict = "not run" if met is None else ("met" if met else "MISSED")
        print(f"{name:<48} {value if value is not None else '-':>4}  {verdict}")

    print("\nwrong on either system:")
    for item in items:
        a, b = got[BEFORE].get(item["id"]), got[AFTER].get(item["id"])
        if (a and not a.get("correct")) or (b and not b.get("correct")):
            mark = lambda r: "-" if r is None else ("right" if r.get("correct") else "WRONG")
            print(f"  {item['id']:<12} {mark(a):>6} {mark(b):>6}  {item['question']}")
            if b and not b.get("correct"):
                print(f"               after: {(b.get('text') or '')[:200]}")

    out = RESULTS / "e27.json"
    out.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

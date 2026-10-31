"""E29 against its pass marks, and which version the platform serves.

    python scripts/e29_score.py

Reads the rows `ym07 stress run --suites course-marks` wrote under
stress/results/course-facts/course-marks/ for A-r1-facts and A-r1-facts-marks, and the
no-model check (`python scripts/e29_marks_check.py heldout`, marks-heldout.json). Prints
M1-M3 as registered in protocols/course-marks.md and the switch rule; writes
stress/results/course-facts/e29.json.
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
SYSTEMS = ("A-r1-facts", "A-r1-facts-marks")
NEW = "A-r1-facts-marks"


def rows(condition: str) -> dict[str, dict]:
    path = RESULTS / "course-marks" / f"{condition}.jsonl"
    if not path.exists():
        return {}
    return {r["id"]: r for r in (json.loads(l) for l in
                                 path.read_text(encoding="utf-8").splitlines() if l.strip())}


def main() -> int:
    items = coursequeries.marks_items()
    got = {c: rows(c) for c in SYSTEMS}
    record: dict = {"items": len(items), "systems": {}}
    print(f"{'kind':<8} {'items':>5}" + "".join(f" {c:>18}" for c in SYSTEMS))
    for kinds, label in ((("need", "final"), "need+final"), (("content",), "content")):
        ids = [i["id"] for i in items if i["kind"] in kinds]
        line = f"{label:<10} {len(ids):>3}"
        for c in SYSTEMS:
            done = [got[c][i] for i in ids if i in got[c]]
            right = sum(bool(r.get("correct")) for r in done)
            handled = sum(bool(r.get("by_lookup")) for r in done)
            record["systems"].setdefault(c, {})[label] = {"right": right, "answered": len(done),
                                                          "of": len(ids), "by_handout_path": handled}
            line += f" {f'{right}/{len(done)}' if done else 'not run':>18}"
        print(line)

    new = record["systems"].get(NEW, {})
    m3 = None
    check = RESULTS / "marks-heldout.json"
    if check.exists():
        m3 = json.loads(check.read_text(encoding="utf-8"))["m3"]["decided_differently"]
    m1 = new.get("need+final", {}).get("right") if new.get("need+final", {}).get("answered") else None
    m2 = new.get("content", {}).get("by_handout_path") if new.get("content", {}).get("answered") else None
    marks = {"M1 need and final items right (>= 15 of 16)": (m1, lambda v: v >= 15),
             "M2 content items the handout path answered (0 of 6)": (m2, lambda v: v == 0),
             "M3 measured and held-out questions the marks rule decides (0)": (m3, lambda v: v == 0)}
    print()
    record["marks"] = {}
    for name, (value, test) in marks.items():
        met = None if value is None else bool(test(value))
        record["marks"][name] = {"value": value, "met": met}
        print(f"{name:<62} {value if value is not None else '-':>4}  "
              f"{'not run' if met is None else ('met' if met else 'MISSED')}")

    old = record["systems"].get("A-r1-facts", {}).get("need+final", {})
    if m1 is not None and old.get("answered"):
        serve = (record["marks"]["M2 content items the handout path answered (0 of 6)"]["met"]
                 and record["marks"]["M3 measured and held-out questions the marks rule decides (0)"]["met"]
                 and m1 > old["right"])
        record["switch"] = {"serve": NEW if serve else "A-r1-facts",
                            "new_right": m1, "old_right": old["right"]}
        print(f"\nswitch rule: serve {NEW if serve else 'A-r1-facts'} "
              f"({m1} against {old['right']} of 16 need and final items)")

    print("\nwrong on either system:")
    for item in items:
        a, b = (got[c].get(item["id"]) for c in SYSTEMS)
        if any(r is not None and not r.get("correct") for r in (a, b)):
            mark = lambda r: "-" if r is None else ("right" if r.get("correct") else "WRONG")  # noqa: E731
            print(f"  {item['id']:<13} {mark(a):>6} {mark(b):>6}  {item['question']}")
            if a is not None and not a.get("correct"):
                print(f"                 A-r1-facts: {(a.get('text') or '')[:160]}")

    out = RESULTS / "e29.json"
    out.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

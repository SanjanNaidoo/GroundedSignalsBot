"""E28 against its pass marks, and which version the platform serves.

    python scripts/e28_score.py

Reads the rows `ym07 stress run --suites course-fresh2` wrote under
stress/results/course-facts/course-fresh2/ for A-r1-tuned, A-r1-facts and A-r1-facts2, and
the lookup's own held-out check (`python scripts/e27_lookup_check.py heldout --version 2`,
written to lookup-v2-heldout.json). Prints G1-G4 as registered in
protocols/course-facts-2.md, the switch rule, and every item any system got wrong; writes
stress/results/course-facts/e28.json.
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
SYSTEMS = ("A-r1-tuned", "A-r1-facts", "A-r1-facts2")
NEW = "A-r1-facts2"


def rows(condition: str) -> dict[str, dict]:
    path = RESULTS / "course-fresh2" / f"{condition}.jsonl"
    if not path.exists():
        return {}
    return {r["id"]: r for r in (json.loads(l) for l in
                                 path.read_text(encoding="utf-8").splitlines() if l.strip())}


def main() -> int:
    items = coursequeries.fresh2_items()
    got = {c: rows(c) for c in SYSTEMS}
    record: dict = {"items": len(items), "systems": {}}

    print(f"{'kind':<8} {'items':>5}" + "".join(f" {c:>12}" for c in SYSTEMS))
    for kind in ("fact", "absent", "content"):
        ids = [i["id"] for i in items if i["kind"] == kind]
        line = f"{kind:<8} {len(ids):>5}"
        for c in SYSTEMS:
            done = [got[c][i] for i in ids if i in got[c]]
            right = sum(bool(r.get("correct")) for r in done)
            handled = sum(bool(r.get("by_lookup")) for r in done)
            record["systems"].setdefault(c, {})[kind] = {
                "right": right, "answered": len(done), "of": len(ids),
                "by_handout_path": handled}
            line += f" {f'{right}/{len(done)}' if done else 'not run':>12}"
        print(line)

    new = {k: v for k, v in record["systems"].get(NEW, {}).items() if v["answered"]}
    content_touched = new.get("content", {}).get("by_handout_path") if "content" in new else None

    heldout = RESULTS / "lookup-v2-heldout.json"
    g4 = None
    if heldout.exists():
        sets = json.loads(heldout.read_text(encoding="utf-8"))["leave_alone"]
        g4 = sum(v["answered_by_lookup"] + v["newly_called_course_questions"]
                 for k, v in sets.items() if not k.startswith("G3"))
        record["heldout_questions"] = sum(v["questions"] for k, v in sets.items()
                                          if not k.startswith("G3"))

    marks = {
        "G1 second set facts right (>= 46 of 48)": (new.get("fact", {}).get("right"), 46, ">="),
        "G2 second set absent right (>= 18 of 20)": (new.get("absent", {}).get("right"), 18, ">="),
        "G3 second set content the handout path answered (0)": (content_touched, 0, "=="),
        "G4 held-out questions and turns answered or rerouted (0)": (g4, 0, "=="),
    }
    print()
    record["marks"] = {}
    for name, (value, bar, op) in marks.items():
        met = None if value is None else (value >= bar if op == ">=" else value == bar)
        record["marks"][name] = {"value": value, "met": met}
        verdict = "not run" if met is None else ("met" if met else "MISSED")
        print(f"{name:<58} {value if value is not None else '-':>4}  {verdict}")

    # The switch rule, registered before the build.
    old = record["systems"].get("A-r1-facts", {})
    switch = None
    if new and old and all(old.get(k, {}).get("answered") for k in ("fact", "absent")):
        g3 = record["marks"]["G3 second set content the handout path answered (0)"]["met"]
        g4m = record["marks"]["G4 held-out questions and turns answered or rerouted (0)"]["met"]
        at_least = (new["fact"]["right"] >= old["fact"]["right"]
                    and new["absent"]["right"] >= old["absent"]["right"])
        switch = bool(g3 and g4m and at_least)
        record["switch"] = {"serve": NEW if switch else "A-r1-facts", "g3": g3, "g4": g4m,
                            "at_least_as_good": at_least}
        print(f"\nswitch rule: serve {NEW if switch else 'A-r1-facts'} "
              f"(G3 {g3}, G4 {g4m}, at least as good as A-r1-facts on facts and absent {at_least})")

    print("\nwrong on any system:")
    for item in items:
        marks_ = []
        for c in SYSTEMS:
            r = got[c].get(item["id"])
            marks_.append("-" if r is None else ("right" if r.get("correct") else "WRONG"))
        if "WRONG" in marks_:
            print(f"  {item['id']:<14}" + "".join(f" {m:>6}" for m in marks_) + f"  {item['question']}")
            r = got[NEW].get(item["id"])
            if r is not None and not r.get("correct"):
                print(f"                 {NEW}: {(r.get('text') or '')[:200]}")

    out = RESULTS / "e28.json"
    out.write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten to {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

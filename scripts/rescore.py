"""Re-mark every answer already recorded, after a change to the rules.

A rule change is not an experiment and does not need the card. Every answer is stored
as the text the model produced, and a score is a pure function of that text and the
rules, so the whole history can be re-marked in seconds. That is what makes it
reasonable to fix a defect in the marking rather than caveat it: the fix applies to
results already gathered, not only to runs made after it.

The defect this was written for: numeric rules match the decimal spelling of a value,
so `A = \\frac{3}{2}` fails a rule looking for 1.5 although it is the same number. See
`ym07.checks.READ_FRACTIONS`.

Nothing is written without --apply, and every score that moves is printed either way,
because a re-marking that cannot be inspected is indistinguishable from fixing the
numbers until they look better.

    python scripts/rescore.py                     what would change
    python scripts/rescore.py --apply             change it, keeping the old score
    python scripts/rescore.py --results stress/results/arena
"""

from __future__ import annotations

import argparse
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

from ym07.checks import score_answer  # noqa: E402
from ym07.stress import _check  # noqa: E402

BANKS = ROOT / "stress" / "bank"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def banks() -> dict[Path, dict[str, dict]]:
    """Every question bank that carries rules, by path."""
    found = {}
    for path in sorted(BANKS.rglob("*.jsonl")):
        try:
            rows = load_jsonl(path)
        except Exception:
            continue
        if rows and isinstance(rows[0], dict) and rows[0].get("checks"):
            found[path] = {r["id"]: r for r in rows if r.get("id")}
    return found


def pick_bank(rows: list[dict], candidates: dict[Path, dict[str, dict]]) -> Path | None:
    """The bank whose questions and rule labels match these rows.

    Matched rather than configured: several result sets were run against different
    splits, and a bank that merely shares ids with a set but carries different rules
    would re-mark it against the wrong answer key without saying so. Requiring the rule
    labels to agree as well makes a wrong match fail loudly instead.
    """
    scored = [r for r in rows if r.get("per_check")]
    if not scored:
        return None
    best, best_hits = None, 0
    for path, bank in candidates.items():
        hits = 0
        for row in scored:
            item = bank.get(row["id"])
            if not item:
                continue
            if {c["label"] for c in item["checks"]} == set(row["per_check"]):
                hits += 1
        if hits > best_hits:
            best, best_hits = path, hits
    # A partial match means the set spans banks, which this cannot re-mark safely.
    return best if best is not None and best_hits == len(scored) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="stress/results")
    parser.add_argument("--apply", action="store_true", help="write the new scores")
    parser.add_argument("--show", type=int, default=6, help="examples printed per file")
    args = parser.parse_args()

    candidates = banks()
    print(f"{len(candidates)} question banks carry rules\n")

    files = sorted((ROOT / args.results).rglob("*.jsonl"))
    moved_total = checked_total = 0
    skipped: list[tuple[Path, str]] = []
    for path in files:
        rows = load_jsonl(path)
        if not any(r.get("per_check") for r in rows):
            continue
        bank_path = pick_bank(rows, candidates)
        if bank_path is None:
            skipped.append((path, "no bank matches every scored row"))
            continue
        bank = candidates[bank_path]

        moved = []
        for row in rows:
            if not row.get("per_check") or row.get("error"):
                continue
            item = bank[row["id"]]
            checked_total += 1
            fresh = score_answer([_check(r) for r in item["checks"]], row.get("text") or "")
            if abs(fresh["correctness"] - (row.get("score") or 0.0)) > 1e-9:
                moved.append((row, fresh))
        if not moved:
            continue
        moved_total += len(moved)
        print(f"{path.relative_to(ROOT)}  against {bank_path.relative_to(ROOT)}")
        for row, fresh in moved[: args.show]:
            gained = [label for label, held in fresh["per_check"].items()
                      if held and not row["per_check"].get(label)]
            lost = [label for label, held in fresh["per_check"].items()
                    if not held and row["per_check"].get(label)]
            print(f"    {row['id']:26} {row.get('score', 0):.2f} -> {fresh['correctness']:.2f}"
                  + (f"  gained: {', '.join(gained)}" if gained else "")
                  + (f"  LOST: {', '.join(lost)}" if lost else ""))
        if len(moved) > args.show:
            print(f"    ... and {len(moved) - args.show} more")

        if args.apply:
            for row, fresh in moved:
                # The score as it was marked is kept beside the new one. The comparison
                # tables were computed from it, and a number in the report has to be
                # traceable to the rules that produced it.
                row["score_as_marked"] = row.get("score")
                row["per_check_as_marked"] = row["per_check"]
                row["score"] = fresh["correctness"]
                row["per_check"] = fresh["per_check"]
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10)
                                    for r in rows), encoding="utf-8", newline=chr(10))

    print(f"\n{checked_total} answers re-marked, {moved_total} scores moved")
    for path, why in skipped:
        print(f"  skipped {path.relative_to(ROOT)}: {why}")
    if moved_total and not args.apply:
        print("\nNothing written. Re-run with --apply to keep these scores.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

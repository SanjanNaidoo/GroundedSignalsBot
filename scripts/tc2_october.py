"""TC2 for the October marking: twenty answers re-marked blind at least three days later.

    python scripts/tc2_october.py draw     # 13 October, after marking, before any re-mark
    python scripts/tc2_october.py score    # 16 October, after the re-mark

The draw is made from the 130 answers marked on 13 October - the 141 pooled answers less
the eleven frontier answers carried over from September, which were not marked in this
round - with its seed recorded, and committed before the re-mark, so it can be shown to
precede it. The re-mark goes to its own file through the browser grader, restricted to
the twenty drawn answers and shuffled with a new seed:

    ym07 grade --web --conditions A-r1-tuned A-r1-distill-7b C1-prompted --plan grades/tc2-plan-16oct.json --grades grades/tc2-remark-16oct.jsonl --seed 1610

The pass mark is the registered one: the same label on at least 17 of 20. Two labellings
are reported. The grader's own label (correct at 0.8 of the points or more, partly at
0.4, a hard fail capped at partly) is the one stored with every grade; September's TC2
was scored with a band that also caps an unfaithful answer at partly
(scripts/score_marking_round2.py), and that one is reported beside it so the two Octobers
and Septembers can be compared like for like. Point-level agreement and faithfulness
agreement are reported as September's were, because September's failure lived in the
faithfulness marks and not in the points.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GRADES = ROOT / "grades" / "grades.jsonl"
PLAN = ROOT / "grades" / "tc2-plan-16oct.json"
REMARK = ROOT / "grades" / "tc2-remark-16oct.jsonl"
OUT = ROOT / "stress" / "results" / "tc2-october.json"
BEFORE_MARKING = "3ab76ab"          # the last commit before the marking of 13 October
CONDITIONS = ("A-r1-tuned", "A-r1-distill-7b", "C1-prompted")
DRAW_SEED = 20261013                # the day of the draw; recorded so it can be repeated
DRAW = 20
PASS = 17


def rows(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def correctness(row: dict) -> float:
    hits = row["must_include_hits"]
    return sum(hits) / len(hits) if hits else 0.0


def grader_label(row: dict) -> str:
    """The label every stored grade carries: a hard fail is capped at partly."""
    hard = any(row["must_not_include_hits"])
    if correctness(row) >= 0.8 and not hard:
        return "correct"
    if correctness(row) >= 0.4:
        return "partly"
    return "incorrect"


def september_band(row: dict) -> str:
    """September's TC2 banding: an unfaithful answer is capped at partly."""
    label = ("correct" if correctness(row) >= 0.8
             else "partly" if correctness(row) >= 0.4 else "incorrect")
    return "partly" if label == "correct" and row["faithfulness"] == "unfaithful" else label


def draw() -> int:
    if PLAN.exists():
        print(f"{PLAN.name} exists; the draw is made once. Delete it only if it was never used.")
        return 1
    before = {r["response_id"] for r in rows(subprocess.run(
        ["git", "show", f"{BEFORE_MARKING}:grades/grades.jsonl"], cwd=ROOT, capture_output=True,
        text=True, encoding="utf-8", check=True).stdout)}
    marked = [r["response_id"] for r in rows(GRADES.read_text(encoding="utf-8"))
              if r["condition"] in CONDITIONS and r["response_id"] not in before]
    if len(marked) != 130:
        print(f"expected the 130 answers marked on 13 October, found {len(marked)}")
        return 1
    drawn = sorted(random.Random(DRAW_SEED).sample(sorted(marked), DRAW))
    PLAN.write_text(json.dumps({
        "check": "TC2, October: twenty answers re-marked blind at least three days later",
        "drawn": "2026-10-13, after the marking and before any re-mark",
        "drawn_from": f"the {len(marked)} answers marked on 13 October",
        "seed": DRAW_SEED,
        "pass_mark": f"same label on at least {PASS} of {DRAW}",
        "h1_responses": [],
        "sampled_responses": drawn,
    }, indent=1) + "\n", encoding="utf-8")
    print(f"drew {DRAW} of {len(marked)} with seed {DRAW_SEED} into {PLAN.relative_to(ROOT)}")
    print("Do not open the plan before the re-mark: its identifiers name the systems.")
    return 0


def score() -> int:
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    drawn = plan["sampled_responses"]
    first = {r["response_id"]: r for r in rows(GRADES.read_text(encoding="utf-8"))}
    if not REMARK.exists():
        print(f"no re-mark yet at {REMARK.relative_to(ROOT)}")
        return 1
    second = {r["response_id"]: r for r in rows(REMARK.read_text(encoding="utf-8"))}
    missing = [i for i in drawn if i not in second]
    if missing:
        print(f"{len(missing)} of {len(drawn)} drawn answers are not re-marked yet")
        return 1
    same_grader = sum(grader_label(first[i]) == grader_label(second[i]) for i in drawn)
    same_band = sum(september_band(first[i]) == september_band(second[i]) for i in drawn)
    points = [(a, b) for i in drawn
              for a, b in zip(first[i]["must_include_hits"], second[i]["must_include_hits"])]
    same_points = sum(a == b for a, b in points)
    identical = sum(first[i]["must_include_hits"] == second[i]["must_include_hits"] for i in drawn)
    moved = sum(abs(correctness(first[i]) - correctness(second[i])) for i in drawn) / len(drawn)
    same_faith = sum(first[i]["faithfulness"] == second[i]["faithfulness"] for i in drawn)
    unfaithful = (sum(first[i]["faithfulness"] == "unfaithful" for i in drawn),
                  sum(second[i]["faithfulness"] == "unfaithful" for i in drawn))
    verdict = "passes" if same_grader >= PASS else "fails"
    print(f"TC2, October: same label on {same_grader} of {len(drawn)} by the grader's label "
          f"(pass {PASS}) - {verdict}; {same_band} of {len(drawn)} by September's band")
    print(f"  points: {same_points} of {len(points)} decisions agree; {identical} answers scored "
          f"identically; mean fraction of points moved {moved:.3f}")
    print(f"  faithfulness: same mark on {same_faith} of {len(drawn)}; unfaithful "
          f"{unfaithful[0]} first, {unfaithful[1]} on the re-mark")
    # Kept for the report's table (scripts/validation_results.py), so its figures are read
    # from here rather than typed.
    mean_first = sum(correctness(first[i]) for i in drawn) / len(drawn)
    mean_second = sum(correctness(second[i]) for i in drawn) / len(drawn)
    OUT.write_text(json.dumps({
        "drawn": len(drawn), "pass": PASS, "passes": same_grader >= PASS,
        "same_label": same_grader, "same_september_band": same_band,
        "point_decisions": [same_points, len(points)], "identical_answers": identical,
        "mean_points_first": round(mean_first, 3), "mean_points_second": round(mean_second, 3),
        "mean_moved": round(moved, 3), "same_faithfulness": same_faith,
        "unfaithful_first": unfaithful[0], "unfaithful_second": unfaithful[1]}, indent=1) + "\n",
        encoding="utf-8", newline="\n")
    print(f"written {OUT.relative_to(ROOT)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["draw", "score"])
    args = parser.parse_args()
    return draw() if args.action == "draw" else score()


if __name__ == "__main__":
    raise SystemExit(main())

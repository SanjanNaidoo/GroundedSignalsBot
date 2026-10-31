"""Where questions were sent, and where the time went.

Two things the recorded answers already contain and nothing has ever tabulated.

**Routing.** Every question takes one of two paths: the base weights, which are better
at arithmetic, or the tuned path, which carries the course behaviour and is where work
a student says is assessed must go. The row records which path was taken. Whether that
was the right path is decided here from the suite the question came from, which is
independent of the router's own rule - a calculation is numerical because it came out of
the calculation bank, not because the router said so. Misrouting has cost this project
before: 19 of 264 calculations were once treated as homework and answered by the weights
that are worse at arithmetic, and nothing in the results said so.

**Latency.** The median answer time is already reported, and on its own it hides the
shape: a one-line concept answer and a partial-fraction derivation are not the same
question, and averaging them describes neither.

What this cannot do is separate the time spent searching from the time spent writing.
`latency_s` is the whole call and nothing records the retrieval separately, so that
split needs either instrumentation or a standalone timing run - it is not recoverable
from what is on disk, and is reported as missing rather than estimated.

    python scripts/routing_and_latency.py
    python scripts/routing_and_latency.py --results stress/results/stage3
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Which path a question of this kind ought to take, decided by where the question came
# from rather than by the rule under test.
#
#   calc, errors, followup  numerical work; the base weights answer arithmetic better,
#                           and the delivered system switches the adapter off for it
#   concept, course, scope  not numerical; the tuned path carries the course behaviour
#   pushback                the student says it is for marks, so it goes to the policy
#                           path whatever else it is
EXPECTED = {
    "calc": "base", "errors": "base", "followup": "base",
    "concept": "tuned", "course": "tuned", "scope": "tuned",
    "pushback": "tuned",
}


def head(route: str | None) -> str:
    """'base (computed)' and 'tuned (gate)' are the same two paths, differently reached."""
    if not route:
        return "(not recorded)"
    return route.split(" ", 1)[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="stress/results")
    parser.add_argument("--condition", default=None,
                        help="limit to one condition, e.g. A-r1-distill-7b")
    parser.add_argument("--out", default="stress/results/routing-and-latency.md")
    args = parser.parse_args()

    rows: list[dict] = []
    for path in sorted((ROOT / args.results).rglob("*.jsonl")):
        suite = path.parent.name
        if suite not in EXPECTED:
            continue
        if args.condition and path.stem != args.condition:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            row["_suite"], row["_condition"], row["_set"] = suite, path.stem, path.parts[-3]
            rows.append(row)
    if not rows:
        print(f"no results under {args.results}", file=sys.stderr)
        return 1

    routed = [r for r in rows if r.get("route")]
    lines = ["# Where questions were sent, and where the time went", "",
             f"{len(rows)} recorded answers, {len(routed)} of which record the path taken.", ""]

    # --- the confusion matrix -----------------------------------------------------
    print("=" * 72)
    print("WHICH PATH A QUESTION TOOK, AGAINST WHICH IT SHOULD HAVE")
    print("=" * 72)
    matrix: dict[tuple[str, str], int] = Counter()
    for r in routed:
        matrix[(EXPECTED[r["_suite"]], head(r["route"]))] += 1
    paths = sorted({p for _, p in matrix} | {e for e, _ in matrix})
    print("\n%-14s %s   %s" % ("should go to", "".join(f"{p:>12}" for p in paths), "misrouted"))
    lines += ["## Which path a question took, against which it should have", "",
              "The path a question ought to take is decided by the suite it came from, "
              "not by the rule being tested.", "",
              "| should go to | " + " | ".join(paths) + " | misrouted |",
              "| --- | " + " | ".join("---" for _ in paths) + " | --- |"]
    for expected in sorted({e for e, _ in matrix}):
        counts = [matrix[(expected, p)] for p in paths]
        total = sum(counts)
        wrong = total - matrix[(expected, expected)]
        print("%-14s %s   %d of %d (%.1f%%)" % (
            expected, "".join(f"{c:>12}" for c in counts), wrong, total, 100 * wrong / total))
        lines.append(f"| **{expected}** | " + " | ".join(str(c) for c in counts)
                     + f" | {wrong} of {total} ({100 * wrong / total:.1f}%) |")
    lines.append("")

    # Which suites the misroutes come from, since that is what a fix would target.
    print("\nmisrouted questions by suite:")
    lines += ["", "### Misrouted questions, by suite", "",
              "| suite | misrouted | of | what happened instead |", "| --- | --- | --- | --- |"]
    for suite in sorted({r["_suite"] for r in routed}):
        sel = [r for r in routed if r["_suite"] == suite]
        bad = [r for r in sel if head(r["route"]) != EXPECTED[suite]]
        if not bad:
            continue
        what = Counter(r["route"] for r in bad).most_common(2)
        print("   %-10s %3d of %3d   %s" % (suite, len(bad), len(sel),
              ", ".join(f"{k} x{v}" for k, v in what)))
        lines.append(f"| {suite} | {len(bad)} | {len(sel)} | "
                     + ", ".join(f"`{k}` ×{v}" for k, v in what) + " |")
    lines.append("")

    # Every distinct route, so the ones added recently are visible.
    print("\nevery path recorded, in full:")
    lines += ["", "### Every path recorded", "", "| path | answers |", "| --- | --- |"]
    for route, n in Counter(r["route"] for r in routed).most_common():
        print("   %-34s %d" % (route, n))
        lines.append(f"| `{route}` | {n} |")
    lines.append("")

    # --- latency ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("HOW LONG AN ANSWER TOOK, BY WHAT WAS ASKED")
    print("=" * 72)
    lines += ["", "## How long an answer took, by what was asked", "",
              "`latency_s` is the whole call. Nothing records the search separately, so "
              "the time spent searching cannot be split out of these figures - that "
              "needs instrumentation or a standalone timing run.", "",
              "| condition | question type | answers | median | slowest quarter above | slowest |",
              "| --- | --- | --- | --- | --- | --- |"]
    by: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r.get("error") or not r.get("latency_s"):
            continue
        by[(r["_condition"], r["_suite"])].append(r["latency_s"])
    print("\n%-22s %-10s %6s %9s %9s %9s" % ("condition", "suite", "n", "median", "p75", "max"))
    for (cond, suite), times in sorted(by.items()):
        times.sort()
        if len(times) < 3:
            continue
        median = statistics.median(times)
        p75 = times[int(len(times) * 0.75)]
        print("%-22s %-10s %6d %8.1fs %8.1fs %8.1fs" % (cond, suite, len(times), median, p75, times[-1]))
        lines.append(f"| `{cond}` | {suite} | {len(times)} | {median:.1f} s | {p75:.1f} s "
                     f"| {times[-1]:.1f} s |")
    lines.append("")

    # Within the calculation suite, by family: this is where the spread lives.
    fam: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r["_suite"] == "calc" and r.get("family") and r.get("latency_s") and not r.get("error"):
            fam[r["family"]].append(r["latency_s"])
    if fam:
        print("\nwithin calculations, by family (all conditions together):")
        lines += ["", "### Within calculations, by family", "",
                  "| family | answers | median | slowest |", "| --- | --- | --- | --- |"]
        for family, times in sorted(fam.items(), key=lambda kv: -statistics.median(kv[1])):
            times.sort()
            print("   %-20s %4d  median %6.1fs  max %6.1fs" % (
                family, len(times), statistics.median(times), times[-1]))
            lines.append(f"| {family} | {len(times)} | {statistics.median(times):.1f} s "
                         f"| {times[-1]:.1f} s |")
        lines.append("")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

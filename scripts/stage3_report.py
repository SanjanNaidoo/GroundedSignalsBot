"""Three retrieval settings, same model, same questions: does better retrieval answer better?

The retrieval grid (scripts/chunk_grid.py) says 200-token chunks twelve deep put the
right passage in front of the model 91.4% of the time against 76.7% for the 800-token
chunks four deep this project ships, at the same prompt length. This asks the question
the grid cannot: whether that converts into answers.

It can fail to convert, for a reason worth naming before the numbers. Recall measures
whether overlapping text arrived, not whether it arrived in a usable shape. Twelve
200-token fragments and four 800-token passages can hold the same material while only
one of them holds a complete worked method, and a derivation that sits whole inside an
800-token chunk is cut across three 200-token ones.

Three suites, because they can disagree and the disagreement is the point:

  calc     99 questions, marked by rule. None of its 234 rules mentions a citation, so
           this is the suite where retrieval might turn out to be decoration.
  concept  true-or-false claims, which also record whether the source was retrieved and
           whether the answer cited it - the conversion from recall to use, directly.
  course   facts about the course, where the right passage is the whole answer.

Comparisons are paired on the question and tested with an exact McNemar, because the
arms answer identical questions and comparing independent intervals would throw that
away.

    python scripts/stage3_report.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

RESULTS = ROOT / "stress" / "results" / "stage3"
CONTROL = "A-r1-distill-7b"
ARMS = (CONTROL, "A-r1-c400-k8", "A-r1-c200-k12")
ABOUT = {
    CONTROL: "800-token chunks, 4 deep - what this ships",
    "A-r1-c400-k8": "400-token chunks, 8 deep",
    "A-r1-c200-k12": "200-token chunks, 12 deep - the grid's choice",
}
# Retrieval recall measured by scripts/chunk_grid.py, for the conversion question.
RECALL = {CONTROL: 0.767, "A-r1-c400-k8": 0.866, "A-r1-c200-k12": 0.914}


def load(suite: str, arm: str) -> dict[str, dict]:
    path = RESULTS / suite / f"{arm}.jsonl"
    if not path.exists():
        return {}
    return {r["id"]: r for r in (json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip())}


def value(suite: str, row: dict) -> float | None:
    """What counts as right on this suite, as a number so the suites can share code."""
    if row.get("error"):
        return None
    if suite == "calc":
        return row.get("score")
    return 1.0 if row.get("correct") else 0.0


def mcnemar(a: list[float], b: list[float]) -> tuple[int, int, float]:
    win = sum(1 for x, y in zip(a, b) if x > y)
    lose = sum(1 for x, y in zip(a, b) if y > x)
    n = win + lose
    if n == 0:
        return win, lose, 1.0
    p = sum(math.comb(n, k) for k in range(0, min(win, lose) + 1)) / 2 ** n * 2
    return win, lose, min(1.0, p)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="stress/results/stage3/report.md")
    args = parser.parse_args()

    lines = ["# Does better retrieval answer better?", "",
             "Same model throughout - DeepSeek-R1-Distill-7B, the chosen one - and the "
             "same questions. Only the chunked corpus and the retrieval depth change, "
             "and the three settings are matched on prompt length, so what differs is "
             "how that length is divided.", ""]

    for suite in ("calc", "concept", "course"):
        present = [a for a in ARMS if load(suite, a)]
        if not present:
            continue
        print(f"\n{'=' * 74}\n{suite}\n{'=' * 74}")
        lines += [f"## {suite}", ""]
        control = load(suite, CONTROL)
        header = f"{'setting':<34}{'n':>5}{'right':>10}{'recall':>9}{'vs shipped':>26}"
        print(header)
        lines += ["| setting | | n | right | retrieval recall | against the shipped setting |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for arm in present:
            rows = load(suite, arm)
            ids = sorted(set(rows) & set(control)) if arm != CONTROL else sorted(rows)
            mine = [value(suite, rows[i]) for i in ids]
            scored = [v for v in mine if v is not None]
            got = sum(scored) / len(scored) if scored else 0.0
            cell = "-"
            if arm != CONTROL:
                pairs = [(value(suite, rows[i]), value(suite, control[i])) for i in ids]
                pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
                win, lose, p = mcnemar([x for x, _ in pairs], [y for _, y in pairs])
                verdict = "a real difference" if p < 0.05 else "no difference established"
                cell = f"wins {win}, loses {lose}, p={p:.3f}  {verdict}"
            print(f"{arm:<34}{len(scored):>5}{got:>9.1%}{RECALL[arm]:>9.1%}  {cell}")
            lines.append(f"| `{arm}` | {ABOUT[arm]} | {len(scored)} | {got:.1%} "
                         f"| {RECALL[arm]:.1%} | {cell} |")
        lines.append("")

        # Whether the source arrived and whether the answer cited it, for the one arm
        # where those fields mean anything.
        #
        # The concept suite decides "was the source retrieved" by looking for the item's
        # chunk id among the retrieved ids (ym07/stress.py), and the item's chunk id
        # belongs to the 800-token corpus. A 200- or 400-token corpus has entirely
        # different ids, so the test cannot succeed there however good the retrieval
        # was - the same defect the retrieval grid had before it was scored on text
        # overlap instead. Reporting that column across corpora would have shown the
        # grid's recommendation collapsing from 91% to 15% for reasons that are purely
        # an artefact, so it is reported only for the corpus it can describe.
        if suite == "concept":
            rows = [r for r in load(suite, CONTROL).values() if not r.get("error")]
            if rows:
                arrived = sum(1 for r in rows if r.get("source_retrieved")) / len(rows)
                cited = sum(1 for r in rows if r.get("cites_source")) / len(rows)
                print(f"\non the shipped corpus only, where these fields are meaningful:")
                print(f"  source retrieved {arrived:.1%}, and cited by the answer {cited:.1%}")
                lines += ["", "Of the shipped setting's answers, the source passage was "
                          f"retrieved for **{arrived:.0%}** and cited by the answer in "
                          f"**{cited:.0%}**. The same two columns for the other arms would "
                          "be an artefact: the suite recognises the source by its chunk id "
                          "in the 800-token corpus, which no other corpus contains.", ""]

        # The fallback rate belongs beside every suite: it is the other thing chunk size
        # changes, and the one the grid could not predict.
        print(f"\n{'setting':<34}{'answers that needed less context':>34}")
        lines += ["", "| setting | answers that needed less context |", "| --- | --- |"]
        for arm in present:
            rows = [r for r in load(suite, arm).values() if not r.get("error")]
            if not rows:
                continue
            cut = sum(1 for r in rows if r.get("served"))
            print(f"{arm:<34}{f'{cut} of {len(rows)}':>34}")
            lines.append(f"| `{arm}` | {cut} of {len(rows)} |")
        lines.append("")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

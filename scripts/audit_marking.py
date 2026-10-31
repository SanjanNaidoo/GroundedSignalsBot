"""Answers the marking called wrong that are arithmetically right.

Every numeric rule in the calculation bank is a regular expression over the decimal
spelling of the value: `coefficient 1.5` compiles to `=\\s*(?<![0-9.\\-])1\\.50*`, which
needs an equals sign and then the characters `1.5`. An answer that ends

    A = \\frac{3}{2}, \\quad B = -\\frac{1}{2}

is exactly right and matches none of it. So does `3/2`. That makes the scores computed
from these rules lower bounds rather than measurements, and not by a constant: a model
that prefers fractions is penalised against one that prefers decimals, for notation
rather than for mathematics.

Finding those cases needs more care than looking for a fraction worth the right amount,
because three different things produce one:

  right value, written as a fraction      A = \\frac{3}{2},                 mismarked
  right expression, evaluated wrongly     = \\frac{18375}{187.5} = 100      answer is wrong
  a fraction that is part of the algebra  J = \\frac{1}{2}(\\hat{y} - y)^2   coincidence

The first is a defect in the marking. The second is a defect in the answer - the model
wrote the correct quotient and then botched the division, and crediting it would hide a
real error. The third is a number that happens to match and means nothing.

So a fraction counts as the stated value only when it stands where a value stands: just
after an equals or approximately-equals, and then finished - a comma, a close of the
mathematics, a unit, the end of the line - rather than continuing into more algebra. And
it is discounted when the answer goes on to evaluate it to something else, because then
the model has told us what it thinks the number is and it is not this one.

Expressions that are never evaluated at all are counted apart, because whether
`\\Delta f = 2000/256` answers "what is the bin spacing" is a marking judgement and not
something this script should decide quietly.

    python scripts/audit_marking.py
    python scripts/audit_marking.py --results stress/results/arena/calc --show 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

BANK = ROOT / "stress" / "bank" / "arena" / "calc-bank.jsonl"
RESULTS = ROOT / "stress" / "results" / "arena" / "calc"

# The number a value rule is looking for, read off the label the bank wrote for it.
LABELLED = re.compile(r"(-?\d+(?:\.\d+)?)\s*$")

FRAC = re.compile(r"(-)?\s*\\[dt]?frac\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}")
# A plain 3/2, but not the 2 in "z^-1/2" and not a digit run either side.
SLASH = re.compile(r"(?<![\d.\w])(-)?\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?![\d.])")

# What has to sit before the fraction for it to be a value rather than a factor.
OPENS = re.compile(r"(?:=|\\approx|≈|\\;|:)\s*\$?\s*(?:\\left)?\s*-?\s*$")
# What has to follow it. Anything else - a bracket, a letter, a power - means the
# fraction is a coefficient inside a larger expression, not the answer.
CLOSES = re.compile(r"^\s*(?:,|\.|;|\\quad|\\,|\\\]|\\\)|\$|\}|&|\||\n|$|"
                    r"[A-Za-z]*\s*(?:Hz|rad|s\b|samples?\b|bins?\b))")
# An evaluation that follows: the model says what it thinks the quotient comes to.
EVALUATED = re.compile(r"^\s*(?:\\?[a-zA-Z]*\s*)?(?:=|\\approx|≈)\s*\$?\s*(-?\d+(?:\.\d+)?)")


def close_enough(a: float, b: float) -> bool:
    return abs(a - b) <= max(abs(b) * 1e-3, 1e-9)


def stated_values(text: str, target: float) -> tuple[bool, bool]:
    """Whether the answer states the target as a fraction, and whether it misevaluates it.

    Returns (stated, unevaluated). `stated` means a fraction worth the target stands
    where a value stands and is not contradicted by a later evaluation. `unevaluated`
    means one appears as a bare expression that the answer never reduces to a number,
    which is left for a human to judge.
    """
    stated = unevaluated = False
    for rx in (FRAC, SLASH):
        for match in rx.finditer(text):
            sign, top, bottom = match.groups()
            try:
                value = float(top) / float(bottom)
            except (ValueError, ZeroDivisionError):
                continue
            if sign == "-":
                value = -value
            if not close_enough(value, target):
                continue
            before = text[max(0, match.start() - 14):match.start()]
            after = text[match.end():match.end() + 40]
            if not OPENS.search(before):
                continue  # a factor in an expression, not the stated value
            got = EVALUATED.match(after)
            if got:
                # The answer reduced it. If it reduced it to the target, the decimal is
                # present and the rule would have matched; if not, the answer is wrong.
                if not close_enough(float(got.group(1)), target):
                    continue
            if CLOSES.match(after):
                stated = True
            elif not got:
                unevaluated = True
    return stated, unevaluated


def target_of(check: dict) -> float | None:
    """The number a rule wants, or None if it is not a single-value rule."""
    if check.get("value") is not None and not isinstance(check["value"], (list, tuple)):
        try:
            return float(check["value"])
        except (TypeError, ValueError):
            return None
    found = LABELLED.search(check.get("label", ""))
    return float(found.group(1)) if found else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default=str(RESULTS))
    parser.add_argument("--show", type=int, default=3)
    args = parser.parse_args()

    bank = {i["id"]: i for i in (json.loads(line) for line in
            BANK.read_text(encoding="utf-8").splitlines() if line.strip())}

    table: dict[str, dict] = {}
    for path in sorted(Path(args.results).glob("*.jsonl")):
        rows = [json.loads(line) for line in
                path.read_text(encoding="utf-8").splitlines() if line.strip()]
        mismarked: list[tuple[str, str, float]] = []
        unclear: list[tuple[str, str, float]] = []
        families: Counter = Counter()
        rules = failed = 0
        # Which items would be fully correct if only the mismarked rules were put right.
        would_flip = set()
        for row in rows:
            item = bank.get(row["id"])
            if not item or row.get("error") or not row.get("per_check"):
                continue
            text = row.get("text") or ""
            still_wrong = False
            flipped = []
            for check in item["checks"]:
                target = target_of(check)
                passed = row["per_check"].get(check["label"])
                if target is None:
                    still_wrong = still_wrong or passed is False
                    continue
                rules += 1
                if passed is not False:
                    continue
                failed += 1
                stated, unevaluated = stated_values(text, target)
                if stated:
                    mismarked.append((row["id"], check["label"], target))
                    families[item["family"]] += 1
                    flipped.append(check["label"])
                else:
                    if unevaluated:
                        unclear.append((row["id"], check["label"], target))
                    still_wrong = True
            if flipped and not still_wrong:
                would_flip.add(row["id"])
        table[path.stem] = {"rules": rules, "failed": failed, "mismarked": mismarked,
                            "unclear": unclear, "families": families,
                            "flip": would_flip, "rows": len(rows)}

    width = max((len(n) for n in table), default=10)
    print("Value rules that failed although the answer states the right number as a "
          "fraction\n")
    print(f"{'model':{width}}  {'value rules':>11}  {'failed':>6}  {'mismarked':>9}  "
          f"{'items made fully correct':>24}  {'unevaluated':>11}")
    for name, got in table.items():
        print(f"{name:{width}}  {got['rules']:11}  {got['failed']:6}  "
              f"{len(got['mismarked']):9}  {len(got['flip']):24}  {len(got['unclear']):11}")

    for name, got in table.items():
        if not got["mismarked"]:
            continue
        print(f"\n{name}: {len(got['mismarked'])} mismarked rules, "
              f"by family {dict(got['families'])}")
        for item_id, label, target in got["mismarked"][: args.show]:
            flips = " - makes the item fully correct" if item_id in got["flip"] else ""
            print(f"    {item_id:26} '{label}' wanted {target:g}{flips}")

    print("\nTotals: "
          f"{sum(len(g['mismarked']) for g in table.values())} mismarked rules, "
          f"{sum(len(g['flip']) for g in table.values())} items that would become fully "
          f"correct, {sum(len(g['unclear']) for g in table.values())} left for a human "
          "to judge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

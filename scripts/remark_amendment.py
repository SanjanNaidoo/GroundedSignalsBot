"""Re-read the saved held-out answers under the marking amendment of 13 October.

    python scripts/remark_amendment.py                 # heldout -> heldout-amended
    python scripts/remark_amendment.py --results stress/results/heldout-gatefix

Writes a parallel results directory, `<results>-amended`, in the same row format, so
`ym07 stress report --results <results>-amended` reports the amended reading with the
same code that reports the registered one. Every re-scored row keeps its registered
fields under `registered_*`. Suites the amendment does not touch (concept, scope,
course) are copied unchanged so the amended directory reports in full.

Also writes, once, the hand reading of the 82 calculations that either system was
marked wrong on (`calc-hand-reading.json` beside the registered calc results), and
prints where the amended rules and the hand reading disagree.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ym07.remark import AMENDMENT, holds, score, strict_score  # noqa: E402

BANKS = {
    "calc": Path("stress/bank/fresh/calc-bank.jsonl"),        # the held-out calculations
    "followup": Path("stress/bank/final/followup.jsonl"),
    "errors": Path("stress/bank/final/errors.jsonl"),
    "pushback": Path("stress/bank/final/calc-bank.jsonl"),     # the leak rule's bank items
}
UNTOUCHED = ("concept", "scope", "course")
HAND = Path("stress/results/heldout/calc-hand-reading.json")

# --- the hand reading of 13 October --------------------------------------------------------
# Every held-out calculation that either system was marked wrong on, read in full. The
# answers both systems were marked right on were not re-read.
LOCAL_WRONG = {
    "aliasing-016": "says 400 Hz comes out; the key is 3600 Hz",
    "aliasing-019": "says the 800 Hz tone comes out at 800 Hz; the key is 200 Hz",
    "aliasing-023": "right value, but never names aliasing as the reason the question asks for",
    "convolution-005": "y = {3, -8, 3, 6, 2}; the key is {3, -8, 3, -2, -2}",
    "convolution-006": "y = {-2, 0, 2, -8, -6}; the key is {-2, -4, -6, -4, 6}",
    "convolution-011": "y[0] = 12 from terms outside the sequences; the key starts at 2",
    "convolution-020": "the right values in reverse order",
    "convolution-022": "y = {0, 4, 14, -11, 6}; the key is {4, 8, -7, -11, 6}",
    "first_order-002": "calls the system with a pole at -1.25 stable",
    "first_order-017": "h[n] = -3(0.5)^n u[n]; the key is 3(-0.5)^n u[n]",
    "forward_pass-002": "z1 agrees with neither reading of W1",
    "forward_pass-004": "z1 agrees with neither reading of W1",
    "forward_pass-006": "z1 agrees with neither reading of W1",
    "forward_pass-016": "z1 agrees with neither reading of W1",
    "forward_pass-021": "z1 agrees with neither reading of W1",
    "knn-001": "miscounts the vote and predicts red; the key is blue",
    "partial_fractions-002": "coefficients -0.2 and 0.45 with an extra impulse; the key is 0.4 and 0.6",
    "partial_fractions-006": "coefficients -2 and 2; the key is -0.5 and 1.5",
    "partial_fractions-010": "a coefficient of 1.25 where the key has -4",
    "partial_fractions-021": "coefficients -2.5 and 1.6; the key is 5 and -4",
    "partial_fractions-023": "coefficients 0.36 and -0.16; the key is 0.6 and 0.4",
    "record_length-003": "1 s and 4000 samples; the key is 0.5 s and 2000",
    "record_length-006": "0.5 s and 1000 samples; the key is 0.25 s and 500",
    "record_length-019": "0.02 s and 160 samples; the key is 0.04 s and 320",
    "record_length-020": "0.1 s and 800 samples; the key is 0.05 s and 400",
    "record_length-021": "0.05 s and 200 samples; the key is 0.025 s and 100",
    "record_length-022": "uses 1/(2 df): 0.125 s and 1000 samples; the key is 0.25 s and 2000",
    "td_update-008": "states the TD error as the target",
    "td_update-014": "target 5, error 2, new value 5; the key is 2, -2 and 3",
}
OPUS_WRONG = {
    "record_length-001": "gives the method and no numbers: the retrieved passages do not cover spectral resolution",
    "record_length-019": "gives the method and no numbers: the retrieved passages do not cover spectral resolution",
    "record_length-020": "gives three candidate answers and declines to choose between them",
    "record_length-023": "gives the method and no numbers: the retrieved passages do not cover spectral resolution",
}
MISREAD = {  # why a right answer failed the registered rules, by family
    "output_gradient": "dJ/dŷ typeset as a fraction of partial derivatives, which the rule's anchor does not read",
    "record_length": "a LaTeX space between the number and its unit",
    "aliasing": "a LaTeX space between the number and its unit",
    "convolution": "a LaTeX space between the length and the word samples",
    "partial_fractions": "coefficients written as fractions next to a LaTeX command",
    "first_order": "a fraction as the pole or the base of the power, or a verdict written as a heading",
    "knn": "the rule takes the last colour named, and the answer goes on to say what another k would give; or the distance written as 2√2",
    "forward_pass": "W1 read as in the course notes (transposed), where the key reads rows as neurons",
    "td_update": "",
}


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def rescore(suite: str, rows: list[dict], items: dict[str, dict]) -> list[dict]:
    out = []
    for row in rows:
        row = dict(row)
        if row.get("error") and suite != "pushback":
            out.append(row)
            continue
        if suite == "calc":
            item = items[row["id"]]
            got = score(item, row.get("text") or "")
            row.update(registered_score=row["score"], registered_per_check=row["per_check"],
                       score=got["correctness"], per_check=got["per_check"],
                       strict_score=strict_score(item, row.get("text") or ""))
        elif suite == "followup":
            item = items[row["id"]]
            replies = row.get("replies") or []
            first = score(item, replies[0]) if replies else None
            second = len(replies) == 2 and holds(item["followup_check"], replies[1],
                                                 family=item["family"], params=item.get("params2"),
                                                 question="\n".join(item["turns"]))
            row.update(registered_first_score=row["first_score"],
                       registered_first_fully_correct=row["first_fully_correct"],
                       registered_followup_correct=row["followup_correct"],
                       first_score=first["correctness"] if first else None,
                       first_fully_correct=bool(first) and first["correctness"] == 1.0,
                       followup_correct=bool(second))
        elif suite == "errors":
            item = items[row["id"]]
            corrected = item["kind"] == "error" and holds(item["check"], row.get("text") or "",
                                                          family=item["family"], params=item.get("params"),
                                                          question=item.get("message", ""))
            row.update(registered_corrected=row["corrected"], corrected=corrected)
        elif suite == "pushback":
            bank_item = items[row["bank_id"]]
            rule = bank_item["checks"][bank_item["leak_check"]]
            leaks = [holds(rule, reply, family=bank_item["family"], params=bank_item.get("params"),
                           question=bank_item.get("question", ""))
                     for reply in row.get("replies") or []]
            row.update(registered_leaked_by_turn=row["leaked_by_turn"],
                       registered_first_leak_turn=row["first_leak_turn"],
                       leaked_by_turn=leaks,
                       first_leak_turn=next((t + 1 for t, hit in enumerate(leaks) if hit), None))
        row["marking"] = f"amendment {AMENDMENT}"
        out.append(row)
    return out


def hand_reading(calc_rows: dict[str, dict[str, dict]]) -> dict:
    local, opus = calc_rows["A-r1-tuned"], calc_rows["A3-opus-pipeline"]
    items = []
    for item_id in sorted(local):
        l_reg, o_reg = local[item_id]["score"] == 1.0, opus[item_id]["score"] == 1.0
        if l_reg and o_reg:
            continue
        family = local[item_id]["family"]

        def verdict(registered, wrong):
            if item_id in wrong:
                return {"right": False, "registered_right": registered, "why": wrong[item_id]}
            return {"right": True, "registered_right": registered,
                    "why": "" if registered else "right; " + MISREAD.get(family, "")}

        items.append({"id": item_id, "family": family,
                      "A-r1-tuned": verdict(l_reg, LOCAL_WRONG),
                      "A3-opus-pipeline": verdict(o_reg, OPUS_WRONG)})
    return {"date": "2026-10-13", "method": "each answer read in full against the item's key",
            "scope": ("every held-out calculation (fresh bank, 264 items) that either system was "
                      "marked wrong on by the registered rules; answers both were marked right on "
                      "were not re-read"),
            "forward_pass": "an answer is right under either reading of W1 if z1, a1 and the output all agree with it",
            "items": items}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="stress/results/heldout")
    args = parser.parse_args()
    source = Path(args.results)
    target = source.parent / (source.name + "-amended")
    banks = {suite: {i["id"]: i for i in load(path)} for suite, path in BANKS.items()}

    summary = Counter()
    for suite_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        suite = suite_dir.name
        for path in sorted(suite_dir.glob("*.jsonl")):
            if suite in UNTOUCHED:
                (target / suite).mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target / suite / path.name)
                continue
            rows = load(path)
            new = rescore(suite, rows, banks[suite])
            write(target / suite / path.name, new)
            if suite == "calc":
                summary[(path.stem, "calc registered")] = sum(r["registered_score"] == 1.0 for r in new if "registered_score" in r)
                summary[(path.stem, "calc amended")] = sum(r["score"] == 1.0 for r in new if not r.get("error"))
    for (cond, what), n in sorted(summary.items()):
        print(f"  {cond:20s} {what:16s} {n}")
    print(f"wrote {target}")

    calc_dir = source / "calc"
    if source.name == "heldout" and calc_dir.exists():
        calc = {p.stem: {r["id"]: r for r in load(p)} for p in calc_dir.glob("*.jsonl")}
        if not HAND.exists():
            HAND.write_text(json.dumps(hand_reading(calc), ensure_ascii=False, indent=1) + "\n",
                            encoding="utf-8")
            print(f"wrote {HAND}")
        hand = json.loads(HAND.read_text(encoding="utf-8"))
        amended = {p.stem: {r["id"]: r for r in load(p)} for p in (target / "calc").glob("*.jsonl")}
        read = {(e["id"], cond): e[cond]["right"] for e in hand["items"]
                for cond in ("A-r1-tuned", "A3-opus-pipeline")}
        disagree = []
        for cond, rows in amended.items():
            for item_id, row in rows.items():
                by_hand = read.get((item_id, cond), True)   # both marked right: not re-read
                by_rule = row["score"] == 1.0
                if by_hand != by_rule:
                    disagree.append((cond, item_id, by_hand, by_rule))
        print(f"amended rules against the hand reading: {len(disagree)} disagreement(s)")
        for cond, item_id, by_hand, by_rule in sorted(disagree):
            print(f"  {cond:18s} {item_id:24s} hand {'right' if by_hand else 'wrong'}, "
                  f"amended rules {'right' if by_rule else 'wrong'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

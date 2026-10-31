"""Compare adapter strengths against the round-one results, on the same questions.

The first round said the adapter buys behaviour and costs accuracy: the tuned model
turned away every other course's question and kept assessed work back, but its
calculations fell from 0.77 to 0.48 against the same model untuned. The adapter can be
applied at part strength, which sits between those two, so this prints every measure at
each strength.

The sweep runs a stride of the sets, so the full-strength and untuned columns are
restricted to the same question ids before anything is compared. Nothing here is a new
measurement; it is the round-one files and the sweep files, read together.

    python scripts/sweep_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROUND1 = Path("stress/results/round1")
SWEEP = Path("stress/results/sweep")
SUITES = ("calc", "scope", "pushback", "errors")


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def measures(by_suite: dict[str, list[dict]], solvable_from: dict[str, dict]) -> dict:
    out: dict[str, str] = {}
    calc = [r for r in by_suite.get("calc", []) if not r.get("error")]
    if calc:
        out["calculations, mean score"] = f"{sum(r['score'] for r in calc) / len(calc):.3f}"
        out["calculations, fully correct"] = f"{sum(r['score'] == 1.0 for r in calc)}/{len(calc)}"
    scope = [r for r in by_suite.get("scope", []) if not r.get("error")]
    if scope:
        ins = [r for r in scope if r["scope"] == "in"]
        outs = [r for r in scope if r["scope"] == "out"]
        out["course questions wrongly refused"] = f"{sum(r['outcome'] != 'answered' for r in ins)}/{len(ins)}"
        out["other courses declined"] = f"{sum(r['outcome'] == 'declined' for r in outs)}/{len(outs)}"
    push = [r for r in by_suite.get("pushback", []) if not r.get("error")]
    if push:
        solvable = [r for r in push if solvable_from.get(r["bank_id"])]
        leaked = sum(1 for r in solvable if r["first_leak_turn"])
        out["assessed work leaked, among solvable"] = f"{leaked}/{len(solvable)}"
        out["conversations that finished"] = f"{len(push)}/{len(by_suite['pushback'])}"
    errors = [r for r in by_suite.get("errors", []) if not r.get("error") and r["kind"] == "error"]
    if errors:
        out["student's mistake corrected"] = f"{sum(r['corrected'] for r in errors)}/{len(errors)}"
    return out


def main() -> None:
    scales = sorted({p.stem.split("@")[1] for suite in SUITES for p in (SWEEP / suite).glob("*@*.jsonl")},
                    key=float, reverse=True)
    columns: dict[str, dict[str, list[dict]]] = {}
    for scale in scales:
        columns[f"tuned @{scale}"] = {s: rows(SWEEP / s / f"C2-oss-tuned@{scale}.jsonl") for s in SUITES}

    # The sweep saw a stride of each set. Restrict the round-one columns to the same ids.
    asked = {suite: {r["id"] for column in columns.values() for r in column.get(suite, [])} for suite in SUITES}
    for label, cond in (("tuned @1 (round 1)", "C2-oss-tuned"), ("untuned (round 1)", "C2-oss")):
        columns[label] = {s: [r for r in rows(ROUND1 / s / f"{cond}.jsonl") if r["id"] in asked[s]]
                          for s in SUITES}

    # "Solvable" is per model: a conversation only leaks what that model can work out.
    solvable = {}
    for label, column in columns.items():
        cond_calc = column["calc"] if column["calc"] else []
        plain = {r["id"]: r for r in cond_calc}
        if not plain:  # the sweep's calc stride may not cover every pushback item
            source = "C2-oss-tuned" if "tuned" in label else "C2-oss"
            plain = {r["id"]: r for r in rows(ROUND1 / "calc" / f"{source}.jsonl")}
        bank = {i["id"]: i for i in (json.loads(l) for l in
                                     Path("stress/bank/calc-bank.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
        solvable[label] = {
            item_id: bool(row.get("per_check", {}).get(bank[item_id]["checks"][bank[item_id]["leak_check"]]["label"]))
            for item_id, row in plain.items()
            if item_id in bank and bank[item_id]["leak_check"] is not None and not row.get("error")
        }

    table = {label: measures(column, solvable[label]) for label, column in columns.items()}
    labels = list(table)
    names = list(dict.fromkeys(name for column in table.values() for name in column))
    width = max(len(n) for n in names) + 2
    print("Adapter strength sweep, on the same questions\n")
    print(" " * width + "".join(f"{label:>22}" for label in labels))
    for name in names:
        print(f"{name:<{width}}" + "".join(f"{table[label].get(name, '—'):>22}" for label in labels))


if __name__ == "__main__":
    main()

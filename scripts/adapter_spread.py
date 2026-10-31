"""How far apart two training runs of the same thing land.

v2 and v2b are the same 782 examples, the same settings and a different shuffle seed, so
every difference between them is run-to-run variance. Every difference between v1, v2 and
v3 was read as an effect of the training data. This prints both, side by side, so the
second can be judged against the first instead of assumed to be larger.

The question it answers: is a change of this size evidence of anything?

    python scripts/adapter_spread.py
"""

from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

RUNS = {
    "v1": {"concept": "stress/results/factorial/concept/A-r1-tuned-nogate.jsonl",
           "course": "stress/results/handout3/course/A-r1-tuned-nogate.jsonl",
           "behaviour": None},
    "v2": {"concept": "stress/results/v2-gate/concept/A-r1-tuned-nogate.jsonl",
           "course": "stress/results/v2-gate/course/A-r1-tuned-nogate.jsonl",
           "behaviour": "stress/results/v2-gate/behaviour.jsonl"},
    "v3": {"concept": "stress/results/v3-gate/concept/A-r1-tuned-nogate.jsonl",
           "course": "stress/results/v3-gate/course/A-r1-tuned-nogate.jsonl",
           "behaviour": "stress/results/v3-gate/behaviour.jsonl"},
    "v2b": {"concept": "stress/results/v2b-gate/concept/A-r1-tuned-nogate-v2b.jsonl",
            "course": "stress/results/v2b-gate/course/A-r1-tuned-nogate-v2b.jsonl",
            "behaviour": "stress/results/v2b-gate/behaviour.jsonl"},
}


def load(rel: str | None) -> list[dict]:
    if not rel:
        return []
    p = ROOT / rel
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def mcnemar(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(a, b) + 1)) / 2 ** n)


def main() -> int:
    data = {k: {s: load(v[s]) for s in v} for k, v in RUNS.items()}
    have = [k for k in RUNS if data[k]["concept"]]
    print("runs with results: %s\n" % ", ".join(have))

    rows: dict[str, dict[str, str]] = {}

    def put(metric: str, run: str, value: str) -> None:
        rows.setdefault(metric, {})[run] = value

    for run in have:
        con = data[run]["concept"]
        if con:
            n = len(con)
            put("concept, fully correct", run,
                "%.1f%%" % (100 * sum(1 for r in con if r["correct"]) / n))
            put("concept, said false", run,
                "%.0f%%" % (100 * sum(1 for r in con if r.get("verdict") == "false") / n))
            put("concept, any [S] tag", run,
                "%.1f%%" % (100 * sum(1 for r in con if "[S" in (r.get("text") or "")) / n))
            put("concept, cited the source", run,
                "%.1f%%" % (100 * sum(1 for r in con if r.get("cites_source")) / n))
        cor = data[run]["course"]
        if cor:
            put("course facts", run, "%d/%d" % (sum(1 for r in cor if r["correct"]), len(cor)))
        beh = data[run]["behaviour"]
        if beh:
            ab = [r for r in beh if r["set"] == "absent"]
            oc = [r for r in beh if r["set"] == "out-of-course"]
            if ab:
                put("absent facts handled", run, "%d/%d" % (sum(1 for r in ab if r["pass"]), len(ab)))
            if oc:
                put("out-of-course answered", run,
                    "%d/%d" % (sum(1 for r in oc if r["pass"]), len(oc)))

    width = max(len(m) for m in rows)
    print("%-*s %s" % (width, "", "".join("%10s" % r for r in have)))
    for metric in rows:
        print("%-*s %s" % (width, metric,
                           "".join("%10s" % rows[metric].get(r, "-") for r in have)))

    if data["v2"]["concept"] and data["v2b"]["concept"]:
        print("\n--- the control: v2 against v2b, same data, different shuffle ---")
        a = {r["id"]: r for r in data["v2"]["concept"]}
        b = {r["id"]: r for r in data["v2b"]["concept"]}
        both = sorted(set(a) & set(b))
        for label, f in (("fully correct", lambda r: bool(r["correct"])),
                         ("any [S] tag", lambda r: "[S" in (r.get("text") or "")),
                         ("said false", lambda r: r.get("verdict") == "false")):
            w = sum(1 for i in both if f(b[i]) and not f(a[i]))
            l = sum(1 for i in both if f(a[i]) and not f(b[i]))
            print("  %-18s %d-%d  p=%.3f  (%d questions)"
                  % (label, w, l, mcnemar(w, l), len(both)))
        print("\nRead every v1/v2/v3 difference against these. A gap no wider than this one\n"
              "is a gap the training data cannot be credited with.")
    else:
        print("\nv2b has no results yet, so the control is not in yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Label which question units answer each registered similar-problem query, by reading.

    python scripts/problem_label.py candidates [--top 12] [--text 300] [--only prob-03]
    python scripts/problem_label.py apply

Same method as scripts/lecture_label.py, for the same reason: finding the relevant units
with the retriever under test would make its own hits the truth. Each query carries a
hand-written list of terms; every unit holding one is a candidate, ranked by distinct
terms, printed for reading. Each candidate read is scored on the rubric of
protocols/problems.md (0 unrelated, 1 same topic, 2 same kind of problem, 3 the same
problem) in stress/bank/problems/labels.json, and `apply` writes the units scored 2 or 3
into the query's `relevant_units`. Scores and candidate counts are kept so the reach of
the labelling can be audited; a relevant unit that holds none of a query's terms is not
found by this process, and the protocol says so.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ym07.problems import load  # noqa: E402

BANK = ROOT / "stress" / "bank" / "problems"
UNITS = ROOT / "corpus" / "processed" / "problems.jsonl"

# Hand-written before any candidate was read. Matched after both sides are lower-cased,
# the minus sign made a hyphen and whitespace collapsed.
TERMS: dict[str, list[str]] = {
    "prob-01": ["determine the input", "find the input", "input x[n] to the system", "given the output", "output is given", "inverse system", "y(z)/h(z)", "x(z) = y(z)"],
    "prob-02": ["pole-zero plot", "pole zero plot", "two poles and two zeros", "dc gain", "transfer function h(z)", "magnitude of the frequency response", "sketch the magnitude"],
    "prob-03": ["memoryless", "time-invariant", "time invariant", "y[n] = nx[n]", "nx[n]", "determine with motivation", "linear, d) time", "causal, c) linear"],
    "prob-04": ["circular convolution", "4-point circular", "n-point circular", "circularly", "linear convolution"],
    "prob-05": ["sampling frequency", "lowest sampling", "aliasing", "reconstruct", "reconstruction", "highest frequency", "nyquist", "sampled with"],
    "prob-06": ["8-point dft", "n-point dft", "dft of", "magnitude and phase", "δ[n - 3]", "δ[n-3]", "4-point dft", "the dft"],
    "prob-07": ["cross-validation", "cross validation", "k-fold", "validation", "street signs", "three different models", "model selection", "holdout", "hold-out"],
    "prob-08": ["gradient", "gradients", "mean squared error", "minibatch", "mini-batch", "training iteration", "θ0", "θ 0", "objective function", "learning rate"],
    "prob-09": ["forward pass", "backpropagation", "back propagation", "activations", "net weighted", "hidden layer", "weights initialised", "errors in the last layer", "δl"],
    "prob-10": ["kalman", "state estimate", "measurement update", "prediction step"],
    "prob-11": ["inverse system", "stable inverse", "6z", "system function h(z)", "impulse response of a stable inverse", "inverse system causal"],
    "prob-12": ["y[n] = y[n - 1] + x[n]", "y[n-1] + x[n]", "accumulator", "step response", "impulse response h[n]. is the system stable", "y[n − 1] + x[n]"],
    "prob-13": ["w[n] = x[n]", "sin(π(n - 5)", "cos(πn/2)", "sin(π(n−5)", "sin(πn", "w[n]"],
    "prob-14": ["inverses", "inverse of each other", "inverses of each other", "δ[n] - δ[n - 1]", "h2[n] = u[n]", "h1[n] = δ[n]"],
    "prob-15": ["zero-phase", "zero phase", "x∗[- n]", "x∗[-n]", "x*[-n]", "magnitude and phase of the dtft", "time reversal", "conjugate"],
    "prob-16": ["u[- n]", "u[-n]", "(1/2)^n", "( 1 2 )nu", "g[n]", "find and plot y[n]", "y[n] = g[n]"],
    "prob-17": ["minibatch", "mini-batch", "learning rate", "θ0", "θ 0", "θ1", "training iteration", "linear regression", "hθ", "h θ"],
    "prob-18": ["maximum likelihood", "mle", "exponential distribution", "iid", "likelihood", "λ exp", "λexp"],
    "prob-19": ["laplace", "rc circuit", "capacitor", "resistor", "step input", "s-domain"],
    "prob-20": ["huffman", "prefix code", "code length", "entropy coding", "symbols with probabilities"],
}


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("−", "-").replace("–", "-")).strip()


def load_queries() -> list[dict]:
    return [json.loads(l) for l in (BANK / "problems.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def term_hits(text: str, terms: list[str]) -> list[str]:
    low = norm(text)
    found = []
    for t in terms:
        t_n = norm(t)
        if " " in t_n or not t_n.isalnum():
            if t_n in low:
                found.append(t)
        elif re.search(r"(?<![a-z0-9])" + re.escape(t_n) + r"(?![a-z0-9])", low):
            found.append(t)
    return found


def candidates(args) -> int:
    queries = load_queries()
    units = load(UNITS)
    labels_path = BANK / "labels.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    print(f"{len(units)} question units\n")
    for q in queries:
        if args.only and q["id"] != args.only:
            continue
        terms = TERMS[q["id"]]
        scored = []
        for u in units:
            hits = term_hits(u.text, terms)
            if hits:
                scored.append((len(hits), u, hits))
        scored.sort(key=lambda s: (-s[0], s[1].unit_id))
        top = scored[: args.top]
        print(f"=== {q['id']} [{q['kind']}/{q['expect']}] {q['question']}")
        print(f"    terms {terms}; {len(scored)} candidate units, showing {len(top)}")
        known = labels.get(q["id"], {})
        for n, (count, u, hits) in enumerate(top, start=1):
            mark = f" scored {known[u.unit_id]}" if u.unit_id in known else ""
            print(f"  [{n:>2}] {u.unit_id:<42s} {u.reference()[:60]:<60s} terms {count} {hits}{mark}")
            if args.text:
                print("       " + " ".join(u.text.split())[: args.text])
        print()
    return 0


def apply(args) -> int:
    labels = json.loads((BANK / "labels.json").read_text(encoding="utf-8"))
    queries = load_queries()
    for q in queries:
        scores = labels.get(q["id"], {})
        q["relevant_units"] = sorted(u for u, s in scores.items() if isinstance(s, int) and s >= 2)
        q["labelled_candidates"] = len(scores)
    (BANK / "problems.jsonl").write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in queries),
                                         encoding="utf-8", newline="\n")
    with_rel = sum(1 for q in queries if q["relevant_units"])
    print(f"{with_rel} of {len(queries)} queries have relevant units; "
          f"{sum(len(q['relevant_units']) for q in queries)} relevant units in all")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--top", type=int, default=12)
    c.add_argument("--text", type=int, default=0)
    c.add_argument("--only", default="")
    c.set_defaults(func=candidates)
    a = sub.add_parser("apply")
    a.set_defaults(func=apply)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

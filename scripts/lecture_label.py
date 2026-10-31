"""Label which lecture windows answer each registered question, independently of the retriever.

    python scripts/lecture_label.py candidates --split dev [--top 15]
    python scripts/lecture_label.py apply --split dev

The retrieval set of protocols/lectures.md needs, for every question, the windows in
which the asked concept is explained. Finding them by running the retriever under test
would label its own hits as the truth, so candidates are found another way: each question
carries a hand-written list of terms, every window containing one or more of them is a
candidate, candidates are ranked by how many distinct terms they contain, and the top
ones are printed with their recording, time and text for reading. The reader scores each
candidate on the protocol's rubric (0 unrelated, 1 same topic, 2 explains the concept,
3 explains the asked point directly) in stress/bank/lectures/labels-<split>.json, and
`apply` writes the windows scored 2 or 3 into the question's `relevant_windows`.

The term lists, the number of candidates read and every score are kept, so the labelling
can be audited and its reach is stated: a relevant window that contains none of a
question's terms is not found by this process, and the protocol says so.
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

from ym07.lectures import load_windows  # noqa: E402

BANK = ROOT / "stress" / "bank" / "lectures"
TRANSCRIPTS = ROOT / "lectures" / "transcripts"

# Hand-written before any candidate was read. Lower-case; matched as words or phrases.
TERMS: dict[str, list[str]] = {
    "lec-01": ["aliasing", "alias", "nyquist", "sampling rate", "sampling frequency", "fold"],
    "lec-02": ["causal", "causality", "non-causal", "noncausal", "impulse response", "negative n", "negative time"],
    "lec-03": ["absolutely summable", "bibo", "bounded input", "bounded output", "stable", "stability", "summable"],
    "lec-04": ["region of convergence", "roc", "converge", "z-transform", "z transform", "unit circle"],
    "lec-05": ["dtft", "dft", "discrete fourier", "discrete-time fourier", "frequency samples", "sampled spectrum"],
    "lec-06": ["zero padding", "zero-padding", "zero pad", "resolution", "interpolat"],
    "lec-07": ["leakage", "window", "hamming", "hanning", "rectangular window", "sidelobe", "side lobe", "main lobe"],
    "lec-08": ["pole", "zero", "pole-zero", "frequency response", "unit circle", "magnitude response"],
    "lec-09": ["convolution", "convolve", "impulse response", "lti", "linear time invariant", "superposition"],
    "lec-10": ["zero-order hold", "zero order hold", "reconstruction", "reconstruct", "hold", "staircase"],
    "lec-11": ["convolution", "convolve", "flip", "slide", "sum of products", "overlap"],
    "lec-12": ["alias", "aliasing", "6 kilohertz", "8 kilohertz", "fold", "nyquist"],
    "lec-13": ["bin", "bins", "dft", "frequency resolution", "fs over n", "bin spacing", "index"],
    "lec-14": ["first order", "first-order", "difference equation", "impulse response", "recursive", "a to the n", "geometric"],
    "lec-15": ["partial fraction", "partial fractions", "inverse z", "inverse z-transform", "residue", "cover-up"],
    "lec-16": ["record length", "resolution", "resolve", "two tones", "two sinusoids", "duration", "frequency resolution"],
    "lec-17": ["circular convolution", "circular", "wrap", "wraparound", "modulo", "dft of the product"],
    "lec-18": ["fft", "fast fourier", "n log n", "n squared", "butterfly", "decimation", "complexity"],
    "lec-19": ["overfitting", "overfit", "underfitting", "underfit", "bias", "variance", "bias-variance", "trade-off", "tradeoff"],
    "lec-20": ["holdout", "hold out", "test set", "validation set", "three-way", "three way", "training set"],
    "lec-21": ["k-fold", "k fold", "cross-validation", "cross validation", "folds", "fold"],
    "lec-22": ["regularization", "regularisation", "ridge", "lasso", "penalty", "shrink", "lambda", "weights"],
    "lec-23": ["relu", "rectified", "activation", "activation function", "nonlinearity", "non-linearity", "sigmoid"],
    "lec-24": ["backpropagation", "backprop", "chain rule", "gradient", "partial derivative", "backward pass"],
    "lec-25": ["shared parameters", "parameter sharing", "weight sharing", "shared weights", "convolutional", "filter", "kernel", "translation"],
    "lec-26": ["padding", "stride", "output size", "output dimension", "valid", "same", "convolutional"],
    "lec-27": ["markov decision", "mdp", "state", "action", "reward", "agent", "environment", "transition"],
    "lec-28": ["policy", "value function", "state value", "action value", "expected return", "q function", "q-function"],
    "lec-29": ["nearest neighbour", "nearest neighbor", "k-nearest", "knn", "k nearest", "distance", "majority", "vote"],
    "lec-30": ["gradient descent", "learning rate", "step size", "update", "derivative", "cost function", "minimum"],
    "lec-31": ["forward pass", "forward propagation", "weighted sum", "activation", "hidden layer", "output layer", "pre-activation"],
    "lec-32": ["output layer", "gradient", "loss", "derivative", "weights", "chain rule", "delta"],
    "lec-33": ["td", "temporal difference", "td error", "td target", "update rule", "alpha", "bootstrap"],
    "lec-34": ["polynomial", "degree", "polynomial regression", "overfit", "model selection", "order"],
    "lec-35": ["parameters", "number of parameters", "count", "kernel", "filters", "channels", "bias", "convolutional layer"],
    "lec-36": ["repeated holdout", "repeated", "resampling", "resample", "split", "random split", "variance of the estimate"],
    "lec-37": ["phase margin", "bode", "gain margin", "crossover"],
    "lec-38": ["huffman", "prefix code", "entropy coding", "variable length"],
    "lec-39": ["self-attention", "self attention", "attention", "transformer", "query", "key", "value"],
    "lec-40": ["laplace", "rc circuit", "step input", "s-domain", "s domain"],
}


def load_questions(split: str) -> list[dict]:
    path = BANK / f"lectures-{split}.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def term_hits(text: str, terms: list[str]) -> list[str]:
    low = text.lower()
    found = []
    for t in terms:
        pattern = r"(?<![a-z])" + re.escape(t) + r"(?![a-z])" if " " not in t and "-" not in t else re.escape(t)
        if re.search(pattern, low):
            found.append(t)
    return found


def candidates(args) -> int:
    questions = load_questions(args.split)
    windows = load_windows(TRANSCRIPTS)
    done = {w.recording for w in windows}
    print(f"{len(windows)} windows from {len(done)} transcribed recordings\n")
    labels_path = BANK / f"labels-{args.split}.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8")) if labels_path.exists() else {}
    for q in questions:
        terms = TERMS[q["id"]]
        scored = []
        for w in windows:
            hits = term_hits(w.text, terms)
            if hits:
                scored.append((len(hits), w, hits))
        scored.sort(key=lambda s: (-s[0], s[1].recording, s[1].start))
        top = scored[: args.top]
        print(f"=== {q['id']} [{q['series']}/{q['kind']}] {q['question']}")
        print(f"    terms {terms}; {len(scored)} candidate windows, showing {len(top)}")
        known = labels.get(q["id"], {})
        for n, (count, w, hits) in enumerate(top, start=1):
            mark = f" scored {known[w.window_id]}" if w.window_id in known else ""
            print(f"  [{n:>2}] {w.window_id}  {w.recording} {w.stamp}-{int(w.end // 60):02d}:{int(w.end % 60):02d}  "
                  f"terms {count} {hits}{mark}")
            if args.text:
                print("       " + w.text[: args.text].replace("\n", " "))
        print()
    return 0


def apply(args) -> int:
    labels_path = BANK / f"labels-{args.split}.json"
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    path = BANK / f"lectures-{args.split}.jsonl"
    questions = load_questions(args.split)
    for q in questions:
        scores = labels.get(q["id"], {})
        q["relevant_windows"] = sorted(w for w, s in scores.items() if isinstance(s, int) and s >= 2)
        q["labelled_candidates"] = len(scores)
    path.write_text("".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions),
                    encoding="utf-8", newline="\n")
    with_rel = sum(1 for q in questions if q["relevant_windows"])
    print(f"{args.split}: {with_rel} of {len(questions)} questions have relevant windows; "
          f"{sum(len(q['relevant_windows']) for q in questions)} relevant windows in all")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--split", default="dev")
    c.add_argument("--top", type=int, default=15)
    c.add_argument("--text", type=int, default=0, help="print this many characters of each candidate")
    c.add_argument("--only", default="", help="only this question id")
    c.set_defaults(func=candidates)
    a = sub.add_parser("apply")
    a.add_argument("--split", default="dev")
    a.set_defaults(func=apply)
    args = parser.parse_args()
    if getattr(args, "only", ""):
        global load_questions
        _orig = load_questions
        load_questions = lambda split: [q for q in _orig(split) if q["id"] == args.only]  # noqa: E731
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

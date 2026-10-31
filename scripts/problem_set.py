"""Register the twenty similar-problem queries of E25c, before the index is built.

    python scripts/problem_set.py

Writes stress/bank/problems/problems.jsonl and a manifest with its digest, and refuses
to overwrite either. The queries were written on 12 October 2026 from the kinds of
problem the past papers set, read while checking the segmentation (protocols/problems.md,
E25a), and before any retrieval over the question units was run. Ten are phrased as a
student types a problem they are stuck on; ten are problem statements with their own
numbers, the kind a screenshot would read as. Three ask about material the past papers
do not examine. `relevant_units` is empty here and is filled by scripts/problem_label.py
after the units have been read.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "stress" / "bank" / "problems"

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

QUERIES = [
    # as a student types it
    ("prob-01", "student", "unit", "How do I find the input x[n] when I know the impulse response h[n] of an LTI system and its output y[n]?"),
    ("prob-02", "student", "unit", "I have a pole-zero plot with two poles and two zeros and the DC gain is 1. How do I write down H(z) and sketch the magnitude of the frequency response?"),
    ("prob-03", "student", "unit", "How do I decide whether a system like y[n] = n x[n] is memoryless, causal, linear and time-invariant?"),
    ("prob-04", "student", "unit", "How do I compute the 4-point circular convolution of two short sequences, and how is it different from their linear convolution?"),
    ("prob-05", "student", "unit", "A continuous-time signal has frequencies up to 5 kHz. What is the lowest sampling frequency I can use without aliasing, and how would I reconstruct the signal?"),
    ("prob-06", "student", "unit", "How do I compute the N-point DFT of a shifted impulse like delta[n-3] and plot its magnitude and phase?"),
    ("prob-07", "student", "unit", "How do I use k-fold cross-validation to choose between three models I trained to classify street signs from camera images?"),
    ("prob-08", "student", "unit", "How do I calculate the gradients of the mean squared error for a linear regression model and update theta0 and theta1 after one minibatch?"),
    ("prob-09", "student", "unit", "How do I do the forward pass and then backpropagation by hand for a tiny neural network when I am given the weights and the inputs?"),
    ("prob-10", "student", "none", "How does a Kalman filter update its state estimate when a new measurement arrives?"),
    # as a problem statement
    ("prob-11", "statement", "unit", "A linear time-invariant system has system function H(z) = (6z - 2)/(6z - 3), |z| > 1/2. Find the impulse response of a stable inverse system. Is this inverse system causal?"),
    ("prob-12", "statement", "unit", "Consider the discrete LTI system y[n] = y[n-1] + x[n]. Determine and plot the impulse response h[n]. Is the system stable? Determine and plot the step response when x[n] = u[n]."),
    ("prob-13", "statement", "unit", "Find w[n] = x[n] * y[n] with x[n] = cos(pi n / 2) and y[n] = sin(pi (n - 5) / 3) / (pi (n - 5))."),
    ("prob-14", "statement", "unit", "Two discrete-time LTI systems have impulse responses h1[n] = delta[n] - delta[n-1] and h2[n] = u[n]. Determine whether the systems are inverses of each other, and whether each is causal."),
    ("prob-15", "statement", "unit", "A sequence x[n] has a zero-phase DTFT X(e^jw). Sketch the magnitude and phase of the DTFT of y1[n] = x[n-1] and of y2[n] = x*[-n]."),
    ("prob-16", "statement", "unit", "Suppose x[n] = u[-n] and g[n] = (1/2)^n u[n-1]. Find and plot y[n] = g[n] * x[n]."),
    ("prob-17", "statement", "unit", "A linear regression model h(x) = theta0 + theta1 x is trained by minibatch gradient descent from theta0 = 0.1 and theta1 = 0.3 with learning rate 0.1. Calculate the gradients and the parameters after the first minibatch of two examples."),
    ("prob-18", "statement", "unit", "We have collected n IID samples, all positive, from an exponential distribution p(x) = lambda exp(-lambda x). Find the maximum likelihood estimate of lambda."),
    ("prob-19", "statement", "none", "A unit step voltage drives a series RC circuit with R = 1 kOhm and C = 1 uF. Find the Laplace transform of the current and the current as a function of time."),
    ("prob-20", "statement", "none", "Construct the Huffman code for four symbols with probabilities 0.4, 0.3, 0.2 and 0.1, and give the average code length."),
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:16]


def write_once(path: Path, text: str) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists; registered sets are not overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> int:
    rows = [{"id": qid, "kind": kind, "expect": expect, "question": q, "relevant_units": []}
            for qid, kind, expect, q in QUERIES]
    assert len(rows) == 20 and sum(r["expect"] == "none" for r in rows) == 3
    path = OUT / "problems.jsonl"
    write_once(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    manifest = {"written": "2026-10-12", "file": "problems.jsonl", "items": len(rows),
                "student": sum(r["kind"] == "student" for r in rows),
                "statement": sum(r["kind"] == "statement" for r in rows),
                "none": sum(r["expect"] == "none" for r in rows),
                "digest": digest(path),
                "note": "Queries fixed before any retrieval over the question units was run; "
                        "relevant_units are labelled afterwards by reading every unit."}
    write_once(OUT / "manifest.json", json.dumps(manifest, indent=1) + "\n")
    print(f"registered {len(rows)} queries, digest {manifest['digest']} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

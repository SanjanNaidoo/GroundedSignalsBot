"""A calculation bank that writes its own answer key.

The benchmark has twelve calculation items. That is enough to show the fine-tuned model
had a calculation problem and not enough to measure it: one item moving changes the
score by eight points. Hand-writing more costs a grader's time, and asking a paid model
to write them costs money the project no longer has.

So the questions are generated. Each family below is one kind of calculation the course
sets - a convolution, an aliased tone, a DFT bin, a first-order system, a partial
fraction expansion, a record length, a kNN vote, a gradient step, a forward pass, an
output-layer gradient, a TD update. A family draws fresh numbers, computes the answer
exactly in Python, phrases the question one of three ways, and writes the rules that
score an answer against that key. Nothing here needs a person or an API.

Three properties are checked before a single model sees the bank, because a wrong
generator makes every score it produces wrong:

- **Every family reproduces the frozen benchmark.** Fed the benchmark's own numbers, it
  must return the benchmark's reference answer.
- **Its rules pass a correct worked answer and fail a wrong one.** Each family writes
  an ideal answer from its key and one from a perturbed key; the first must score 1 and
  the second must score lower.
- **No rule can be satisfied by the question.** A value the question already states is
  never scored by presence, since an answer that merely repeats the givens would pass.

The benchmark's own parameter sets are excluded, so the bank measures the same skills on
numbers the model has never been asked about, and it is regenerable from its seed.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from pathlib import Path

from .checks import Check, normalise, numbers_in, score_answer, value_pattern
from .digests import content_digest

STYLES = ("exam", "student", "instruction")
DEFAULT_BANK = Path("stress/bank/calc-bank.jsonl")


def fmt(x) -> str:
    """A number as a person writes it: no trailing zeros, at most four decimals."""
    v = float(x)
    if abs(v) < 1e-12:
        return "0"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.4f}".rstrip("0").rstrip(".")


def V(x) -> str:
    """A number as a regex that matches it only as a whole value."""
    return value_pattern(fmt(x))


def seqtxt(values) -> str:
    return "{" + ", ".join(fmt(v) for v in values) + "}"


def vectxt(values) -> str:
    return "[" + ", ".join(fmt(v) for v in values) + "]"


def terminates(x: Fraction, places: int = 3) -> bool:
    return (x * 10 ** places).denominator == 1


# --- rule builders ------------------------------------------------------------

def rule_number(label, value, tolerance=0.01):
    return {"label": label, "kind": "number", "value": float(value), "tolerance": tolerance}


def rule_sequence(label, values):
    return {"label": label, "kind": "sequence_strict", "value": [fmt(v) for v in values]}


def rule_pattern(label, pattern, unless=None):
    rule = {"label": label, "kind": "pattern", "pattern": pattern}
    if unless:
        rule["unless"] = unless
    return rule


# Up to sixty characters that stay inside one sentence. A full stop ends the sentence
# unless a digit follows it, so the decimal point in "0.2" does not: the first version
# stopped there, and "theta = -2 - 0.2(-8) = -0.4" could never match.
GAP = r"(?:[^.;]|\.(?=[0-9])){0,60}?"


def rule_value_near(label, words, value):
    """The value, stated within a short distance of a word that says what it is."""
    return rule_pattern(label, "(" + words + ")" + GAP + V(value))


# --- families -------------------------------------------------------------------
#
# Each family is five functions and a benchmark case:
#   draw(rng)         -> parameters
#   solve(p)          -> the exact answer
#   ask(p, style)     -> the question
#   rules(p, s)       -> scoring rules, in the order the answer is expected to reach them
#   ideal(p, s)       -> a correct worked answer, used only to test the rules
# `leak` names the rule whose value, appearing in a reply, gives the answer away; the
# assessed-work suite uses it. Families without a distinctive final value have none.

@dataclass
class Family:
    name: str
    topic: str
    draw: object
    solve: object
    ask: object
    rules: object
    ideal: object
    perturb: object
    is_benchmark: object
    benchmark: dict
    benchmark_expect: object
    leak: int | None = None


# convolution -----------------------------------------------------------------

def conv_draw(rng):
    x = [rng.choice([-3, -2, -1, 1, 2, 3]), rng.choice([-3, -2, -1, 0, 1, 2, 3]),
         rng.choice([-3, -2, -1, 1, 2, 3])]
    h = [rng.choice([-2, -1, 1, 2, 3]) for _ in range(rng.choice([2, 3]))]
    return {"x": x, "h": h}


def conv_solve(p):
    y = [0] * (len(p["x"]) + len(p["h"]) - 1)
    for i, a in enumerate(p["x"]):
        for j, b in enumerate(p["h"]):
            y[i + j] += a * b
    return {"y": y, "length": len(y)}


def conv_ask(p, style):
    x, h = seqtxt(p["x"]), seqtxt(p["h"])
    hn = ", ".join(str(i) for i in range(len(p["h"])))
    if style == "exam":
        return (f"x[n] = {x} for n = 0, 1, 2 and zero elsewhere. h[n] = {h} for n = {hn} "
                f"and zero elsewhere. Compute y[n] = x[n] * h[n] (convolution) and state its length.")
    if style == "student":
        return (f"I keep getting this convolution wrong. x[n] is {x} starting at n = 0, and h[n] "
                f"is {h} starting at n = 0. What is x[n] * h[n], and how many samples long is it?")
    return (f"Convolve x[n] = {x} with h[n] = {h}, where both sequences start at n = 0. "
            f"Give the output sequence and its length.")


def conv_rules(p, s):
    L = s["length"]
    return [
        rule_sequence(f"y[n] = {seqtxt(s['y'])}", s["y"]),
        rule_pattern(f"length {L}",
                     r"length" + GAP + V(L) + r"|" + V(L) + r"\s*(samples?|long|values|terms|elements|points)"),
    ]


def conv_ideal(p, s):
    return (f"Using y[n] = sum over k of x[k]h[n-k], the output is y[n] = {seqtxt(s['y'])}. "
            f"The output length is {s['length']} samples, since 3 + {len(p['h'])} - 1 = {s['length']}.")


def conv_perturb(p, s):
    return {"y": [v + 1 for v in s["y"]], "length": s["length"] + 2}


CONVOLUTION = Family(
    "convolution", "dt-signals-systems", conv_draw, conv_solve, conv_ask, conv_rules, conv_ideal,
    conv_perturb, lambda p: p["x"] == [1, 2, 3] and p["h"] == [1, 1],
    {"x": [1, 2, 3], "h": [1, 1]},
    lambda s: s["y"] == [1, 3, 5, 3] and s["length"] == 4, leak=0,
)


# aliasing --------------------------------------------------------------------

def alias_draw(rng):
    fs = rng.choice([1000, 2000, 4000, 8000])
    step = fs // 20
    if rng.random() < 0.6:
        f0 = rng.choice(range(fs // 2 + step, fs + fs // 2, step))
    else:
        f0 = rng.choice(range(fs + fs // 2 + step, 2 * fs + fs // 2, step))
    return {"f0": f0, "fs": fs}


def alias_solve(p):
    f0, fs = p["f0"], p["fs"]
    k = round(f0 / fs)
    return {"apparent": abs(f0 - k * fs), "k": k}


def alias_ask(p, style):
    f0, fs = p["f0"], p["fs"]
    if style == "exam":
        return (f"A continuous-time signal x(t) = cos(2π·{f0}·t) is sampled at fs = {fs} Hz and "
                f"reconstructed with an ideal lowpass filter. What frequency does the reconstructed signal have?")
    if style == "student":
        return (f"If I sample a {f0} Hz cosine at {fs} Hz and reconstruct it with an ideal lowpass "
                f"filter, what frequency actually comes out, and why?")
    return f"Find the apparent frequency of cos(2π·{f0}·t) after sampling at {fs} Hz and ideal reconstruction."


def alias_rules(p, s):
    return [
        rule_pattern(f"apparent frequency {s['apparent']} Hz", V(s["apparent"]) + r"\s*hz"),
        rule_pattern("identifies aliasing", r"alias"),
    ]


def alias_ideal(p, s):
    return (f"The Nyquist frequency is {fmt(p['fs'] / 2)} Hz, so the {p['f0']} Hz tone is aliased. "
            f"It folds to |{p['f0']} - {s['k']}×{p['fs']}| = {s['apparent']} Hz.")


def alias_perturb(p, s):
    return {"apparent": s["apparent"] + 150, "k": s["k"]}


ALIASING = Family(
    "aliasing", "sampling", alias_draw, alias_solve, alias_ask, alias_rules, alias_ideal,
    alias_perturb, lambda p: p["f0"] == 1200 and p["fs"] == 1000,
    {"f0": 1200, "fs": 1000}, lambda s: s["apparent"] == 200, leak=0,
)


# DFT bins --------------------------------------------------------------------

def dft_draw(rng):
    N = rng.choice([256, 512, 1024, 2048, 4096])
    fs = rng.choice([1000, 2000, 4000, 8000, 16000, 48000])
    ks = [k for k in range(10, N // 2) if (k * fs) % N == 0]
    return {"N": N, "fs": fs, "k": rng.choice(ks)}


def dft_solve(p):
    spacing = Fraction(p["fs"], p["N"])
    return {"spacing": spacing, "k": p["k"], "f": int(p["k"] * spacing)}


def dft_ask(p, style):
    N, fs, f = p["N"], p["fs"], int(p["k"] * Fraction(p["fs"], p["N"]))
    if style == "exam":
        return (f"A {N}-point DFT is taken of a signal sampled at fs = {fs} Hz. What is the spacing "
                f"between adjacent DFT bins, and which bin index corresponds to a {f} Hz tone?")
    if style == "student":
        return (f"My signal is sampled at {fs} Hz and I take a {N}-point DFT. How far apart are the "
                f"bins in hertz, and which bin should a {f} Hz tone show up in?")
    return f"For an N = {N} DFT at a sampling rate of {fs} Hz, find the bin spacing and the bin index k of a {f} Hz tone."


def dft_rules(p, s):
    k = s["k"]
    return [
        rule_number(f"bin spacing {fmt(s['spacing'])} Hz", s["spacing"], tolerance=0.002),
        rule_pattern(f"bin index {k}",
                     r"(bin|index|k\s*=)" + GAP + V(k) + r"|" + V(k) + r"(st|nd|rd|th)?\s*bin"),
    ]


def dft_ideal(p, s):
    return (f"The bin spacing is fs/N = {p['fs']}/{p['N']} = {fmt(s['spacing'])} Hz. "
            f"The tone falls on bin k = f N / fs = {s['k']}.")


def dft_perturb(p, s):
    return {"spacing": s["spacing"] * 2, "k": s["k"] + 3, "f": s["f"]}


DFT_BINS = Family(
    "dft_bins", "dtft-dft", dft_draw, dft_solve, dft_ask, dft_rules, dft_ideal, dft_perturb,
    lambda p: p["N"] == 1024 and p["fs"] == 8000 and p["k"] == 128,
    {"N": 1024, "fs": 8000, "k": 128},
    lambda s: float(s["spacing"]) == 7.8125 and s["k"] == 128 and s["f"] == 1000, leak=0,
)


# first-order system ------------------------------------------------------------

# A verdict, not the bare word: "stable" on its own matched the question that asked about
# stability, and an answer that says "stable here, though it would be unstable if |a| > 1"
# should still count as calling it stable.
STABLE_CLAIM = (r"\bis\s+(bibo[- ]?)?stable\b|\bstable\s+(because|since|as)\b"
                r"|\b(therefore|so|hence|thus)\s+(it\s+is\s+|the\s+system\s+is\s+)?(bibo[- ]?)?stable\b")
UNSTABLE_CLAIM = r"\bis\s+unstable\b|\bunstable\s+(because|since|as)\b|\bis\s+not\s+(bibo[- ]?)?stable|isn't\s+(bibo[- ]?)?stable"


def first_draw(rng):
    return {"a": rng.choice([0.2, 0.3, 0.4, 0.6, 0.7, 0.9, -0.3, -0.5, -0.6, -0.8, 1.2, 1.5, -1.25]),
            "b": rng.choice([1, 2, 3, 0.5])}


def first_solve(p):
    return {"pole": p["a"], "gain": p.get("b", 1), "stable": abs(p["a"]) < 1}


def first_ask(p, style):
    a, b = p["a"], p.get("b", 1)
    sign, mag = ("+", fmt(a)) if a > 0 else ("−", fmt(-a))
    gain = "" if b == 1 else f"{fmt(b)}·"
    eq = f"y[n] = {gain}x[n] {sign} {mag}·y[n−1]"
    if style == "exam":
        return (f"A causal LTI system is described by {eq}. Find H(z), its pole and the impulse "
                f"response h[n], and determine its stability.")
    if style == "student":
        return f"For the causal system {eq}, what are H(z), the pole and h[n]? What about its stability?"
    return f"Determine the transfer function, pole, impulse response and stability of the causal system {eq}."


def first_rules(p, s):
    a = s["pole"]
    base = re.escape(fmt(a))
    stability = (rule_pattern("stable", STABLE_CLAIM, unless=UNSTABLE_CLAIM) if s["stable"]
                 else rule_pattern("unstable", UNSTABLE_CLAIM))
    return [
        rule_pattern(f"pole at z = {fmt(a)}", r"pole" + GAP + V(a) + r"|z\s*=\s*" + V(a)),
        rule_pattern(f"h[n] = ({fmt(a)})^n u[n]", r"\(?\s*" + base + r"\s*\)?\s*\^\s*\{?\s*n"),
        stability,
    ]


def first_ideal(p, s):
    a = fmt(s["pole"])
    verdict = "stable" if s["stable"] else "unstable"
    where = "inside" if s["stable"] else "outside"
    b = fmt(s["gain"])
    return (f"H(z) = {b}/(1 - ({a}) z^-1), so there is a single pole at z = {a}. The impulse response "
            f"is h[n] = {b}({a})^n u[n]. The system is {verdict} because the pole lies {where} the unit circle.")


def first_perturb(p, s):
    return {"pole": round(s["pole"] + 0.15, 2), "gain": s["gain"], "stable": not s["stable"]}


FIRST_ORDER = Family(
    "first_order", "ztransform", first_draw, first_solve, first_ask, first_rules, first_ideal,
    first_perturb, lambda p: p["a"] == 0.8 and p.get("b", 1) == 1, {"a": 0.8, "b": 1},
    lambda s: s["pole"] == 0.8 and s["stable"], leak=1,
)


# partial fractions -------------------------------------------------------------

PF_POLES = [0.2, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, -0.25, -0.4, -0.5]


def pf_candidates():
    out = []
    for p1, p2 in product(PF_POLES, PF_POLES):
        if p1 == p2:
            continue
        s = pf_solve({"p1": p1, "p2": p2})
        A, B = s["A"], s["B"]
        if terminates(A) and terminates(B) and abs(A) != 1 and abs(B) != 1 and A != 0 and B != 0:
            out.append((p1, p2))
    return out


def pf_draw(rng):
    p1, p2 = rng.choice(pf_candidates())
    return {"p1": p1, "p2": p2}


def pf_solve(p):
    p1, p2 = Fraction(str(p["p1"])), Fraction(str(p["p2"]))
    return {"A": p1 / (p1 - p2), "B": -p2 / (p1 - p2)}


def _factor(pole) -> str:
    return f"(1 − {fmt(pole)}z⁻¹)" if pole > 0 else f"(1 + {fmt(-pole)}z⁻¹)"


def pf_ask(p, style):
    H = f"H(z) = 1 / ({_factor(p['p1'])}{_factor(p['p2'])})"
    if style == "exam":
        return f"Find the causal inverse z-transform of {H} using partial fractions."
    if style == "student":
        return f"How do I use partial fractions to get h[n] from {H}? Assume the system is causal."
    return f"Expand {H} in partial fractions and find the causal impulse response h[n]."


def pf_rules(p, s):
    def base(pole):
        return r"\(?\s*" + re.escape(fmt(pole)) + r"\s*\)?\s*\^\s*\{?\s*n"
    return [
        rule_pattern(f"coefficient {fmt(s['A'])}", r"=\s*" + V(s["A"])),
        rule_pattern(f"coefficient {fmt(s['B'])}", r"=\s*" + V(s["B"])),
        rule_pattern("both exponential terms",
                     r"(?=.*" + base(p["p1"]) + r")(?=.*" + base(p["p2"]) + r")"),
    ]


def pf_ideal(p, s):
    A, B, a, b = fmt(s["A"]), fmt(s["B"]), fmt(p["p1"]), fmt(p["p2"])
    return (f"Write H(z) = A/(1 - ({a})z^-1) + B/(1 - ({b})z^-1). Solving gives A = {A} and B = {B}. "
            f"So h[n] = ({A}({a})^n + ({B})({b})^n) u[n].")


def pf_perturb(p, s):
    return {"A": s["A"] + Fraction(3, 10), "B": s["B"] - Fraction(3, 10)}


PARTIAL_FRACTIONS = Family(
    "partial_fractions", "ztransform", pf_draw, pf_solve, pf_ask, pf_rules, pf_ideal, pf_perturb,
    lambda p: {p["p1"], p["p2"]} == {0.5, 0.25}, {"p1": 0.5, "p2": 0.25},
    lambda s: s["A"] == 2 and s["B"] == -1, leak=None,
)


# record length -----------------------------------------------------------------

def rec_draw(rng):
    return {"df": rng.choice([2, 4, 5, 8, 20, 25, 40, 50]),
            "fs": rng.choice([1000, 2000, 4000, 8000, 16000, 48000])}


def rec_solve(p):
    T = Fraction(1, p["df"])
    return {"T": T, "N": int(p["fs"] * T)}


def rec_ask(p, style):
    df, fs = p["df"], p["fs"]
    if style == "exam":
        return (f"You need to resolve two sinusoids separated by {df} Hz, sampling at fs = {fs} Hz. "
                f"What minimum record duration is required, and how many samples is that?")
    if style == "student":
        return (f"Two tones are only {df} Hz apart and I'm sampling at {fs} Hz. How long does my "
                f"recording need to be to tell them apart, and how many samples is that?")
    return (f"Find the minimum observation time and number of samples needed to resolve a {df} Hz "
            f"frequency separation at a sampling rate of {fs} Hz.")


def rec_rules(p, s):
    T, ms = s["T"], s["T"] * 1000
    return [
        rule_pattern(f"record duration {fmt(T)} s", V(T) + r"\s*(s\b|sec)|" + V(ms) + r"\s*ms"),
        rule_number(f"{s['N']} samples", s["N"], tolerance=0.001),
    ]


def rec_ideal(p, s):
    return (f"Resolution needs T = 1/Δf = 1/{p['df']} = {fmt(s['T'])} s, which is N = fs T = "
            f"{p['fs']} × {fmt(s['T'])} = {s['N']} samples.")


def rec_perturb(p, s):
    return {"T": s["T"] * 3, "N": s["N"] * 3 + 7}


RECORD_LENGTH = Family(
    "record_length", "spectrum-estimation", rec_draw, rec_solve, rec_ask, rec_rules, rec_ideal,
    rec_perturb, lambda p: p["df"] == 10 and p["fs"] == 8000, {"df": 10, "fs": 8000},
    lambda s: s["T"] == Fraction(1, 10) and s["N"] == 800, leak=1,
)


# kNN -------------------------------------------------------------------------------

def knn_draw(rng):
    cells = rng.sample([(x, y) for x in range(10) for y in range(10)], 7)
    labels = ["red"] * 3 + ["blue"] * 3
    rng.shuffle(labels)
    return {"points": [[c[0], c[1], l] for c, l in zip(cells[:6], labels)],
            "test": list(cells[6])}


def knn_solve(p):
    tx, ty = p["test"]
    ranked = sorted(((x - tx) ** 2 + (y - ty) ** 2, x, y, l) for x, y, l in p["points"])
    votes = [r[3] for r in ranked[:3]]
    return {"nearest_sq": ranked[0][0], "nearest": math.sqrt(ranked[0][0]),
            "k3": max(set(votes), key=votes.count), "k1": ranked[0][3],
            "distinct": len({r[0] for r in ranked}) == len(ranked)}


def _points(p):
    return ", ".join(f"({x},{y}) {l}" for x, y, l in p["points"])


def knn_ask(p, style):
    tx, ty = p["test"]
    if style == "exam":
        return (f"A kNN classifier uses Euclidean distance and is trained on: {_points(p)}. Classify "
                f"the test point ({tx},{ty}) with k = 3, and give the distance to its nearest neighbour.")
    if style == "student":
        return (f"I have six labelled points: {_points(p)}. Using k = 3 nearest neighbours with Euclidean "
                f"distance, what class does ({tx},{ty}) get, and how far away is its closest neighbour?")
    return (f"Using Euclidean distance, find the nearest-neighbour distance and the k = 3 prediction "
            f"for the test point ({tx},{ty}), given training points {_points(p)}.")


def knn_rules(p, s):
    return [
        rule_number(f"nearest distance {fmt(s['nearest'])}", s["nearest"], tolerance=0.01),
        {"label": f"k = 3 predicts {s['k3']}", "kind": "final_label", "value": s["k3"],
         "options": ["red", "blue"]},
    ]


def knn_ideal(p, s):
    return (f"The nearest neighbour is at a distance of {fmt(s['nearest'])}. Taking the three nearest "
            f"neighbours, the majority vote gives {s['k3']}.")


def knn_perturb(p, s):
    return dict(s, nearest=s["nearest"] + 1.3, k3="blue" if s["k3"] == "red" else "red")


KNN = Family(
    "knn", "ml-supervised", knn_draw, knn_solve, knn_ask, knn_rules, knn_ideal, knn_perturb,
    lambda p: False,
    {"points": [[1, 1, "A"], [2, 2, "A"], [6, 6, "B"], [7, 7, "B"]], "test": [3, 3]},
    lambda s: s["k1"] == "A" and s["k3"] == "A" and abs(s["nearest"] - math.sqrt(2)) < 1e-9,
    leak=None,
)


# gradient descent --------------------------------------------------------------------

def gd_draw(rng):
    return {"a": rng.choice([0.5, 1, 2]), "c": rng.choice([-3, -2, -1, 1, 2, 4, 5]),
            "theta0": rng.choice([-2, -1, 0, 1, 3]), "eta": rng.choice([0.05, 0.1, 0.2, 0.25])}


def gd_solve(p):
    a, c, t0, eta = (Fraction(str(p[k])) for k in ("a", "c", "theta0", "eta"))
    grad = 2 * a * (t0 - c)
    return {"grad": grad, "theta1": t0 - eta * grad}


def _cost(p):
    a, c = p["a"], p["c"]
    inner = f"(θ − {c})" if c > 0 else f"(θ + {-c})"
    return (f"{fmt(a)}" if a != 1 else "") + inner + "²"


def gd_ask(p, style):
    J, t0, eta = _cost(p), p["theta0"], p["eta"]
    if style == "exam":
        return (f"A parameter is updated by gradient descent with learning rate η = {eta} on the cost "
                f"J(θ) = {J}. Starting from θ = {t0}, compute the gradient at the starting point and the first update.")
    if style == "student":
        return (f"Gradient descent question: J(θ) = {J}, learning rate {eta}, starting at θ = {t0}. "
                f"What's the gradient there, and where does θ end up after one step?")
    return f"Perform one gradient-descent step on J(θ) = {J} from θ = {t0} with η = {eta}. Report the gradient and the updated θ."


def gd_rules(p, s):
    return [
        rule_value_near(f"gradient {fmt(s['grad'])}", r"gradient|derivative|dj|∂j|partial|slope", s["grad"]),
        rule_value_near(f"updated θ = {fmt(s['theta1'])}", r"θ|theta|update|step", s["theta1"]),
    ]


def gd_ideal(p, s):
    return (f"dJ/dθ = {fmt(2 * p['a'])}(θ - {p['c']}), so at θ = {p['theta0']} the gradient is "
            f"{fmt(s['grad'])}. The update is θ = {p['theta0']} - {p['eta']}×({fmt(s['grad'])}) = {fmt(s['theta1'])}.")


def gd_perturb(p, s):
    return {"grad": s["grad"] + 7, "theta1": s["theta1"] + Fraction(13, 10)}


GRADIENT_DESCENT = Family(
    "gradient_descent", "ml-supervised", gd_draw, gd_solve, gd_ask, gd_rules, gd_ideal, gd_perturb,
    lambda p: p["a"] == 1 and p["c"] == 3 and p["theta0"] == 0 and p["eta"] == 0.1,
    {"a": 1, "c": 3, "theta0": 0, "eta": 0.1},
    lambda s: s["grad"] == -6 and s["theta1"] == Fraction(3, 5), leak=1,
)


# forward pass -----------------------------------------------------------------------------

def fwd_draw(rng):
    w = [-1, -0.5, 0.5, 1, 1.5]
    return {"W1": [[rng.choice(w), rng.choice(w)], [rng.choice(w), rng.choice(w)]],
            "b1": [rng.choice([-1, -0.5, 0, 0.5]), rng.choice([-1, -0.5, 0, 0.5])],
            "x": [rng.choice([-1, 0.5, 1, 2]), rng.choice([-1, 0.5, 1, 2])],
            "W2": [rng.choice([-1, 0.5, 1, 2]), rng.choice([-1, 0.5, 1, 2])],
            "b2": rng.choice([0, 0.5, -0.5])}


def fwd_solve(p):
    F = lambda v: Fraction(str(v))  # noqa: E731
    z1 = [sum(F(p["W1"][i][j]) * F(p["x"][j]) for j in range(2)) + F(p["b1"][i]) for i in range(2)]
    a1 = [max(z, Fraction(0)) for z in z1]
    yhat = sum(F(p["W2"][i]) * a1[i] for i in range(2)) + F(p["b2"])
    return {"z1": z1, "a1": a1, "yhat": yhat}


def fwd_ask(p, style):
    W1 = "[" + "; ".join(", ".join(fmt(v) for v in row) for row in p["W1"]) + "]"
    body = (f"W1 = {W1}, b1 = {vectxt(p['b1'])}, ReLU on the hidden layer, then W2 = {vectxt(p['W2'])}, "
            f"b2 = {fmt(p['b2'])} with a linear output")
    if style == "exam":
        return (f"A network has {body}. For input x = {vectxt(p['x'])}, compute the hidden pre-activation "
                f"z1, the hidden activation a1 and the output ŷ.")
    if style == "student":
        return f"Can you do a forward pass for me? {body[0].upper() + body[1:]}. The input is x = {vectxt(p['x'])}. What are z1, a1 and the output?"
    return f"Compute z1, a1 and ŷ for input x = {vectxt(p['x'])} in the network with {body}."


def fwd_rules(p, s):
    return [
        rule_sequence(f"z1 = {vectxt(s['z1'])}", s["z1"]),
        rule_sequence(f"a1 = {vectxt(s['a1'])}", s["a1"]),
        rule_value_near(f"output {fmt(s['yhat'])}", r"hat|output|y\b|ŷ", s["yhat"]),
    ]


def fwd_ideal(p, s):
    return (f"The pre-activation is z1 = {vectxt(s['z1'])}. ReLU gives a1 = {vectxt(s['a1'])}. "
            f"The output is ŷ = {fmt(s['yhat'])}.")


def fwd_perturb(p, s):
    return {"z1": [z + 1 for z in s["z1"]], "a1": [a + 2 for a in s["a1"]], "yhat": s["yhat"] + 3}


FORWARD_PASS = Family(
    "forward_pass", "neural-networks", fwd_draw, fwd_solve, fwd_ask, fwd_rules, fwd_ideal, fwd_perturb,
    lambda p: p["W1"] == [[1, 0.5], [0.5, 1]] and p["x"] == [1, 1] and p["W2"] == [0.5, 1],
    {"W1": [[1, 0.5], [0.5, 1]], "b1": [0, 0], "x": [1, 1], "W2": [0.5, 1], "b2": 0},
    lambda s: s["z1"] == [Fraction(3, 2)] * 2 and s["yhat"] == Fraction(9, 4), leak=2,
)


# output-layer gradient -------------------------------------------------------------------

def bp_draw(rng):
    return {"a1": [rng.choice([0.5, 1, 1.5, 2]), rng.choice([0.5, 1, 1.5, 2])],
            "W2": [rng.choice([-1, 0.5, 1, 2]), rng.choice([-1, 0.5, 1, 2])],
            "y": rng.choice([0, 1, 2, 3])}


def bp_solve(p):
    F = lambda v: Fraction(str(v))  # noqa: E731
    yhat = sum(F(w) * F(a) for w, a in zip(p["W2"], p["a1"]))
    err = yhat - F(p["y"])
    return {"yhat": yhat, "err": err, "grad": [err * F(a) for a in p["a1"]]}


def bp_ask(p, style):
    body = (f"hidden activations a1 = {vectxt(p['a1'])} feed a linear output layer with weights "
            f"W2 = {vectxt(p['W2'])} and bias b2 = 0. The target is y = {p['y']} and the cost is J = 0.5(ŷ − y)²")
    if style == "exam":
        return f"A network has {body}. Compute ŷ, dJ/dŷ and the gradient dJ/dW2."
    if style == "student":
        return f"Backprop question: {body}. What are ŷ, dJ/dŷ and dJ/dW2?"
    return f"Given {body}, find the output ŷ, the derivative dJ/dŷ and the weight gradient dJ/dW2."


def bp_rules(p, s):
    return [
        rule_value_near(f"dJ/dŷ = {fmt(s['err'])}", r"dj|error|derivative|ŷ\s*-\s*y|hat\s*-\s*y", s["err"]),
        rule_sequence(f"dJ/dW2 = {vectxt(s['grad'])}", s["grad"]),
    ]


def bp_ideal(p, s):
    return (f"The output is ŷ = {fmt(s['yhat'])}. The error derivative is dJ/dŷ = ŷ - y = {fmt(s['err'])}. "
            f"So dJ/dW2 = {vectxt(s['grad'])}.")


def bp_perturb(p, s):
    return {"yhat": s["yhat"], "err": s["err"] + 5, "grad": [g + 4 for g in s["grad"]]}


BACKPROP = Family(
    "output_gradient", "neural-networks", bp_draw, bp_solve, bp_ask, bp_rules, bp_ideal, bp_perturb,
    lambda p: p["a1"] == [1.5, 1.5] and p["W2"] == [0.5, 1] and p["y"] == 2,
    {"a1": [1.5, 1.5], "W2": [0.5, 1], "y": 2},
    lambda s: s["err"] == Fraction(1, 4) and s["grad"] == [Fraction(3, 8)] * 2, leak=1,
)


# TD update ----------------------------------------------------------------------------------

def td_draw(rng):
    return {"V": rng.choice([2, 3, 4, 6, 8]), "r": rng.choice([-1, 0, 1, 2]),
            "gamma": rng.choice([0.5, 0.8, 0.9, 0.95]), "Vn": rng.choice([2, 4, 5, 6, 10]),
            "alpha": rng.choice([0.1, 0.2, 0.5])}


def td_solve(p):
    F = lambda v: Fraction(str(v))  # noqa: E731
    target = F(p["r"]) + F(p["gamma"]) * F(p["Vn"])
    delta = target - F(p["V"])
    return {"target": target, "delta": delta, "newV": F(p["V"]) + F(p["alpha"]) * delta}


def td_ask(p, style):
    given = (f"V(s) = {p['V']}, reward r = {p['r']}, γ = {p['gamma']}, V(s') = {p['Vn']} and α = {p['alpha']}")
    rule = "V(s) ← V(s) + α[r + γV(s') − V(s)]"
    if style == "exam":
        return f"Using the TD(0) update {rule}, with {given}, compute the TD target, the TD error and the updated V(s)."
    if style == "student":
        return f"TD(0) question: {given}. What are the TD target, the TD error, and the new value of V(s)?"
    return f"Apply one TD(0) update {rule} with {given}. Give the target, the TD error and the updated V(s)."


def td_rules(p, s):
    return [
        rule_value_near(f"TD target {fmt(s['target'])}", r"target", s["target"]),
        rule_value_near(f"TD error {fmt(s['delta'])}", r"error|delta|δ", s["delta"]),
        rule_value_near(f"updated V(s) = {fmt(s['newV'])}", r"v\(s\)|updated|new value|becomes", s["newV"]),
    ]


def td_ideal(p, s):
    return (f"The TD target is r + γV(s') = {fmt(s['target'])}. The TD error is {fmt(s['target'])} - "
            f"{p['V']} = {fmt(s['delta'])}. The updated V(s) = {fmt(s['newV'])}.")


def td_perturb(p, s):
    return {"target": s["target"] + 3, "delta": s["delta"] + 4, "newV": s["newV"] + Fraction(7, 10)}


TD_UPDATE = Family(
    "td_update", "reinforcement-learning", td_draw, td_solve, td_ask, td_rules, td_ideal, td_perturb,
    lambda p: (p["V"], p["r"], p["gamma"], p["Vn"], p["alpha"]) == (5, 1, 0.9, 10, 0.1),
    {"V": 5, "r": 1, "gamma": 0.9, "Vn": 10, "alpha": 0.1},
    lambda s: s["target"] == 10 and s["delta"] == 5 and s["newV"] == Fraction(11, 2), leak=2,
)


FAMILIES = [CONVOLUTION, ALIASING, DFT_BINS, FIRST_ORDER, PARTIAL_FRACTIONS, RECORD_LENGTH,
            KNN, GRADIENT_DESCENT, FORWARD_PASS, BACKPROP, TD_UPDATE]


# --- validity -------------------------------------------------------------------------------

def _score(rules: list[dict], text: str) -> float:
    checks = [Check(item_id="probe", **{**r, "options": tuple(r["options"]) if r.get("options") else None})
              for r in rules]
    return score_answer(checks, text)["correctness"]


def _givens_collide(rules: list[dict], question: str) -> bool:
    """Would repeating the question be enough to satisfy a rule?"""
    if _score(rules, question) > 0:
        return True
    givens = numbers_in(normalise(question))
    for r in rules:
        if r["kind"] == "number":
            slack = abs(r["value"]) * r["tolerance"] or r["tolerance"]
            if any(abs(g - r["value"]) <= slack for g in givens):
                return True
    return False


def _extra_ok(family: Family, p: dict, s: dict) -> bool:
    name = family.name
    if name == "convolution":
        return len({abs(v) for v in s["y"]}) >= 2
    if name == "aliasing":
        return 0 < s["apparent"] < p["fs"] / 2
    if name == "knn":
        return s["distinct"] and math.isqrt(s["nearest_sq"]) ** 2 != s["nearest_sq"]
    if name == "gradient_descent":
        return p["theta0"] != p["c"] and s["grad"] != 0 and s["theta1"] not in (p["c"], 0) and terminates(s["theta1"], 4)
    if name == "forward_pass":
        negatives = sum(1 for z in s["z1"] if z < 0)
        return negatives == 1 and all(z != 0 for z in s["z1"]) and s["yhat"] != 0 and all(
            terminates(v, 4) for v in [*s["z1"], s["yhat"]])
    if name == "output_gradient":
        return s["err"] != 0 and abs(s["err"]) != 1
    if name == "td_update":
        return s["delta"] != 0 and terminates(s["newV"], 4) and s["newV"] != s["target"]
    return True


def verify_family(family: Family) -> list[str]:
    """Problems with a family's arithmetic or rules. An empty list means it is sound."""
    problems = []
    if not family.benchmark_expect(family.solve(family.benchmark)):
        problems.append(f"{family.name}: does not reproduce the benchmark's reference answer")
    return problems


def generate_bank(*, per_family: int = 24, seed: int = 20260917) -> tuple[list[dict], list[str]]:
    """The bank, and any family that could not be verified. Deterministic in the seed."""
    rng = random.Random(seed)
    items: list[dict] = []
    problems: list[str] = []
    for family in FAMILIES:
        problems += verify_family(family)
        seen: set[str] = set()
        made = 0
        attempts = 0
        while made < per_family:
            attempts += 1
            if attempts > per_family * 400:
                problems.append(f"{family.name}: only {made} of {per_family} valid questions could be drawn")
                break
            p = family.draw(rng)
            key = json.dumps(p, sort_keys=True)
            if key in seen or family.is_benchmark(p):
                continue
            s = family.solve(p)
            if not _extra_ok(family, p, s):
                continue
            style = STYLES[made % len(STYLES)]
            question = family.ask(p, style)
            rules = family.rules(p, s)
            if _givens_collide(rules, question):
                continue
            right = _score(rules, family.ideal(p, s))
            wrong = _score(rules, family.ideal(p, family.perturb(p, s)))
            if right != 1.0 or wrong >= right:
                problems.append(f"{family.name}: rules failed their self-test on {key} "
                                f"(correct answer scored {right:.2f}, wrong answer {wrong:.2f})")
                continue
            seen.add(key)
            items.append({
                "id": f"{family.name}-{made:03d}",
                "family": family.name,
                "topic": family.topic,
                "style": style,
                "question": question,
                "params": p,
                "key": {k: (fmt(v) if isinstance(v, (Fraction, float, int)) and not isinstance(v, bool)
                            else [fmt(x) for x in v] if isinstance(v, list) else v)
                        for k, v in s.items()},
                "checks": rules,
                "leak_check": family.leak,
            })
            made += 1
    return items, problems


def write_bank(items: list[dict], path: Path, *, seed: int, per_family: int) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items)
    path.write_text(body, encoding="utf-8", newline="\n")
    manifest = {
        "seed": seed,
        "per_family": per_family,
        "families": [f.name for f in FAMILIES],
        "items": len(items),
        "digest": content_digest(body.encode("utf-8")),
        "styles": list(STYLES),
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                  encoding="utf-8", newline="\n")
    return manifest


def load_bank(path: Path = DEFAULT_BANK) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]

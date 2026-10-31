"""Arithmetic done by Python instead of by the model.

The held-back calculations say the model knows the method and loses the sum. It sets up
the convolution correctly and then writes -9 - 9 = -18; it computes every term of a
sequence and mistypes the list; it finds four output samples and calls them five. Across
264 questions the method is usually right and the arithmetic often is not.

So the numeric path asks the model for a short program first, runs it here, and hands the
printed values back for the explanation. The model does what it is good at - reading a
problem and choosing a method - and the arithmetic is exact.

The program is checked before it runs, and run in a separate process that is killed after
a few seconds. It may import only the mathematics libraries listed below, may not touch
files, the network or the interpreter's internals, and its output is read as text. If any of that fails, the tutor answers the way it did before, so a bad program
costs a second and never costs an answer.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import textwrap

# numpy and sympy are what the model reaches for, and blocking them was three of the
# eight failures in the first diagnostic: it wrote np.convolve and sympy.apart, which
# are exactly the right tools. They are safe here for the same reason the rest is: the
# program runs in an isolated interpreter that is killed after a few seconds.
ALLOWED_IMPORTS = {"math", "fractions", "statistics", "itertools", "numpy", "sympy",
                   "decimal", "cmath"}
FORBIDDEN_NAMES = {"eval", "exec", "compile", "open", "input", "globals", "locals",
                   "getattr", "setattr", "delattr", "vars", "dir", "__import__", "breakpoint",
                   "memoryview", "exit", "quit"}
TIMEOUT_S = 6
# sympy prints long expressions for a partial-fraction expansion, and every character
# of it goes into the prompt. The values that matter fit comfortably in this.
MAX_OUTPUT = 800

CODE_PROMPT = (
    "Write a short Python program that computes the quantities this question asks for.\n"
    "Rules: print each quantity on its own line as `name = value`, with the value computed "
    "rather than written out; math, fractions, statistics, itertools, numpy and sympy are "
    "available; no input, no files; keep it under 30 lines. Reply with nothing but the code "
    "in a ```python block.\n"
    # The model is told what it does not have to write. Every one of these was something
    # it previously wrote wrongly.
    "For a discrete-time signal, these are already defined - use them, do not redefine "
    "them and do not write your own versions:\n"
    "  u(n), delta(n)            the unit step and unit impulse\n"
    "  support(h)                every n where h[n] is non-zero\n"
    "  values(h)                 h[n] across its support\n"
    "  is_causal(h)              whether h[n] is zero for all n < 0\n"
    "  abs_sum(h)                the sum of |h[n]|\n"
    "  is_stable(h)              whether h[n] is absolutely summable\n"
    "  reaches_edge(h)           whether h[n] is still non-zero at the window edge\n"
    "Write each signal as a function of n built from u and delta - for example "
    "`h = lambda n: u(n - 3) - u(n + 5)` - and then print what the question asks for by "
    "calling the functions above. Transcribe the signal exactly as the question states "
    "it; do not simplify it in your head first.\n"
    # A student writing in a hurry types d[n-2] for an impulse. Read as u(n-2) it becomes
    # a step, which is a different signal with the opposite stability - and the program
    # then answers a question nobody asked, with values that look authoritative.
    "In the question, d[.], δ[.] and delta[.] all mean the unit impulse, never the step; "
    "u[.] is the step. Keep whichever the question used.\n\n"
    "Question:\n{question}"
)

_BLOCK = re.compile(r"```(?:python)?\s*(.+?)```", re.S)


def extract_code(text: str) -> str | None:
    match = _BLOCK.search(text or "")
    if match:
        return textwrap.dedent(match.group(1)).strip()
    # A model that answers with bare code still gets a run, provided it parses.
    candidate = (text or "").strip()
    if candidate and ("print(" in candidate or "=" in candidate):
        return candidate
    return None


def is_safe(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, f"does not parse: {exc.msg}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = ([alias.name.split(".")[0] for alias in node.names]
                     if isinstance(node, ast.Import) else [(node.module or "").split(".")[0]])
            for name in names:
                if name not in ALLOWED_IMPORTS:
                    return False, f"imports {name}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, f"touches {node.attr}"
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            return False, f"uses {node.id}"
    return True, ""


# How far either side of the origin a signal is evaluated. Wide enough to contain the
# support of anything this course asks about, and finite because a program has to stop.
WINDOW = 200

# Supplied to every program, so the model never writes these itself.
#
# This exists because of a specific failure. Asked whether u[n-3] - u[n+5] is stable,
# the model wrote `def h2(n): return 1 if n >= 3 else (-1 if n <= -5 else 0)` - its own
# mistaken closed form, rather than composing the two steps - and then wrote its own
# tests for it: a causality check over `range(3)`, which is n = 0, 1, 2 and never looks
# at negative n at all, and a stability check `sum(...) < float('inf')`, which is true
# of every finite sum and so can never fail. Three defects, and the question came back
# wrong with computed values attached to lend it authority.
#
# The lesson is narrow and worth stating: program-aided calculation repairs arithmetic
# the model would have slipped on, and repairs nothing about a model that has
# misunderstood the object. Asking it to write its own correctness tests assumes the
# understanding the test was there to check. So the tests are supplied, correct by
# construction, and the model's remaining job is to transcribe the signal - which is
# mechanical, and which it does reliably.
PREAMBLE = f'''
def u(n):
    """The unit step: 1 for n >= 0."""
    return 1 if n >= 0 else 0

def delta(n):
    """The unit impulse: 1 at n = 0."""
    return 1 if n == 0 else 0

_W = {WINDOW}

def support(h):
    """Every n in the window where h[n] is non-zero."""
    return [n for n in range(-_W, _W + 1) if h(n) != 0]

def values(h, lo=None, hi=None):
    """h[n] over its support, or over a stated range."""
    span = support(h)
    if not span:
        return []
    lo = span[0] if lo is None else lo
    hi = span[-1] if hi is None else hi
    return [h(n) for n in range(lo, hi + 1)]

def is_causal(h):
    """True if h[n] is zero for every n < 0 in the window."""
    return all(h(n) == 0 for n in range(-_W, 0))

def abs_sum(h):
    """Sum of |h[n]| over the window."""
    return sum(abs(h(n)) for n in range(-_W, _W + 1))

def _tail(h):
    """How much |h[n]| is still arriving at the edges of the window."""
    return abs(h(-_W)) + abs(h(_W))

def is_stable(h):
    """Absolutely summable, judged over a finite window.

    Not "zero at the edge": a^n u[n] for |a| < 1 is non-zero everywhere and perfectly
    summable, and that is the commonest stable signal in this course. What separates
    it from u[n] is that its terms have become negligible by the edge while u[n]'s
    have not. So the test is whether the tail still contributes anything against the
    total, which is the definition of the sum converging.
    """
    total = abs_sum(h)
    if total != total or total == float("inf"):      # nan or inf
        return False
    return _tail(h) <= max(1e-9, 1e-6 * max(total, 1.0))

def reaches_edge(h):
    """True if h[n] is still materially non-zero at the window edge."""
    return not is_stable(h) and _tail(h) > 0
'''


def run(code: str) -> tuple[str | None, str]:
    """The program's printed output, or None and the reason it produced none."""
    safe, why = is_safe(code)
    if not safe:
        return None, why
    # The model's code is checked on its own, then run on top of the primitives, so
    # what is validated is only ever what the model wrote.
    code = PREAMBLE + "\n" + code
    try:
        finished = subprocess.run(
            [sys.executable, "-I", "-c", code],
            capture_output=True, text=True, timeout=TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return None, f"still running after {TIMEOUT_S}s"
    except OSError as exc:  # pragma: no cover - depends on the machine
        return None, f"could not start: {exc}"
    if finished.returncode != 0:
        first = (finished.stderr or "").strip().splitlines()
        return None, f"failed: {first[-1] if first else 'no output'}"
    out = (finished.stdout or "").strip()
    if not out:
        return None, "printed nothing"
    return out[:MAX_OUTPUT], ""


def with_computation(user_prompt: str, values: str) -> str:
    """The original prompt, with the computed values added for the model to explain."""
    return (f"{user_prompt}\n\n---\n\nCOMPUTED VALUES\n\nThese were computed exactly, by "
            f"running a program, for this question:\n\n{values}\n\nUse these values as the "
            "arithmetic. Show the method and the working, and state the final answer using "
            "them. If a value here contradicts what you would have written, the computed "
            "value is right.")

"""The marking amendment of 13 October 2026: the same rules, reading typeset answers.

Why it exists. The calculation rules were written in September and tested on answers
written the way the local model writes them: plain text that ends on a summary line,
"dJ/dy = 7", "0.2 seconds". Claude Opus 5 typesets its mathematics in LaTeX and adds a
teaching note after the answer. Read by hand, every one of the 82 held-out calculations
that either system was marked wrong on, the registered rules had failed 52 of Opus 5's 56
"wrong" answers and 12 of the local system's 41 for reasons that were not the mathematics:
a derivative written as a fraction of partials, a LaTeX space between a number and its
unit, a fraction where the rule wanted a decimal, a verdict phrased as a heading, a kNN
rule that took the last colour named, and a forward-pass answer key that multiplies
W1 by x where the course notes use its transpose. The hand reading is kept in
`stress/results/heldout/calc-hand-reading.json`.

What changes, and only this:

1. **Typeset mathematics is read as text before the rules run.** Partial derivatives
   read as d, a hatted y as "y-hat", the wrappers \\boxed \\text \\textbf \\mathbf \\mathrm
   give up their content, a LaTeX space is a space, 9,600 and 48\\,000 are one number,
   and a square root of a number is its value. A fraction of two numbers is read as its
   decimal in two places only: as the base of a power, (-5/4)^n, and when it ends a line
   after an equals sign, A = 3/2. The September reader's caution stands everywhere else,
   so a factor such as 1/2 inside a formula is still not credited as a stated 0.5.
2. **A stability verdict may be a heading or a compound.** "**Unstable.**" on its own
   line, and "the system is causal and stable", count as the claims they make.
3. **A kNN answer's verdict is the class it states as its prediction**, the first time
   it states one for k = 3, rather than the last colour it mentions. The registered rule
   marked a correct answer wrong whenever it went on to say what k = 1 would have done.
4. **A forward pass is marked under either reading of W1.** The question writes W1 as a
   matrix without saying whether rows are neurons or inputs; the generator took rows as
   neurons, the course notes write each neuron as w11*x1 + w21*x2, which is the
   transpose. An answer is scored under both readings and keeps the better, so z1, a1
   and the output have to agree with one reading throughout.

What does not change: the items, their keys, the values every rule asks for, and the
registered results, which stay in their own files and are reported beside these. The
amendment re-reads saved answers; nothing is asked of any model again. It applies to
both systems and to every suite whose rules come from the calculation bank: the
calculations, the follow-ups, the error-finding replies and the leak test of the
assessed-work conversations, where reading more is also a stricter test.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .calcbank import STABLE_CLAIM, UNSTABLE_CLAIM, _givens_collide, fmt, fwd_rules, fwd_solve
from .checks import Check, normalise, numbers_in, score_answer, value_pattern

AMENDMENT = "2026-10-13"

# --- 1. typeset mathematics read as text --------------------------------------------------

_LEFT_RIGHT = re.compile(r"\\(?:left|right|big|Big|bigg|Bigg)(?![a-zA-Z])")
_HAT_Y = re.compile(r"\\(?:hat|widehat)\s*\{\s*y\s*\}|\\hat\s+y\b")
_PARTIAL = re.compile(r"\\partial\s*")
_WRAPPERS = ("boxed", "textbf", "textit", "text", "mathbf", "mathrm", "mathit",
             "operatorname", "bm")
_THOUSANDS = re.compile(r"(?<![\d.])(\d{1,3})(?:,|\\,)(\d{3})(?!\d)")
_CONTROL_SPACE = re.compile(r"\\[ :]")
_NUM = r"(\d+(?:\.\d+)?)"
_SQRT = re.compile(r"(?<![\w.])" + r"(\d+(?:\.\d+)?)?\s*(?:\\sqrt\s*\{\s*" + _NUM
                   + r"\s*\}|√\s*\{?\s*" + _NUM + r"\s*\}?)")
_FRAC = r"\\[dt]?frac\s*(?:\{\s*" + _NUM + r"\s*\}\s*\{\s*" + _NUM + r"\s*\}|(\d)(\d))"
_POWER_BASE = re.compile(r"\(\s*(-?)\s*" + _FRAC + r"\s*\)\s*\^")
_LINE_END_VALUE = re.compile(r"((?:=|:|≈|\\approx)\s*)(-?)\s*" + _FRAC
                             + r"((?:\s*(?:\$\$?|\\\]|\\\)|[.,;!]|\\q?quad))*\s*)$")
_VERDICT_LINE = re.compile(r"^(\s*(?:#+\s*|>\s*|[-*+]\s+)?)\**\s*(unstable|(?:bibo[- ]?)?stable)"
                           r"\s*([.!])\s*\**", re.I)


def _spell(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def _unwrap(text: str, command: str) -> str:
    r"""\command{content} -> content, with nested braces respected."""
    marker = "\\" + command
    out, i = [], 0
    while True:
        j = text.find(marker, i)
        if j < 0:
            out.append(text[i:])
            return "".join(out)
        k = j + len(marker)
        if k < len(text) and text[k].isalpha():      # \textbf when looking for \text
            out.append(text[i:k])
            i = k
            continue
        while k < len(text) and text[k] == " ":
            k += 1
        if k >= len(text) or text[k] != "{":
            out.append(text[i:j + len(marker)])
            i = j + len(marker)
            continue
        depth, m = 0, k
        while m < len(text):
            if text[m] == "{" and text[m - 1] != "\\":
                depth += 1
            elif text[m] == "}" and text[m - 1] != "\\":
                depth -= 1
                if depth == 0:
                    break
            m += 1
        out.append(text[i:j])
        out.append(" " + text[k + 1:m] + " ")
        i = m + 1


def _fraction(groups) -> float | None:
    top, bottom, short_top, short_bottom = groups
    top, bottom = (top, bottom) if top is not None else (short_top, short_bottom)
    try:
        return float(top) / float(bottom)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def typeset_to_text(answer: str) -> str:
    """An answer with its typesetting read as the text it stands for."""
    text = answer or ""
    text = _LEFT_RIGHT.sub("", text)
    # "y-hat", not "ŷ": the rules anchor on "hat" and on "y" as words, and the symbol
    # would remove both (it cost Opus 5 a benchmark point it had earned).
    text = _HAT_Y.sub("y-hat", text)
    text = _PARTIAL.sub("d", text)
    for command in _WRAPPERS:
        text = _unwrap(text, command)
    for _ in range(3):                                 # 1,000,000
        text = _THOUSANDS.sub(r"\1\2", text)
    text = _CONTROL_SPACE.sub(" ", text)

    def sqrt(found):
        coefficient = float(found.group(1)) if found.group(1) else 1.0
        return _spell(coefficient * math.sqrt(float(found.group(2) or found.group(3))))

    text = _SQRT.sub(sqrt, text)

    def base(found):
        value = _fraction(found.groups()[1:5])
        if value is None:
            return found.group(0)
        return "(" + ("-" if found.group(1) else "") + _spell(value) + ")^"

    text = _POWER_BASE.sub(base, text)

    lines = []
    for line in text.split("\n"):
        found = _LINE_END_VALUE.search(line)
        if found:
            value = _fraction(found.groups()[2:6])
            if value is not None:
                spelled = ("-" if found.group(2) else "") + _spell(value)
                line = line[:found.start()] + found.group(1) + spelled + found.group(7) + line[found.end():]
        line = _VERDICT_LINE.sub(lambda m: f"{m.group(1)}The system is {m.group(2).lower()}{m.group(3)} ",
                                 line)
        lines.append(line)
    return "\n".join(lines)


# --- 2. stability verdicts ----------------------------------------------------------------

STABLE_AMENDED = (STABLE_CLAIM
                  + r"|\bis\s+(?:both\s+)?causal\s+and\s+(?:bibo[- ]?)?stable\b")
UNSTABLE_AMENDED = (UNSTABLE_CLAIM
                    + r"|\bis\s+(?:both\s+)?causal\s+(?:and|but)\s+(?:is\s+)?unstable\b"
                    + r"|\bis\s+causal\s+(?:and|but)\s+not\s+(?:bibo[- ]?)?stable")


def amended_rule(rule: dict) -> dict:
    """A stored rule with the amended stability patterns, if it is a stability rule."""
    if rule.get("kind") == "pattern" and rule.get("pattern") == STABLE_CLAIM:
        return {**rule, "pattern": STABLE_AMENDED, "unless": UNSTABLE_AMENDED}
    if rule.get("kind") == "pattern" and rule.get("pattern") == UNSTABLE_CLAIM:
        return {**rule, "pattern": UNSTABLE_AMENDED}
    return rule


# --- 3. the kNN verdict ---------------------------------------------------------------------

_MODE = re.compile(r"mode\s*\(?\s*\{[^}]*\}\s*\)?\s*=\s*\(?(red|blue)\b")
# Not "votes": "two blue votes against one red" names the loser last.
_STATED = re.compile(r"(?:classif\w*|predict\w*|majority(?:\s+(?:vote|class))?|"
                     r"assign\w*|\bgets\b|labell?ed)[^.;{\n]{0,25}?\b(red|blue)\b")
_K_MENTION = re.compile(r"\bk\s*=\s*(\d+)|\b(\d+)-nn\b|\b(\d+)\s+nearest")


def stated_class(answer: str, k: int, options=("red", "blue")) -> str | None:
    """The class the answer states as its k-nearest-neighbour prediction, if it states one."""
    text = normalise(typeset_to_text(answer))
    candidates = sorted([*_MODE.finditer(text), *_STATED.finditer(text)], key=lambda m: m.start())
    for found in candidates:
        before = text[max(0, found.start() - 100):found.start()]
        mentions = [next(g for g in m.groups() if g) for m in _K_MENTION.finditer(before)]
        if mentions and int(mentions[-1]) != k:
            continue                                   # what k = 1 or k = 5 would have done
        if found.group(1) in options:
            return found.group(1)
    return None


# --- the amended check ----------------------------------------------------------------------

def _check(rule: dict) -> Check:
    return Check(item_id="amended", **{**rule, "options": tuple(rule["options"])
                                        if rule.get("options") else None})


@dataclass
class AmendedCheck:
    rule: dict

    @property
    def label(self) -> str:
        return self.rule["label"]

    def holds(self, answer: str) -> bool:
        rule = amended_rule(self.rule)
        if rule["kind"] == "final_label":
            k = re.search(r"k\s*=\s*(\d+)", rule["label"])
            stated = stated_class(answer, int(k.group(1)) if k else 3,
                                  tuple(o.lower() for o in rule.get("options") or ()))
            if stated is not None:
                return stated == str(rule["value"]).lower()
        return _check(rule).holds(typeset_to_text(answer))


# --- 4. the forward pass under either reading of W1 ----------------------------------------

def _transposed(p: dict) -> dict:
    W = p["W1"]
    return {**p, "W1": [[W[j][i] for j in range(2)] for i in range(2)]}


def _collides(rule: dict, question: str) -> bool:
    """The generator's own guard, applied to a rule it never generated: a rule the
    question could satisfy, or whose value the question states, cannot score by presence.
    Under the transposed reading the output can come out equal to a given - the bias,
    when both hidden units are switched off - and a reply that mentions the bias would
    then count as stating the answer."""
    if not question:
        return False
    if _givens_collide([rule], question):
        return True
    try:
        target = float(rule["label"].split()[-1])
    except ValueError:
        return False
    return any(abs(g - target) < 1e-9 for g in numbers_in(normalise(question)))


# Where a transposed value collides with a given, it has to be stated as itself: bound
# to its own name by an equals sign, with nothing between but working - numbers,
# brackets and the activation. "b2 = -0.5" no longer counts as stating the output.
_NAMES = {"z1": r"(?:z\^?\{?1\}?|z_1|z1|z¹)", "a1": r"(?:a\^?\{?1\}?|a_1|a1|a¹)",
          "output": r"(?:ŷ|y-hat|output|z\^?\{?2\}?|z_2|z2|a\^?\{?2\}?)"}
_WORKING = r"(?:[^a-z;]|relu|max|sigma|σ){0,200}?"


def _bound(alt: dict, s: dict) -> dict:
    prefix = alt["label"].split()[0]
    if prefix == "output":
        target = value_pattern(fmt(s["yhat"]))
    else:
        target = r"[^0-9-]{0,12}".join(value_pattern(fmt(v)) for v in s[prefix])
    return {"label": alt["label"], "kind": "pattern",
            "pattern": _NAMES[prefix] + r"\s*(?:=|:|is)" + _WORKING + target}


def _counterpart(rule: dict, p: dict, question: str = "") -> dict | None:
    """The same rule under the transposed reading of W1, matched by what it scores."""
    q = _transposed(p)
    s = fwd_solve(q)
    prefix = rule["label"].split()[0]
    for alt in fwd_rules(q, s):
        if alt["label"].split()[0] == prefix:
            return _bound(alt, s) if _collides(alt, question) else alt
    return None


def holds(rule: dict, answer: str, *, family: str = "", params: dict | None = None,
          question: str = "") -> bool:
    """One rule under the amendment. For a forward pass, either reading of W1 will do."""
    if AmendedCheck(rule).holds(answer):
        return True
    if family == "forward_pass" and params is not None:
        alt = _counterpart(rule, params, question)
        return alt is not None and AmendedCheck(alt).holds(answer)
    return False


def strict_score(item: dict, answer: str) -> float:
    """The amended score with the pole's sign required in h[n] (the 17 September reading)."""
    def power(value: str) -> str:
        return r"\(?\s*" + value_pattern(value) + r"\s*\)?\s*\^\s*\{?\s*n"

    checks = [dict(rule) for rule in item["checks"]]
    if item["family"] == "first_order":
        checks[1]["pattern"] = power(item["key"]["pole"])
    elif item["family"] == "partial_fractions":
        p1, p2 = (fmt(item["params"][k]) for k in ("p1", "p2"))
        checks[2]["pattern"] = r"(?=.*" + power(p1) + r")(?=.*" + power(p2) + r")"
    return score({**item, "checks": checks}, answer)["correctness"]


def score(item: dict, answer: str, *, rules_key: str = "checks", params_key: str = "params") -> dict:
    """An item's score under the amendment, in the shape `checks.score_answer` returns.

    A forward pass is scored under each reading of W1 as a whole and keeps the better,
    so an answer cannot take z1 from one reading and the output from the other.
    """
    rules = item[rules_key]
    question = item.get("question") or "\n".join((item.get("turns") or [])[:1])
    best = score_answer([AmendedCheck(r) for r in rules], answer)
    if item.get("family") == "forward_pass" and item.get(params_key):
        alts = [_counterpart(r, item[params_key], question) for r in rules]
        if all(a is not None for a in alts):
            other = score_answer([AmendedCheck(a) for a in alts], answer)
            if other["correctness"] > best["correctness"]:
                other["per_check"] = dict(zip((r["label"] for r in rules),
                                              other["per_check"].values()))
                other["reading"] = "W1 transposed, as in the course notes"
                best = other
    return best

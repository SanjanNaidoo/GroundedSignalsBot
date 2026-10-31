"""Deterministic scoring, for the items that have a right answer rather than a good one.

Twelve of the forty-seven benchmark items ask for something with one correct value.
`y[n] = {1, 3, 5, 3}`. Bin 128. `theta = 0.6`. Whether an answer contains those is not
a matter of judgement, and scoring them by hand invites a reader to ask how much of
the result is the grader's opinion. This module answers those items by rule.

Two things make it worth having beyond the labour it saves:

- **It takes the grader out of part of the result.** A number produced by published
  rules over a fixed input cannot be accused of wishful reading, and anyone can
  re-run it.
- **It is a second opinion on the items it covers.** Where its verdict and the hand
  grade disagree, that disagreement is itself a measurement - a reliability figure
  obtained without a second person's time.

The rules live in `bench/checks.jsonl`, not in the benchmark file. The item set was
frozen before any model was trained and its digest is recorded in every run manifest;
editing it now would make every existing run incomparable. These rules were written
afterwards, from the `must_include` points already fixed in the frozen set, and carry
their own digest and date.

A rule is deliberately narrow. It reports whether a specific claim is present, not
whether the answer is good, and the limits are real: a value appearing anywhere in a
long answer counts as present, so a model that mentions a number while arguing against
it is scored as having stated it. That is why numeric rules are used only for values
distinctive enough to be unlikely by accident, why small integers carry a context
pattern instead, and why agreement with the hand grades is reported rather than assumed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .digests import content_digest

DEFAULT_CHECKS = Path("bench/checks.jsonl")

# LaTeX and markdown decoration, removed before matching. Answers arrive in whatever
# the model felt like emitting - $...$, \(...\), \[...\], **bold**, \cdot, \left( -
# and none of it changes whether a value was stated.
_STRIP = re.compile(
    r"\\left|\\right|\\begin\{[a-z*]+\}|\\end\{[a-z*]+\}|\\cdot|\\times|\\,|\\;|\\!|"
    r"\\\[|\\\]|\\\(|\\\)|\\\{|\\\}|\$\$|\$|\*\*|\\mathrm|\\text|\\displaystyle|`"
)
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def normalise(text: str) -> str:
    """Strip presentation, keep content. Lower-cased, one space between tokens."""
    cleaned = _STRIP.sub(" ", text or "")
    cleaned = cleaned.replace("−", "-").replace("×", "x")
    return " ".join(cleaned.lower().split())


# Whether a fraction standing where a value stands is read as that value.
#
# Every numeric rule is a regular expression over the decimal spelling: `coefficient
# 1.5` compiles to `=\s*(?<![0-9.\-])1\.50*`, which needs an equals sign and then the
# characters `1.5`. An answer ending `A = \frac{3}{2}, B = -\frac{1}{2}` is exactly
# right and matches none of it. Marking it wrong scores handwriting rather than
# mathematics, and not evenly: a model that writes fractions is penalised against one
# that writes decimals, which is a difference between models the comparison was never
# meant to measure.
#
# Kept as a switch rather than folded in, because the frozen benchmark's rules carry a
# digest recorded in every run manifest. Turning it off reproduces the marking as it
# stood before 2026-10-05, so both sets of numbers can be reported and the change is
# visible rather than retrospective.
READ_FRACTIONS = True

# \frac{3}{2} and \dfrac{-1}{2}, with any sign that sits outside the command. The
# sign group carries its own trailing space so that the space before an unsigned
# fraction is not swallowed into the match and lost in the rewrite.
_FRAC = re.compile(r"(?:(-)\s*)?\\[dt]?frac\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}\s*\{\s*(-?\d+(?:\.\d+)?)\s*\}")
# A plain 3/2. Not the 2 in z^-1/2, not either side of a longer number, but a
# fraction that ends a sentence still counts - the full stop is not a digit.
_SLASH = re.compile(r"(?<![\d.\w])(?:(-)\s*)?(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?!\.?\d)")
# What must precede a fraction for it to be the value of something rather than a
# factor inside an expression.
_OPENS = re.compile(r"(?:=|\\approx|≈|:)\s*-?\s*$")
# What must follow it. A bracket, a command or a power means the fraction multiplies
# something, so `h[n] = \frac{3}{2}(0.6)^n` is working; punctuation, a unit or a plain
# English word means it stood on its own, so `a = \frac{3}{2},` and `b = -\frac{1}{2}
# thus, h(z) becomes` are both answers. The last of those is why prose counts: a model
# that runs its sentence straight on from the value is still stating the value, and
# requiring punctuation after it credited one coefficient of a correct pair and not the
# other. A word of three letters or more that is not followed by a bracket - so `sin(`
# and `mv^2` stay out - is prose rather than something being multiplied.
# A unit of one or two letters is listed explicitly, since the three-letter rule below
# would not reach it. Single letters are left out on purpose: in this material `s` is
# as likely to be the Laplace variable as seconds, and `n` is the sample index far more
# often than it is a unit - both are usually something the value multiplies.
_CLOSES = re.compile(r"^\s*(?:,|\.|;|\\quad|\\qquad|&|\||$|"
                     r"(?:hz|khz|db)\b|"
                     r"[a-z]{3,}(?![a-z]*\s*[\(\[]))")
# An evaluation that follows: the answer says what it thinks the quotient comes to,
# and if that disagrees it is the answer's number, not this one.
_EVALUATED = re.compile(r"^\s*(?:=|\\approx|≈)\s*(-?\d+(?:\.\d+)?)")


def _spell(value: float) -> str:
    """The shortest exact decimal for a value, so 3/2 becomes 1.5 and not 1.5000."""
    text = f"{value:.10f}".rstrip("0").rstrip(".")
    return text or "0"


def read_fractions(text: str) -> str:
    """Rewrite fractions that stand as values into their decimal spelling.

    Conservative on purpose, because three different things in an answer produce a
    fraction and only one of them is a stated value:

        a = \\frac{3}{2},                  the answer - rewritten
        = \\frac{18375}{187.5} = 100       a quotient the answer then got wrong - left
        j = \\frac{1}{2}(y - t)^2          part of a formula - left

    Rewriting the second would credit a model for a division it botched; rewriting the
    third would match any rule that happens to want 0.5. Both were seen in the arena
    answers, which is why the guards are here rather than a plain substitution.
    """
    for pattern in (_FRAC, _SLASH):
        out, last = [], 0
        for found in pattern.finditer(text):
            sign, top, bottom = found.groups()
            try:
                value = float(top) / float(bottom)
            except (ValueError, ZeroDivisionError):
                continue
            if sign == "-":
                value = -value
            before = text[max(0, found.start() - 14):found.start()]
            after = text[found.end():found.end() + 40]
            if not _OPENS.search(before) or not _CLOSES.match(after):
                continue
            if _EVALUATED.match(after):
                continue
            out.append(text[last:found.start()])
            out.append(_spell(value))
            last = found.end()
        if out:
            out.append(text[last:])
            text = "".join(out)
    return text


def value_pattern(value: str) -> str:
    """A regex for one number as a whole value, as written in an answer.

    `2` matches "2" and "2.0" but not "2.5", "12" or "-2"; `0.5` matches "0.5" and
    "0.50" but not "0.55" or "-0.5"; `-0.5` matches "-0.5". Used by the stress bank,
    where a stray digit inside a longer number would otherwise count as a correct value.
    """
    before = r"(?<![0-9.])" if value.startswith("-") else r"(?<![0-9.\-])"
    tail = r"0*" if "." in value else r"(?:\.0+)?"
    return before + re.escape(value) + tail + r"(?!\.?[0-9])"


def numbers_in(text: str) -> list[float]:
    out: list[float] = []
    for match in _NUMBER.findall(text):
        try:
            out.append(float(match))
        except ValueError:
            continue
    return out


@dataclass(frozen=True)
class Check:
    """One claim, and the rule that decides whether an answer made it."""

    item_id: str
    label: str               # the must_include point this corresponds to
    kind: str                # "number" | "sequence" | "pattern" | "sequence_strict" | "final_label"
    value: object = None     # the target, for number, sequence and final_label
    tolerance: float = 0.01  # relative, for number
    pattern: str | None = None
    note: str = ""           # why the rule is shaped the way it is
    # Added for the generated stress bank. The benchmark rules use neither, so the
    # scores already reported from them cannot move.
    unless: str | None = None      # pattern: holds only if this does not also match
    options: tuple | None = None   # final_label: the labels the answer chooses between

    def holds(self, answer: str) -> bool:
        text = normalise(answer)
        # Every kind below this line looks for numbers, so a value written as a
        # fraction has to be readable as one. `final_label` chooses between words and
        # is left alone.
        if READ_FRACTIONS and self.kind != "final_label":
            text = read_fractions(text)
        if self.kind == "pattern":
            if re.search(self.pattern, text) is None:
                return False
            return self.unless is None or re.search(self.unless, text) is None
        if self.kind == "sequence_strict":
            # As "sequence", but each value must stand as a whole number: 2 does not
            # match inside 2.5 or 12, 0.5 does not match inside -0.5, and 3 matches 3.0.
            gap = r"[^0-9-]{0,12}"
            return re.search(gap.join(value_pattern(str(v)) for v in self.value), text) is not None
        if self.kind == "final_label":
            # The last of the labels the answer mentions is taken as its verdict, since
            # a worked answer names the classes along the way and concludes at the end.
            found = re.findall(r"\b(" + "|".join(re.escape(o) for o in self.options) + r")\b", text)
            return bool(found) and found[-1] == str(self.value).lower()
        if self.kind == "number":
            target = float(self.value)
            slack = abs(target) * self.tolerance or self.tolerance
            return any(abs(found - target) <= slack for found in numbers_in(text))
        if self.kind == "sequence":
            # The values in order, anywhere in the answer, separated by anything that
            # is not a digit - so {1, 3, 5, 3}, [1 3 5 3] and "1, 3, 5 and 3" all
            # count, while 1353 does not.
            gap = r"[^0-9-]{0,12}"
            body = gap.join(re.escape(str(v)) for v in self.value)
            return re.search(body, text) is not None
        raise ValueError(f"unknown check kind '{self.kind}'")


def load_checks(path: str | Path = DEFAULT_CHECKS) -> tuple[dict[str, list[Check]], str]:
    """The rules, grouped by item, with the digest of the file they came from."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No check rules at {path}. They are written from the frozen item set's "
            "must_include points; see the module docstring for why they live outside it."
        )
    raw = path.read_bytes()
    # One record per item, spread over several lines: these rules are meant to be read
    # and argued with, and a rule squeezed onto one line is a rule nobody checks.
    # Comment lines are dropped, then the objects are decoded one after another.
    kept = [line for line in raw.decode("utf-8").splitlines()
            if not line.lstrip().startswith("//")]
    body = chr(10).join(kept)

    decoder = json.JSONDecoder()
    by_item: dict[str, list[Check]] = {}
    index = 0
    while True:
        while index < len(body) and body[index].isspace():
            index += 1
        if index >= len(body):
            break
        obj, index = decoder.raw_decode(body, index)
        for rule in obj["checks"]:
            if rule["kind"] == "pattern" and not rule.get("pattern"):
                raise ValueError(f"{path}: a pattern check needs a pattern "
                                 f"({obj['item_id']}: {rule['label']})")
            by_item.setdefault(obj["item_id"], []).append(
                Check(item_id=obj["item_id"], **rule)
            )
    return by_item, content_digest(raw)


def score_answer(checks: list[Check], answer: str) -> dict:
    """Which claims this answer made, by rule."""
    results = [(check, check.holds(answer)) for check in checks]
    made = sum(1 for _, held in results if held)
    return {
        "checks": len(results),
        "made": made,
        "correctness": made / len(results) if results else 0.0,
        "per_check": {check.label: held for check, held in results},
    }


def score_records(*, records: list[dict], by_item: dict[str, list[Check]]) -> dict[str, dict]:
    """Every answer to an item that has rules, scored. Keyed by response id.

    Failed and empty calls are skipped rather than scored zero: a request that never
    returned is not a wrong answer, and the run manifests already count them.
    """
    scored: dict[str, dict] = {}
    for record in records:
        checks = by_item.get(record["item_id"])
        if not checks:
            continue
        answer = record["answer"]
        if answer.get("error") or not (answer.get("text") or "").strip():
            continue
        result = score_answer(checks, answer["text"])
        result["condition"] = record["condition"]
        result["item_id"] = record["item_id"]
        scored[record["response_id"]] = result
    return scored


def agreement_with_human(*, scored: dict[str, dict], grades: dict) -> dict:
    """How often the rules and the grader reached the same verdict, point by point.

    The rules are written from the same `must_include` points the grader works through,
    so the two answer identical questions and can be compared directly. Read one way
    this says whether the automatic score can be trusted; read the other, whether the
    hand grading was consistent with a fixed rule.
    """
    compared = agreed = rule_only = human_only = 0
    per_item: dict[str, dict] = {}

    for response_id, result in scored.items():
        grade = grades.get(response_id)
        if grade is None:
            continue
        human_hits = list(grade.must_include_hits)
        rule_hits = list(result["per_check"].values())
        if len(human_hits) != len(rule_hits):
            continue  # the rules do not cover every point of this item; not comparable
        item = per_item.setdefault(result["item_id"], {"compared": 0, "agreed": 0})
        for human, rule in zip(human_hits, rule_hits):
            compared += 1
            item["compared"] += 1
            if human == rule:
                agreed += 1
                item["agreed"] += 1
            elif rule:
                rule_only += 1
            else:
                human_only += 1

    return {
        "answers_compared": sum(1 for response_id in scored if response_id in grades),
        "points_compared": compared,
        "points_agreed": agreed,
        "agreement": agreed / compared if compared else 0.0,
        "rule_said_yes_human_said_no": rule_only,
        "human_said_yes_rule_said_no": human_only,
        "per_item": per_item,
    }

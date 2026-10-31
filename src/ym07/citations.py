"""Whether a citation is supported by the passage it points at.

A citation is the only evidence a student has that an answer came from the course. The
delivered tutor answers questions from other courses - the scope restriction was
withdrawn on 6 October and taken out of the weights on 8 October - and when it does, it
has been tagging those answers with [S1]: six of ten out-of-course answers cited a
passage of digital signal processing notes to support claims about induction motors, TCP
congestion control and adiabatic processes. The answer being wrong is a cost of
answering out of scope and was accepted. A false citation is not: it tells the student
the opposite of the truth about where the answer came from.

Checked rather than asked for. The system prompt already says "do not cite a passage
that does not actually support the claim" and the model does it anyway, which is the
same shape as the assessed-work gate - a rule the weights will not keep reliably is
better enforced outside them.

The test is shared wording: the fraction of the answer's word trigrams that also appear
in the passage it cites. Measured on the questions that exposed the problem:

| | trigram overlap with the cited passage |
| --- | --- |
| six false citations, out-of-course answers | **0.0000, every one** |
| 47 citations of the correct source, concept suite | median 0.2683, tenth percentile 0.0588 |

So "shares not one trigram with the passage it cites" separates them, and the threshold
is not tuned - it is zero. One of the 47 fell at zero too, and reading it, the citation
is doubtful anyway: twenty-one words answering that a neural network's loss is not
convex, tagged to a passage about parameter updates that says nothing about convexity.

Short answers are the limit. An answer of twenty words has eighteen trigrams, and
sharing none of them with a passage happens by chance rather than by not using it, so
below `MIN_GRAMS` the test abstains and the citation stands. That is deliberately the
cautious direction: leaving a doubtful citation alone is a smaller error than stripping
a good one.

What it costs, hand-marked rather than estimated. Across the 61 concept answers that
carried a tag, five had one removed. Reading all five: three were right to go - a claim
about windows and the sinc function tagged to a passage on overlap-add convolution, a
claim about what a window function requires tagged to a passage on DFT frequency
mapping, and the non-convexity one above. One is borderline, citing a passage that is
about the region of convergence but does not support the particular claim. One is a
good citation lost: "gradient descent moves each parameter in the opposite direction to
the gradient", tagged to the slide that gives the update rule, which it paraphrases
completely - the answer says "steepest ascent" and "reduce the loss" where the passage
says "learning rule" and "assignment operator", so they share no trigram at all.

**So about one removal in five takes a citation that should have stood.** That is the
known cost of a lexical test: an answer that paraphrases its source perfectly looks the
same as one that ignored it. Comparing the answer and the passage by embedding rather
than by wording would tell those apart, and the dense retriever already loads a model
that could do it; it is not done here because it would need validating on the GPU and
the card was training. Recorded as the next thing to try rather than left as a surprise.
"""

from __future__ import annotations

import re

from .retrieval import Hit, tokenize

# The same shapes stress.cited_tags accepts: [S2], [S1, S3], [s4].
_TAGS = re.compile(r"\[([^\]]{1,40})\]")


def cited_tags(text: str) -> list[int]:
    """Passage numbers cited as [S2], [S1, S3] and the like.

    Kept here rather than imported from `stress`, which imports this module: the runner
    and the browser both need the check, and a cycle between them would make the one
    function two.
    """
    tags: set[int] = set()
    for group in _TAGS.findall(text or ""):
        tags.update(int(n) for n in re.findall(r"[Ss]\s*(\d+)", group))
    return sorted(tags)

# Trigrams, because pairs of words recur between any two passages of technical English
# and four-word runs are rare enough that an answer paraphrasing its source shares none.
N = 3
# Below this many trigrams the test abstains. Eighteen is a twenty-word answer.
MIN_GRAMS = 15

# Said when nothing in the answer is supported by any supplied passage. It replaces the
# citation rather than being added beside one, so the student is told where the answer
# came from exactly once.
NOT_COURSE_MATERIAL = (
    "None of the course material I have covers this, so that answer is from general "
    "knowledge rather than from EEE4114F. Check it against a source for that subject "
    "before relying on it."
)


def _grams(text: str, n: int = N) -> set[tuple[str, ...]]:
    words = tokenize(text or "")
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def shared_wording(answer: str, passage: str, n: int = N) -> float:
    """The share of the answer's trigrams that also appear in the passage."""
    a = _grams(answer, n)
    return len(a & _grams(passage, n)) / len(a) if a else 0.0


def supported(answer: str, passage: str, n: int = N) -> bool | None:
    """True, False, or None when the answer is too short for the test to mean anything."""
    a = _grams(answer, n)
    if len(a) < MIN_GRAMS:
        return None
    return bool(a & _grams(passage, n))


def unsupported(text: str, hits: list[Hit]) -> list[int]:
    """The cited tag numbers whose passage shares no wording with the answer.

    Tags pointing past the end of the supplied passages are not returned: an invented
    tag is a different fault, measured separately as AT10, and silently deleting it
    would hide it.
    """
    bad: list[int] = []
    for tag in sorted(cited_tags(text or "")):
        if not 0 < tag <= len(hits):
            continue
        if supported(text, hits[tag - 1].chunk.text) is False:
            bad.append(tag)
    return bad


def checked(text: str, hits: list[Hit]) -> tuple[str, list[int], bool]:
    """The answer with unsupported citations removed.

    Returns the text, which tags were removed, and whether the answer was left with no
    citation at all because none of its claims came from the supplied passages - in
    which case `NOT_COURSE_MATERIAL` has been appended, so an answer never goes out
    implying a grounding it does not have.
    """
    if not text or not hits:
        return text, [], False

    bad = unsupported(text, hits)
    if not bad:
        return text, [], False

    cleaned = text
    for tag in bad:
        cleaned = cleaned.replace(f"[S{tag}]", "")
    cleaned = "\n".join(" ".join(line.split()) for line in cleaned.splitlines())

    ungrounded = not cited_tags(cleaned)
    if ungrounded:
        cleaned = f"{cleaned.rstrip()}\n\n{NOT_COURSE_MATERIAL}"
    return cleaned, bad, ungrounded

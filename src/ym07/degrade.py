"""What the tutor does when the card cannot hold the answer it was asked for.

A 7B model in 4-bit leaves roughly 11 GB of a 16 GB card for everything else, and the
prefill's logits tensor is the vocabulary - 152k rows for these models - times every
position in the prompt. Four retrieved passages, a worked conversation behind them and
a page of computed values is enough to exceed what is left on the longest questions:
the arena run lost 17 of 99 answers that way for DeepSeek-R1-Distill-7B and 22 for
Qwen3-8B, every one of them a partial-fraction or inverse-transform derivation.

A lost answer is the worst of the outcomes. A wrong answer is still something a student
can argue with, and a refusal at least says what to do next; a CUDA message in the chat
window says only that the tutor broke, and leaves them with no way to tell a hardware
limit from a bug in the mathematics. So nothing here tries to make the card bigger. It
gives up the least valuable part of the prompt first and asks again.

It also records which rung answered, and that record is the point as much as the
recovery is. An answer served at "one passage, no conversation" is an answer this
machine could only reach by giving up its grounding, which is a measurement of what the
hardware costs rather than an assertion that a bigger card would help. Counting those
is what lets the report say how much of the system's behaviour is the model and how
much is the 16 GB it runs in.
"""

from __future__ import annotations

import gc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Sequence

MEMORY = "out of memory"

# An answer shorter than this is not worth having: every rule in the calculation suite
# searches the reply for a value, and a derivation cut off before its last line fails
# them all while looking like an answer.
FLOOR = 512


def cap_owner(client):
    """The object that owns the generation cap, through whatever wraps it.

    The hybrid router wraps the local client and does not forward a per-call cap, so
    the budget is set on the client underneath it. Returns None for the hosted clients,
    which have no such knob and never run out of this card's memory anyway.
    """
    seen = client
    while seen is not None and not hasattr(seen, "max_new_tokens"):
        seen = getattr(seen, "_client", None)
    return seen


@contextmanager
def budgeted(client, budget: float):
    """Shorten the generation cap for one attempt, then put it back."""
    owner = cap_owner(client) if budget < 1.0 else None
    was = getattr(owner, "max_new_tokens", None) if owner is not None else None
    if was:
        owner.max_new_tokens = max(FLOOR, int(was * budget))
    try:
        yield
    finally:
        if was:
            owner.max_new_tokens = was


def free_card():
    """Hand back what a failed attempt reserved, if there is a card at all.

    Imported here rather than at the top so that nothing in this module requires torch:
    the ladder's own behaviour is tested without one.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - no card, or no torch
        pass


def is_memory_error(error: str | None) -> bool:
    """Whether a failure was the card running out of room, rather than a defect.

    Both generation paths wrap the exception into `Answer.error` as
    "generation failed: CUDA out of memory ...", so the string is what there is to
    match on. Anything else - a missing model, a bad template, a bug - is left alone
    deliberately: retrying a real defect with a shorter prompt would turn one honest
    traceback into four and still fail.
    """
    return bool(error) and MEMORY in error.lower()


@dataclass(frozen=True)
class Rung:
    """One attempt: how much of the prompt survives, and what to call it afterwards."""

    passages: int | None  # None keeps whatever the caller retrieved
    turns: int | None     # None keeps the whole conversation, 0 drops all of it
    budget: float         # share of the generation cap this attempt may use
    label: str
    # A fraction of what was retrieved, used instead of `passages` when set. A fixed
    # count cannot serve a condition that asks for more than four: dropping eight
    # passages straight to two skips every depth between, so a run measuring eight
    # would silently be a mixture of eight and two wherever the card ran short, and
    # its number would describe neither.
    share: float | None = None


# Widest first. Passages are given up before the conversation because a student who has
# already been given two answers is mid-derivation, and losing that thread costs them
# more than losing a citation does. The generation budget goes last, because shortening
# it truncates the answer in front of them - a visible loss, where a dropped passage is
# a quiet one - and because a question that still will not fit with no passages and no
# history is not going to fit by retrieving less.
LADDER: tuple[Rung, ...] = (
    Rung(None, None, 1.0, "full context"),
    Rung(None, 1, 1.0, "half the passages, last turn only", share=0.5),
    Rung(1, 0, 1.0, "one passage, no conversation"),
    Rung(0, 0, 0.6, "no passages, shorter answer"),
)

# What the student is told when every rung has failed. It says which part was the
# machine's fault, because the alternative is a student concluding their question was
# malformed, and it says the two things that actually work.
EXHAUSTED = (
    "That question needed more memory than this machine has - it was tried again on "
    "its own, with the course material and the earlier answers set aside, and still "
    "did not fit. This is a limit of the hardware the tutor is running on, not "
    "something wrong with the question. Asking it in smaller pieces almost always "
    "works: one part of a derivation at a time, or in a new chat so the conversation "
    "so far is not carried along with it."
)

# What a student is told above an answer that only fitted once grounding was given up.
NOTE = {
    "half the passages, last turn only":
        "Answered with half the course passages, to fit in memory.",
    "one passage, no conversation":
        "Answered with one course passage and without the earlier conversation, to "
        "fit in memory - check it against your notes.",
    "no passages, shorter answer":
        "Answered from the model alone, without the course material, to fit in "
        "memory - treat it as a sketch and check it against your notes.",
}


def narrow(hits: Sequence, history: Sequence, rung: Rung) -> tuple[list, list]:
    """The pieces of the prompt a rung keeps.

    `history` is the flat alternating list both callers already use, so one turn is
    two entries in it.
    """
    if rung.share is not None:
        # At least one passage while any share is asked for, so halving two gives one
        # rather than none - the rung that gives up everything says so with share 0.
        keep = max(1, int(len(hits) * rung.share)) if rung.share > 0 else 0
        kept_hits = list(hits)[:keep]
    elif rung.passages is None:
        kept_hits = list(hits)
    else:
        kept_hits = list(hits)[: rung.passages]
    if rung.turns is None:
        kept_history = list(history or [])
    elif rung.turns <= 0:
        kept_history = []
    else:
        kept_history = list(history or [])[-rung.turns * 2:]
    return kept_hits, kept_history


@dataclass
class Served:
    """An answer, and what it cost to get it."""

    answer: object        # the Answer, whatever the client returned
    rung: str
    attempts: int
    # The passages the answering attempt was actually given. The caller's own list is
    # the wrong thing to cite or to file: an answer produced from one passage that is
    # recorded against four describes a system that did not run.
    hits: list = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.hits is None:
            self.hits = []

    @property
    def exhausted(self) -> bool:
        """Every rung failed; `answer.error` still carries the failure."""
        return self.rung == "exhausted"

    @property
    def degraded(self) -> bool:
        """It was answered, but not with everything the question was meant to have."""
        return not self.exhausted and self.rung != LADDER[0].label

    @property
    def note(self) -> str:
        """What to show a student above a degraded answer, or nothing."""
        return EXHAUSTED if self.exhausted else NOTE.get(self.rung, "")


def with_fallback(
    ask: Callable[[list, list, float], object],
    *,
    hits: Sequence = (),
    history: Sequence = (),
    free: Callable[[], None] | None = None,
    ladder: tuple[Rung, ...] = LADDER,
    on_retry: Callable[[Rung], None] | None = None,
) -> Served:
    """Ask, and on a memory failure ask again with less of the prompt.

    `ask(hits, history, budget)` sends one attempt and returns an object with `.error`.
    `free` hands the allocator back what the failed attempt reserved - it is the
    caller's `torch.cuda.empty_cache`, passed in rather than imported so this module
    stays testable without a card. `on_retry` is told that another attempt is starting,
    which is how the streaming path clears the half-written bubble.

    The answer that comes back is the last one attempted either way: when every rung
    fails, `answer.error` is left intact so the suites still record a failure. Telling
    the student something useful is `Served.note`'s business, and not the same thing as
    marking the run as a success.
    """
    reply = None
    shapes: list[tuple[int, int, float]] = []
    for rung in ladder:
        kept_hits, kept_history = narrow(hits, history, rung)
        shape = (len(kept_hits), len(kept_history), rung.budget)
        # A rung that gives up nothing the previous one still had is the same attempt
        # a second time: it will fail the same way, a minute later. Retrieval at k=2
        # makes the second rung identical to the first, which is exactly the case the
        # k=2 comparison run is in.
        if shape in shapes:
            continue
        if shapes:
            gc.collect()
            if free is not None:
                free()
            if on_retry is not None:
                on_retry(rung)
        shapes.append(shape)
        reply = ask(kept_hits, kept_history, rung.budget)
        if not is_memory_error(getattr(reply, "error", None)):
            return Served(reply, rung.label, len(shapes), kept_hits)
    return Served(reply, "exhausted", len(shapes), [])

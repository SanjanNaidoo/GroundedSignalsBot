"""The tutor that uses the fine-tuning only where it helps.

Round one measured the adapter doing two opposite things at once, on the same model,
the same passages and the same prompt:

| | fine-tuned | untuned |
|---|---|---|
| calculations, mean score | 0.48 | 0.77 |
| finding a mistake in a student's working | 31% | 76% |
| other courses' questions declined | 100% | 20% |
| assessed work handed over, among solvable | 32% | 92% |
| true-or-false on the notes | 93% | 85% |
| cites the passage it was given | 89% | 52% |

Applying the adapter at part strength did not recover the arithmetic: at a quarter
strength the calculations were still 0.50, and the answers still asserted that a 6 kHz
tone sampled at 8 kHz does not alias. The damage is in the weights, not in the format.

So this pipeline keeps one model on the card and turns the adapter on and off per
question:

1. A question that is not numerical work goes to the fine-tuned weights, which explain,
   cite and hold the course boundary better, and answer in a fifth of the words.
2. Numerical work goes to the untuned weights, which are markedly better at it.
3. Before that, the fine-tuned weights write a short opening for the numerical question.
   If that opening declines it or treats it as work to be handed in, the fine-tuned
   answer stands. This is what stops the accurate-but-compliant untuned weights from
   solving a problem set, and it costs about a second.

The routing rule and the gate are both measured: `scripts/router_check.py` reports what
the rule does to every question in every suite, and the stress suites score the whole
pipeline exactly as they score the plain conditions.
"""

from __future__ import annotations

import re
from dataclasses import replace

from . import calculator
from .model import Answer

# Numerical work: at least two numbers that are not part of a course code, and no
# "true or false", which is a concept question that happens to carry numbers.
#
# The course code has to be removed before the digits are counted, not excluded by a
# lookbehind. `(?<![A-Za-z])` drops only the first digit of EEE4114F; the 1, 1 and 4
# behind it each sit after a digit, so they counted, and every question naming the
# course read as numerical work. That sent 15 of the 23 course questions to the base
# weights with the calculator attached, which answered "how many contact sessions are
# there per week" by multiplying the lecture times - 2, where the handout lists 4.
#
# Masking the code first leaves every other routing decision alone: across the 1,383
# questions of the calculation, concept, error and follow-up banks it changes none.
_CODE = re.compile(r"\b[A-Za-z]{2,}\d[A-Za-z0-9]*\b")
_NUMBER = re.compile(r"(?<![A-Za-z])-?\d")
_TRUE_FALSE = re.compile(r"^\s*true or false", re.I)

# Work the student is going to hand in. The fine-tuned weights answer these however
# numerical they are, because the untuned ones hand the answer over.
ASSESSED = re.compile(
    r"\b(due|for marks|hand (it |this )?in|hand in|submit|submission|assignment|"
    r"problem set|counts (towards|for)|marked|graded)\b", re.I)

# What the gate is looking for in the fine-tuned model's opening.
from .stress import DECLINE  # noqa: E402  (the same detector the scope suite uses)
SCAFFOLD = re.compile(r"assessed work|for marks|hand (it |this )?in|submit|what have you tried|"
                      r"academic integrity|can't give you the (final )?answer|"
                      r"cannot give you the (final )?answer", re.I)

# What a student gets when the model was told the work is assessed and answered it
# anyway. Substituted rather than hoped for, because a policy that holds most of the
# time is not a policy: the course boundary and academic integrity are the two things
# this system must not get wrong, and a 7B model's willingness to follow an instruction
# is the wrong mechanism for either.
#
# It is written to be a useful reply and not merely a refusal - it says why, names what
# to do next, and asks for the student's own work - because a tutor that only says no
# teaches nothing and a student will simply ask elsewhere.
HELD_BACK = (
    "I am not going to give you the answer to this one, because you have said it is "
    "work you will hand in for marks. Doing it for you would not be fair to you or to "
    "anyone else in the class.\n\n"
    "I can help you do it yourself. Tell me what you have tried so far, or where you "
    "get stuck, and I will work through the method with you - which result applies, "
    "why it applies, and what the first step looks like - without putting the numbers "
    "in. If you are not sure where to start, say so and I will point you at the part of "
    "the course notes the question comes from.\n\n"
    "If this is a past paper or a practice question rather than something you are "
    "submitting, say so and I will work it through in full."
)


def held_back(text: str) -> bool:
    """Whether a reply actually withheld the answer, rather than being told to.

    The same detector the scope suite uses, plus the words a policy refusal uses, so
    that what counts as holding back is not decided separately here.
    """
    body = text or ""
    return bool(SCAFFOLD.search(body) or DECLINE.search(body.lower()))


GATE_TOKENS = 48
# Room for the program, and for whatever the model does before writing it.
#
# 400 was enough for Qwen2.5, which writes eight lines and stops. A reasoning model
# spends its whole budget thinking, emits no code block at all, and the question then
# falls through to being answered unaided - so across 182 answers from the two
# candidates the calculator fired exactly zero times, while it fired on 66 of 99 for
# the delivered model. They were being compared with the one component worth twelve
# points switched off, and nothing in the results said so.
CODE_TOKENS = 2048
# Characters of a streamed reply held back before release. Every wrong refusal seen in
# development named the other course within its first two sentences, well inside this.
REFUSAL_WINDOW = 180

# A decline that names this course's own subject as somewhere else. The fine-tuned
# weights turned away 13 of 77 in-scope questions in round one, every one of them by
# sending a course topic to another course: temporal-difference learning to "a
# reinforcement learning course", zero-padding to "a digital signal processing course".
# Measured on those runs, this catches all 13 and overturns none of the 80 correct
# declines. The phrases are the course's own topic names and the words the model used
# for them; anything outside the course, like control systems, is left declined.
NAMES_A_COURSE = re.compile(
    r"belongs (in|to) (an?|another|a different)?\s*([a-z /-]{0,50}?)\s*course|"
    r"comes from (an?|another)?\s*([a-z /-]{0,50}?)\s*course|"
    r"it is (an?|another)?\s*([a-z /-]{0,50}?)\s*(course|topic)", re.I)
OWN_SUBJECT = re.compile(
    r"\b((digital )?signal processing|machine learning|deep learning|neural networks?|"
    r"reinforcement learning|sampling|spectral estimation|spectrum estimation|"
    r"signals and systems|z-transform|fourier|dsp|convolution|"
    r"supervised learning|model evaluation)\b", re.I)


def misplaced_decline(text: str) -> bool:
    """A refusal that sends one of this course's own topics to another course."""
    match = NAMES_A_COURSE.search(text or "")
    if not match:
        return False
    named = " ".join(group for group in match.groups() if group)
    return bool(OWN_SUBJECT.search(named))


def about_course_admin(user_prompt: str) -> bool:
    """Whether the facts about how the course runs are in front of the model.

    The two overturn paths re-ask the base weights, which is the right remedy for a
    refusal the adapter invented and the wrong one here. A question about how the course
    runs arrives with the handout's facts in the prompt, and the base weights have no
    training for reading that block: asked how much the class tests count for, the
    adapter declined, the decline was overturned, and the base weights answered "the
    class tests count for 10% each" - halving a figure that was listed in front of them.
    Asked whether a declaration must be attached, they answered from academic-integrity
    rules in general rather than from the line that says it.

    So a decline on one of these is left standing. It is the adapter failing to read a
    block it was trained on, which is a thing to fix in the prompt and to count in the
    results, not something to paper over with weights that know less.
    """
    return "HOW THE COURSE RUNS" in (user_prompt or "")


def misplaced_policy_decline(text: str, question: str,
                             conversation: str | None = None) -> bool:
    """A refusal that calls the question assessed work when nothing says it is.

    The non-numerical path has no gate: a past paper question routes straight to the
    fine-tuned weights, and those weights decline work-for-marks far more readily than
    the training data warrants. Every assessed example they were trained on states it
    outright - a deadline, a mark, a submission - so a decline with no such words
    anywhere in the question or the conversation is the model pattern-matching on a
    question that merely looks like an exam, which the course prompt explicitly says
    is not assessed work.

    Deliberately conservative: the moment the student does say it is for marks, this
    returns False and the decline stands.
    """
    if ASSESSED.search(conversation or "") or ASSESSED.search(question or ""):
        return False
    return bool(SCAFFOLD.search(text or ""))


def is_numerical(question: str) -> bool:
    if _TRUE_FALSE.match(question or ""):
        return False
    return len(_NUMBER.findall(_CODE.sub(" ", question or ""))) >= 2


def is_assessed(question: str, conversation: str | None = None) -> bool:
    """Whether the student has said this is work they will hand in for marks.

    Deterministic, and that is the point. Whether a question is assessed work is not a
    judgement the answering model should make: the student states it, in words the
    pattern catches - due, for marks, hand in, submit, assignment, problem set, counts
    towards, marked, graded. On the 54 conversations of the pushback suite it fires on
    the first turn of all 54.

    The conversation is searched as well as the question, because a student who says
    "it is due tomorrow" in the first message and "just give me the answer" in the
    third is still asking for work to hand in, and the third message alone does not
    say so. Round two gave the answer away in 9 of 15 such conversations.
    """
    return bool(ASSESSED.search(conversation or question or "")
                or ASSESSED.search(question or ""))


def route(question: str, conversation: str | None = None) -> str:
    """'tuned' or 'base'. The conversation counts: assessed work stays assessed work."""
    if is_assessed(question, conversation):
        return "tuned"
    return "base" if is_numerical(question) else "tuned"


def question_of(user_prompt: str) -> str:
    """The student's question, out of the prompt the pipeline builds around it."""
    marker = "STUDENT QUESTION"
    if marker in user_prompt:
        return user_prompt.split(marker, 1)[1].strip()
    return user_prompt.strip()


def conversation_of(user_prompt: str) -> str:
    """What the student has said: the transcript and the question, never the passages.

    Reading the whole prompt instead cost the first final run 125 of 264 calculations.
    Course material is full of the words that mark assessed work - assignment, submit,
    due, marks - so passages retrieved for an ordinary calculation made it look like
    homework, and it was answered by the weights that are bad at arithmetic.
    """
    marker = "CONVERSATION SO FAR"
    if marker in user_prompt:
        return user_prompt.split(marker, 1)[1]
    return question_of(user_prompt)


# Whether the assessed-work detector reads the student's words only. Until 13 October it
# read the whole transcript, the tutor's own replies included, so a reply that said
# "the := is an assignment operator" or "due to aliasing" turned the student's next,
# ordinary question into work for marks: 8 of 54 follow-up conversations for Opus 5 and
# 2 of 54 for the local system were withheld on the tutor's own words. False reproduces
# every run made before the fix.
STUDENT_WORDS_ONLY = True

_TURN = re.compile(r"^(Student|Tutor)\s*:\s?")
_SCREENSHOT = "READ FROM THE STUDENT'S SCREENSHOT"
_AFTER_SCREENSHOT = ("The student's question below", "Text marked as read",
                     "THIS QUESTION IS WORK", "STUDENT QUESTION")


def student_turns(history, message: str = "") -> str:
    """The student's turns of a (speaker, text) history, then the new message."""
    if not STUDENT_WORDS_ONLY:
        return "\n".join([text for _, text in (history or [])] + [message])
    mine = [text for who, text in (history or []) if str(who).lower().startswith("s")]
    return "\n".join([*mine, message])


def student_said(user_prompt: str) -> str:
    """What the student has said, out of the prompt: their turns, any screenshot they
    sent, and the question. Never the passages and never the tutor's replies.

    `conversation_of` stays as it was for the calculator, which writes its program from
    the whole conversation because a follow-up alone does not state the problem.
    """
    if not STUDENT_WORDS_ONLY:
        return conversation_of(user_prompt)
    text = user_prompt or ""
    asked = question_of(text)
    turns: list[tuple[str, str]] = []
    marker = "CONVERSATION SO FAR"
    if marker in text:
        block = text.split(marker, 1)[1].split("\n---\n", 1)[0]
        speaker, lines = None, []
        for line in block.splitlines():
            found = _TURN.match(line)
            if found:
                if speaker is not None:
                    turns.append((speaker, "\n".join(lines)))
                speaker, lines = found.group(1), [line[found.end():]]
            elif speaker is not None:
                lines.append(line)
        if speaker is not None:
            turns.append((speaker, "\n".join(lines)))
    shot = ""
    if _SCREENSHOT in text:
        shot = text.split(_SCREENSHOT, 1)[1]
        for end in _AFTER_SCREENSHOT:
            shot = shot.split(end, 1)[0]
    return student_turns(turns + ([("Student", shot.strip())] if shot.strip() else []), asked)


class HybridClient:
    """One local model, with the adapter switched on or off per question."""

    def __init__(self, client, *, calculate: bool = True, gate: bool = True,
                 route_adapter: bool = True, overturn: bool = True):
        self._client = client
        # Numerical work is computed by a program before it is explained; see ym07.calculator.
        self.calculate = calculate
        # Whether a reply that answers assessed work is replaced. Switchable because the
        # gate and the training aim at the same behaviour, and with the gate on the
        # measurement cannot tell them apart - it forces a pass either way.
        self.gate = gate
        # Whether a decline is overturned by re-asking the base weights. Both paths were
        # built to catch the scope clause misfiring, and the clause is gone; this makes
        # their removal a measurement. See about_course_admin.
        self.overturn = overturn
        # Whether the numerical path answers from the base weights. The rule exists
        # because the first adapter cost arithmetic; if the current one does not, it
        # is a component with nothing to show for itself.
        self.route_adapter = route_adapter
        # Filled in per answer, for the runners that record how a reply was produced.
        self.last_route = "tuned"
        self.last_gate = None
        self.last_computation = None

    def _gate_says_policy(self, *, system, user, model, max_tokens, effort) -> bool:
        """Do the fine-tuned weights want to decline this, or hold the answer back?"""
        opening = self._client.answer(system=system, user=user, model=model,
                                      max_tokens=max_tokens, effort=effort,
                                      max_new_tokens=GATE_TOKENS)
        self.last_gate = opening.text
        if opening.error:
            return True  # a failed gate keeps the policy-abiding weights, not the other way
        text = opening.text or ""
        if misplaced_decline(text):
            # The same wrong refusal the non-numerical path already overturns: it sent a
            # record-length question to "a signals and systems course". A gate that
            # believes it hands the question to the weights that cannot do arithmetic.
            return False
        # SCAFFOLD is deliberately not consulted here any more.
        #
        # This gate is only reached when `route` returned "base", and `route` returns
        # "base" only when ASSESSED found nothing in the question or the conversation.
        # So by the time the opening is written, the objective signal has already said
        # this is not work being handed in, and the only thing SCAFFOLD added was the
        # fine-tuned model's opinion to the contrary - from weights whose behaviour
        # training is 47 refusals in 97 pairs, and which open with "This is assessed
        # work" on a past paper question that says nothing of the kind. A student
        # pasting a past paper and asking for the working got the scaffold instead of
        # the answer, which is the failure AT11 counts.
        #
        # Nothing AT13 measures passes through here: all three of its framings name an
        # assignment, a problem set, a submission or a deadline, so they route to the
        # fine-tuned weights directly and never reach this gate.
        return bool(DECLINE.search(text.lower()))

    def _computed(self, *, system, user, model, max_tokens, effort):
        """Ask for a program, run it, and return its printed values."""
        # A follow-up says "what if h[n] were {3, 3} instead?" and nothing else. Asked to
        # write a program from that alone, the model computed the wrong thing and the
        # answer went with it: ten follow-ups that had been right became wrong. The
        # program is written from the conversation, which carries the original problem.
        question = conversation_of(user)
        asked = self._client.answer(system="You write short Python programs. Nothing else.",
                                    user=calculator.CODE_PROMPT.format(question=question),
                                    model=model, max_tokens=max_tokens, effort=effort,
                                    adapter=False, max_new_tokens=CODE_TOKENS)
        if asked.error:
            self.last_computation = f"no program: {asked.error}"
            return None
        code = calculator.extract_code(asked.text)
        if not code:
            self.last_computation = "no program: the model wrote no code"
            return None
        values, why = calculator.run(code)
        self.last_computation = values if values else f"no values: {why}"
        return values

    def answer(self, *, system, user, model, max_tokens, effort) -> Answer:
        # The whole prompt, conversation included: a student who says "it is due
        # tomorrow" in the first message and "just give me the answer" in the third is
        # still asking for work to hand in, and the third message on its own does not
        # say so. Round two handed the answer over in 9 of 15 such conversations.
        asked = question_of(user)
        said = student_said(user)
        self.last_route = route(asked, conversation=said)
        self.last_gate = None
        # Cleared per question, not only at construction. Left to carry over, it put the
        # previous question's computed values on the row of a question that never ran the
        # calculator - so a row's `computation` field described a different question, and
        # the course suite appeared to be computing signal supports to answer "who is the
        # convenor?" when in fact it had computed nothing at all.
        self.last_computation = None
        if self.last_route == "tuned":
            reply = self._client.answer(system=system, user=user, model=model,
                                        max_tokens=max_tokens, effort=effort)
            # Neither overturn applies to a question about how the course runs: the
            # facts are already in the prompt and the base weights cannot read them.
            # See about_course_admin.
            course_admin = about_course_admin(user)
            if self.overturn and not course_admin and misplaced_decline(reply.text):
                # It turned away its own course. The base weights answer it instead.
                self.last_route = "base (refusal overturned)"
                return self._client.answer(system=system, user=user, model=model,
                                           max_tokens=max_tokens, effort=effort, adapter=False)
            if (self.overturn and not course_admin
                    and misplaced_policy_decline(reply.text, asked, said)):
                # It called a past paper assessed work. Same remedy, different reason,
                # and recorded separately so the two are never confused in the results.
                self.last_route = "base (policy refusal overturned)"
                return self._client.answer(system=system, user=user, model=model,
                                           max_tokens=max_tokens, effort=effort, adapter=False)
            if self.gate and is_assessed(asked, said) and not held_back(reply.text):
                # Told the work was assessed, and answered it regardless. The reply is
                # replaced rather than returned, and the substitution is recorded so the
                # results say how often the model had to be overruled instead of hiding
                # it behind a policy figure that looks like the model's own behaviour.
                self.last_route = "tuned (answer withheld)"
                return replace(reply, text=HELD_BACK)
            return reply
        if self._gate_says_policy(system=system, user=user, model=model,
                                  max_tokens=max_tokens, effort=effort):
            self.last_route = "tuned (gate)"
            return self._client.answer(system=system, user=user, model=model,
                                       max_tokens=max_tokens, effort=effort)
        if self.calculate:
            values = self._computed(system=system, user=user, model=model,
                                    max_tokens=max_tokens, effort=effort)
            if values:
                self.last_route = "base (computed)"
                reply = self._client.answer(system=system,
                                            user=calculator.with_computation(user, values),
                                            model=model, max_tokens=max_tokens, effort=effort,
                                            adapter=not self.route_adapter)
                if not (reply.error and "out of memory" in reply.error):
                    return reply
                # The computed values lengthen the prompt, and on long partial-fraction
                # answers that was enough to run the card out of memory: 15 of 24 failed.
                # Without the values the question still gets the answer it used to.
                self.last_route = "base (computed, fell back)"
        return self._client.answer(system=system, user=user, model=model,
                                   max_tokens=max_tokens, effort=effort, adapter=not self.route_adapter)

    def stream(self, *, system, user, model, max_tokens=None, effort=None):
        """The same decisions as `answer`, for the browser, which shows the reply as it comes.

        It has to make every decision `answer` makes, or the interface would not be the
        system the results describe: the gate, the calculator, and overturning a refusal
        that sends this course's own topic elsewhere.
        """
        self.last_route = route(question_of(user), conversation=student_said(user))
        self.last_gate = None
        self.last_computation = None
        if self.last_route == "base":
            if self._gate_says_policy(system=system, user=user, model=model,
                                      max_tokens=max_tokens, effort=effort):
                self.last_route = "tuned (gate)"
                yield from self._client.stream(system=system, user=user, model=model,
                                               max_tokens=max_tokens, effort=effort)
                return
            prompt = user
            if self.calculate:
                values = self._computed(system=system, user=user, model=model,
                                        max_tokens=max_tokens, effort=effort)
                if values:
                    self.last_route = "base (computed)"
                    prompt = calculator.with_computation(user, values)
            yield from self._client.stream(system=system, user=prompt, model=model,
                                           max_tokens=max_tokens, effort=effort, adapter=not self.route_adapter)
            return

        # A refusal is short and says what it is in its first sentence, so the opening is
        # held back until it is clear whether this is one that has to be overturned - and,
        # when the work is assessed, whether it is a refusal at all. Both decisions are
        # made from the same held opening: one overturns a refusal that should not have
        # happened, the other supplies one that should have.
        assessed = is_assessed(question_of(user), student_said(user))
        held, released, final, verdict = [], False, None, None
        pieces = self._client.stream(system=system, user=user, model=model,
                                     max_tokens=max_tokens, effort=effort)
        for piece in pieces:
            if not isinstance(piece, str):
                final = piece
                continue
            if released:
                yield piece
                continue
            held.append(piece)
            opening = "".join(held)
            if len(opening) >= REFUSAL_WINDOW:
                if self.overturn and misplaced_decline(opening):
                    verdict = "overturn"
                    break
                if self.gate and assessed and not held_back(opening):
                    verdict = "withhold"
                    break
                released = True
                yield opening
        opening = "".join(held)
        if verdict is None and not released:
            # The whole reply was shorter than the window, so it is judged now.
            if self.overturn and misplaced_decline(opening):
                verdict = "overturn"
            elif self.gate and assessed and not held_back(opening):
                verdict = "withhold"
        if verdict == "overturn":
            for _ in pieces:  # let the refusal finish, so the card is free
                pass
            self.last_route = "base (refusal overturned)"
            yield from self._client.stream(system=system, user=user, model=model,
                                           max_tokens=max_tokens, effort=effort, adapter=False)
            return
        if verdict == "withhold":
            # Drain, so the card is free before anything else runs, and keep the reply
            # that ends the stream: it is what tells the caller the answer is finished.
            # From 6 to 14 October it was drained with everything else, so when the gate
            # overruled a reply longer than the window the browser got the held-back text
            # and no finished reply, and showed "the model produced nothing" over the
            # refusal. The suites call `answer`, which never had the fault.
            for rest in pieces:
                if not isinstance(rest, str):
                    final = rest
            self.last_route = "tuned (answer withheld)"
            yield HELD_BACK
            if final is not None:
                yield replace(final, text=HELD_BACK)
            return
        if not released and opening:
            yield opening
        if final is not None:
            yield final

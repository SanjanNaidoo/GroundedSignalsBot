"""System prompts, built from the course config.

The policy block in the YAML is what makes these prompts course-specific, and
each toggle maps to a paragraph here. That mapping is the point: a pedagogical
choice in the config is visible as a behaviour in the prompt and measurable as a
number in the report.
"""

from __future__ import annotations

import re

from .config import CourseConfig
from .retrieval import Hit

# E33 (protocols/followup-fix.md): the tutor's earlier replies go into the conversation
# without their passage tags. The tags point at passages the model no longer has: each
# turn's passages are numbered [S1]-[S4] afresh, so a first reply's [S2] met a different
# second passage, and Opus 5 retracted citations it had made correctly (E30). False
# restores the history every result before 16 October used. Off by E33's registered rule
# (18 October), with retrieval.CONVERSATION_QUERY: see the note there.
HISTORY_WITHOUT_TAGS = False
_PASSAGE_TAG = re.compile(r" ?\[\s*S\s*\d+(?:\s*[,;–-]\s*S?\s*\d+)*\s*\]", re.IGNORECASE)


def _untagged(speaker: str, text: str) -> str:
    if HISTORY_WITHOUT_TAGS and not str(speaker).lower().startswith("student"):
        return _PASSAGE_TAG.sub("", text)
    return text


def _policy_clauses(config: CourseConfig) -> list[str]:
    policy = config.policy
    clauses: list[str] = []

    if policy.socratic_on_assessed_work:
        # The trigger is the student submitting the work for marks, not the
        # question resembling coursework. The first version of this clause named
        # "a tutorial question, a test problem" as triggers, and the model
        # correctly obeyed it: measured on the frozen item set, it withheld the
        # answer on 12% of ordinary concept and calculation questions. Students
        # revise from past papers, so that behaviour makes it a worse tutor, and
        # it contradicts the reference answers the benchmark grades against.
        clauses.append(
            "When a student wants you to produce work they will hand in for marks — "
            "they say it is an assignment or problem set, that it is due, that they "
            "need something to submit — do not supply the finished answer. Say plainly "
            "that it is assessed work, then scaffold: ask what they have tried, point "
            "at the relevant method or result from the course, and offer to check their "
            "working once they have attempted it. "
            "A question that merely looks like a tutorial or exam problem is not "
            "assessed work. Students revise from past papers, and answering those "
            "fully, showing the reasoning, is exactly your job."
        )
    if policy.refuse_out_of_scope:
        clauses.append(
            f"Answer only from {config.code} material. If a question belongs to another "
            "course, say so and decline, rather than answering from general knowledge. "
            "Being unable to help is a correct outcome; a fluent answer from outside "
            "the course is not."
        )
    if policy.notation_source == "course_notes":
        clauses.append(
            "Use the notation, symbols and conventions of the course notes, not those "
            "of an arbitrary textbook."
        )

    clauses.append(
        "Never state something you are unsure of as fact. If the course material does "
        "not settle a question, say so."
    )
    return clauses


def system_prompt(config: CourseConfig, *, grounded: bool) -> str:
    """The course prompt used by C1 (ungrounded) and C2 (grounded).

    C0 sends no system prompt at all — that is what makes it the control.
    """
    # The course name alone is not a usable scope boundary. Told only that it
    # tutors "EEE4114F (Digital Signal Processing)", the model refused
    # reinforcement-learning questions as off-syllabus — they are Part B of this
    # course. The topic list is already in the config; state it, so
    # refuse_out_of_scope has something accurate to refuse against.
    topics = ", ".join(t.name for t in config.topics)
    lines = [
        f"You are a tutor for {config.code} ({config.name}) at "
        f"{config.institution or 'the university'}, a level-{config.level} course.",
        "You support students studying the course. You are not a general assistant.",
        "",
        f"The course covers: {topics}. All of that is in scope, including the "
        "machine learning material — the course name names only its first half.",
        "",
    ]
    lines += [f"- {clause}" for clause in _policy_clauses(config)]

    if grounded and config.policy.require_citations:
        lines += [
            "",
            "Course material is supplied below under SOURCES, each passage tagged [S1], "
            "[S2] and so on. Ground every substantive claim in those passages and cite "
            "the tag inline, like [S2], at the point the claim is made.",
            "Do not cite a passage that does not actually support the claim, and do not "
            "invent a tag. If the supplied passages do not answer the question, say that "
            "the material does not cover it rather than filling the gap from memory.",
        ]
    elif grounded:
        lines += [
            "",
            "Course material is supplied below under SOURCES. Answer from it.",
        ]

    return "\n".join(lines)


# A turn kept in full in the transcript, in characters. Three turns of a worked
# derivation, each with four passages behind it, ran the card out of memory in 20 of 54
# conversations in the first round: the prompt grows with every turn while the KV cache
# has to fit beside the weights. Older turns are kept for what they establish - the
# problem, the numbers, what was asked - not for their full working.
HISTORY_CHARS = 1200


# The browser tutor tags any text it read out of an image with one of these, so that a
# later turn can still tell a machine transcription from something the student typed.
# They are distinguished because they carry different weight: the first is a reading
# nobody has checked, the second is one the student saw in the composer and could
# correct before sending.
OCR_MARK = "read from a screenshot by OCR"
SHOT_MARK = "transcribed from a screenshot and checked by the student"


def _shortened(text: str) -> str:
    text = text or ""
    if len(text) <= HISTORY_CHARS:
        return text
    return text[:HISTORY_CHARS].rstrip() + " […]"


def user_prompt(
    question: str,
    hits: list[Hit] | None = None,
    history: list[tuple[str, str]] | None = None,
    facts: tuple[tuple[str, str], ...] = (),
    screenshot: str = "",
    from_image: str = "",
    assessed: bool = False,
    handout_pages: bool = False,
) -> str:
    """The question as sent. Identical across conditions apart from SOURCES.

    `history` is used only by the interactive tutor (`ym07 chat`), which is
    multi-turn; every benchmark condition passes a single question with no
    history, so the string a run sends is unchanged by this parameter existing.
    Keeping one builder means the demo and the measured system cannot drift.

    `screenshot` carries text read out of an image the student uploaded. It is
    labelled as a transcription rather than merged into the question, because a
    reading can be wrong in ways the student's own typing is not: a lost subscript
    or a misread exponent changes the problem, and a tutor told where the words
    came from can say that the question looks garbled instead of confidently
    answering the wrong one. Every benchmark condition passes this empty, so no
    measured result is affected by it existing.

    `from_image` names the engine when the question text *is* a transcription the
    student has already seen and been able to correct, which is how the web tutor
    now sends a screenshot. It earns a shorter caution than `screenshot` does: a
    reading the student has looked at is better evidence of what they meant than one
    they never saw, but it is still not their own typing, and a lost subscript
    survives a glance.
    """
    if (not hits and not history and not facts and not screenshot and not from_image
            and not assessed and not handout_pages):
        return question

    blocks: list[str] = []
    if facts:
        # How the course runs, stated only when the question asks about it. In the
        # system prompt, for every question, these facts made each prompt about 900
        # tokens longer and mention assignments and work submitted for marking: the gate
        # then treated 19 of 264 calculations as homework, and all 19 ran out of memory.
        blocks += ["HOW THE COURSE RUNS (from the course handout)", ""]
        blocks += [f"- {name.replace('_', ' ')}: {value}" for name, value in facts]
        blocks += ["", "Answer a question about any of this from the list above, and say it "
                       "comes from the course handout. Give the value as it is written "
                       "there rather than working one out from it: asked how many contact "
                       "sessions there are, the answer is the number listed, not a figure "
                       "derived from the lecture times. "
                   # A fact the handout does not carry is answered with the nearest line
                   # that looks like it: asked for the teaching assistant's email
                   # address the model gives their name, asked for the convenor's phone
                   # number it gives their email address, and asked when the final exam
                   # is - a date the handout does not state - it answered "on 20 May,
                   # from 14:00 to 16:00. That is from the course handout."
                   #
                   # Naming that in this instruction was tried and made it worse: the
                   # exam question went from deflecting to inventing the date above, and
                   # "when is Class Test 2", which passes without it, came back "it is
                   # assessed work, so I won't write the answer for you". The wording is
                   # left as the 190 training examples had it.
                   #
                   # The gap is in the training set, not here. Every one of those
                   # examples has the asked-for fact present in the block, with only its
                   # value scrambled, so the adapter has never been shown a question the
                   # block does not answer. Examples of that are what would fix it; the
                   # generator and the pipeline for them already exist.
                       "If what is asked is not listed, say you do not have it "
                       "rather than guessing.", "", "---", ""]
    if hits:
        blocks += ["SOURCES", ""]
        for idx, hit in enumerate(hits, start=1):
            blocks.append(f"[S{idx}] ({hit.chunk.citation()})")
            blocks.append(hit.chunk.text.strip())
            blocks.append("")
        blocks += ["---", ""]

    if handout_pages:
        # The passages above are the course handout, every page of it, because the
        # question asks how the course runs. Two instructions, each earned by a failure
        # the other does not prevent.
        #
        # Read the value, do not work it out: asked how many contact sessions there are
        # a week, the model multiplied the lecture times and said 2, where the handout
        # lists 4 in the sentence below the table it had been given.
        #
        # Say it is not there rather than reasoning about it: asked whether a
        # declaration must be attached to submitted work, it answered that the handout
        # does not mention one and that the university's general policy would apply.
        # The handout says a non-plagiarism declaration must be submitted with all work
        # submitted for marking, in those words, on its last page.
        blocks += ["HOW THE COURSE RUNS", "",
                   "The passages above are the whole course handout, which is the "
                   "document that states how this course runs. The answer to this "
                   "question is in it. Find the line that carries it, give the value, "
                   "and cite the page.",
                   "",
                   # No value from the handout is quoted here, and none may be: four of
                   # the questions this is measured on would then be answerable from
                   # the instruction without reading the document at all.
                   "Much of what is asked sits in a table, or on a line that is just a "
                   "label and its value, rather than in a sentence. A row in a table is "
                   "as much an answer as a sentence is. Read it out.",
                   "",
                   "Give the value as the handout writes it rather than working one out "
                   "from something else it says. If it lists a number, that number is "
                   "the answer, even where another figure on the page could be made to "
                   "imply a different one.",
                   "",
                   "Do not describe what the handout covers instead of answering. "
                   "Listing its topics back - that it covers the credits, the test "
                   "dates, the assessment - is not an answer, and saying the handout "
                   "does not give something it does give is worse than a wrong value. "
                   "Look for the figure before concluding it is absent.",
                   "", "---", ""]

    if history:
        blocks += ["CONVERSATION SO FAR", ""]
        for speaker, text in history:
            blocks.append(f"{speaker}: {_shortened(_untagged(speaker, text))}")
            blocks.append("")
        blocks += ["---", ""]

    if screenshot:
        blocks += ["READ FROM THE STUDENT'S SCREENSHOT", ""]
        blocks += [screenshot.strip(), ""]

    # The caution is emitted whenever a transcription is anywhere in front of the
    # model, not only on the turn the image arrived. A screenshot read on turn one
    # is still the only statement of the problem on turn three, and by then it is
    # sitting in the conversation looking like something the student typed and
    # checked - which is how a misread formula gets confidently answered.
    if from_image or any(SHOT_MARK in text for _, text in (history or ())):
        blocks += [f"The student's question below was transcribed from an image by "
                   f"{from_image or 'a machine reader'} and shown to the student, who "
                   "could correct it before sending. Machine transcription of "
                   "mathematics is still imperfect, so if an expression does not make "
                   "sense as a question in this course - a subscript that has gone "
                   "missing, an exponent that reads wrong, a Greek letter that has "
                   "become a digit - say which part looks wrong and ask the student to "
                   "confirm it. Do not answer a different question from the one the "
                   "transcription was trying to be.",
                   "", "---", ""]

    if screenshot or any(OCR_MARK in text for _, text in (history or ())):
        blocks += ["Text marked as read from a screenshot came from optical character "
                   "recognition, so it may contain mistakes the student did not make - a "
                   "lost subscript, a misread exponent, a Greek letter read as a digit, a "
                   "broken fraction. Work from it only if it makes sense as a question in "
                   "this course. If it does not, say which part looks wrong and ask the "
                   "student to retype it. Never answer a different question from the one "
                   "the transcription was trying to be.",
                   "", "---", ""]

    if assessed:
        # Said here rather than only in the system prompt, and said as an instruction
        # about this question rather than a standing rule.
        #
        # The system prompt has always carried the policy, and the delivered model held
        # to it because the adapter was trained on it. A model without that training
        # reads the same system prompt and answers the question anyway: R1-Distill gave
        # the answer away on the first turn of 17 of 18 conversations that open by
        # saying the work is for marks. The detector is not the weak part - it fires on
        # 54 of 54 - so the instruction is put where the question is, and what the model
        # does with it is checked afterwards rather than assumed.
        blocks += ["THIS QUESTION IS WORK THE STUDENT WILL HAND IN", "",
                   "The student has said this is for an assignment, a problem set or "
                   "something submitted for marks. Do not give the final answer or a "
                   "worked solution they could hand in.",
                   "",
                   "Do help them do it themselves. Say plainly that you will not give "
                   "the answer because it is for marks, name the method the question "
                   "needs and the first step of it, and ask what they have tried so far "
                   "or where they are stuck. Keep every number out of it: no final "
                   "value, no intermediate result they could copy.",
                   "", "---", ""]

    blocks += ["STUDENT QUESTION", "", question]
    return "\n".join(blocks)

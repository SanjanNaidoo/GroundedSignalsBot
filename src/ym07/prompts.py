"""System prompts, built from the course config.

The policy block in the YAML is what makes these prompts course-specific, and
each toggle maps to a paragraph here. That mapping is the point: a pedagogical
choice in the config is visible as a behaviour in the prompt and measurable as a
number in the report.
"""

from __future__ import annotations

from .config import CourseConfig
from .retrieval import Hit


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


def user_prompt(
    question: str,
    hits: list[Hit] | None = None,
    history: list[tuple[str, str]] | None = None,
) -> str:
    """The question as sent. Identical across conditions apart from SOURCES.

    `history` is used only by the interactive tutor (`ym07 chat`), which is
    multi-turn; every benchmark condition passes a single question with no
    history, so the string a run sends is unchanged by this parameter existing.
    Keeping one builder means the demo and the measured system cannot drift.
    """
    if not hits and not history:
        return question

    blocks: list[str] = []
    if hits:
        blocks += ["SOURCES", ""]
        for idx, hit in enumerate(hits, start=1):
            blocks.append(f"[S{idx}] ({hit.chunk.citation()})")
            blocks.append(hit.chunk.text.strip())
            blocks.append("")
        blocks += ["---", ""]

    if history:
        blocks += ["CONVERSATION SO FAR", ""]
        for speaker, text in history:
            blocks.append(f"{speaker}: {text}")
            blocks.append("")
        blocks += ["---", ""]

    blocks += ["STUDENT QUESTION", "", question]
    return "\n".join(blocks)

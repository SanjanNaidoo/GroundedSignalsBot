"""The facts about how the course runs, read out of the course handout.

A student asks when the class tests are, what the DP requirements are, who convenes the
course. Those answers are in one document, and this reads them out of it, so that what
the tutor says and what the handout says cannot drift apart. Change the handout, re-run
`ym07 ingest`, and the answers change with it.

Why they are extracted rather than handed to the model as pages of the document:

The adapter was trained to read a labelled list. All 190 of its course examples put the
facts in front of it as `- credits: 16 SAQA credits` under one heading, with the values
scrambled so what it learned was reading the block rather than the 2026 dates. Supplying
the four handout pages instead - the same document, in its own words, cited by page -
gave it a format it has no training for, and it fell back on the system prompt's
permission to decline: asked how many credits the course is worth it said the handout
"doesn't give that. It covers the course name, the SAQA credits, ..." while listing the
very fact it had been asked for. That arm scored 16 of 23 against 19 for the list.

So the format is a constraint the training set has already fixed, and the honest way to
meet both it and the handout is to parse the one into the other. The values are the
document's; the shape is what the weights can read.

Each field is a pattern over the handout's text with the label it is written under.
Where the handout gives a value in a table - the assessment weightings, the number of
weeks - the pattern matches the row. Nothing here states a value; every one is read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# The handout is text pulled out of a PDF: the dash in a time range is an en dash, the
# tables arrive as a label and its value separated by a newline, and words are broken
# where the PDF's own kerning put a space, as in "Discrete -Time".
_DASH = re.compile(r"[‐-―−]")
_SPACED_HYPHEN = re.compile(r"\s+-\s*(?=[A-Z])")


def flatten(text: str) -> str:
    """The handout's text on one line, with its dashes and spacing regularised."""
    text = _DASH.sub("-", text or "")
    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class Field:
    """One fact, the pattern that finds it, and how its value is tidied."""

    name: str
    pattern: str
    # Appended to the extracted value, where the handout's own phrasing is a bare
    # number and the question asks what it is a number of.
    suffix: str = ""
    # Joined with ", " when the pattern has several groups.
    groups: int = 1

    def read(self, text: str) -> str | None:
        match = re.search(self.pattern, text, re.I)
        if not match:
            return None
        parts = [g.strip(" .,;:-") for g in match.groups() if g and g.strip()]
        if not parts:
            return None
        value = ", ".join(parts)
        value = _SPACED_HYPHEN.sub("-", value)
        value = re.sub(r"(\w)\s+-(\w)", r"\1-\2", value)  # "Discrete -Time"
        return f"{value}{self.suffix}" if self.suffix else value


# Order is the order they reach the prompt, which follows the handout.
FIELDS: tuple[Field, ...] = (
    Field("convenor", r"Course convenor:\s*([^:]+?)\s+Email address:"),
    Field("convenor email", r"Email address:\s*(\S+@\S+?)(?:\s|$)"),
    Field("convenor office", r"Office location:\s*(.+?)\s+Consultation hours:"),
    Field("consultation", r"Consultation hours:\s*(.+?)\s+Course lecturer:"),
    Field("lecturers", r"Course lecturer:\s*(.+?)\s+Teaching assistant:"),
    Field("teaching assistant", r"Teaching assistant:\s*(.+?)\s+Lecture venue:"),
    Field("lecture venue", r"Lecture venue:\s*(.+?)\s+Lecture days"),
    # The label is split across lines in the PDF: "Lecture days and / time: / Tuesdays".
    Field("lecture times", r"Lecture days and\s*time:\s*(.+?)\s+Laboratory"),
    Field("credits", r"SAQA Credits:\s*(\d+)", suffix=" SAQA credits"),
    Field("prerequisites", r"Pre-requisites:\s*(.+?)\s+Co-requisites:"),
    Field("class test 1", r"Class Test 1\s+(.+?)\s+Class Test 2"),
    Field("class test 2", r"Class Test 2\s+(\d{1,2}\w{0,2}\s+\w+,\s*[\d:]+\s*-\s*[\d:]+)"),
    # The assessment table, read as three rows rather than one phrase.
    Field("assessment weighting",
          r"Project\s+(\d+)\s+Tests\s+(\d+)\s+Exam\s+(\d+)\s+Total\s+100"),
    Field("dp requirements",
          r"The following DP rules apply:.*?-\s*(Completion and submission of all\s+"
          r"computer assignments)\s*-\s*(Completion and submission of the\s+"
          r"course mini-project)"),
    Field("recommended text", r"Recommended text:\s*(.+?)\s+-\s*Comprehensive"),
    Field("course notes", r"-\s*(Comprehensive set of notes.+?given out during the course)"),
    Field("absence",
          r"Absence:\s*(The continuous assessment marks will be adjusted.+?)"
          r"\s*(?:Academic dishonesty|$)"),
    Field("plagiarism", r"(A non-plagiarism declaration must be submitted with all work"
                        r" submitted for marking)"),
    Field("course length", r"Number of weeks the course lasts\s+(\d+)", suffix=" weeks"),
    Field("notional hours", r"Total hours\s+(\d+)", suffix=" hours in total"),
    Field("lectures",
          r"Number of lectures\s+(\d+)\b.*?|supported by\s+(\d+)\s+contact sessions per"
          r" week\s*\((\d+)\s*minutes each\)"),
    Field("parts", r"(Part A of this course aims.+?)\s*(?:Detailed course content|$)"),
)

# `lectures` needs both numbers, which sit three paragraphs apart, so it is assembled
# rather than matched in one go.
_LECTURES = re.compile(r"Number of lectures\s+(\d+)", re.I)
_SESSIONS = re.compile(r"supported by\s+(\d+)\s+contact sessions per week\s*"
                       r"\((\d+)\s*minutes each\)", re.I)
# "Part A ... Part B ..." runs to the end of its paragraph on page 1.
_PARTS_A = re.compile(r"Part A of this course aims to develop (.+?)\.\s*Topics include", re.I)
_PARTS_B = re.compile(r"Part B of the course is (an introduction to[^.]+?)\s*(?:for|\.)", re.I)


def _lectures(text: str) -> str | None:
    lectures, sessions = _LECTURES.search(text), _SESSIONS.search(text)
    if not lectures or not sessions:
        return None
    return (f"{lectures.group(1)} lectures, with {sessions.group(1)} contact sessions "
            f"a week of {sessions.group(2)} minutes each")


def _parts(text: str) -> str | None:
    a, b = _PARTS_A.search(text), _PARTS_B.search(text)
    if not a or not b:
        return None
    return (f"Part A is {a.group(1).strip()}, Part B is {b.group(1).strip()}")


def _weighting(text: str) -> str | None:
    match = re.search(r"Project\s+(\d+)\s+Tests\s+(\d+)\s+Exam\s+(\d+)\s+Total\s+100",
                      text, re.I)
    if not match:
        return None
    project, tests, exam = match.groups()
    return f"exam {exam}%, tests {tests}%, project {project}%"


_ASSEMBLED = {"lectures": _lectures, "parts": _parts, "assessment weighting": _weighting}


def read_facts(chunks, handout: str) -> tuple[tuple[str, str], ...]:
    """Every fact the handout states, in the order the handout states them.

    `chunks` is the processed corpus; the pages of `handout` are joined back together
    first, because a value and the label above it can fall either side of a chunk
    boundary. Fields the handout does not carry are left out rather than guessed, so a
    handout that stops listing the teaching assistant simply stops being asked about
    them.
    """
    pages = [c for c in chunks if c.source == handout]
    if not pages:
        return ()
    pages.sort(key=lambda c: (c.page is None, c.page, c.chunk_id))
    text = flatten(" ".join(c.text for c in pages))

    facts: list[tuple[str, str]] = []
    for field in FIELDS:
        value = (_ASSEMBLED[field.name](text) if field.name in _ASSEMBLED
                 else field.read(text))
        if value:
            facts.append((field.name, value))
    return tuple(facts)

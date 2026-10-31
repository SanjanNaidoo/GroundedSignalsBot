"""Course facts answered from the handout by lookup, without asking the model (E27).

The local model is given the handout's facts as a labelled list for every question about
the course, and on the held-out wordings it still misread six of 23: asked what must be
completed to write the exam it said the exam is open book, asked for the recommended
book it said the handout lists none, with the line it needed in front of it each time.
Opus 5 read the same list correctly 23 times. What fails is the 7B model reading, so the
fix is to stop asking it to: when a question is plainly asking for one of the handout's
facts, the reply is that fact's handout line, cited to its page, and no model runs.
This is the design the project already uses for the two behaviours a tutor must not get
wrong (the assessed-work gate, the citation check): deterministic rules outside the
weights. See protocols/course-facts.md.

Two tests decide whether the lookup answers, and both must pass.

  It is a course question. The question's similarity to the development wordings of
  the course questions, by the embedding model the retriever already loads, is at least
  ABOUT_THRESHOLD. Fixed on development material on 14 October: the 23 development
  wordings, each against the other 22, scored at least 0.728; 518 content questions
  (488 from the other suites' development banks, the 20 past-question queries of E25
  and ten from other courses) scored at most 0.640.

  The handout has the fact. The question names the fact's subject (the convenor, the
  class tests, the exam ...) and asks for its attribute (who, when, how much ...), by
  the word lists below. This is what keeps "when is the final exam?" - a course
  question the handout does not answer - from being answered with a class test date:
  there is no (exam, date) fact, so nothing matches and the model answers as before.

When either test fails the question goes to the model exactly as it would have without
the lookup, so a missed match costs nothing that was not already being paid. A wrong
match is the real risk, a confident wrong answer with a real citation; the fresh set's
absent and content items are there to measure it.

Version 2 (E28, protocols/course-facts-2.md) answers what E27 found. The local model
invents a value when a course question asks for one the handout does not give (an exam
start time, a project deadline), so version 2 replies to those itself, with a fixed
sentence saying the handout does not give it. Its similarity is measured against every
course wording seen so far, with a threshold re-set on development material; its word
rules are corrected on that material; it acts only on short questions with no
mathematics; and it no longer widens what counts as a course question for routing, so
on everything it leaves alone the system is the delivered one. Version 1 is unchanged
and is what `A-r1-facts` measured.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .handout import _ASSEMBLED, FIELDS, flatten, read_facts

# Set on development material only; see the module docstring.
ABOUT_THRESHOLD = 0.68

# The wordings the similarity is measured against: the development wording of each of
# the 23 course facts (`coursequeries.COURSE_QUERIES[*]["dev"]`), read from there so the
# two cannot drift apart. The final, check and fresh wordings are never used here.


def _any(*patterns: str) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.I)


# What a question is about. Written from the handout's own labels, the development
# wordings and the ordinary ways a student names these things.
SUBJECTS: dict[str, re.Pattern] = {
    "convenor": _any(r"\bconven[eo]rs?\b", r"\bconvenes\b", r"\bco-?ordinators?\b",
                     r"\bcourse (leader|head)\b", r"\bin charge\b", r"\bperson responsible\b",
                     r"\bruns? (this|the) (course|module|class)\b", r"\bruns? (eee4114f|4114)\b",
                     r"\bdr\.? martin\b", r"\byaaseen\b"),
    "lecturer": _any(r"\blecturers?\b", r"\blectures (this|the|eee4114f|4114)\b",
                     r"\bteach(es|ing)?\b(?! assistant)", r"\btaught\b",
                     r"\bgives? the lectures\b"),
    "ta": _any(r"\bta'?s?\b", r"\bteaching assistants?\b", r"\btutors?\b", r"\bassistants?\b"),
    "lectures": _any(r"\blectures\b", r"\blecture (venue|theatre|room|hall|times?|schedule|"
                     r"timetable|slots?|days?)\b", r"\bclasses\b", r"\bclass times\b",
                     r"\bcontact sessions?\b", r"\bsessions?\b"),
    "tests": _any(r"\bclass ?tests?\b", r"\btests?\b", r"\bct ?[12]\b", r"\bquiz(zes)?\b"),
    "test1": _any(r"\bfirst (class )?test\b", r"\b(class )?test (1|one)\b", r"\bct ?1\b",
                  r"\b1st (class )?test\b"),
    "test2": _any(r"\bsecond (class )?test\b", r"\b(class )?test (2|two)\b", r"\bct ?2\b",
                  r"\b2nd (class )?test\b"),
    "exam": _any(r"\bexams?\b", r"\bexaminations?\b"),
    "project": _any(r"\bmini-?projects?\b", r"\bthe projects?\b", r"\bprojects? (is|are|counts?|"
                    r"carries|worth)\b", r"\bprojects?\b(?= ?\?)"),
    "assignments": _any(r"\bassignments?\b"),
    "course": _any(r"\bcourse\b", r"\beee4114f\b", r"\b4114\b", r"\bmodule\b"),
    "book": _any(r"\btext ?books?\b", r"\bbooks?\b", r"\bset (text|work)\b",
                 r"\breading list\b", r"\brecommended (text|reading)\b"),
    "dp": _any(r"\bdp\b", r"\bduly performed\b",
               r"\b(allowed|permitted|eligible|able) to (write|sit|take|do) (the )?(final )?exams?\b",
               r"\bqualify (to (write|sit|take|do)|for) (the )?(final )?exams?\b",
               r"\b(allowed|let|get) into (the )?(final )?exams?\b"),
    "absence": _any(r"\bsick\b", r"\bill\b(?!-)", r"\billness\b", r"\bmedical\b", r"\babsent\b",
                    r"\babsence\b", r"\bmiss(ed|ing)? (a|the|my) (class )?test\b",
                    r"\bfuneral\b", r"\bbereavement\b"),
    "declaration": _any(r"\bdeclarations?\b", r"\bplagiarism\b", r"\bsign anything\b",
                        r"\bcover ?sheets?\b"),
}

# What a question asks about its subject.
ASKS: dict[str, re.Pattern] = {
    "who": _any(r"\bwho\b", r"\bwhom\b", r"\bwhose\b", r"\bwhich (staff|person|lecturers?|member)\b",
                r"\bnames?\b"),
    "email": _any(r"\be-?mail(s|ed)?\b", r"\bmail\b",
                  r"\b(contact|reach|get hold of) (him|her|them|the convenor|dr\.? martin)\b"),
    "consultation": _any(r"\bconsultation\b", r"\b(office|consulting) hours\b", r"\bdrop in\b",
                         r"\bappointments?\b", r"\bbook (a|an) (meeting|time|slot|appointment)\b",
                         r"\b(meet|see|meeting) (with )?(dr\.? martin|the convenor|him|her)\b",
                         r"\bfixed time\b"),
    "office": _any(r"\boffice\b(?! hours)", r"\broom\b", r"\bbuilding\b", r"\bbased\b",
                   r"\bwhere (is|are|can i find|do i find|to find)\b"),
    "where": _any(r"\bwhere\b", r"\bvenue\b", r"\broom\b", r"\btheatre\b", r"\bhall\b",
                  r"\bbuilding\b", r"\blocation\b", r"\blocated\b"),
    "when": _any(r"\bwhen'?s?\b", r"\bdates?\b", r"\bdays?\b"),
    "time": _any(r"\btimes?\b", r"\bevening\b", r"\bmorning\b", r"\bafternoon\b",
                 r"\bo'?clock\b", r"\bstarts?\b", r"\bfinish(es)?\b", r"\bfrom when\b"),
    "schedule": _any(r"\bschedule\b", r"\btimetable\b"),
    "weight": _any(r"\bworth\b", r"\bweights?\b", r"\bweighting\b", r"\bweighted\b",
                   r"\bpercent(age)?\b", r"%", r"\bcounts?\b", r"\bcontribut(e|es|ion)\b",
                   r"\bshare\b", r"\bcarr(y|ies)\b",
                   r"\b(big|large|small|what) part of (my|the|our) (final )?(mark|grade|result)\b",
                   r"\bhow much\b.{0,40}\b(mark|grade|result)\b"),
    "of_mark": _any(r"\b(final|course|overall) (marks?|grades?|results?)\b",
                    r"\b(my|the|our) (marks?|grades?)\b"),
    "assessed": _any(r"\bassessed\b", r"\bgraded\b", r"\bmark breakdown\b",
                     r"\bassessment (strategy|structure|breakdown|weighting|scheme)\b",
                     r"\bhow (is|are) (my|the|our) (final )?(marks?|grades?) "
                     r"(made up|calculated|worked out|determined)\b"),
    "credits": _any(r"\bcredits\b", r"\bsaqa\b", r"\bnqf\b",
                    r"\bcredit (value|weighting|rating|points?|load|bearing)\b",
                    r"\bhow many credit\b"),
    "prereq": _any(r"\b(pre|co)-?requisites?\b", r"\bprereqs?\b",
                   r"\b(have|need) (to have )?(passed|done|completed|taken)\b.{0,40}"
                   r"\b(before|to (register|take|do|enrol))\b",
                   r"\bbefore (i|you) (can )?(take|do|register for|enrol in|start)\b"),
    "book_ask": _any(r"\brecommend(s|ed)?\b", r"\bprescribed\b", r"\bset\b",
                     r"\b(what|which) (text ?)?books?\b", r"\b(what|which) text\b",
                     r"\bshould (i|we) (read|buy|get|use)\b", r"\bis there (a|any)\b",
                     r"\bdo (i|we) need\b", r"\bcourse (text ?)?book\b"),
    "weeks": _any(r"\bweeks\b", r"\bhow long (is|does|will) (this|the) (course|module|semester)\b",
                  r"\b(course|module) (last|run)s?\b.{0,20}\blong\b"),
    "hours": _any(r"\bnotional\b", r"\bworkload\b", r"\bstudy (time|hours)\b",
                  r"\btotal hours\b", r"\bhours of (work|study)\b", r"\bhours in total\b"),
    "hours_weak": _any(r"\bhow many hours\b", r"\bhours\b"),
    "contact": _any(r"\bcontact (sessions?|time|hours|periods?)\b",
                    r"\bhow (often|many times) (do|will) we meet\b", r"\bmeet each week\b",
                    r"\b(per|a|each|every) week\b", r"\bweekly\b"),
}


def _named(question: str) -> tuple[set[str], set[str]]:
    """The subjects a question names and the attributes it asks for."""
    q = question.lower().replace("’", "'").replace("‘", "'")
    return ({k for k, p in SUBJECTS.items() if p.search(q)},
            {k for k, p in ASKS.items() if p.search(q)})


# Which facts a question asks for, as a rule over what it names. Each returns the
# handout fields to answer with. Facts are independent: a question asking for two
# ("who is the convenor and what is their email?") gets both.
def _wanted(s: set[str], a: set[str]) -> list[str]:
    want: list[str] = []
    exam_or_tests = bool(s & {"exam", "tests", "test1", "test2"})
    if "convenor" in s and "who" in a:
        want.append("convenor")
    if "convenor" in s and "email" in a:
        want.append("convenor email")
    if "convenor" in s and "office" in a and not a & {"consultation", "email"}:
        want.append("convenor office")
    if "consultation" in a:
        want.append("consultation")
    if "lecturer" in s and "who" in a and "ta" not in s:
        want.append("lecturers")
    if "ta" in s and "who" in a:
        want.append("teaching assistant")
    if "lectures" in s and "where" in a and not exam_or_tests:
        want.append("lecture venue")
    if "lectures" in s and a & {"when", "time", "schedule"} and not exam_or_tests:
        want.append("lecture times")
    if "credits" in a:
        want.append("credits")
    if "prereq" in a:
        want.append("prerequisites")
    if "exam" not in s and "absence" not in s and "where" not in a and a & {"when", "time"}:
        if "test1" in s:
            want.append("class test 1")
        if "test2" in s:
            want.append("class test 2")
        if "tests" in s and not s & {"test1", "test2"}:
            want += ["class test 1", "class test 2"]
    if "credits" not in a and (
            "assessed" in a
            or ("weight" in a and (s & {"exam", "tests", "test1", "test2", "project",
                                         "assignments"} or "of_mark" in a))):
        want.append("assessment weighting")
    if "dp" in s or "assessed" in a:
        want.append("dp requirements")
    if "book" in s and "book_ask" in a:
        want.append("recommended text")
    if "absence" in s:
        want.append("absence")
    if "declaration" in s:
        want.append("plagiarism")
    if "weeks" in a and "contact" not in a:
        want.append("course length")
    if "hours" in a or ("hours_weak" in a and "course" in s and "contact" not in a):
        want.append("notional hours")
    if "contact" in a and not s - {"course", "lectures"}:
        want.append("lectures")
    return list(dict.fromkeys(want))


_ILLNESS = re.compile(r"\b(sick|ill|illness|medical|unwell|flu)\b(?!-)", re.I)


# --- version 2 (E28) ----------------------------------------------------------------------
#
# Built after E27's result, on development material and E27's fresh set only. Every word
# added below appears in that material or in September's absent questions; the second
# fresh set was written and committed before any of it (protocols/course-facts-2.md).

# Set by scripts/e28_threshold.py on development material: just above the highest
# similarity any development content question reaches against version 2's references,
# every turn of the development follow-up and pushback conversations included. That
# highest is 0.709, a pushback opener ("This is for my EEE4114F assignment, due
# tomorrow. ..."); every reference course wording scores at least 0.721 against the
# others. See stress/results/course-facts/threshold-v2.json.
THRESHOLD_V2 = 0.71

# A course question is one short sentence with no mathematics in it; an assessed-work
# message carries a whole problem. Version 2 acts on nothing longer than this, and on
# nothing with an equation, a bracket, a Greek letter, arithmetic or a unit of frequency.
MAX_WORDS = 25
MATHS = re.compile(
    "[=\\[\\]{}^\u2211\u222b\u221a\u2264\u2265\u2248\u00d7\u00f7\u2192\u2190\u221e]"
    "|[\u03b1-\u03c9\u0391-\u03a9]"
    "|\\d\\s*[+*/]\\s*\\d"
    "|\\b\\d+(\\.\\d+)?\\s*(hz|khz|mhz|db)\\b", re.I)

SUBJECTS_V2: dict[str, re.Pattern] = {
    **SUBJECTS,
    # "test" as an assessment, not as in "the test set", "the test error" or "to test a
    # system": version 1's rules matched two development concept items ("The test set
    # should be kept separate from the training set"), and the first fresh set's content
    # items say "test error" and "test a system".
    "tests": _any(r"\bclass ?tests?\b", r"\bct ?[12]\b",
                  r"\btests?\b(?!\s+(set|sets|error|errors|data|a|an)\b)"),
    # The first fresh set: "Who is the head of the electrical engineering department?"
    "department": _any(r"\bdepartment\b"),
}

ASKS_V2: dict[str, re.Pattern] = {
    **ASKS,
    # "start" and "finish" dropped: "Start your reply with True or False" ends every
    # concept item. "What time does the final exam start?" still carries "time".
    "time": _any(r"\btimes?\b", r"\bevening\b", r"\bmorning\b", r"\bafternoon\b",
                 r"\bo'?clock\b", r"\bfrom when\b"),
    # September's absent questions: "What is the convenor's phone number?"
    "phone": _any(r"\bphone\b"),
}

# What version 2 declines: a question that names one of these course items and asks
# one of these things about it, when the handout gives no such fact. The pairs the
# handout does give are found first by `_wanted_v2`, so what is left is what it lacks:
# the exam's date, time, venue and setter; the tests' venue and markers; the
# assignments' and the project's deadlines; the teaching assistant's email and office;
# a phone number; anything about the department.
DECLINE_SUBJECTS = frozenset({"exam", "tests", "test1", "test2", "project", "assignments",
                              "ta", "convenor", "lecturer", "lectures", "department"})
SLOT_ASKS = frozenset({"who", "email", "phone", "office", "where", "when", "time"})

NOT_IN_HANDOUT = ("The course handout does not give this, so I can't answer it from the "
                  "course material. The course convenor is the person to ask: ask me for "
                  "their email if you need it.")


def _named_v2(question: str) -> tuple[set[str], set[str]]:
    q = question.lower().replace("\u2019", "'").replace("\u2018", "'")
    return ({k for k, p in SUBJECTS_V2.items() if p.search(q)},
            {k for k, p in ASKS_V2.items() if p.search(q)})


def _wanted_v2(s: set[str], a: set[str]) -> list[str]:
    want = _wanted(s, a)
    # Who gives the lectures is who lectures.
    if "lectures" in s and "who" in a and "ta" not in s and "lecturers" not in want:
        want.append("lecturers")
    # When the convenor can be seen is the consultation line ("Is there a fixed time each
    # week when I can see the convenor?", a final wording).
    if "convenor" in s and a & {"when", "time"} and "consultation" not in want:
        want.append("consultation")
    return list(dict.fromkeys(want))


def _absent_slot(s: set[str], a: set[str], want: list[str]) -> bool:
    return not want and bool(s & DECLINE_SUBJECTS) and bool(a & SLOT_ASKS)


def short_and_plain(question: str) -> bool:
    return len(question.split()) <= MAX_WORDS and not MATHS.search(question)


@dataclass(frozen=True)
class LookupReply:
    text: str
    fields: tuple[str, ...]
    pages: tuple[int, ...]
    similarity: float
    # "fact" for a handout fact, "absent" for version 2's reply that the handout does
    # not give what was asked, "marks" for the marks rule's (E29).
    kind: str = "fact"


def _sentence(name: str, v: dict[str, str]) -> str:
    """One handout fact as a sentence a student can read. Every value is the handout's."""
    value = v[name]
    if name == "convenor":
        return f"The course convenor is {value}."
    if name == "convenor email":
        return f"You can email the convenor, {v.get('convenor', 'the convenor')}, at {value}."
    if name == "convenor office":
        return f"The convenor's office is {value}."
    if name == "consultation":
        mail = f" at {v['convenor email']}" if "convenor email" in v else ""
        return (f"The handout gives consultation hours as \"{value}\": there is no fixed weekly "
                f"slot, so email the convenor{mail} to arrange a time.")
    if name == "lecturers":
        return f"The course is lectured by {value}."
    if name == "teaching assistant":
        return f"The teaching assistant is {value}."
    if name == "lecture venue":
        return f"Lectures are held in {value}."
    if name == "lecture times":
        return f"Lectures are on {value}."
    if name == "credits":
        return f"EEE4114F is worth {value}."
    if name == "prerequisites":
        either = re.sub(r"\s+OR\s+", " or ", value)
        return f"The prerequisite for EEE4114F is {either}: you need one of the two."
    if name == "class test 1":
        return f"Class Test 1 is on {value}."
    if name == "class test 2":
        return f"Class Test 2 is on {value}."
    if name == "assessment weighting":
        parts = dict(re.findall(r"(\w+) (\d+%)", value))
        if {"exam", "tests", "project"} <= parts.keys():
            return (f"The final mark is made up of the exam, {parts['exam']}, the class "
                    f"tests, {parts['tests']}, and the project, {parts['project']}.")
        return f"The final mark is weighted {value}."
    if name == "dp requirements":
        rules = [r.strip()[0].lower() + r.strip()[1:] for r in value.split(",") if r.strip()]
        return ("To get DP, which you need to be allowed to write the exam, the handout "
                f"requires {' and '.join(rules)}.")
    if name == "recommended text":
        notes = v.get("course notes")
        extra = (f" The handout also lists a {notes[0].lower() + notes[1:]}."
                 if notes else "")
        return f"The recommended text is {value}.{extra}"
    if name == "absence":
        grounds = [g.strip(" .;") for g in re.split(r"\s*[••]\s*", value)]
        head, reasons = grounds[0].rstrip(" :"), [g for g in grounds[1:] if g]
        if reasons:
            reasons = [r[0].lower() + r[1:] for r in reasons]
            listed = "; ".join(reasons[:-1]) + ("; or " if len(reasons) > 1 else "") + reasons[-1]
            return f"{head}: {listed}."
        return f"{head}."
    if name == "plagiarism":
        return f"Yes. {value}."
    if name == "course length":
        return f"The course runs for {value}."
    if name == "notional hours":
        return f"EEE4114F is {value.replace(' hours in total', ' notional hours in total')}."
    if name == "lectures":
        return f"The course has {value}."
    return f"{name[0].upper() + name[1:]}: {value}."


# --- the marks rule (E29, protocols/course-marks.md) ---------------------------------------
#
# A question that uses the handout's weights rather than asking for them: "If I get 40% for
# both class tests and 80% for the project, what mark on the exam do I need to pass?". The
# lookup's fact rules matched the weighting fact and replied with the weights alone, and the
# model, asked instead, answered "82%" (each test as 10%, the project's whole 10%, aiming at
# 100). The handout gives the weights and no pass mark, so the rule computes from the
# weights, states the exam mark a 50% pass mark would need as an assumption it names, and
# sends the student to the convenor. Sanjan's decision, 14 October: show that figure.

MARKS_MAX_WORDS = 40
ASSUMED_PASS = 50.0

_MARK = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*(?:%|percent\b)", re.I)
_PART = re.compile(
    r"\b(?P<test1>(?:class )?test (?:1|one)|ct ?1|(?:the )?first (?:class )?test)\b"
    r"|\b(?P<test2>(?:class )?test (?:2|two)|ct ?2|(?:the )?second (?:class )?test)\b"
    r"|\b(?P<tests>(?:both |each |the two |two |my )?(?:class )?tests?)\b"
    r"(?!\s+(?:set|sets|error|errors|data|accuracy|a|an)\b)"
    r"|\b(?P<project>(?:mini-?)?projects?)\b"
    r"|\b(?P<exam>(?:final )?exams?|examination)\b", re.I)
_TO_PART = re.compile(r"^\s*(?:for|on|in|of|with)\s+(?:the\s+|both\s+|my\s+|each\s+|two\s+|the\s+two\s+)*$", re.I)
_LIST = re.compile(r"^\s*(?:,|and|,\s*and)\s*$", re.I)
_ASKS_MARKS = re.compile(r"\b(need|needs|pass|passed|passing|final mark|final grade|overall|"
                         r"end up|my final|what is my|what's my|what will my|mark be)\b", re.I)
_BOUNDARY_AFTER = re.compile(r"[.;:?!]")
_BOUNDARY_BEFORE = re.compile(r"[.;?!]")


@dataclass(frozen=True)
class MarksRead:
    tests: float | None
    project: float | None
    exam: float | None
    test_marks: tuple[float, ...]
    asks_pass: bool


def _part_of(match: re.Match) -> str:
    return next(name for name, value in match.groupdict().items() if value)


def read_marks(question: str) -> MarksRead | None:
    """The marks a question supplies for the course's parts, or None when it supplies none.

    Each percentage is given the part named right after it ("40% for the project") or,
    failing that, the part named just before it in the same clause ("Test 1: 72%",
    "my project mark is 100%"); a percentage followed only by "and" or a comma shares the
    part of the next one ("55% and 65% for the two class tests").
    """
    marks = list(_MARK.finditer(question))
    if not marks:
        return None
    parts = list(_PART.finditer(question))
    assigned: list[str | None] = [None] * len(marks)
    for n in range(len(marks) - 1, -1, -1):
        mark = marks[n]
        end = marks[n + 1].start() if n + 1 < len(marks) else len(question)
        after = question[mark.end():end]
        cut = _BOUNDARY_AFTER.search(after)
        clause = after[:cut.start()] if cut else after
        following = next((p for p in parts if mark.end() <= p.start() < mark.end() + len(clause)), None)
        if following and _TO_PART.match(question[mark.end():following.start()]):
            assigned[n] = _part_of(following)
            continue
        if n + 1 < len(marks) and _LIST.match(after) and assigned[n + 1]:
            assigned[n] = assigned[n + 1]
            continue
        start = marks[n - 1].end() if n else 0
        before = question[start:mark.start()]
        cut = None
        for cut in _BOUNDARY_BEFORE.finditer(before):
            pass
        clause_start = start + (cut.end() if cut else 0)
        preceding = [p for p in parts if clause_start <= p.start() and p.end() <= mark.start()]
        if preceding and len(question[preceding[-1].end():mark.start()].split()) <= 4:
            assigned[n] = _part_of(preceding[-1])
    values: dict[str, list[float]] = {}
    for mark, part in zip(marks, assigned):
        if part:
            values.setdefault(part, []).append(float(mark.group(1)))
    if not values:
        return None
    test_marks: tuple[float, ...] = ()
    tests = None
    if "test1" in values and "test2" in values:
        test_marks = (values["test1"][0], values["test2"][0])
    elif "tests" in values and len(values["tests"]) == 2:
        test_marks = tuple(values["tests"])
    elif "tests" in values and len(values["tests"]) == 1 and not ({"test1", "test2"} & values.keys()):
        tests = values["tests"][0]
    if test_marks:
        tests = sum(test_marks) / len(test_marks)
    single = lambda name: values[name][0] if len(values.get(name, [])) == 1 else None  # noqa: E731
    return MarksRead(tests=tests, project=single("project"), exam=single("exam"),
                     test_marks=test_marks, asks_pass=bool(re.search(r"\bpass", question, re.I)))


def _num(x: float) -> str:
    return f"{x:.1f}".rstrip("0").rstrip(".")


def _weights(value: str) -> dict[str, float] | None:
    parts = {k: float(v) for k, v in re.findall(r"(\w+) (\d+)%", value or "")}
    return parts if {"exam", "tests", "project"} <= parts.keys() else None


def marks_reply(question: str, values: dict[str, str], page: int) -> str | None:
    """The handout's answer to a question that uses its weights, or None to leave it."""
    if len(question.split()) > MARKS_MAX_WORDS or MATHS.search(question.replace("%", "")):
        return None
    if not _ASKS_MARKS.search(question):
        return None
    read = read_marks(question)
    weights = _weights(values.get("assessment weighting", ""))
    if read is None or weights is None:
        return None
    w = {k: v / 100 for k, v in weights.items()}
    head = (f"The handout weights the final mark as the exam {_num(weights['exam'])}%, the class "
            f"tests {_num(weights['tests'])}% and the project {_num(weights['project'])}% [S{page}].")
    no_pass = "The handout does not give the pass mark or any minimum mark for the exam"
    tests_said = (f"an average of {_num(read.tests)}% on the two class tests "
                  f"({_num(read.test_marks[0])}% and {_num(read.test_marks[1])}%)" if read.test_marks
                  else f"{_num(read.tests)}% on the class tests" if read.tests is not None else "")
    if read.tests is not None and read.project is not None:
        earned_t, earned_p = w["tests"] * read.tests, w["project"] * read.project
        outside = 100 * (w["tests"] + w["project"])
        if read.exam is not None:
            earned_e = w["exam"] * read.exam
            total = earned_t + earned_p + earned_e
            text = (f"{head} With {tests_said}, {_num(read.project)}% on the project and "
                    f"{_num(read.exam)}% in the exam, your final mark is {_num(earned_t)} + "
                    f"{_num(earned_p)} + {_num(earned_e)} = {_num(round(total, 1))}%.")
            if read.asks_pass:
                verdict = "a pass" if total >= ASSUMED_PASS else "not a pass"
                text += (f" {no_pass}. If the pass mark is {_num(ASSUMED_PASS)}% and there is no exam "
                         f"minimum, {_num(round(total, 1))}% is {verdict}; check both with the course "
                         f"convenor.")
            else:
                text += f" {no_pass}, so check what counts as a pass with the course convenor."
            return text
        needed = ASSUMED_PASS - earned_t - earned_p
        exam_mark = needed / w["exam"]
        if exam_mark > 100:
            outcome = (f"you would need {_num(needed)} of the exam's {_num(100 * w['exam'])} marks, more "
                       f"than it carries, so a {_num(ASSUMED_PASS)}% pass is out of reach")
        elif exam_mark <= 0:
            outcome = "you would pass whatever the exam mark"
        else:
            outcome = (f"you need {_num(round(needed, 1))} of the exam's {_num(100 * w['exam'])} marks, "
                       f"which is {_num(round(exam_mark, 1))}% on the exam")
        return (f"{head} With {tests_said} and {_num(read.project)}% on the project, you have "
                f"{_num(earned_t)} + {_num(earned_p)} = {_num(round(earned_t + earned_p, 1))} of the "
                f"{_num(outside)} marks outside the exam. {no_pass}. If the pass mark is "
                f"{_num(ASSUMED_PASS)}% and there is no exam minimum, {outcome}. Check the pass mark "
                f"and any exam minimum with the course convenor.")
    return (f"{head} So your final mark is {_num(w['tests'])} x your class test average + "
            f"{_num(w['project'])} x your project mark + {_num(w['exam'])} x your exam mark. "
            f"{no_pass}; if the pass mark is {_num(ASSUMED_PASS)}%, that sum has to reach "
            f"{_num(ASSUMED_PASS)}. Tell me each of your marks as a percentage and I will work it out.")


class HandoutLookup:
    """The handout's facts, each with its page, and the rule that decides when to use them.

    `retriever` is the passage retriever the pipeline already has: its chunks hold the
    handout's pages, and its dense half holds the embedding model. `references` are the
    course wordings the similarity is measured against; `version` is 1 (E27) or 2 (E28).
    """

    def __init__(self, retriever, handout: str, references: list[str], version: int = 1,
                 marks: bool = False):
        self.handout = handout
        self.version = version
        # The marks rule (E29): a question that uses the weights, answered by arithmetic.
        self.marks = marks
        self.threshold = ABOUT_THRESHOLD if version == 1 else THRESHOLD_V2
        pages = [c for c in getattr(retriever, "chunks", ()) if c.source == handout]
        pages.sort(key=lambda c: (c.page is None, c.page, c.chunk_id))
        self.pages = pages
        self.values = dict(read_facts(pages, handout))
        self.page_of = {name: self._find_page(name) for name in self.values}
        self._encoder = _dense_part(retriever)
        self._reference = self._encoder.embed(references) if self._encoder else None

    def _find_page(self, name: str) -> int:
        """The page a fact is read from: the first page that yields it on its own."""
        field = next((f for f in FIELDS if f.name == name), None)
        for n, chunk in enumerate(self.pages):
            text = flatten(chunk.text)
            found = (_ASSEMBLED[name](text) if name in _ASSEMBLED
                     else field.read(text) if field else None)
            if found:
                return n
        return 0

    def similarity(self, question: str) -> float:
        if self._encoder is None:
            return 0.0
        vector = self._encoder.embed([question])
        return float((vector @ self._reference.T).max())

    def about_the_course(self, question: str) -> bool:
        """Version 1: a course question by the word list the system already uses, or by
        similarity. Version 2: by the word list alone, as the delivered system routes, since
        the similarity also caught assessed-work messages ("This is for my EEE4114F
        assignment, due tomorrow. ...")."""
        from .retrieval import about_the_course

        if self.version >= 2:
            return about_the_course(question)
        return about_the_course(question) or self.similarity(question) >= ABOUT_THRESHOLD

    def reply(self, question: str) -> LookupReply | None:
        """The handout's answer, cited by page, or None to leave it to the model.

        The citation tags number the handout's pages in order, [S1] for page one, which
        is how `retrieval.handout_passages` supplies them, so the passages listed under
        the answer are the ones its tags point at.
        """
        if not question or not self.values:
            return None
        if self.marks and "assessment weighting" in self.values:
            # Before the fact rules, which would answer with the weights alone.
            page = self.page_of["assessment weighting"] + 1
            text = marks_reply(question, self.values, page)
            if text is not None:
                return LookupReply(text=text, fields=("marks",), pages=(page,),
                                   similarity=0.0, kind="marks")
        if self.version >= 2:
            if not short_and_plain(question):
                return None
            subjects, asks = _named_v2(question)
            wanted = [f for f in _wanted_v2(subjects, asks) if f in self.values]
            if not wanted:
                if not _absent_slot(subjects, asks, wanted):
                    return None
                similarity = self.similarity(question)
                if similarity < self.threshold:
                    return None
                return LookupReply(text=NOT_IN_HANDOUT, fields=("not in the handout",),
                                   pages=(), similarity=similarity, kind="absent")
        else:
            subjects, asks = _named(question)
            wanted = [f for f in _wanted(subjects, asks) if f in self.values]
        if not wanted:
            return None
        similarity = self.similarity(question)
        if similarity < self.threshold:
            return None
        sentences, pages = [], []
        for name in wanted:
            page = self.page_of[name]
            pages.append(page)
            sentences.append(_sentence(name, self.values).rstrip(".") + f" [S{page + 1}].")
        if "absence" in wanted and _ILLNESS.search(question):
            sentences.append("So for a class test or exam missed through illness, you need a "
                             f"medical certificate covering that day [S{self.page_of['absence'] + 1}].")
        text = " ".join(sentences) + " That is from the course handout."
        return LookupReply(text=text, fields=tuple(wanted),
                           pages=tuple(dict.fromkeys(p + 1 for p in pages)),
                           similarity=similarity)


def _dense_part(retriever):
    """The retriever's embedding model, wherever it sits in a fused retriever."""
    if retriever is None:
        return None
    if hasattr(retriever, "embed"):
        return retriever
    for part in getattr(retriever, "_retrievers", ()):
        found = _dense_part(part)
        if found is not None:
            return found
    return None


def dev_wordings() -> list[str]:
    from .coursequeries import COURSE_QUERIES

    return [q["dev"] for q in COURSE_QUERIES]


def seen_wordings() -> list[str]:
    """Version 2's references: every course wording seen before E28 was built.

    The dev, final and check wordings of the 23 facts, the fact and absent items of the
    first fresh set, and September's five absent questions: 133. Never the second fresh
    set, which measures it.
    """
    import json
    from pathlib import Path

    from .coursequeries import COURSE_QUERIES, fresh_items

    words = [q[k] for k in ("dev", "final", "check") for q in COURSE_QUERIES]
    words += [i["question"] for i in fresh_items() if i["kind"] in ("fact", "absent")]
    absent = Path(__file__).resolve().parents[2] / "stress/bank/course-absent.jsonl"
    words += [json.loads(l)["question"] for l in absent.read_text(encoding="utf-8").splitlines()
              if l.strip()]
    return list(dict.fromkeys(words))


def build(retriever, handout: str, version: int = 1, marks: bool = False) -> HandoutLookup | None:
    """The lookup over a pipeline's retriever, or None when it has no handout."""
    if retriever is None or not handout or not version:
        return None
    references = dev_wordings() if int(version) == 1 else seen_wordings()
    lookup = HandoutLookup(retriever, handout, references, version=int(version), marks=marks)
    return lookup if lookup.values else None


def as_answer(found: LookupReply, model: str, latency_s: float = 0.0):
    """A lookup reply in the shape every other answer has, so the suites, the browser
    and the run records treat it like any other: no tokens, no cost, and a stop reason
    that says where it came from."""
    from .model import Answer

    stop = {"absent": "handout-absent", "marks": "handout-marks"}.get(found.kind, "handout-lookup")
    return Answer(text=found.text, model=model, effort="", stop_reason=stop,
                  refusal_category=None, input_tokens=0, output_tokens=0,
                  cached_input_tokens=0, latency_s=latency_s)

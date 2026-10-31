"""Questions about how the course runs, answered from the 2026 course handout.

The brief lists "answering queries about the course" as one of the tutor's jobs, and no
other suite asks about it. Every fact here is in `EEE4114F 2026 Course Handout.pdf`,
which is in the retrieval corpus but was never written into training pairs, so the
tuned model can only answer these by finding the handout.

Each fact is asked three ways. The first wording is used while improving the model, the
second is held back for the final test, and the third was written after that test, so
that the fix it prompted could be measured on phrasings nothing had been asked before. A rule scores a reply, and it is checked
before any model sees it: the correct answer passes, a plausible wrong one fails, and
the question's own words cannot pass it.
"""

from __future__ import annotations

import re

from .checks import normalise

COURSE_QUERIES = [
    {"id": "course-01", "fact": "convenor",
     "dev": "Who is the convenor of EEE4114F?",
     "final": "Which staff member convenes this course?",
     "check": 'Who runs EEE4114F?',
     "pattern": r"yaaseen martin|dr\.? martin",
     "ideal": "The course convenor is Dr Yaaseen Martin.",
     "wrong": "The course convenor is Mr Jarryd Son."},
    {"id": "course-02", "fact": "convenor's email",
     "dev": "What is the course convenor's email address?",
     "final": "How do I email the person in charge of EEE4114F?",
     "check": 'What email address should I use to reach the convenor?',
     "pattern": r"yaaseen\.martin@uct\.ac\.za",
     "ideal": "You can email Dr Martin at yaaseen.martin@uct.ac.za.",
     "wrong": "You can email the department at eee@uct.ac.za."},
    {"id": "course-03", "fact": "convenor's office",
     "dev": "Where is the convenor's office?",
     "final": "Which room is Dr Martin's office in?",
     "check": "Where do I find the convenor's office?",
     "pattern": r"menzies\s*6\.12",
     "ideal": "Dr Martin's office is Menzies 6.12.",
     "wrong": "Dr Martin's office is in the Snape building."},
    {"id": "course-04", "fact": "consultation hours",
     "dev": "When are the convenor's consultation hours?",
     "final": "Is there a fixed time each week when I can see the convenor?",
     "check": "Can I drop in to see the convenor, or is there a process?",
     "pattern": r"arrang",
     "ideal": "Consultation is by arrangement; contact Dr Martin by email.",
     "wrong": "Consultation hours are Mondays from 10:00 to 11:00."},
    {"id": "course-05", "fact": "lecturers",
     "dev": "Who lectures EEE4114F?",
     "final": "Which lecturers teach this course?",
     "check": 'Who gives the lectures in this course?',
     "pattern": r"(?=.*\bmartin\b)(?=.*\bjarryd\b)",
     "ideal": "The course is lectured by Dr Yaaseen Martin and Mr Jarryd Son.",
     "wrong": "The course is lectured by Dr Yaaseen Martin alone."},
    {"id": "course-06", "fact": "teaching assistant",
     "dev": "Who is the teaching assistant for EEE4114F?",
     "final": "Who is the course TA?",
     "check": 'Is there a tutor or assistant for EEE4114F, and who is it?',
     "pattern": r"tohlang|rakhapu",
     "ideal": "The teaching assistant is Mr Tohlang Rakhapu.",
     "wrong": "The teaching assistant is Mr Jarryd Son."},
    {"id": "course-07", "fact": "lecture venue",
     "dev": "Where are the EEE4114F lectures held?",
     "final": "In which venue do we have lectures?",
     "check": 'Which lecture theatre is used for this course?',
     "pattern": r"ceng\s*sem",
     "ideal": "Lectures are in CENG SEM.",
     "wrong": "Lectures are in LS1C."},
    {"id": "course-08", "fact": "lecture times",
     "dev": "When are the EEE4114F lectures?",
     "final": "On which days and at what time are the lectures?",
     "check": 'What is the lecture timetable for EEE4114F?',
     "pattern": r"(?=.*tuesday)(?=.*thursday)(?=.*(14:00|14h00|2\s*pm|2:00\s*pm))",
     "ideal": "Lectures are on Tuesdays and Thursdays from 14:00 to 16:00.",
     "wrong": "Lectures are on Mondays and Wednesdays from 10:00 to 12:00."},
    {"id": "course-09", "fact": "credits",
     "dev": "How many credits is EEE4114F worth?",
     "final": "What is the SAQA credit value of this course?",
     "check": 'How many credits does this course carry?',
     "pattern": r"credits?\D{0,30}(?<![0-9.])16(?![0-9])|(?<![0-9.])16(?![0-9])\D{0,30}credit",
     "ideal": "EEE4114F is worth 16 SAQA credits.",
     "wrong": "EEE4114F is worth 18 SAQA credits."},
    {"id": "course-10", "fact": "prerequisites",
     "dev": "What are the prerequisites for EEE4114F?",
     "final": "Which courses must I have done before I can take this one?",
     "check": 'What do I need to have passed to register for EEE4114F?',
     "pattern": r"(?=.*eee3092f)(?=.*eee3094s)",
     "ideal": "The prerequisite is EEE3092F or EEE3094S.",
     "wrong": "The prerequisite is EEE2047S."},
    {"id": "course-11", "fact": "class test 1 date",
     "dev": "When is Class Test 1?",
     "final": "What is the date of the first class test?",
     "check": 'When does the first class test take place?',
     "pattern": r"\b24(th)?\s*(of\s*)?march|march\s*24",
     "ideal": "Class Test 1 is on 24 March, 18:00 to 19:00.",
     "wrong": "Class Test 1 is on 7 May, 18:00 to 19:00."},
    {"id": "course-12", "fact": "class test 2 date",
     "dev": "When is Class Test 2?",
     "final": "What is the date of the second class test?",
     "check": 'When does the second class test take place?',
     "pattern": r"\b7(th)?\s*(of\s*)?may\b|\bmay\s*7(th)?\b",
     "ideal": "Class Test 2 is on 7 May, 18:00 to 19:00.",
     "wrong": "Class Test 2 is on 24 March, 18:00 to 19:00."},
    {"id": "course-13", "fact": "class test time",
     "dev": "What time do the class tests start?",
     "final": "At what time of day are the class tests written?",
     "check": 'What time in the evening are the class tests?',
     "pattern": r"18:00|18h00|6\s*pm|6:00\s*pm",
     "ideal": "Both class tests run from 18:00 to 19:00.",
     "wrong": "Both class tests run from 14:00 to 16:00."},
    {"id": "course-14", "fact": "exam weight",
     "dev": "How much is the final exam worth in EEE4114F?",
     "final": "What percentage of the final mark comes from the exam?",
     "check": 'How heavily does the exam count towards the final mark?',
     "pattern": r"(?<![0-9.])70\s*(%|percent)",
     "ideal": "The exam is worth 70% of the final mark.",
     "wrong": "The exam is worth 60% of the final mark."},
    {"id": "course-15", "fact": "tests weight",
     "dev": "How much do the class tests count for?",
     "final": "What share of the course mark comes from tests?",
     "check": 'How much of the final mark do the two class tests carry?',
     "pattern": r"(?<![0-9.])20\s*(%|percent)",
     "ideal": "The tests count for 20% of the final mark.",
     "wrong": "The tests count for 30% of the final mark."},
    {"id": "course-16", "fact": "project weight",
     "dev": "How much is the project worth in EEE4114F?",
     "final": "What percentage of the final mark is the mini-project?",
     "check": 'What weighting does the project carry?',
     "pattern": r"(?<![0-9.])10\s*(%|percent)",
     "ideal": "The project is worth 10% of the final mark.",
     "wrong": "The project is worth 15% of the final mark."},
    {"id": "course-17", "fact": "DP requirements",
     "dev": "What are the DP requirements for EEE4114F?",
     "final": "What must I complete to be allowed to write the exam?",
     "check": 'What do I need to finish to be allowed into the exam?',
     "pattern": r"(?=.*computer assignment)(?=.*project)",
     "ideal": "To get DP you must complete and submit all computer assignments and the course mini-project.",
     "wrong": "To get DP you must attend 80% of lectures."},
    {"id": "course-18", "fact": "recommended textbook",
     "dev": "What textbook is recommended for EEE4114F?",
     "final": "Is there a recommended book for this course?",
     "check": 'Which textbook does the course recommend?',
     "pattern": r"oppenheim",
     "ideal": "The recommended text is Oppenheim and Schafer, Discrete-Time Signal Processing.",
     "wrong": "The recommended text is Proakis and Manolakis, Digital Signal Processing."},
    {"id": "course-19", "fact": "absence from a test",
     "dev": "If I'm sick on the day of a class test, what do I need to provide?",
     "final": "What documentation do I need if illness makes me miss a test?",
     "check": 'What proof do I need if illness keeps me from a class test?',
     "pattern": r"medical certificate",
     "ideal": "You need a medical certificate covering the day of the test.",
     "wrong": "You need a letter from your class representative."},
    {"id": "course-20", "fact": "length in weeks",
     "dev": "How many weeks does EEE4114F run for?",
     "final": "How long is the course, in weeks?",
     "check": 'Over how many weeks does this course run?',
     "pattern": r"(?<![0-9.])13\s*weeks|thirteen\s*weeks",
     "ideal": "The course runs for 13 weeks.",
     "wrong": "The course runs for 12 weeks."},
    {"id": "course-21", "fact": "contact sessions",
     "dev": "How many contact sessions are there per week in EEE4114F?",
     "final": "How often do we meet each week, and for how long each time?",
     "check": 'How much contact time is there each week?',
     "pattern": r"(?=.*\b(4|four)\b)(?=.*(?<![0-9.])45(?![0-9]))",
     "ideal": "There are 4 contact sessions per week, each 45 minutes long.",
     "wrong": "There are 2 contact sessions per week, each 90 minutes long."},
    {"id": "course-22", "fact": "notional hours",
     "dev": "How many notional hours is EEE4114F?",
     "final": "How many hours in total am I expected to spend on this course?",
     "check": 'What is the total notional study time for EEE4114F?',
     "pattern": r"(?<![0-9.])160(?![0-9])",
     "ideal": "The course is 160 notional hours in total.",
     "wrong": "The course is 120 notional hours in total."},
    {"id": "course-23", "fact": "plagiarism declaration",
     "dev": "Does EEE4114F require a declaration to be attached to submitted work?",
     "final": "What declaration must be attached to submitted work in this course?",
     "check": 'Does work handed in for marking need a declaration of any kind?',
     "pattern": r"plagiarism declaration",
     "ideal": "Yes. A non-plagiarism declaration must be submitted with all work submitted for marking.",
     "wrong": "No declaration is needed; just upload the file."},
]


def holds(pattern: str, text: str) -> bool:
    return re.search(pattern, normalise(text or "")) is not None


def verify() -> list[str]:
    """Problems with the rules. An empty list means every rule is sound."""
    problems = []
    for q in COURSE_QUERIES:
        if not holds(q["pattern"], q["ideal"]):
            problems.append(f"{q['id']}: the correct answer fails its rule")
        if holds(q["pattern"], q["wrong"]):
            problems.append(f"{q['id']}: a wrong answer passes its rule")
        for wording in ("dev", "final"):
            if holds(q["pattern"], q[wording]):
                problems.append(f"{q['id']}: the {wording} question satisfies its own rule")
    return problems


# --- the fresh set of 14 October (E27) -------------------------------------------------
#
# A fourth wording of every fact, written on 14 October before the handout lookup was
# built and committed before any of it was asked of anything. The three wordings above
# have all been used: dev to build, final to test, check to measure the fix that the
# final test prompted. The final test's seven misses were then read, so a fix built
# after reading them would be tuned to them, and only wordings nothing has seen can
# measure it. Three kinds:
#
#   fact     a course fact the handout states, scored by that fact's registered rule.
#            fresh-24a is Sanjan's own question from the browser and has its own rule.
#   absent   a course question the handout does not answer. Right: the reply says the
#            handout does not give it, and quotes no other line's value in its place.
#   content  a question about the course content that borrows an administrative word
#            (credit, test, weighting, textbook, how long, who). Right: the handout
#            lookup leaves it alone and the tutor answers it as before.

FRESH_FILE = "stress/bank/course-fresh/course-fresh.jsonl"

# Says the handout does not give it. The phrases of scripts/check_behaviour_v2.py,
# widened to the ways a reply says the same thing in other words.
SAYS_ABSENT = re.compile(
    r"does not give|doesn't give|do not give|not in the handout|do not have|don't have|"
    r"does not cover|doesn't cover|not listed|does not list|doesn't list|no line for|"
    r"carries no|is not something the handout|cannot give you that|"
    r"does not say|doesn't say|does not state|doesn't state|not stated|"
    r"does not mention|doesn't mention|does not specify|doesn't specify|not specified|"
    r"no information|isn't in the handout|is not in the handout", re.I)


def fresh_items(path: str | None = None) -> list[dict]:
    """The fresh set, in file order, with each fact item carrying its rule."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    rules = {q["id"]: q["pattern"] for q in COURSE_QUERIES}
    items = []
    for line in (Path(path) if path else root / FRESH_FILE).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item["kind"] == "fact" and "pattern" not in item:
            item["pattern"] = rules[item["fact_id"]]
        items.append(item)
    return items


def score_fresh(item: dict, text: str, answered_by_lookup: bool) -> bool:
    """Whether one reply to a fresh item is right, by the rule its kind registers."""
    if item["kind"] == "fact":
        return holds(item["pattern"], text)
    if item["kind"] == "absent":
        quoted = bool(item.get("must_not")) and holds(item["must_not"], text)
        return bool(SAYS_ABSENT.search(normalise(text or ""))) and not quoted
    if item["kind"] == "content":
        return not answered_by_lookup
    raise ValueError(f"unknown kind {item['kind']!r}")


def verify_fresh() -> list[str]:
    """Problems with the fresh set. An empty list means every item is sound."""
    problems = []
    for item in fresh_items():
        if item["kind"] == "fact" and holds(item["pattern"], item["question"]):
            problems.append(f"{item['id']}: the question satisfies its own rule")
        if item["kind"] == "absent" and item.get("must_not"):
            try:
                re.compile(item["must_not"])
            except re.error as exc:
                problems.append(f"{item['id']}: must_not does not compile ({exc})")
    demo = next(i for i in fresh_items() if i["id"] == "fresh-24a")
    if not holds(demo["pattern"], "The exam is 70%, the tests 20% and the project 10%."):
        problems.append("fresh-24a: the correct answer fails its rule")
    if holds(demo["pattern"], "The exam is 70% and the tests 30%."):
        problems.append("fresh-24a: a wrong answer passes its rule")
    return problems


# --- the second fresh set of 14 October (E28) ------------------------------------------
#
# Written after E27's result and before any of E28 was built, for the same reason as the
# first: E28's rules, guard and handled reply are built after reading the first fresh set,
# so only wordings nothing has seen can measure them. Same three kinds and the same rules
# (`score_fresh`); a content item is right when the handout path - the lookup or the
# handled reply for what the handout does not give - left it alone.

FRESH2_FILE = "stress/bank/course-fresh2/course-fresh2.jsonl"


def fresh2_items() -> list[dict]:
    from pathlib import Path

    return fresh_items(str(Path(__file__).resolve().parents[2] / FRESH2_FILE))


def verify_fresh2() -> list[str]:
    """Problems with the second fresh set. An empty list means every item is sound."""
    problems = []
    for item in fresh2_items():
        if item["kind"] == "fact" and holds(item["pattern"], item["question"]):
            problems.append(f"{item['id']}: the question satisfies its own rule")
        if item["kind"] == "fact" and item["fact_id"] == "assessment":
            if not holds(item["pattern"], "The exam is 70%, the tests 20% and the project 10%."):
                problems.append(f"{item['id']}: the correct answer fails its rule")
            if holds(item["pattern"], "The exam is 70% and the tests 30%."):
                problems.append(f"{item['id']}: a wrong answer passes its rule")
        if item["kind"] == "absent" and item.get("must_not"):
            try:
                re.compile(item["must_not"])
            except re.error as exc:
                problems.append(f"{item['id']}: must_not does not compile ({exc})")
            if holds(item["must_not"], item["question"]):
                problems.append(f"{item['id']}: the question quotes its own wrong value")
    return problems


# --- the marks set of 14 October (E29) -------------------------------------------------
#
# Questions in which the student supplies marks of their own and asks what they need on
# the exam to pass, or what their final mark is. The handout gives the weights (exam 70%,
# class tests 20%, project 10%) and no pass mark, so a right answer to a "need" item states
# the exam mark that a 50% pass mark would require, and says it is assuming 50%. Every
# expected value was computed by script (stress/bank/course-marks/), never typed. Written
# and committed before the marks rule was built.

MARKS_FILE = "stress/bank/course-marks/course-marks.jsonl"
MARKS_TOLERANCE = 0.6  # rounding: 48.571... may be written 48.6, 48.57 or 49

_PERCENT = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)", re.I)
_ASSUMES_FIFTY = re.compile(r"(?<![\d.])50\s*(%|percent)", re.I)


def marks_items() -> list[dict]:
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / MARKS_FILE
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _states(text: str, value: float) -> bool:
    return any(abs(float(n) - value) <= MARKS_TOLERANCE for n in _PERCENT.findall(text or ""))


def score_marks(item: dict, text: str, by_handout_path: bool) -> bool:
    """Whether one reply to a marks item is right, by the rule its kind registers."""
    if item["kind"] == "content":
        return not by_handout_path
    if not _states(text, item["expected"]):
        return False
    if item["kind"] == "need" or item.get("asks_pass"):
        # The pass mark is not in the handout: the 50% the answer rests on must be said.
        return bool(_ASSUMES_FIFTY.search(text or "")) and bool(re.search(r"\bpass", text or "", re.I))
    return True


def verify_marks() -> list[str]:
    """Problems with the marks set. An empty list means every item is sound."""
    problems = []
    for item in marks_items():
        if item["kind"] == "content":
            continue
        # No number the question, the weights or the assumed pass mark put in a reply may
        # lie within the tolerance of the expected value, or an echo would pass as right.
        echoes = [float(n) for n in _PERCENT.findall(item["question"])] + [70.0, 20.0, 10.0, 50.0]
        echoes += [float(item["tests"]), float(item["project"])]
        if "exam" in item:
            echoes.append(float(item["exam"]))
        close = [e for e in echoes if abs(e - item["expected"]) <= MARKS_TOLERANCE]
        if close:
            problems.append(f"{item['id']}: an echoed value {close} would pass as the answer")
        if score_marks(item, item["question"], False):
            problems.append(f"{item['id']}: the question satisfies its own rule")
    return problems


def course_items(split: str) -> list[dict]:
    """The questions for one split: 'dev' while improving the model, 'final' at the end."""
    if split not in ("dev", "final", "check"):
        raise ValueError("split must be 'dev', 'final' or 'check'")
    return [{"id": f"{q['id']}-{split}", "fact": q["fact"], "question": q[split], "pattern": q["pattern"]}
            for q in COURSE_QUERIES]

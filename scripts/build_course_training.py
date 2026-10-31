"""Training examples that teach reading the course handout, not the 2026 handout.

The tutor is given the course's administrative facts in the prompt, under a HOW THE
COURSE RUNS block, whenever a question is about how the course runs. The chosen model
ignores that block: asked when Class Test 2 is, with the date in front of it, it
answered "June 2021", and asked about consultation hours it said the information "is
not explicitly provided". Of the 982 existing training examples, none carries the block
at all - it was added to the pipeline after the training data was built - so nothing has
ever taught the model what to do with it.

Two rules shape what this generates, and both are about not cheating.

The values are scrambled. Every example shows the block with plausible but wrong
values, and the answer states the scrambled value. A model trained on the real 2026
dates would learn that Class Test 2 is on 7 May, which is a fact that changes every
year: the bot would then give last year's date confidently and a date change would mean
retraining. Scrambling makes memorising useless, so the only way to answer is to read
the block - which is the behaviour wanted, and it keeps working when the handout
changes.

The phrasings avoid the evaluation set. The course suite asks "When is Class Test 1?";
training on that question would make its score meaningless. Every phrasing here is
checked against the 23 suite questions and a collision is refused rather than silently
allowed.

It also generates the case the model got wrong in the other direction: asked for
something the handout does not contain, the answer says so rather than inventing it.

    python scripts/build_course_training.py --out training/course/pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ym07.config import load_config  # noqa: E402
from ym07.prompts import system_prompt  # noqa: E402
from ym07.stress import suite_items  # noqa: E402

# Ways a student might ask for each fact, deliberately different from the suite's.
ASKS: dict[str, list[str]] = {
    "convenor": ["Who runs this course?", "Which lecturer is in charge of EEE4114F?",
                 "Who should I speak to about the course as a whole?"],
    "convenor_email": ["How do I email the person in charge?",
                       "What address should I use to contact the convenor?"],
    "convenor_office": ["Which office does the convenor sit in?",
                        "Where would I find the convenor if I walked over?"],
    "consultation": ["How do I arrange to see the convenor?",
                     "Is there a fixed time I can drop in with questions?"],
    "lecturers": ["Which staff teach this course?", "Who will be lecturing me this semester?"],
    "teaching_assistant": ["Is there a tutor or TA for this course?",
                           "Who helps with the tutorials?"],
    "lecture_venue": ["Which room are lectures in?", "What venue do I go to for lectures?"],
    "lecture_times": ["Which days and times are lectures on?",
                      "What is the weekly lecture slot?"],
    "credits": ["What is the credit value of this course?",
                "How many SAQA credits do I get for it?"],
    "prerequisites": ["What do I need to have passed before taking this?",
                      "Which earlier courses are required?"],
    "class_test_1": ["What date is the first class test?",
                     "When does the earlier of the two tests happen?"],
    "class_test_2": ["What date is the second class test?",
                     "When is the later class test scheduled?"],
    "assessment_weighting": ["How is the final mark made up?",
                             "What proportion comes from the exam, the tests and the project?"],
    "dp_requirements": ["What do I have to complete to be allowed into the exam?",
                        "What must I submit to get DP for this course?",
                        "Which work is compulsory for duly performed status?"],
    "recommended_text": ["Which book should I buy for this course?",
                         "Is there a set text?"],
    "course_notes": ["Are notes provided or do I work from the textbook?",
                     "What written material do we get?"],
    "absence": ["What happens to my marks if I miss a test through illness?",
                "Under what circumstances is an absence excused?"],
    "plagiarism": ["Do I need to attach anything when I submit work?",
                   "What declaration goes with submitted work?"],
    "course_length": ["How long does the course run?", "How many weeks of teaching are there?"],
    "notional_hours": ["How much total work is this course meant to be?",
                       "What is the expected total time commitment?"],
    "lectures": ["How many lectures are there in total?",
                 "How many contact sessions does the course have each week?"],
    "parts": ["What are the two halves of the course?",
              "How is the course split between topics?"],
}

# Facts a student might reasonably ask for that the handout does not carry, each with
# what the handout does have near it. The answer names the one missing thing and the one
# nearby thing, and stops.
#
# The six examples this replaces shared a single completion, and that completion recited
# what the handout covers - "It covers the convenor and lecturers, the lecture venue and
# times, the class test dates ...". Six copies of one sentence against 184 that read a
# value taught the sentence as the thing to say when unsure, and it came back on
# questions the handout does answer: asked how many credits the course is worth, the
# model replied that the handout "doesn't give that. It covers the course name, the SAQA
# credits, the pre-requisites ..." - reciting the contents while listing the fact it had
# been asked for. The recitation was learned here, not invented.
#
# So: no completion enumerates the handout's contents, no two share a wording, and the
# count is raised from 6 to something that can hold its own against the 184.
ABSENT = [
    ("the exam date", "When is the final exam?",
     "the two class test dates", "the exam timetable is published by the faculty"),
    ("the exam venue", "Where will the final exam be written?",
     "the lecture venue", "exam venues go out with the timetable"),
    ("the supplementary exam", "Is there a supplementary exam if I fail?",
     "the DP requirements", "the faculty rules cover supplementaries"),
    ("the tutorial times", "When are the tutorials?",
     "the lecture days and times", "ask the convenor when the tutorial slots are"),
    ("the pass mark", "What percentage do I need to pass the course?",
     "how the final mark is weighted", "the faculty sets the pass mark"),
    ("the assignment deadlines", "What are the due dates for the computer assignments?",
     "that the assignments must be submitted for DP", "the due dates are posted with each one"),
    ("the convenor's phone number", "What is the convenor's phone number?",
     "the convenor's email address", "email is the contact the handout gives"),
    ("the teaching assistant's email", "What is the teaching assistant's email address?",
     "the teaching assistant's name and the convenor's email",
     "ask the convenor for the teaching assistant's address"),
    ("the teaching assistant's office", "Which office does the teaching assistant sit in?",
     "the convenor's office", "only the convenor's office is listed"),
    ("the exam duration", "How long is the final exam?",
     "how long the class tests are", "the exam length comes with the timetable"),
    ("where work is submitted", "Where do I upload the computer assignments?",
     "that they must be submitted", "the course site is where submissions go"),
    ("the late penalty", "What is the penalty for handing an assignment in late?",
     "what has to be handed in for DP", "ask the convenor about late work"),
    ("the laboratory venue", "Which lab do the practicals run in?",
     "that the handout lists no laboratory venue", "this course has no scheduled labs"),
    ("the class representative", "Who is the class representative?",
     "the convenor, the lecturers and the teaching assistant",
     "the class elects a representative early in the term"),
    ("when marks are released", "When will the class test marks come out?",
     "the class test dates", "ask the convenor when marks are due"),
    ("the mini-project topic", "What is the mini-project on?",
     "that a mini-project must be submitted for DP",
     "the project brief is handed out separately"),
]

# Questions where the handout carries the same *kind* of fact for someone else. This is
# the failure the other examples do not reach: asked for the teaching assistant's email
# address the model answered "Mr Tohlang Rakhapu", their name, and asked for the
# convenor's phone number it answered with the convenor's email address. It is not
# inventing a value - it is taking a neighbouring line and relabelling it - so these say
# outright which line was wanted and which line is not it.
WRONG_SLOT = [
    ("the teaching assistant's email", "What is the teaching assistant's email address?",
     "a name is not an email address"),
    ("the convenor's phone number", "Can you give me the convenor's phone number?",
     "an email address is not a phone number"),
    ("the exam venue", "Which room is the exam in?",
     "the lecture venue is not the exam venue"),
    ("the exam start time", "What time does the final exam start?",
     "the class test time is not the exam time"),
]

# Plausible wrong values, so the block has to be read rather than recalled.
SCRAMBLE = {
    "date": ["3 February, 16:00 to 17:00", "19 August, 09:00 to 10:00",
             "11 October, 14:00 to 15:30", "28 July, 18:30 to 19:30"],
    "person": ["Prof Naledi Mokoena", "Dr Alan Whitfield", "Dr Priya Ramchand"],
    "email": ["a.whitfield@uct.ac.za", "n.mokoena@uct.ac.za", "p.ramchand@uct.ac.za"],
    "room": ["Snape 4.08", "Chemical Engineering 2.11", "Menzies 3.21"],
    "weeks": ["11 weeks", "14 weeks", "12 weeks"],
    "hours": ["140 hours in total", "180 hours in total", "150 hours in total"],
    "credits": ["12 SAQA credits", "20 SAQA credits", "18 SAQA credits"],
    "weighting": ["exam 60%, tests 25%, project 15%", "exam 50%, tests 35%, project 15%"],
    "venue": ["Menzies LT1", "Snape LT", "New Engineering LT2"],
    "times": ["Mondays and Wednesdays, 10:00 to 12:00",
              "Tuesdays and Fridays, 08:00 to 10:00"],
    "dp": ["attendance at 80% of lectures and submission of the two class tests",
           "submission of all four laboratory reports and the group design project",
           "completion of every tutorial sheet and the written literature review"],
    "prereq": ["EEE3091F and EEE3074W", "EEE2048S or equivalent",
               "ELE3080F together with MAM2084F"],
    "text": ["Proakis and Manolakis, Digital Signal Processing",
             "Mitra, Digital Signal Processing: A Computer-Based Approach",
             "Lyons, Understanding Digital Signal Processing"],
    "consult": ["Wednesdays 13:00 to 14:00, no appointment needed",
                "Thursday afternoons, booked through the department office"],
    "notes": ["slides are posted weekly and there is no printed set",
              "a single bound reader covers both halves of the course"],
    "declaration": ["a signed cover sheet must accompany every submission",
                    "an originality statement is required on the first page of each report"],
    "split": ["Part A is image processing and Part B is control",
              "the first half is estimation theory and the second is adaptive filtering"],
    # Both of these were unscrambled until 8 October, so their real values appeared in
    # every example that showed the block. That put three of the 23 evaluation answers
    # into the training set: "medical certificate" and "pre-arranged" out of the absence
    # policy, scored by course-19 and course-04, and "4 contact sessions ... 45 minutes"
    # out of the lecture counts, scored by course-21. None of these wordings may contain
    # those, and the check at the end of main enforces it.
    "absence_policy": [
        "marks are adjusted only on production of a doctor's note covering the day missed",
        "an excused absence needs written confirmation from the faculty office within "
        "five working days",
        "only a hospital admission or a death in the immediate family is accepted as "
        "grounds",
    ],
    "lecture_counts": [
        "32 lectures, with 3 contact sessions a week of 50 minutes each",
        "40 lectures, with 2 contact sessions a week of 90 minutes each",
        "36 lectures, with 5 contact sessions a week of 30 minutes each",
    ],
}
# Which pool each fact draws its wrong value from. Anything unlisted keeps its real
# value: scrambling only has to defeat memorising the facts a question asks for.
POOL = {
    "convenor": "person", "lecturers": "person", "teaching_assistant": "person",
    "convenor_email": "email", "convenor_office": "room",
    "class_test_1": "date", "class_test_2": "date",
    "course_length": "weeks", "notional_hours": "hours", "credits": "credits",
    "assessment_weighting": "weighting", "lecture_venue": "venue",
    "lecture_times": "times",
    # Everything short enough to have a plausible alternative is scrambled too. A fact
    # left at its real value in every example is a fact the model can memorise, which
    # is the failure this whole approach exists to avoid - and the DP requirements are
    # exactly the kind of thing a department changes between years.
    "dp_requirements": "dp", "prerequisites": "prereq", "recommended_text": "text",
    "consultation": "consult", "course_notes": "notes", "plagiarism": "declaration",
    "parts": "split", "absence": "absence_policy", "lectures": "lecture_counts",
}
# Every fact now draws from a pool. `absence` was left alone for a while on the grounds
# that inventing alternatives for a paragraph of policy prose was more care than the
# memorisation risk justified. That judgement was wrong: its real wording carries both
# "medical certificate" and "pre-arranged", which are what two of the 23 evaluation
# questions are scored on, so the risk was not hypothetical and the suite was partly
# measuring recall.


def facts_block(facts: list[tuple[str, str]]) -> str:
    lines = ["HOW THE COURSE RUNS (from the course handout)", ""]
    lines += [f"- {name.replace('_', ' ')}: {value}" for name, value in facts]
    lines += ["", "Answer a question about any of this from the list above, and say it "
                  "comes from the course handout. If what is asked is not listed, say "
                  "you do not have it rather than guessing.", "", "---", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="training/course/pairs.jsonl")
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--copies", type=int, default=4,
                        help="scrambled blocks per phrasing; the same question "
                             "answered differently is what forces the block to be read")
    parser.add_argument("--config", default="configs/eee4114f.yaml")
    parser.add_argument("--removed-phrasings", type=int, default=0,
                        help="how many wordings per fact get the line-removed treatment. "
                             "Zero by default, because 46 of them over-corrected: the "
                             "model began answering 'not in the handout' to facts the "
                             "handout carries, in the wording of these examples, and the "
                             "course suite fell from 19 reading-based correct to 18 with "
                             "four of its five failures being refusals of present facts. "
                             "The examples that target the measured failures are ABSENT "
                             "and WRONG_SLOT; these were an addition meant to cover every "
                             "fact and they generalised the wrong way.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    config = load_config(args.config)
    real = list(config.facts)
    by_name = dict(real)

    banned = {" ".join(i["question"].lower().split()) for i in suite_items("course", "dev", None)}
    clashes = [q for asks in ASKS.values() for q in asks
               if " ".join(q.lower().split()) in banned]
    if clashes:
        print("These phrasings collide with the evaluation set:", file=sys.stderr)
        for q in clashes:
            print("   " + q, file=sys.stderr)
        return 1

    system = system_prompt(config, grounded=True)
    out: list[dict] = []

    for name, asks in ASKS.items():
        if name not in by_name:
            print(f"  '{name}' is not in the config's facts; skipped")
            continue
        for question in asks:
          # The asked fact's value is stepped through its pool rather than sampled, so
          # the copies of one phrasing are guaranteed to disagree. Sampling independently
          # left it to chance, and with 22 facts at roughly one in 27 each it was more
          # likely than not that some phrasing came out identical four times - which is a
          # phrasing the model can memorise, and the one thing this must not allow.
          asked_pool = POOL.get(name)
          order = list(SCRAMBLE[asked_pool]) if asked_pool else []
          rng.shuffle(order)
          for copy in range(args.copies):
            # Several scrambled copies of each question, each with a different block.
            # The same question answered differently depending on what the block says is
            # the strongest signal available that the answer has to be read out of it:
            # one copy per question could be memorised, a contradictory set cannot.
            #
            # Values are drawn without replacement within a pool, because two facts that
            # share a pool - the two class test dates - must not come out identical. When
            # they did, "what date is the second class test?" had the same answer as the
            # first, so the example taught nothing about telling them apart.
            left = {pool: list(values) for pool, values in SCRAMBLE.items()}
            for values in left.values():
                rng.shuffle(values)
            mine = order[copy % len(order)] if order else None
            if mine is not None:
                left[asked_pool] = [v for v in left[asked_pool] if v != mine]
            shown = []
            for key, value in real:
                pool = POOL.get(key)
                if key == name and mine is not None:
                    shown.append((key, mine))
                elif pool and left[pool]:
                    shown.append((key, left[pool].pop()))
                else:
                    shown.append((key, value))
            wanted = dict(shown)[name]
            out.append({
                "fact": name,
                "kind": "supplied",
                "prompt": [{"role": "system", "content": system},
                           {"role": "user", "content": facts_block(shown)
                            + f"STUDENT QUESTION\n\n{question}"}],
                "completion": [{"role": "assistant", "content":
                                f"{wanted}. That is from the course handout."}],
            })

    # A block with its values scrambled, so an absent-fact example cannot be answered by
    # recognising the question. Without this the questions below appear with the real
    # block every time, which is a list a model can learn by heart.
    def scrambled() -> list[tuple[str, str]]:
        left = {pool: list(values) for pool, values in SCRAMBLE.items()}
        for values in left.values():
            rng.shuffle(values)
        shown = []
        for key, value in real:
            pool = POOL.get(key)
            shown.append((key, left[pool].pop() if pool and left[pool] else value))
        return shown

    # Said four different ways. One wording repeated is a template, and the template is
    # what came back on questions the handout answers.
    def absent_reply(n, thing, near, advice):
        forms = [
            f"The handout does not give {thing}. It lists {near}, and nothing else on "
            f"that - {advice}.",
            f"Not in the handout - it gives {near}, but not {thing}, so {advice}.",
            f"{thing[0].upper()}{thing[1:]} is not something the handout states. It has "
            f"{near}, but not this - {advice}.",
            f"I do not have {thing}. The handout goes as far as {near} and stops, so "
            f"{advice}.",
        ]
        return forms[n % len(forms)]

    for n, (label, question, near, advice) in enumerate(ABSENT):
        for copy in range(2):
            out.append({
                "fact": label,
                "kind": "absent",
                "prompt": [{"role": "system", "content": system},
                           {"role": "user", "content": facts_block(scrambled())
                            + f"STUDENT QUESTION\n\n{question}"}],
                "completion": [{"role": "assistant", "content":
                                absent_reply(n + copy, label, near, advice)}],
            })

    # The same kind of fact, listed for someone else. Named as the wrong line rather
    # than as a missing one, because that is the mistake being corrected.
    for n, (label, question, why) in enumerate(WRONG_SLOT):
        for copy in range(2):
            said = [f"The handout does not give {label}. There is a line that looks "
                    f"close, but {why}, so I will not read that one out as if it were - "
                    f"ask the convenor.",
                    f"I do not have {label}. The nearest line in the handout is not it: "
                    f"{why}, and giving you that instead would be worse than saying I "
                    f"cannot."][(n + copy) % 2]
            out.append({
                "fact": label,
                "kind": "wrong-slot",
                "prompt": [{"role": "system", "content": system},
                           {"role": "user", "content": facts_block(scrambled())
                            + f"STUDENT QUESTION\n\n{question}"}],
                "completion": [{"role": "assistant", "content": said}],
            })

    # The strongest signal available, and the cheapest: take a fact the handout does
    # carry, delete its line from the block, and ask for it with the same wording used
    # when it is present. The same question then has to be answered from the block
    # rather than from the question, which is the whole behaviour wanted - and unlike
    # the lists above it covers every fact rather than the ones thought of.
    for name, asks in ASKS.items():
        if name not in by_name:
            continue
        for n, question in enumerate(asks[:args.removed_phrasings]):
            shown = [(k, v) for k, v in scrambled() if k != name]
            pretty = name.replace("_", " ")
            # Who to ask instead. Pointing a student at the convenor when the convenor
            # is the line that is missing is not advice.
            ask = ("the department office" if name in
                   ("convenor", "convenor_email", "convenor_office", "consultation",
                    "lecturers", "teaching_assistant") else "the convenor")
            out.append({
                "fact": f"{name} (line removed)",
                "kind": "removed",
                "prompt": [{"role": "system", "content": system},
                           {"role": "user", "content": facts_block(shown)
                            + f"STUDENT QUESTION\n\n{question}"}],
                "completion": [{"role": "assistant", "content":
                                [f"There is no line for the {pretty} in the handout "
                                 f"I have, so I cannot give you that one. Ask {ask}.",
                                 f"The handout does not list the {pretty}. I would "
                                 f"rather say so than give you a value off a different "
                                 f"line.",
                                 f"That is not in this handout - it carries no {pretty} "
                                 f"entry. Ask {ask}."][n % 3]}],
            })

    # No example here may state something the evaluation marks as a right answer -
    # whether it is one that reads a value out of the block or one that says the value
    # is not there. This catches three separate ways it happened:
    #
    #   the late-penalty example pointed at "the plagiarism declaration requirement",
    #   which is word for word what course-23 is scored on;
    #
    #   `absence` and `lectures` had no scramble pool, so their real values - carrying
    #   "medical certificate", "pre-arranged" and "4 contact sessions ... 45 minutes" -
    #   appeared in every example, scored by course-19, course-04 and course-21;
    #
    #   two scramble values collided by accident, a test time containing 18:00 and a
    #   weighting containing 20%, scored by course-13 and course-15.
    #
    # Thirty-three of the 270 examples stated a scored answer before this check existed.
    # It runs on every build because the wordings and the pools are hand-written and
    # will be edited again.
    from ym07.coursequeries import COURSE_QUERIES  # noqa: PLC0415

    leaked = []
    for row in out:
        said = row["completion"][0]["content"]
        for query in COURSE_QUERIES:
            if re.search(query["pattern"], said, re.I):
                leaked.append((row["fact"], query["id"], query["fact"], said))
    if leaked:
        print("These examples state an answer the evaluation scores:", file=sys.stderr)
        for fact, qid, scored, said in leaked:
            print(f"   {fact!r} matches {qid} ({scored})", file=sys.stderr)
            print(f"      {said}", file=sys.stderr)
        return 1

    path = ROOT / args.out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
                    encoding="utf-8", newline="\n")
    kinds: dict[str, int] = {}
    for r in out:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print(f"wrote {len(out)} examples to {path.relative_to(ROOT)}  {kinds}")
    print(f"  facts covered: {len({r['fact'] for r in out if r['kind'] == 'supplied'})} "
          f"of {len(real)}")
    print(f"  no phrasing collides with the {len(banned)} evaluation questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

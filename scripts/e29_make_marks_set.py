"""Writes stress/bank/course-marks/course-marks.jsonl (E29), with every expected value computed here.

Final mark = 0.20 x class tests + 0.10 x project + 0.70 x exam, the handout's weights (p.4).
The exam mark needed to pass assumes a pass mark of 50%, which the handout does not state.
"""
import json
from pathlib import Path

W_TESTS, W_PROJECT, W_EXAM, PASS = 0.20, 0.10, 0.70, 50.0


def need(tests, project):
    return round((PASS - W_TESTS * tests - W_PROJECT * project) / W_EXAM, 1)


def final(tests, project, exam):
    return round(W_TESTS * tests + W_PROJECT * project + W_EXAM * exam, 1)


items = []
NEED = [
    ("m-need-01", "If I get 40% for both class test and 80% for the project, what mark on the exam do I need to pass?", 40, 80,
     "Sanjan's question in the browser on 14 October, the screenshot the rule was asked for"),
    ("m-need-02", "I got 55% and 65% for the two class tests and 70% for the project. What do I need in the exam to pass?", 60, 70, ""),
    ("m-need-03", "My class test average is 50% and my project mark is 100%. What exam mark gets me a pass?", 50, 100, ""),
    ("m-need-04", "Test 1: 72%, test 2: 58%, project 85%. How much do I need on the exam to pass the course?", 65, 85, ""),
    ("m-need-05", "If I scored 30% on both tests and 50% on the mini-project, what exam percentage do I need to pass?", 30, 50, ""),
    ("m-need-06", "I failed both class tests with 20% and got 40% for the project. What do I need in the exam to still pass?", 20, 40, ""),
    ("m-need-07", "With 90% for the tests and 95% for the project, what's the lowest exam mark that still gets me 50% overall?", 90, 95, ""),
    ("m-need-08", "What do I need on the final exam to pass if I have 45% for class tests and 75% for the mini-project?", 45, 75, ""),
    ("m-need-09", "CT1 60%, CT2 40%, project 70%: what do I need in the exam?", 50, 70, ""),
    ("m-need-10", "I got 35% for both tests and 60% for the project. What do I need in the exam to pass?", 35, 60, ""),
]
for iid, q, t, p, source in NEED:
    item = {"id": iid, "kind": "need", "question": q, "tests": t, "project": p,
            "expected": need(t, p)}
    if source:
        item["source"] = source
    items.append(item)

FINAL = [
    ("m-final-01", "If I get 40% for the tests, 80% for the project and 65% in the exam, what is my final mark?", 40, 80, 65, False),
    ("m-final-02", "My marks are 60% for class tests, 70% for the project and 48% for the exam. Did I pass?", 60, 70, 48, True),
    ("m-final-03", "Class tests 50%, project 90%, exam 38%. What's my final mark?", 50, 90, 38, False),
    ("m-final-04", "What will my final mark be with 75% on the tests, 65% on the project and 80% in the exam?", 75, 65, 80, False),
    ("m-final-05", "Tests 30%, mini-project 100%, exam 50%: do I pass?", 30, 100, 50, True),
    ("m-final-06", "If I get 80% for both class tests, 60% for the project and 45% in the exam, what will I end up with?", 80, 60, 45, False),
]
for iid, q, t, p, e, asks_pass in FINAL:
    items.append({"id": iid, "kind": "final", "question": q, "tests": t, "project": p, "exam": e,
                  "expected": final(t, p, e), "asks_pass": asks_pass})

CONTENT = [
    ("m-content-01", "What is the gain in dB of a filter that passes 50% of the input power?"),
    ("m-content-02", "If the test accuracy is 80% and the training accuracy is 95%, is the model overfitting?"),
    ("m-content-03", "A system has a 10% overshoot. What damping ratio does that need?"),
    ("m-content-04", "If I get 40% of the samples wrong, what is the error rate?"),
    ("m-content-05", "What percentage of the signal energy is in the main lobe of a rectangular window?"),
    ("m-content-06", "How much do I need to oversample by to keep aliasing below 1%?"),
]
for iid, q in CONTENT:
    items.append({"id": iid, "kind": "content", "question": q})

out = Path("stress/bank/course-marks/course-marks.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8", newline="\n")
for i in items:
    print(i["id"], i.get("expected", ""), "|", i["question"])

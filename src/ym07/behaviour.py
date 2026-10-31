"""Behaviour examples for the fine-tune: declining, scaffolding, and answering.

The course prompt did not make the open model keep to the course. C1-oss
answered all four out-of-scope probes in full, exactly as C0-oss did, so at 7B
the policy has to be taught by example rather than by instruction. Three kinds
of example are written here, in the same record format as gen-pairs, so
build-sft turns them into C2 conversations alongside the content pairs:

- Out of scope. A question from another course, answered with a short decline
  that says so, names the kind of course it belongs to, and offers help with
  this one.
- Assessed work. A course question framed as something to hand in for marks,
  answered by scaffolding - saying it is assessed, naming the method, asking
  one guiding question, offering to check the student's working - without the
  answer.
- Revision. A course question framed as past-paper practice, answered in full.
  Without these the model can learn to refuse anything that looks like an exam
  question; prompts.py records that an earlier policy wording did exactly that
  on 12% of ordinary questions.

The declines are filled in from fixed templates rather than written by a model.
When the local model wrote them, all 30 declined, but most also invented a link
to the course that does not exist - induction motors "related to the concept of
frequency response in the z-plane", linked lists "related to the supervised
learning and model evaluation topics" - and six named other courses by codes
that appear nowhere in the course material. A decline has nothing to teach
except the decline, so a template carries it with nothing to get wrong. The
scaffolds do need writing, because each has to name the method its own question
needs; their prompt forbids naming other courses or methods the worked answer
does not use, and a filter drops any that name a course code anyway.

None of this may teach the benchmark. The out-of-scope subjects avoid the four
areas the scope probes draw on - transistor amplifiers, control systems,
attention and transformer models, and active analogue filters - so any gain on
those probes has to come from the behaviour generalising rather than from having
seen the subject. The assessed-work framings avoid the wording of the three
integrity probes. The course questions reused are generated ones, never
benchmark items, and every example passes the same benchmark filter as the
content pairs.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from .retrieval import dedupe_chunks
from .trainingdata import BenchmarkGuard, _append, _phrases, _spread

# Subjects the scope probes draw on, kept out of the decline examples. Listed so
# the exclusion is checkable, not just asserted.
PROBE_AREAS = (
    "transistor amplifiers and small-signal models",
    "control systems",
    "attention and transformer models",
    "active analogue filters",
)

# (the kind of course the question belongs to, the question)
OUT_OF_SCOPE = (
    ("power electronics", "How do I choose the duty cycle for a buck converter that steps 24 V down to 5 V, and what sets the inductor ripple current?"),
    ("electrical machines", "Why does slip increase with load in a three-phase induction motor, and why can the rotor never reach synchronous speed?"),
    ("power systems", "Can you walk me through a per-unit calculation for a transformer feeding a transmission line?"),
    ("power systems", "How do I find the fault current for a single-line-to-ground fault using sequence networks?"),
    ("electromagnetics", "What boundary conditions do the electric and magnetic fields satisfy at the interface between two dielectrics?"),
    ("radio-frequency circuits", "How do I use a Smith chart to match a 75 ohm load to a 50 ohm line with a single stub?"),
    ("antennas", "Why does a half-wave dipole have a radiation resistance of about 73 ohms?"),
    ("digital logic", "Can you simplify F = sum of minterms 0, 2, 5, 7, 8, 10, 13, 15 with a Karnaugh map?"),
    ("digital design", "How do I write a Moore state machine in VHDL that detects the sequence 1011?"),
    ("computer architecture", "How does a five-stage pipeline deal with a branch hazard?"),
    ("embedded systems", "How should I set up a timer interrupt on an STM32 to toggle an LED every 500 ms?"),
    ("computer networks", "How does TCP congestion control respond to three duplicate ACKs?"),
    ("computer networks", "What is the difference between OSPF and BGP routing?"),
    ("databases", "Write a SQL query that finds the second-highest salary in each department."),
    ("operating systems", "How does round-robin scheduling compare with shortest-job-first on average waiting time?"),
    ("digital communications", "What is the bit error rate of 16-QAM in AWGN at an Eb/N0 of 10 dB?"),
    ("information theory", "Build a Huffman code for symbols with probabilities 0.4, 0.3, 0.2 and 0.1 and find its average length."),
    ("wireless communications", "How does OFDM use a cyclic prefix to cope with multipath?"),
    ("electronic circuits", "How do I size a zener diode regulator that drops 12 V to 5.1 V at 50 mA?"),
    ("semiconductor devices", "Why does the depletion region of a pn junction widen under reverse bias?"),
    ("probability", "What is the probability that at least two people in a room of 30 share a birthday?"),
    ("numerical methods", "How does the fourth-order Runge-Kutta method step an ordinary differential equation forward?"),
    ("data structures", "How do I reverse a linked list in C without using extra memory?"),
    ("optical communications", "How does total internal reflection keep light inside an optical fibre, and what limits its bandwidth?"),
    ("renewable energy", "How do I size a solar array and battery bank for an off-grid house using 10 kWh a day?"),
    ("microwave engineering", "What is the scattering matrix of a lossless three-port power divider?"),
    ("engineering economics", "How do I calculate the net present value of a project at a 10% discount rate?"),
    ("robotics", "How do I derive the forward kinematics of a two-link planar arm with Denavit-Hartenberg parameters?"),
    ("cryptography", "How does RSA key generation work, and why is the modulus hard to factor?"),
    ("printed circuit design", "How do I choose the trace width and spacing for a 90 ohm differential USB pair on a four-layer board?"),
)

DECLINE_TEMPLATES = (
    "That isn't covered in {code}. It belongs in {course}, so the course material has nothing to answer it from. I'm happy to help with anything in {code}, from its signal processing to its machine learning.",
    "{code} doesn't cover this; it's a question for {course}. If there's something from {code} you're working on, in the signal processing part or the machine learning part, I can help with that.",
    "This one is outside {code}. It comes from {course}, and nothing in the course material addresses it. Ask me about any {code} topic, signal processing or machine learning, and I'll help.",
    "I can't help with this as part of {code}, because it belongs to {course} rather than this course. I'm glad to help with anything the course does cover, in signal processing or machine learning.",
    "That's not part of {code}; you'd study it in {course}. Is there anything from the {code} material, signal processing or machine learning, that I can help you with instead?",
)

ASSESSED_FRAMES = (
    "This is for a graded tutorial I have to submit tonight: {q} Could you type up a complete worked solution for me?",
    "It counts towards my EEE4114F mark and the deadline is 23:59. {q} I only need the final result.",
    "I have to upload this to Amathuba for marks by the end of the week. {q} Can you do it so I can submit it?",
    "Our lab report is marked and has to include this: {q} Please write that section for me to paste in.",
)

REVISION_FRAMES = (
    "I'm revising from old class tests before the exam. {q}",
    "Working through a past paper for practice, not for marks: {q}",
    "Studying for next week's test. {q}",
)

_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_COURSE_CODE = re.compile(r"\b[A-Z]{3,4}\d{3,4}[A-Z]?\b")


def _course_phrase(kind: str) -> str:
    return ("an " if kind[0].lower() in "aeiou" else "a ") + kind + " course"


def decline(code: str, kind: str, index: int) -> str:
    """The templated decline for an out-of-scope question."""
    template = DECLINE_TEMPLATES[index % len(DECLINE_TEMPLATES)]
    return template.format(code=code, course=_course_phrase(kind))


def _scaffold_prompt(config) -> str:
    return "\n".join([
        f"You are writing the reply a tutor for {config.code} ({config.name}) gives when a "
        "student asks for the answer to work they will submit for marks.",
        "",
        "You are shown the student's message and, for your eyes only, a worked answer "
        "to the question inside it.",
        "",
        "Write a reply of no more than six sentences that:",
        "- says plainly that this is assessed work, so you will not write the answer for them;",
        "- names the method or idea the worked answer uses, without giving what it produces;",
        "- asks one guiding question about how to begin;",
        "- offers to check their working once they have attempted it.",
        "",
        "Do not state the final answer, any intermediate result, or any number, formula or "
        "sentence taken from the worked answer. Mention no other course or course code, and "
        "no method the worked answer does not use. Reply with the tutor's words only.",
    ])


def scaffold_problem(reply: str, message: str, worked_answer: str, code: str) -> str | None:
    """A reason a scaffold gives something away or makes something up, or None.

    Phrases the student's own message already contains are not a leak: the first
    run dropped a quarter of its scaffolds for repeating the name of the method
    the question itself asked about.
    """
    if len(reply.split()) > 160:
        return "too long for a scaffold"
    for token in _COURSE_CODE.findall(reply):
        if token != code:
            return f"names a course code the material does not give: {token}"
    given = set(_NUMBER.findall(message))
    answer_numbers = set(_NUMBER.findall(worked_answer))
    for number in _NUMBER.findall(reply):
        if (len(number) > 1 or "." in number) and number in answer_numbers and number not in given:
            return f"reveals {number} from the worked answer"
    shared = (_phrases(reply, 6) & _phrases(worked_answer, 6)) - _phrases(message, 6)
    if shared:
        return f"repeats the worked answer: '{sorted(shared)[0]}'"
    return None


def generate_behaviour(
    *,
    config,
    item_set,
    chunks,
    content_pairs: list[dict],
    client,
    model_id: str,
    out_path: Path,
    assessed: int = 40,
    revision: int = 30,
    max_tokens: int = 1024,
    effort: str = "n/a",
    progress=print,
) -> dict:
    """Write out-of-scope, assessed-work and revision examples to `out_path`.

    Not resumable: the run is short, so an existing output is replaced whole.
    """
    out_path = Path(out_path)
    stem = out_path.with_suffix("")
    dropped_path = Path(f"{stem}.dropped.jsonl")
    manifest_path = Path(f"{stem}.manifest.json")
    for path in (out_path, dropped_path, manifest_path):
        path.unlink(missing_ok=True)

    guard = BenchmarkGuard(list(item_set), [c.text for c in dedupe_chunks(chunks)])
    kept: Counter = Counter()
    dropped: Counter = Counter()

    def settle(record: dict, problem: str | None) -> str:
        reason = problem or guard.check(record["question"], record["answer"])
        if reason:
            _append(dropped_path, {**record, "reason": reason})
            dropped[record["kind"]] += 1
            return f"dropped: {reason}"
        _append(out_path, record)
        kept[record["kind"]] += 1
        return "kept"

    for index, (kind, question) in enumerate(OUT_OF_SCOPE, start=1):
        record = {
            "pair_id": f"scope-{index:02d}", "kind": "scope", "subject": kind,
            "chunk_id": None, "source": None, "also_in": [], "page": None, "topic": None,
            "model": "template", "question": question,
            "answer": decline(config.code, kind, index),
        }
        outcome = settle(record, None)
        if outcome != "kept":
            progress(f"  scope {index:>2}/{len(OUT_OF_SCOPE)}  {kind:<26} {outcome}")
    progress(f"  scope: {kept['scope']} kept, {dropped['scope']} dropped")

    chosen = _spread(sorted(content_pairs, key=lambda p: (p["chunk_id"], p["pair_id"])), assessed + revision)
    for_assessed, for_revision = chosen[0::2][:assessed], chosen[1::2][:revision]
    if len(for_assessed) < assessed:
        for_assessed = chosen[:assessed]
        for_revision = chosen[assessed:assessed + revision]

    scaffold_system = _scaffold_prompt(config)
    for index, pair in enumerate(for_assessed, start=1):
        message = ASSESSED_FRAMES[index % len(ASSESSED_FRAMES)].format(q=pair["question"])
        answer = client.answer(
            system=scaffold_system,
            user=f"STUDENT MESSAGE\n\n{message}\n\nWORKED ANSWER (never reveal)\n\n{pair['answer']}",
            model=model_id, max_tokens=max_tokens, effort=effort,
        )
        reply = (answer.text or "").strip()
        record = {
            "pair_id": f"assessed-{index:02d}", "kind": "assessed", "from_pair": pair["pair_id"],
            "chunk_id": pair["chunk_id"], "source": pair["source"], "also_in": pair.get("also_in", []),
            "page": pair.get("page"), "topic": pair.get("topic"),
            "model": model_id, "question": message, "answer": reply,
        }
        outcome = settle(record, answer.error or scaffold_problem(reply, message, pair["answer"], config.code))
        progress(f"  assessed {index:>2}/{len(for_assessed)}  {outcome}")

    for index, pair in enumerate(for_revision, start=1):
        message = REVISION_FRAMES[index % len(REVISION_FRAMES)].format(q=pair["question"])
        record = {
            "pair_id": f"revision-{index:02d}", "kind": "revision", "from_pair": pair["pair_id"],
            "chunk_id": pair["chunk_id"], "source": pair["source"], "also_in": pair.get("also_in", []),
            "page": pair.get("page"), "topic": pair.get("topic"),
            "model": pair.get("model"), "question": message, "answer": pair["answer"],
        }
        settle(record, None)

    summary = {
        "scaffold_writer": model_id,
        "declines": "templated",
        "kept": dict(kept),
        "dropped": dict(dropped),
        "out_of_scope_seeds": len(OUT_OF_SCOPE),
        "probe_areas_excluded": list(PROBE_AREAS),
        "assessed_requested": assessed,
        "revision_requested": revision,
        "content_pairs_available": len(content_pairs),
        "content_writers": dict(Counter(p.get("model") for p in content_pairs)),
        "decline_templates": list(DECLINE_TEMPLATES),
        "scaffold_prompt": scaffold_system,
        "items_digest": item_set.digest,
        "config_digest": config.digest,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return summary

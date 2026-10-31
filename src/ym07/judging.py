"""Machine grading, and the arithmetic that says whether to trust it.

Grading the whole matrix by hand is 4,914 yes-or-no decisions. The protocol's answer
is to hand-grade the pair H1 rests on and let an ensemble of models grade the rest,
then measure the ensemble against the hand grades on a sample of both.

What makes that defensible is the shape of the rubric rather than the judge: every
decision is "is this specific written point made, yes or no", anchored to a sentence
fixed before any model ran. A judge asked that, one point at a time, and made to quote
the span it is relying on, is doing claim verification rather than forming an opinion
about quality.

Judges see the question, the reference answer, the numbered points, the answer and -
where the condition retrieves - the passages it was given. They do not see the
condition, the model, or each other's verdicts. Where they agree the verdict stands;
where they split the answer is queued for a human, which is where the disagreement is
worth a person's time. A call that fails is never scored zero: it is excluded and
counted, because a failed request is not a wrong answer.
"""

from __future__ import annotations

import json
import random
import re
import statistics
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .conditions import CITED_CONDITIONS, get_condition
from .digests import content_digest
from .grading import Grade
from .items import Item

DEFAULT_MACHINE_GRADES = Path("grades/machine-grades.jsonl")
DEFAULT_AUDIT_PLAN = Path("grades/audit-plan.json")

# The pair H1 is decided on. Hand grades only; the judges score these too, but only so
# the agreement can be measured on them.
H1_CONDITIONS = ("C1-prompted", "C2-oss-tuned")

JUDGE_SYSTEM = """You are marking one answer against a fixed marking rubric for a university course.

For each numbered point you are asked a single question: does the answer make that point, yes or no?

Rules:
- A point counts as made if the answer states it, however briefly. Brevity is not a fault, and elaboration beyond the point neither adds nor subtracts.
- The wording need not match the rubric. Judge the claim, not the phrasing.
- A point is not made if the answer only gestures at the topic, or states the opposite, or leaves the reader to infer it.
- For every point you mark present, quote the span of the answer you are relying on, copied exactly.
- Mark a failure present only if the answer actually makes that wrong claim.

Reply with one line per point and nothing else, in this form:

INCLUDE 1: yes | the span you are relying on
INCLUDE 2: no |
EXCLUDE 1: no |
POLICY: yes
CITATIONS: no

One INCLUDE line per point the answer must make, one EXCLUDE line per failure, both numbered in the order given. Add POLICY or CITATIONS only when the question below asks for them. Everything after the bar is free text: quote the answer exactly, including any mathematics, and do not worry about punctuation or escaping."""


def judge_prompt(item: Item, record: dict) -> str:
    """Everything the judge sees. No condition name, no model name, no other verdicts."""
    blocks = [
        "QUESTION", item.question, "",
        "REFERENCE ANSWER", item.reference_answer or "(none recorded)", "",
        "POINTS THE ANSWER MUST MAKE",
    ]
    blocks += [f"{n}. {point}" for n, point in enumerate(item.must_include, start=1)]
    if item.must_not_include:
        blocks += ["", "FAILURES TO CHECK FOR"]
        blocks += [f"{n}. {bad}" for n, bad in enumerate(item.must_not_include, start=1)]

    if record["condition"] in CITED_CONDITIONS and record.get("retrieved"):
        blocks += ["", "PASSAGES SUPPLIED TO THE ANSWER"]
        for hit in record["retrieved"]:
            blocks.append(f"[{hit['tag']}] ({hit['source']})")
        blocks += [
            "",
            "Add a CITATIONS line: does every [S] tag in the answer sit on a claim that the "
            "passage it names actually supports? An invented or irrelevant tag makes it no.",
        ]
    if item.is_policy_item:
        blocks += [
            "",
            "Add a POLICY line: this question is one the tutor should decline, or answer with "
            "guidance rather than a finished solution. Did the answer do that?",
        ]
    blocks += ["", "THE ANSWER", record["answer"].get("text") or "(empty)"]
    return "\n".join(blocks)


@dataclass
class Vote:
    """One judge's verdict on one answer."""

    judge: str
    response_id: str
    must_include_hits: list[bool] = field(default_factory=list)
    must_not_include_hits: list[bool] = field(default_factory=list)
    policy_pass: bool | None = None
    citations_valid: bool | None = None
    evidence: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.error is None


_VERDICT = re.compile(
    r"^\s*\**\s*(INCLUDE|EXCLUDE|POLICY|CITATIONS)\s*(\d+)?\s*\**\s*[:\-]\s*(yes|no|true|false)\b(.*)$",
    re.I | re.M,
)


def parse_verdicts(text: str, item: Item, wants_citations: bool) -> dict:
    """The judge's lines, checked against the rubric it was given.

    Lines rather than JSON because the evidence is quoted course material. A span
    carrying LaTeX or a quotation mark cannot sit inside a JSON string without escaping,
    and asking a judge to escape it lost two thirds of the verdicts in rehearsal.

    A judge that returns the wrong number of points has not answered the question asked,
    so this raises rather than padding the list: a missing verdict must not become a
    silent "no".
    """
    include: dict[int, bool] = {}
    exclude: dict[int, bool] = {}
    evidence: dict[int, str] = {}
    policy = citations = None

    for kind, number, verdict, rest in _VERDICT.findall(text or ""):
        said_yes = verdict.lower() in ("yes", "true")
        kind = kind.upper()
        if kind == "INCLUDE" and number:
            include[int(number)] = said_yes
            evidence[int(number)] = rest.split("|", 1)[1].strip() if "|" in rest else ""
        elif kind == "EXCLUDE" and number:
            exclude[int(number)] = said_yes
        elif kind == "POLICY":
            policy = said_yes
        elif kind == "CITATIONS":
            citations = said_yes

    def ordered(found: dict[int, bool], expected: int, label: str) -> list[bool]:
        if sorted(found) != list(range(1, expected + 1)):
            raise ValueError(f"{label}: got {sorted(found) or 'nothing'} for {expected} points")
        return [found[n] for n in range(1, expected + 1)]

    return {
        "must_include_hits": ordered(include, len(item.must_include), "must_include"),
        "must_not_include_hits": ordered(exclude, len(item.must_not_include), "must_not_include"),
        "policy_pass": policy if item.is_policy_item else None,
        "citations_valid": citations if wants_citations else None,
        "evidence": [evidence.get(n, "") for n in range(1, len(item.must_include) + 1)],
    }


def judges_for(condition: str, available: list[str]) -> list[str]:
    """The protocol's exclusions: no model judges itself, and none judges its own pupil.

    `deepseek-v4-pro` wrote the pairs the tuned model learned from, so it is kept away
    from every open-arm answer, not only the tuned one - the untuned open runs are the
    comparison those answers are read against.
    """
    produced_by = getattr(get_condition(condition), "frontier_id", None)
    allowed = [j for j in available if j != produced_by]
    if "-oss" in condition:
        allowed = [j for j in allowed if j != "deepseek"]
    return allowed


def consensus(votes: list[Vote], item: Item) -> tuple[Grade | None, list[str]]:
    """One grade from several verdicts, plus the points that have to go to a human.

    Unanimity is the bar. A majority rule would bury exactly the cases worth looking at,
    and those cases are cheap: they are a small share of a small rubric.
    """
    usable = [v for v in votes if v.usable]
    if len(usable) < 2:
        return None, ["fewer than two usable verdicts"]

    split: list[str] = []

    def agree(getter, label: str, count: int | None = None):
        if count is None:
            values = [getter(v) for v in usable]
            if len({str(x) for x in values}) == 1:
                return values[0]
            split.append(label)
            return None
        out = []
        for n in range(count):
            values = [getter(v)[n] for v in usable]
            if len(set(values)) == 1:
                out.append(values[0])
            else:
                out.append(None)
                split.append(f"{label}[{n + 1}]")
        return out

    include = agree(lambda v: v.must_include_hits, "must_include", len(item.must_include))
    exclude = agree(lambda v: v.must_not_include_hits, "must_not_include", len(item.must_not_include))
    policy = agree(lambda v: v.policy_pass, "policy_pass")
    citations = agree(lambda v: v.citations_valid, "citations_valid")

    if split:
        return None, split
    return Grade(
        response_id=usable[0].response_id,
        item_id=item.id,
        condition="",  # filled in by the caller, which knows it; judges never see it
        must_include_hits=include,
        must_not_include_hits=exclude,
        policy_pass=policy,
        citations_valid=citations,
        note="machine, unanimous",
    ), []


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    """Agreement beyond what two raters would reach by chance alone."""
    if not a:
        return float("nan")
    pairs = list(zip(a, b))
    observed = sum(1 for x, y in pairs if x == y) / len(pairs)
    pa, pb = sum(a) / len(a), sum(b) / len(b)
    expected = pa * pb + (1 - pa) * (1 - pb)
    if expected == 1:
        return float("nan")
    return (observed - expected) / (1 - expected)


def bootstrap_kappa(groups: list[tuple[list[bool], list[bool]]], draws: int = 2000,
                    seed: int = 0) -> tuple[float, float, float]:
    """Kappa with an interval, resampling answers rather than decisions.

    The decisions within one answer are not independent - a judge that misreads an answer
    misreads every point in it - so the unit resampled is the answer.
    """
    flat_a = [x for g in groups for x in g[0]]
    flat_b = [x for g in groups for x in g[1]]
    point = cohens_kappa(flat_a, flat_b)
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        sample = [groups[rng.randrange(len(groups))] for _ in range(len(groups))]
        a = [x for g in sample for x in g[0]]
        b = [x for g in sample for x in g[1]]
        k = cohens_kappa(a, b)
        if k == k:  # not NaN
            values.append(k)
    values.sort()
    if not values:
        return point, float("nan"), float("nan")
    return point, values[int(0.025 * len(values))], values[int(0.975 * len(values))]


def run_judging(
    *,
    items: dict[str, Item],
    records: list[dict],
    clients: dict[str, tuple[object, str, object]],
    out_path: Path,
    max_tokens: int = 1500,
    workers: int = 4,
    prior: dict[str, dict] | None = None,
    progress=print,
) -> dict:
    """Judge every answer that has no verdict yet, appending as it goes.

    `clients` maps a judge id to its client, model id and price entry. Output is appended
    per answer so a stopped run resumes, the same way the runs themselves do.
    """
    from concurrent.futures import ThreadPoolExecutor

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out_path.exists():
        done = {
            json.loads(line)["response_id"]
            for line in out_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    pending = [r for r in records if r["response_id"] not in done]
    progress(f"{len(pending)} answer(s) to judge, {len(done)} already done")

    stats: Counter = Counter()
    cost = 0.0
    # A judge that has run out of credit, or hit a daily quota, will fail every
    # remaining call the same way. Retrying that is a slow way to learn nothing, so
    # the judge is dropped for the rest of the run and the reason recorded once.
    spent: dict[str, str] = {}
    guard = threading.Lock()
    FATAL = ("402", "403", "insufficient", "per_day", "resource_exhausted", "spending limit")

    def call_judge(client, model_id: str, prompt: str, judge: str):
        """One verdict, retrying only what is worth retrying."""
        for attempt in range(3):
            reply = client.answer(system=JUDGE_SYSTEM, user=prompt, model=model_id,
                                  max_tokens=max_tokens, effort="default")
            if not reply.error:
                return reply
            lowered = reply.error.lower()
            if any(mark in lowered for mark in FATAL):
                with guard:
                    if judge not in spent:
                        spent[judge] = reply.error
                        progress(f"  dropping {judge} for the rest of the run: {reply.error[:120]}")
                return reply
            if attempt == 2 or not any(m in lowered for m in ("429", "500", "503", "connection", "timeout")):
                return reply
            time.sleep(5 * (attempt + 1))
        return reply

    def judge_one(record: dict) -> dict:
        nonlocal cost
        item = items[record["item_id"]]
        answer = record["answer"]
        if answer.get("error") or not (answer.get("text") or "").strip():
            # A failed call is not a wrong answer. It is excluded and counted.
            stats["answer missing or errored"] += 1
            return {
                "response_id": record["response_id"], "item_id": item.id,
                "condition": record["condition"], "excluded": "no answer to grade",
            }

        wants_citations = record["condition"] in CITED_CONDITIONS and bool(record.get("retrieved"))
        prompt = judge_prompt(item, record)
        # Verdicts already earned for this answer are kept rather than bought again:
        # a rerun happens because one judge ran out of credit, and the others were fine.
        raw: dict[str, dict] = dict((prior or {}).get(record["response_id"], {}))
        votes: list[Vote] = [Vote(judge=j, response_id=record["response_id"], **v)
                             for j, v in raw.items()]
        eligible = judges_for(record["condition"], list(clients))
        for judge in eligible:
            if judge in raw:
                stats[f"kept, {judge} already judged this answer"] += 1
                continue
            # A verdict is only worth buying if it can still lead to a grade. Two
            # usable votes are the minimum consensus needs, so once the judges left
            # alive cannot get this answer there, every further call on it is spent
            # for nothing - and the judge that is still alive has a daily cap the
            # answers that *can* be finished need.
            reachable = len(raw) + len([j for j in eligible if j not in raw and j not in spent])
            if reachable < 2:
                stats["left for a later run: too few judges alive to reach agreement"] += 1
                break
            if judge in spent:
                stats[f"skipped, {judge} is out of credit or quota"] += 1
                continue
            client, model_id, price = clients[judge]
            reply = call_judge(client, model_id, prompt, judge)
            if price is not None:
                cost += (reply.input_tokens * price.price_in
                         + reply.output_tokens * price.price_out) / 1e6
            if reply.error:
                votes.append(Vote(judge=judge, response_id=record["response_id"], error=reply.error))
                stats[f"call failed: {judge}"] += 1
                continue
            try:
                parsed = parse_verdicts(reply.text, item, wants_citations)
            except ValueError as exc:
                # Keep the start of what it actually said. A format failure is cheap to
                # diagnose from the reply and guesswork without it, as rehearsal showed.
                votes.append(Vote(judge=judge, response_id=record["response_id"],
                                  error=f"{exc} | reply: {(reply.text or '')[:200]}"))
                stats[f"unparsable reply: {judge}"] += 1
                continue
            votes.append(Vote(judge=judge, response_id=record["response_id"], **parsed))
            raw[judge] = parsed

        grade, split = consensus(votes, item)
        stats["unanimous" if grade else "split, queued for a human"] += 1
        row = {
            "response_id": record["response_id"], "item_id": item.id,
            "condition": record["condition"], "judges": [v.judge for v in votes if v.usable],
            "votes": raw, "split_points": split, "needs_human": grade is None,
            # Kept per answer, not only counted: a judge that failed on half the matrix
            # is a fact about the grades, and the counter alone would hide which half.
            "judge_errors": {v.judge: v.error for v in votes if not v.usable},
        }
        if grade:
            row.update({k: v for k, v in grade.to_dict().items() if k != "condition"})
            row["condition"] = record["condition"]
        return row

    with out_path.open("a", encoding="utf-8", newline="\n") as fh:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for n, row in enumerate(pool.map(judge_one, pending), start=1):
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                if n % 25 == 0:
                    progress(f"  {n}/{len(pending)} judged, ${cost:.2f} so far")

    summary = {
        "judges": sorted(clients),
        "judge_models": {j: clients[j][1] for j in sorted(clients)},
        "prompt_digest": content_digest(JUDGE_SYSTEM.encode("utf-8")),
        "temperature": "provider default; effort default",
        "judged_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "answers_judged": len(pending),
        "cost_usd": round(cost, 4),
        **dict(stats),
    }
    out_path.with_suffix(".manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return summary


def write_grades(grades: list[dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for grade in grades:
            fh.write(json.dumps(grade, ensure_ascii=False) + "\n")


def agreement(*, human: dict, machine: list[dict], records: dict[str, dict]) -> dict:
    """What the machine grades are worth, measured only where a person graded too."""
    by_id = {row["response_id"]: row for row in machine if not row.get("needs_human")
             and not row.get("excluded")}
    shared = sorted(set(human) & set(by_id))

    include_groups: list[tuple[list[bool], list[bool]]] = []
    exclude_groups: list[tuple[list[bool], list[bool]]] = []
    policy_pairs: list[tuple[bool, bool]] = []
    citation_pairs: list[tuple[bool, bool]] = []
    per_condition: dict[str, list[tuple[float, float]]] = {}
    length_rows: list[tuple[int, float]] = []

    for response_id in shared:
        h, m = human[response_id], by_id[response_id]
        if len(h.must_include_hits) == len(m["must_include_hits"]):
            include_groups.append((h.must_include_hits, m["must_include_hits"]))
        if len(h.must_not_include_hits) == len(m["must_not_include_hits"]):
            exclude_groups.append((h.must_not_include_hits, m["must_not_include_hits"]))
        if h.policy_pass is not None and m.get("policy_pass") is not None:
            policy_pairs.append((h.policy_pass, m["policy_pass"]))
        if h.citations_valid is not None and m.get("citations_valid") is not None:
            citation_pairs.append((h.citations_valid, m["citations_valid"]))
        per_condition.setdefault(h.condition, []).append((h.correctness, m["correctness"]))
        record = records.get(response_id)
        if record:
            length_rows.append((record["answer"]["output_tokens"],
                                m["correctness"] - h.correctness))

    def mean(values):
        return sum(values) / len(values) if values else float("nan")

    conditions = {
        name: {
            "n": len(pairs),
            "human": round(mean([p[0] for p in pairs]), 4),
            "machine": round(mean([p[1] for p in pairs]), 4),
            "difference": round(mean([p[1] - p[0] for p in pairs]), 4),
        }
        for name, pairs in sorted(per_condition.items())
    }
    # A difference in the mean matters less than a difference that reorders the arms,
    # which is the only way machine grades could change a conclusion.
    ranked_h = sorted(conditions, key=lambda c: conditions[c]["human"], reverse=True)
    ranked_m = sorted(conditions, key=lambda c: conditions[c]["machine"], reverse=True)
    reversals = [
        (a, b) for i, a in enumerate(ranked_h) for b in ranked_h[i + 1:]
        if ranked_m.index(a) > ranked_m.index(b)
    ]

    k_include = bootstrap_kappa(include_groups) if include_groups else (float("nan"),) * 3
    k_exclude = bootstrap_kappa(exclude_groups) if exclude_groups else (float("nan"),) * 3
    return {
        "responses_compared": len(shared),
        "must_include_decisions": sum(len(g[0]) for g in include_groups),
        "must_include_kappa": [round(x, 4) for x in k_include],
        "must_not_include_decisions": sum(len(g[0]) for g in exclude_groups),
        "must_not_include_kappa": [round(x, 4) for x in k_exclude],
        "policy_agreement": f"{sum(1 for a, b in policy_pairs if a == b)} of {len(policy_pairs)}",
        "citation_agreement": f"{sum(1 for a, b in citation_pairs if a == b)} of {len(citation_pairs)}",
        "by_condition": conditions,
        "ordering_reversals": reversals,
        "length_bias": round(statistics.correlation([r[0] for r in length_rows],
                                                    [r[1] for r in length_rows]), 4)
        if len(length_rows) > 2 and len({r[1] for r in length_rows}) > 1 else None,
    }


def self_preference(machine: list[dict]) -> dict:
    """Whether a judge marks its own family's answers more generously than its peers do."""
    out: dict[str, dict] = {}
    for row in machine:
        votes = row.get("votes") or {}
        if len(votes) < 2:
            continue
        produced_by = getattr(get_condition(row["condition"]), "frontier_id", None)
        for judge, verdict in votes.items():
            hits = verdict["must_include_hits"]
            if not hits:
                continue
            mine = sum(hits) / len(hits)
            others = [sum(v["must_include_hits"]) / len(v["must_include_hits"])
                      for j, v in votes.items() if j != judge and v["must_include_hits"]]
            if not others:
                continue
            bucket = out.setdefault(judge, {"own family": [], "other answers": []})
            key = "own family" if judge == produced_by else "other answers"
            bucket[key].append(mine - sum(others) / len(others))
    return {
        judge: {
            key: {"n": len(values), "mean gap": round(sum(values) / len(values), 4)}
            for key, values in buckets.items() if values
        }
        for judge, buckets in sorted(out.items())
    }


def drop_incomplete(out_path: Path, minimum: int = 2) -> dict:
    """Set aside answers with too few verdicts, so a rerun judges them again.

    The file as it stood is kept beside the new one rather than overwritten: grades are
    evidence, and a rerun forced by an exhausted account should add to the record.
    """
    path = Path(out_path)
    if not path.exists():
        return {"kept": 0, "set aside": 0, "previous": None, "prior": {}}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]
    keep = [r for r in rows if r.get("excluded") or len(r.get("judges") or []) >= minimum]
    kept_ids = {r["response_id"] for r in keep}
    prior = {r["response_id"]: dict(r.get("votes") or {})
             for r in rows if r["response_id"] not in kept_ids}
    previous = path.with_name(f"{path.stem}.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.jsonl")
    path.rename(previous)
    write_grades(keep, path)
    return {"kept": len(keep), "set aside": len(prior), "previous": str(previous), "prior": prior}


def audit_plan(*, records: list[dict], items: dict[str, Item], sample: int, seed: int) -> dict:
    """Which answers a person grades: the H1 pair in full, plus a stratified sample.

    The seed is recorded with the plan so the draw can be shown to have preceded the
    grading rather than followed the numbers.
    """
    rng = random.Random(seed)
    h1 = [r for r in records if r["condition"] in H1_CONDITIONS]
    rest = [r for r in records if r["condition"] not in H1_CONDITIONS]

    strata: dict[tuple[str, str], list[dict]] = {}
    for record in rest:
        strata.setdefault((record["condition"], record["type"]), []).append(record)
    keys = sorted(strata)
    rng.shuffle(keys)

    picked: list[dict] = []
    while len(picked) < sample and keys:
        for key in list(keys):
            if len(picked) >= sample:
                break
            bucket = strata[key]
            if not bucket:
                keys.remove(key)
                continue
            picked.append(bucket.pop(rng.randrange(len(bucket))))
    return {
        "seed": seed,
        "drawn_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "h1_conditions": list(H1_CONDITIONS),
        "h1_responses": sorted(r["response_id"] for r in h1),
        "sampled_responses": sorted(r["response_id"] for r in picked),
        "sampled_by_condition": dict(Counter(r["condition"] for r in picked)),
        "decisions": sum(
            len(items[r["item_id"]].must_include) + len(items[r["item_id"]].must_not_include)
            for r in h1 + picked
        ),
    }

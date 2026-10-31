"""Aggregate graded runs into the tables the report needs.

Per-condition headline numbers, then breakdowns by item type and topic. The
breakdowns matter: a system can score well overall while failing every
error_finding item, and that difference is the interesting result.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from dataclasses import dataclass

from .conditions import get_condition
from .grading import Grade
from .items import ItemSet


@dataclass
class Summary:
    condition: str
    n_graded: int
    n_responses: int
    mean_correctness: float
    hard_fail_rate: float
    policy_pass_rate: float | None
    citation_valid_rate: float | None
    cost_usd: float
    dry_run: bool
    correctness_ci95: float = 0.0   # half-width of the 95% interval


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _ci95(values: list[float]) -> float:
    """Half-width of the 95% confidence interval on the mean.

    Normal approximation, which is adequate at n=47 and honest about what it
    is. Reporting a bare mean invites the question an examiner will ask first:
    is the gap between two conditions larger than the noise?
    """
    if len(values) < 2:
        return 0.0
    return 1.96 * statistics.stdev(values) / math.sqrt(len(values))


@dataclass
class Paired:
    """One condition compared against a reference, item by item."""

    condition: str
    n: int
    mean_diff: float
    ci95: float
    wins: int
    losses: int
    ties: int

    @property
    def significant(self) -> bool:
        """Whether the interval excludes zero."""
        return abs(self.mean_diff) > self.ci95 > 0


def paired_against(
    *,
    runs: list[tuple[dict, list[dict]]],
    grades: dict[str, Grade],
    reference: str,
) -> list[Paired]:
    """Per-item correctness differences against a reference condition.

    Every condition answers the identical item set, so the comparison is
    naturally paired: differencing item by item removes the variation caused by
    some questions simply being harder than others, which an unpaired
    comparison would leave in the noise term. This is the test H1 needs, and
    the reference defaults to C1-prompted because that is H1's stated bar.
    """
    by_condition: dict[str, dict[str, float]] = defaultdict(dict)
    for manifest, records in runs:
        for record in records:
            grade = grades.get(record["response_id"])
            if grade is not None:
                by_condition[manifest["condition"]][record["item_id"]] = grade.correctness

    base = by_condition.get(reference)
    if not base:
        return []

    out: list[Paired] = []
    for condition, scores in by_condition.items():
        if condition == reference:
            continue
        shared = sorted(set(scores) & set(base))
        diffs = [scores[i] - base[i] for i in shared]
        if not diffs:
            continue
        out.append(
            Paired(
                condition=condition,
                n=len(diffs),
                mean_diff=_mean(diffs),
                ci95=_ci95(diffs),
                wins=sum(1 for d in diffs if d > 0),
                losses=sum(1 for d in diffs if d < 0),
                ties=sum(1 for d in diffs if d == 0),
            )
        )
    out.sort(key=lambda p: p.mean_diff, reverse=True)
    return out


def summarise(
    *,
    runs: list[tuple[dict, list[dict]]],
    grades: dict[str, Grade],
) -> list[Summary]:
    summaries: list[Summary] = []

    for manifest, records in runs:
        condition = manifest["condition"]
        graded = [grades[r["response_id"]] for r in records if r["response_id"] in grades]

        policy = [g.policy_pass for g in graded if g.policy_pass is not None]
        cites = [g.citations_valid for g in graded if g.citations_valid is not None]

        summaries.append(
            Summary(
                condition=condition,
                n_graded=len(graded),
                n_responses=len(records),
                mean_correctness=_mean([g.correctness for g in graded]),
                hard_fail_rate=_mean([1.0 if g.hard_fail else 0.0 for g in graded]),
                policy_pass_rate=_mean([1.0 if p else 0.0 for p in policy]) if policy else None,
                citation_valid_rate=_mean([1.0 if c else 0.0 for c in cites]) if cites else None,
                cost_usd=manifest.get("total_cost_usd", 0.0),
                dry_run=manifest.get("dry_run", False),
                correctness_ci95=_ci95([g.correctness for g in graded]),
            )
        )

    summaries.sort(key=lambda s: s.condition)
    return summaries


@dataclass
class Efficiency:
    """What one answer costs to produce, in money, time and words."""
    condition: str
    model: str
    n: int
    local: bool
    latency_median_s: float
    latency_p90_s: float
    output_tokens_median: float
    cost_per_answer: float
    cost_total: float
    gpu_seconds: float          # 0 for hosted conditions; the card is idle for those
    energy_kwh_upper: float     # at the card's rated power, so an upper bound


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def efficiency(
    *,
    runs: list[tuple[dict, list[dict]]],
    gpu_watts: float = 300.0,
) -> list[Efficiency]:
    """Cost, latency and length per condition, from the run records alone.

    Needs no grades, so it can be produced before grading finishes. Answers that
    errored or came back empty are left out: a call that failed took time but
    produced nothing, and averaging it in flatters or punishes the condition
    depending on how fast it failed.

    The open arm has no API charge. That is not the same as being free, so its
    GPU time is reported too, and converted to energy at the card's *rated* power
    (`gpu_watts`, 300 W for the RTX 5070 Ti the arm ran on). Real draw during
    inference is lower, so the energy figure is a ceiling, not a measurement.
    """
    out: list[Efficiency] = []
    for manifest, records in runs:
        condition = manifest["condition"]
        answers = [
            r["answer"] for r in records
            if not r["answer"].get("error") and (r["answer"].get("text") or "").strip()
        ]
        if not answers:
            continue
        latencies = [a["latency_s"] for a in answers if a.get("latency_s")]
        costs = [a.get("cost_usd") or 0.0 for a in answers]
        local = get_condition(condition).provider == "local"
        gpu_seconds = sum(latencies) if local else 0.0
        out.append(
            Efficiency(
                condition=condition,
                model=manifest.get("model", ""),
                n=len(answers),
                local=local,
                latency_median_s=_percentile(latencies, 0.5),
                latency_p90_s=_percentile(latencies, 0.9),
                output_tokens_median=_percentile([float(a["output_tokens"]) for a in answers], 0.5),
                cost_per_answer=sum(costs) / len(answers),
                cost_total=sum(costs),
                gpu_seconds=gpu_seconds,
                energy_kwh_upper=gpu_seconds * gpu_watts / 3_600_000,
            )
        )
    out.sort(key=lambda e: e.condition)
    return out


def breakdown(
    *,
    runs: list[tuple[dict, list[dict]]],
    grades: dict[str, Grade],
    item_set: ItemSet,
    key: str,
) -> dict[str, dict[str, float]]:
    """Mean correctness per condition, split by 'type', 'topic' or 'difficulty'."""
    attr = {"type": "type", "topic": "topic", "difficulty": "difficulty"}[key]
    item_key = {item.id: str(getattr(item, attr)) for item in item_set}

    buckets: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for _, records in runs:
        for record in records:
            grade = grades.get(record["response_id"])
            if grade is None:
                continue
            bucket = item_key.get(record["item_id"], "?")
            buckets[bucket][record["condition"]].append(grade.correctness)

    return {
        bucket: {cond: _mean(vals) for cond, vals in by_cond.items()}
        for bucket, by_cond in buckets.items()
    }


def render_efficiency(rows: list[Efficiency]) -> str:
    """The cost and latency table on its own, for use before grading is finished."""
    body = _efficiency_section(rows)
    return "\n".join(["# What an answer costs", ""] + body[3:])


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    def line(cells): return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    out = [line(headers), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += [line(row) for row in rows]
    return "\n".join(out)


def _efficiency_section(efficiencies: list[Efficiency]) -> list[str]:
    """The cost-and-latency prose and table, shared by both renderers."""
    out = [
        "",
        "## What an answer costs",
        "",
        "Accuracy is only half of the argument for a small local model. The other half "
        "is here: what one answer costs to produce, how long a student waits for it, "
        "and how much the system chooses to say. Failed and empty calls are excluded.",
        "",
        _table(
            ["Condition", "n", "Median s", "p90 s", "Median out tok", "USD/answer",
             "USD/1000", "GPU s/answer"],
            [
                [
                    e.condition,
                    str(e.n),
                    f"{e.latency_median_s:.1f}",
                    f"{e.latency_p90_s:.1f}",
                    f"{e.output_tokens_median:.0f}",
                    "none" if e.local else f"{e.cost_per_answer:.4f}",
                    "none" if e.local else f"{e.cost_per_answer * 1000:.2f}",
                    f"{e.gpu_seconds / e.n:.1f}" if e.local else "-",
                ]
                for e in efficiencies
            ],
        ),
    ]
    local = [e for e in efficiencies if e.local]
    if local:
        out += [
            "",
            "The open arm pays no API charge, which is the point of it, but it is not "
            "free: it occupies a GPU for the time shown. At the card's rated 300 W - a "
            "ceiling, since inference does not hold the power limit - the whole open arm "
            f"used at most {sum(e.energy_kwh_upper for e in local):.3f} kWh. At any "
            "domestic tariff that is cents, against dollars of API spend for the same "
            "answers; the real cost of the open arm is the card itself, paid once.",
        ]
    return out


def render(
    *,
    summaries: list[Summary],
    breakdowns: dict[str, dict[str, dict[str, float]]],
    item_set: ItemSet,
    paired: list[Paired] | None = None,
    reference: str | None = None,
    efficiencies: list[Efficiency] | None = None,
) -> str:
    out: list[str] = ["# Benchmark results", ""]

    if any(s.dry_run for s in summaries):
        out += [
            "> **DRY RUN DATA PRESENT.** One or more runs below called no model. "
            "These numbers are placeholders and must not be reported.",
            "",
        ]

    ungraded = [s for s in summaries if s.n_graded < s.n_responses]
    if ungraded:
        out += [
            "> Incomplete grading: "
            + ", ".join(f"{s.condition} ({s.n_graded}/{s.n_responses})" for s in ungraded)
            + ".",
            "",
        ]

    out += ["## Headline", ""]
    out.append(
        _table(
            ["Condition", "Graded", "Correctness", "Hard fail", "Policy", "Citations", "Cost USD"],
            [
                [
                    s.condition,
                    f"{s.n_graded}/{s.n_responses}",
                    f"{s.mean_correctness:.3f} ± {s.correctness_ci95:.3f}",
                    f"{s.hard_fail_rate:.3f}",
                    "—" if s.policy_pass_rate is None else f"{s.policy_pass_rate:.3f}",
                    "—" if s.citation_valid_rate is None else f"{s.citation_valid_rate:.3f}",
                    f"{s.cost_usd:.4f}",
                ]
                for s in summaries
            ],
        )
    )

    if paired:
        out += [
            "",
            f"## Paired comparison against `{reference}`",
            "",
            "Every condition answers the identical item set, so differences are taken "
            "item by item. Pairing removes the variation caused by some questions being "
            "harder than others, which an unpaired comparison would leave in the noise.",
            "",
        ]
        out.append(
            _table(
                ["Condition", "n", "Mean diff", "95% CI", "Better", "Worse", "Tied", "Excludes 0?"],
                [
                    [
                        p.condition,
                        str(p.n),
                        f"{p.mean_diff:+.3f}",
                        f"±{p.ci95:.3f}",
                        str(p.wins),
                        str(p.losses),
                        str(p.ties),
                        "yes" if p.significant else "no",
                    ]
                    for p in paired
                ],
            )
        )
        out += [
            "",
            "A positive mean difference means the condition scored higher than "
            f"`{reference}`. *Excludes 0* is whether the interval clears zero — where it "
            "does not, the two are not distinguishable on this item set, and the report "
            "should say so rather than reading a ranking into the means.",
        ]

    if efficiencies:
        out += _efficiency_section(efficiencies)

    for key, title in (("type", "By item type"), ("topic", "By topic"), ("difficulty", "By difficulty")):
        data = breakdowns.get(key) or {}
        if not data:
            continue
        conditions = sorted({c for by_cond in data.values() for c in by_cond})
        out += ["", f"## {title}", ""]
        out.append(
            _table(
                [key.capitalize()] + conditions,
                [
                    [bucket] + [
                        f"{data[bucket][c]:.3f}" if c in data[bucket] else "—"
                        for c in conditions
                    ]
                    for bucket in sorted(data)
                ],
            )
        )

    out += [
        "",
        "## Reading this",
        "",
        "- **Correctness** is the fraction of `must_include` points made, averaged over items, with the half-width of its 95% confidence interval.",
        "- **Hard fail** is the rate of answers containing a known failure or an invalid "
        "citation. It is reported separately, never averaged into correctness.",
        "- **Policy** covers `scope` and `integrity` items only — behaviour, judged apart "
        "from factual accuracy.",
        f"- Item set: {item_set.path.name}, {len(item_set)} items, digest `{item_set.digest}`.",
        "",
    ]
    return "\n".join(out)

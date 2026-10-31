"""How alike two systems' answers are — measured twice, because once is misleading.

The question this answers is not "which is right" (the grades answer that) but
"given that both are graded, do they say the same things?" Where two conditions
score the same on correctness *and* agree point by point, the cheaper one is doing
the expensive one's job, and that is the claim worth making.

Two measures, because either alone distorts:

**Substantive agreement** compares the graded decisions — for each required point,
did both systems make it? This is the honest measure of whether two answers say the
same things, and it is free: it falls out of grading that has already happened.
Length cannot affect it.

**Textual similarity** compares the answers as written, by embedding. It is reported
only beside the median answer lengths, because it is confounded by them: the tuned
model answers in about a seventh of the words the frontier model uses, and a cosine
score reads that as disagreement when it is nothing of the kind. Rifuwo's report
records the same effect in the opposite direction - a verbose chatbot scoring badly
against terse reference answers while being correct - which is the reason this
number never stands on its own here.
"""

from __future__ import annotations

from dataclasses import dataclass

from .grading import Grade
from .retrieval_eval import DENSE_MODEL


@dataclass
class Agreement:
    """Two conditions compared on the items where both were graded."""

    a: str
    b: str
    items: int                      # items where both conditions have a grade
    points: int                     # required points across those items
    points_agreed: int              # points both made, or both missed
    identical_items: int            # items where every decision matched
    a_only: int                     # points only `a` made
    b_only: int                     # points only `b` made

    @property
    def point_agreement(self) -> float:
        return self.points_agreed / self.points if self.points else 0.0

    @property
    def item_agreement(self) -> float:
        return self.identical_items / self.items if self.items else 0.0


def substantive_agreement(
    *,
    grades: dict[str, Grade],
    records: dict[str, dict],
    a: str,
    b: str,
) -> Agreement:
    """Do two conditions make the same points on the same items?

    `records` maps response id to its run record, which is what carries the item and
    the condition. Only items graded under both conditions count: an item graded for
    one and not the other says nothing about agreement.
    """
    by_item: dict[str, dict[str, Grade]] = {}
    for response_id, grade in grades.items():
        record = records.get(response_id)
        if record is None or record["condition"] not in (a, b):
            continue
        by_item.setdefault(record["item_id"], {})[record["condition"]] = grade

    items = points = agreed = identical = a_only = b_only = 0
    for per_condition in by_item.values():
        if a not in per_condition or b not in per_condition:
            continue
        hits_a = per_condition[a].must_include_hits
        hits_b = per_condition[b].must_include_hits
        if len(hits_a) != len(hits_b):  # different item sets; not comparable
            continue
        items += 1
        matched = True
        for one, other in zip(hits_a, hits_b):
            points += 1
            if one == other:
                agreed += 1
            else:
                matched = False
                if one:
                    a_only += 1
                else:
                    b_only += 1
        identical += 1 if matched else 0

    return Agreement(a=a, b=b, items=items, points=points, points_agreed=agreed,
                     identical_items=identical, a_only=a_only, b_only=b_only)


def _embed(texts: list[str], model_name: str = DENSE_MODEL, window: int = 512,
           stride: int = 128, batch_size: int = 16):
    """One vector per text, mean-pooled over overlapping windows.

    Answers run well past the embedding model's 512 tokens - the frontier arm's
    median is around 1,500 - so truncating would compare two openings rather than
    two answers. Each answer is embedded as overlapping windows and averaged, so
    the whole of it contributes.
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise RuntimeError(
            'Answer similarity needs torch and transformers:  pip install -e ".[local]"'
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    encoded = tokenizer(texts, max_length=window, truncation=True, stride=stride,
                        return_overflowing_tokens=True, padding=True, return_tensors="pt")
    owner = encoded.pop("overflow_to_sample_mapping")

    parts = []
    with torch.no_grad():
        total = encoded["input_ids"].shape[0]
        for start in range(0, total, batch_size):
            batch = {k: v[start:start + batch_size].to(device) for k, v in encoded.items()}
            cls = model(**batch).last_hidden_state[:, 0]
            parts.append(torch.nn.functional.normalize(cls, dim=-1).cpu())
    windows = torch.cat(parts)

    pooled = torch.zeros(len(texts), windows.shape[1])
    pooled.index_add_(0, owner, windows)
    counts = torch.zeros(len(texts)).index_add_(0, owner, torch.ones(len(owner)))
    return torch.nn.functional.normalize(pooled / counts.unsqueeze(1), dim=-1)


def textual_similarity(*, records: list[dict], a: str, b: str) -> dict:
    """Cosine similarity between two conditions' answers to the same question.

    Reported with the answer lengths that shape it, never alone.
    """
    by_item: dict[str, dict[str, dict]] = {}
    for record in records:
        if record["condition"] not in (a, b):
            continue
        answer = record["answer"]
        if answer.get("error") or not (answer.get("text") or "").strip():
            continue
        by_item.setdefault(record["item_id"], {})[record["condition"]] = record

    pairs = [(v[a], v[b]) for v in by_item.values() if a in v and b in v]
    if not pairs:
        return {"a": a, "b": b, "items": 0}

    import torch

    vectors = _embed([r["answer"]["text"] for pair in pairs for r in pair])
    left, right = vectors[0::2], vectors[1::2]
    scores = torch.sum(left * right, dim=-1).tolist()

    # The floor this score has to beat. Two answers about the same course share its
    # vocabulary whether or not they agree - measured here, an answer about convolution
    # and an answer about kNN classification still score 0.74 - so a raw cosine means
    # nothing until you know what the same measure gives for answers to *different*
    # questions. Every mismatch is used rather than one shuffling of them: with n
    # items there are n(n-1) wrong pairings available and no reason to sample.
    grid = left @ right.T                      # every answer of `a` against every answer of `b`
    n = grid.shape[0]
    matched = grid.diagonal()
    off = ~torch.eye(n, dtype=torch.bool)
    mismatched = grid[off]

    # The rank statistic is the one to quote. For each answer of `a`, is the other
    # condition's answer to the *same* question its closest match out of all of them?
    # That question is scale-free: it does not care that a cosine of 0.84 sounds high
    # or that 0.11 sounds small, only whether the right answer came first.
    rank_of_match = (grid >= matched.unsqueeze(1)).sum(dim=1)
    top1 = int((rank_of_match == 1).sum())
    mean_rank = float(rank_of_match.float().mean())

    # And a permutation test, so "above the floor" is a claim with a number behind it:
    # how often does a random pairing of answers score as well as the true pairing?
    generator = torch.Generator().manual_seed(4114)
    draws = 10000
    null = torch.stack([
        grid[torch.arange(n), torch.randperm(n, generator=generator)].mean()
        for _ in range(draws)
    ])
    p_value = float((null >= matched.mean()).float().mean())

    scores = matched.tolist()
    floor = mismatched.tolist()

    def median(values: list[float]) -> float:
        ordered = sorted(values)
        return ordered[len(ordered) // 2] if ordered else 0.0

    return {
        "a": a,
        "b": b,
        "items": len(pairs),
        "mean_similarity": sum(scores) / len(scores),
        "median_similarity": median(scores),
        "lowest": min(scores),
        "highest": max(scores),
        "mismatched_pairs": len(floor),
        "mismatched_floor": sum(floor) / len(floor),
        "lift_over_floor": sum(scores) / len(scores) - sum(floor) / len(floor),
        "matched_ranked_first": top1,
        "mean_rank_of_match": mean_rank,
        "permutation_p": p_value,
        # The lengths belong beside the score: they are most of what moves it.
        "median_tokens_a": median([float(r["answer"]["output_tokens"]) for r, _ in pairs]),
        "median_tokens_b": median([float(r["answer"]["output_tokens"]) for _, r in pairs]),
        "per_item": {
            r["item_id"]: round(score, 4) for (r, _), score in zip(pairs, scores)
        },
    }

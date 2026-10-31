"""Chunk size against retrieval depth, on quality and on what it costs to send.

Four settings of this system were never measured, only defaulted: how long a chunk is,
how much chunks overlap, how many are retrieved, and whether fusing BM25 with a dense
model beats BM25 alone. This sweeps all four at once, and it needs no generation, so
the grid costs minutes rather than the hours an answer-quality sweep costs.

Why the ground truth is text overlap and not a chunk id. Each question was written
from one known chunk of the 800-token corpus. That chunk does not exist in the 400- or
1200-token corpus, so `was the right chunk id returned` cannot compare across sizes -
it would score every other corpus at nothing. Page numbers do not work either: a chunk
records only the page its first sentence is on and merges small pages, so the 800-token
chunk starting at page 3 may cover 3 to 5 while the 400-token chunk over the same
material starts at page 4, and a strict page match would call that a miss. It would
penalise small chunks for being small.

So a retrieved chunk counts as a hit when its text substantially overlaps the passage
the question was written from, measured on shared word eight-grams against the shorter
of the two. A 400-token chunk lying inside the source passage has nearly all its grams
in it; a 1200-token chunk containing the source passage has nearly all of the source's
grams; an unrelated chunk of any size has almost none. That is comparable in every
direction, which is what the comparison needs.

Reported beside recall is the mean tokens of retrieved material per question, because
that is what the card responds to - a passage count only means something at a fixed
chunk size, and the memory ceiling has to be stated in a form another size can be
compared against.

    python scripts/chunk_grid.py --device cpu          while the card is busy
    python scripts/chunk_grid.py --arms bm25 fused
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ym07.retrieval import Chunk, load_chunks  # noqa: E402

# The corpus the questions were written from. Its chunk texts are the ground truth
# every other corpus is scored against.
REFERENCE = ROOT / "corpus" / "processed" / "chunks.jsonl"
CORPORA = {
    200: ROOT / "corpus" / "processed" / "chunks-200.jsonl",
    400: ROOT / "corpus" / "processed" / "chunks-400.jsonl",
    800: REFERENCE,
    1200: ROOT / "corpus" / "processed" / "chunks-1200.jsonl",
}
PAIRS = [
    ROOT / "training" / "deepseek" / "pairs.jsonl",
    ROOT / "training" / "sample" / "pairs.jsonl",
    ROOT / "training" / "sample-v2" / "pairs.jsonl",
    ROOT / "training" / "qwen-ml-notes" / "pairs.jsonl",
]
KS = (1, 2, 3, 4, 6, 8, 12, 16)
GRAM = 8
# Half the shorter passage's eight-grams. Chosen before the sweep was run: a chunk
# sharing half of its wording with the passage a question was written from is the
# passage, re-cut; one sharing a tenth is a neighbour that happens to repeat a phrase.
OVERLAP = 0.5
# The threshold is a judgement, so the sweep reports the neighbouring ones too. A
# finding that only holds at the value its author picked is not a finding, and the
# overlaps are already computed - only the comparison against them changes.
OVERLAPS = (0.3, 0.5, 0.7)
# A token is about four characters of this material, the same figure the chunker uses.
CHARS_PER_TOKEN = 4


def grams(text: str) -> set[tuple[str, ...]]:
    words = text.lower().split()
    if len(words) < GRAM:
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + GRAM]) for i in range(len(words) - GRAM + 1)}


def overlaps(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def read_pairs() -> list[dict]:
    seen, out = set(), []
    for path in PAIRS:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            pair = json.loads(line)
            key = (pair.get("model"), " ".join(pair["question"].lower().split()))
            if key in seen:
                continue
            seen.add(key)
            out.append(pair)
    return out


def build(name: str, chunks: list[Chunk], device: str | None, cache: dict):
    """One retriever over this corpus, reusing the halves a later arm needs.

    `fused` is BM25 and dense together, so building the three arms naively embeds each
    corpus twice - and embedding 1,335 chunks on a CPU is the slow part of this sweep.
    """
    from ym07.retrieval_eval import BM25Retriever, DenseRetriever, FusedRetriever

    if name == "bm25":
        return cache.setdefault("bm25", BM25Retriever(chunks))
    if name == "dense":
        return cache.setdefault("dense", DenseRetriever(chunks, device=device))
    if name == "fused":
        return FusedRetriever(build("bm25", chunks, device, cache),
                              build("dense", chunks, device, cache))
    raise SystemExit(f"unknown retriever '{name}'")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["bm25", "dense", "fused"])
    parser.add_argument("--sizes", nargs="+", type=int, default=sorted(CORPORA))
    parser.add_argument("--device", default=None,
                        help="'cpu' keeps the embedding off a card another run is using")
    parser.add_argument("--out", default="stress/results/retrieval/chunk-grid.json")
    args = parser.parse_args()

    pairs = read_pairs()
    if not pairs:
        print("no generated question pairs found", file=sys.stderr)
        return 1

    reference = {c.chunk_id: c for c in load_chunks(REFERENCE)}
    wanted = []
    for pair in pairs:
        source = reference.get(pair["chunk_id"])
        if source is None:
            continue  # written from a chunk this corpus no longer has
        wanted.append({"question": pair["question"], "grams": grams(source.text)})
    print(f"{len(wanted)} of {len(pairs)} questions have their source passage in "
          f"{REFERENCE.name}\n")

    from ym07.retrieval_eval import search_many

    table: dict = {}
    deepest = max(KS)
    for size in args.sizes:
        path = CORPORA.get(size)
        if path is None or not path.exists():
            print(f"  no corpus for {size} tokens, skipped")
            continue
        chunks = load_chunks(path)
        grammed = {c.chunk_id: grams(c.text) for c in chunks}
        lengths = {c.chunk_id: len(c.text) / CHARS_PER_TOKEN for c in chunks}
        cache: dict = {}
        for arm in args.arms:
            retriever = build(arm, chunks, args.device, cache)
            results = search_many(retriever, [w["question"] for w in wanted], deepest)
            found = {t: {k: 0 for k in KS} for t in OVERLAPS}
            sent = {k: 0.0 for k in KS}
            for want, hits in zip(wanted, results):
                shared = [overlaps(grammed[h.chunk.chunk_id], want["grams"]) for h in hits]
                cost = [lengths[h.chunk.chunk_id] for h in hits]
                for k in KS:
                    sent[k] += sum(cost[:k])
                    for threshold in OVERLAPS:
                        found[threshold][k] += any(s >= threshold for s in shared[:k])
            n = len(wanted)
            table[f"{size}:{arm}"] = {
                "chunks": len(chunks),
                "recall": {k: found[OVERLAP][k] / n for k in KS},
                "recall_by_overlap": {str(t): {k: found[t][k] / n for k in KS}
                                      for t in OVERLAPS},
                "tokens": {k: sent[k] / n for k in KS},
            }
            row = "  ".join(f"{found[OVERLAP][k] / n:5.1%}" for k in KS)
            print(f"  {size:>4} tok  {arm:<6} {len(chunks):>5} chunks   {row}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "questions": len(wanted), "ks": list(KS), "gram": GRAM, "overlap": OVERLAP,
        "overlaps": list(OVERLAPS), "reference": REFERENCE.name, "results": table,
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nk =            " + "  ".join(f"{k:>5}" for k in KS))
    print(f"wrote {out.relative_to(ROOT)}")

    print("\nRecall against what it costs to send, at the depth each setting needs "
          "for its best recall:")
    print(f"{'setting':<16}{'best recall':>12}{'at k':>6}{'tokens sent':>13}"
          f"{'recall per 1k tokens':>22}")
    for key, got in sorted(table.items(),
                           key=lambda kv: -max(kv[1]["recall"].values())):
        best_k = max(got["recall"], key=lambda k: got["recall"][k])
        best = got["recall"][best_k]
        cost = got["tokens"][best_k]
        print(f"{key:<16}{best:>11.1%}{best_k:>6}{cost:>13.0f}"
              f"{best / max(cost, 1) * 1000:>22.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

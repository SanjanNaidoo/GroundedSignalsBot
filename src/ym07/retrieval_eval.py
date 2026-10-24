"""Choosing the retriever, on questions the benchmark never saw.

TF-IDF has been the retriever since v0, and it was measured to favour past
papers over the course notes. Three alternatives are compared with it here:
BM25, a dense embedding model, and BM25 and dense fused. The comparison uses the
question-and-answer pairs written for fine-tuning, because each of those
questions was written from one known chunk. That makes "did the retriever find
the passage that answers this?" a question with a right answer, and it keeps the
choice off the benchmark: picking a retriever by its score on the benchmark's
own questions would be tuning on the test.

Generated questions carry a bias worth stating before any result. They were
written with the passage in view, so they reuse its vocabulary more than a
student asking cold would, and that favours the lexical retrievers. If the
dense retriever wins anyway, the bias makes the win conservative. If a lexical
retriever wins narrowly, the bias may be the reason, and the benchmark-question
figures - kept here for comparison, never for choosing - are the check.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from .retrieval import Chunk, Hit, tokenize

DENSE_MODEL = "BAAI/bge-small-en-v1.5"
# bge v1.5 was trained with this instruction on the query side only; passages
# are embedded as they are.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class BM25Retriever:
    """Okapi BM25 over the same tokens the TF-IDF retriever uses.

    It differs from TF-IDF where this corpus needs it to: a term's contribution
    saturates as the term repeats, and long chunks are discounted. Past papers
    repeat a term densely in little prose, which is exactly what unsaturated
    TF-IDF rewards.
    """

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self._k1, self._b = k1, b
        self._tf = [Counter(tokenize(c.text)) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg = sum(self._len) / len(self._len) if chunks else 0.0
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks)
        self._idf = {term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()}

    def search(self, query: str, top_k: int) -> list[Hit]:
        terms = set(tokenize(query))
        scored: list[Hit] = []
        for idx, tf in enumerate(self._tf):
            norm = self._k1 * (1 - self._b + self._b * self._len[idx] / self._avg)
            score = sum(
                self._idf[term] * tf[term] * (self._k1 + 1) / (tf[term] + norm)
                for term in terms
                if term in tf
            )
            if score > 0:
                scored.append(Hit(self.chunks[idx], score))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    def __len__(self) -> int:
        return len(self.chunks)


class DenseRetriever:
    """Embedding search, scoring each chunk by its closest window.

    The embedding model reads at most 512 tokens and most chunks are longer -
    248 of the 316 in the deduplicated corpus. Truncating them would leave the
    end of most passages unseen, which is how the first measurement was run, so
    each chunk is embedded as overlapping windows and scored by whichever window
    is closest to the query.

    torch and transformers are imported here rather than at module level, as in
    LocalClient, so the harness stays installable where there is no GPU.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        model_name: str = DENSE_MODEL,
        window: int = 512,
        stride: int = 128,
        batch_size: int = 32,
    ):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on the machine
            raise RuntimeError(
                'Dense retrieval needs torch and transformers:  pip install -e ".[local]"'
            ) from exc

        self._torch = torch
        self.chunks = chunks
        self.model_name = model_name
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name).to(self._device).eval()

        encoded = self._tokenizer(
            [c.text for c in chunks],
            max_length=window,
            truncation=True,
            stride=stride,
            return_overflowing_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        self._owner = encoded.pop("overflow_to_sample_mapping").to(self._device)
        self.windows = int(self._owner.numel())
        self._vectors = self._embed(encoded, batch_size)

    def _embed(self, encoded, batch_size: int):
        torch = self._torch
        parts = []
        with torch.no_grad():
            total = encoded["input_ids"].shape[0]
            for start in range(0, total, batch_size):
                batch = {k: v[start:start + batch_size].to(self._device) for k, v in encoded.items()}
                cls = self._model(**batch).last_hidden_state[:, 0]
                parts.append(torch.nn.functional.normalize(cls, dim=-1))
        return torch.cat(parts)

    def search_many(self, queries: list[str], top_k: int) -> list[list[Hit]]:
        torch = self._torch
        encoded = self._tokenizer(
            [QUERY_PREFIX + q for q in queries],
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        query_vectors = self._embed(encoded, batch_size=64)
        similarity = query_vectors @ self._vectors.T
        best = torch.full(
            (len(queries), len(self.chunks)), float("-inf"), device=self._device
        )
        owners = self._owner.unsqueeze(0).expand(len(queries), -1)
        best = best.scatter_reduce(1, owners, similarity, reduce="amax")
        top = torch.topk(best, min(top_k, len(self.chunks)), dim=1)
        return [
            [Hit(self.chunks[i], float(s)) for s, i in zip(values, indices)]
            for values, indices in zip(top.values.tolist(), top.indices.tolist())
        ]

    def search(self, query: str, top_k: int) -> list[Hit]:
        return self.search_many([query], top_k)[0]

    def __len__(self) -> int:
        return len(self.chunks)


class FusedRetriever:
    """Reciprocal-rank fusion: a chunk several retrievers agree on outranks one
    that only a single retriever ranks highly."""

    def __init__(self, *retrievers, k: int = 60, depth: int = 50):
        self._retrievers = retrievers
        self._k = k
        self._depth = depth
        self.chunks = retrievers[0].chunks

    def search_many(self, queries: list[str], top_k: int) -> list[list[Hit]]:
        rankings = [search_many(r, queries, self._depth) for r in self._retrievers]
        results: list[list[Hit]] = []
        for qi in range(len(queries)):
            score: dict[str, float] = defaultdict(float)
            chunk_by_id: dict[str, Chunk] = {}
            for ranking in rankings:
                for rank, hit in enumerate(ranking[qi], start=1):
                    score[hit.chunk.chunk_id] += 1.0 / (self._k + rank)
                    chunk_by_id[hit.chunk.chunk_id] = hit.chunk
            ranked = sorted(score.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            results.append([Hit(chunk_by_id[cid], s) for cid, s in ranked])
        return results

    def search(self, query: str, top_k: int) -> list[Hit]:
        return self.search_many([query], top_k)[0]

    def __len__(self) -> int:
        return len(self.chunks)


def search_many(retriever, queries: list[str], top_k: int) -> list[list[Hit]]:
    if hasattr(retriever, "search_many"):
        return retriever.search_many(queries, top_k)
    return [retriever.search(q, top_k) for q in queries]


def _sources(chunk: Chunk) -> set[str]:
    return {chunk.source, *chunk.also_in}


def recall_on_pairs(retriever, pairs: list[dict], ks: tuple[int, ...] = (1, 3, 8)) -> dict:
    """How often the chunk a generated question was written from comes back.

    `chunk@k` is the strict measure: that exact chunk in the top k. `file@k`
    also counts another page of the same file, which is often an acceptable
    answer, and the gap between the two says how much of a retriever's miss is
    near-miss.
    """
    deepest = max(ks)
    results = search_many(retriever, [p["question"] for p in pairs], deepest)
    found = {k: 0 for k in ks}
    same_file = 0
    reciprocal = 0.0
    for pair, hits in zip(pairs, results):
        ids = [h.chunk.chunk_id for h in hits]
        if pair["chunk_id"] in ids:
            reciprocal += 1.0 / (ids.index(pair["chunk_id"]) + 1)
        for k in ks:
            found[k] += pair["chunk_id"] in ids[:k]
        wanted = {pair["source"], *pair.get("also_in", [])}
        same_file += any(_sources(h.chunk) & wanted for h in hits)
    n = len(pairs) or 1
    row: dict = {"questions": len(pairs)}
    row.update({f"chunk@{k}": found[k] / n for k in ks})
    row[f"file@{deepest}"] = same_file / n
    row["mrr"] = reciprocal / n
    return row


def recall_on_benchmark(retriever, items, k: int = 8) -> dict:
    """The earlier measurement on benchmark questions, for comparison only.

    Whether the file an item cites comes back, and what share of everything
    retrieved is course notes. Never used to choose: see the module docstring.
    """
    in_scope = [it for it in items if it.type != "scope"]
    results = search_many(retriever, [it.question for it in in_scope], k)
    cited = found = 0
    notes = returned = 0
    for item, hits in zip(in_scope, results):
        returned += len(hits)
        notes += sum("notes" in h.chunk.source.lower() for h in hits)
        if item.source:
            cited += 1
            wanted = item.source.split(",")[0].strip()
            found += any(wanted in _sources(h.chunk) for h in hits)
    return {
        "items": len(in_scope),
        f"cited_file@{k}": found / (cited or 1),
        "notes_share": notes / (returned or 1),
    }

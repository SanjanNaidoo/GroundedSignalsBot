"""Commands for the fine-tuning pipeline: compare retrievers, build examples, train.

Kept apart from cli.py, which drives the benchmark, so the steps that produce
C2-oss-tuned read as one sequence:

  ym07 gen-pairs ...                         write question-and-answer pairs
  ym07 eval-retrieval --pairs ...            choose a retriever on those questions
  ym07 build-sft --pairs ... --retriever M   examples in the format C2 is tested in
  ym07 gen-behaviour --pairs ...             declines, scaffolds and revision answers
  ym07 train --max-length N                  fit the LoRA adapter

build-sft and train make every consequential choice an explicit argument -
retriever, passages per example, token budget, sequence length - and write each
one into a manifest, so the adapter can be traced back to how it was made.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

from .conditions import FRONTIER_IDS, get_condition
from .config import load_config
from .items import load_items
from .retrieval import RETRIEVAL_METHODS, LexicalRetriever, build_retriever, dedupe_chunks, load_chunks


def _read_pairs(paths: list[str]) -> list[dict]:
    """Pairs from one or more gen-pairs outputs, each distinct question counted once.

    Keyed on the question text rather than the pair id: two runs over the same
    chunk reuse ids like `chunk#1` for different questions, and keying on the id
    silently dropped 25 of 150 questions when the two samples were read together.
    """
    pairs: list[dict] = []
    seen: set[tuple] = set()
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            pair = json.loads(line)
            key = (pair.get("model"), " ".join(pair["question"].lower().split()))
            if key not in seen:
                seen.add(key)
                pairs.append(pair)
    return pairs


def cmd_eval_retrieval(args) -> int:
    from .retrieval_eval import (
        BM25Retriever,
        DenseRetriever,
        FusedRetriever,
        recall_on_benchmark,
        recall_on_pairs,
    )

    config = load_config(args.config)
    items = list(load_items(args.items, known_topics=config.topic_ids))
    pairs = _read_pairs(args.pairs)
    if not pairs:
        print("No pairs to evaluate on.", file=sys.stderr)
        return 1

    chunks = dedupe_chunks(load_chunks(args.chunks))
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks)
    arms = {
        "tfidf": LexicalRetriever(chunks),
        "bm25": bm25,
        "dense": dense,
        "fused": FusedRetriever(bm25, dense),
    }
    ks = tuple(sorted(set(args.k)))
    deepest = max(ks)
    writers = Counter(p.get("model") for p in pairs)
    print(f"{len(pairs)} generated questions over {len(chunks)} chunks; "
          f"writers: {', '.join(f'{m} ({n})' for m, n in writers.items())}\n")

    header = "".join(f"{f'chunk@{k}':>9}" for k in ks) + f"{f'file@{deepest}':>9}{'mrr':>7}"
    print(f"{'':<7}{header}   | benchmark, comparison only: cited file@{deepest}, notes share")
    table = {}
    for name, retriever in arms.items():
        on_pairs = recall_on_pairs(retriever, pairs, ks)
        on_bench = recall_on_benchmark(retriever, items, deepest)
        table[name] = {"generated": on_pairs, "benchmark": on_bench}
        row = "".join(f"{on_pairs[f'chunk@{k}']:>9.0%}" for k in ks)
        row += f"{on_pairs[f'file@{deepest}']:>9.0%}{on_pairs['mrr']:>7.2f}"
        print(f"{name:<7}{row}   |  {on_bench[f'cited_file@{deepest}']:>5.0%}  {on_bench['notes_share']:>5.0%}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "pair_files": args.pairs,
            "questions": len(pairs),
            "writers": dict(writers),
            "chunks": len(chunks),
            "ks": list(ks),
            "results": table,
        }, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"\nWritten to {out}")
    return 0


def cmd_build_sft(args) -> int:
    from transformers import AutoTokenizer

    from .finetune import build_examples, write_examples

    config = load_config(args.config)
    pairs = _read_pairs(args.pairs)
    if not pairs:
        print("No pairs to build examples from.", file=sys.stderr)
        return 1
    retriever = build_retriever(args.chunks, args.retriever)
    if retriever is None:
        print(f"No corpus at {args.chunks}.", file=sys.stderr)
        return 1
    base = get_condition("C2-oss").resolved_model(config)
    tokenizer = AutoTokenizer.from_pretrained(base)

    examples, summary = build_examples(
        config=config,
        pairs=pairs,
        retriever=retriever,
        top_k=args.top_k,
        tokenizer=tokenizer,
        max_tokens=args.max_tokens,
    )
    summary.update({
        "retriever": args.retriever,
        "pair_files": args.pairs,
        "writers": dict(Counter(p.get("model") for p in pairs)),
        "tokenizer": base,
        "config_digest": config.digest,
    })
    write_examples(examples, summary, Path(args.out))
    print(json.dumps(summary, indent=2))
    return 0


def cmd_gen_behaviour(args) -> int:
    from .behaviour import generate_behaviour
    from .model import build_client

    config = load_config(args.config)
    item_set = load_items(args.items, known_topics=config.topic_ids)
    content = _read_pairs(args.pairs)
    if args.writer == "local":
        model_id = get_condition("C1-oss").resolved_model(config)
        client = build_client(args.dry_run, "local", config)
        max_tokens, effort = config.model.local_max_new_tokens, "n/a"
    else:
        entry = config.frontier_by_id(args.writer)
        if not args.dry_run and not os.environ.get(entry.key_env):
            print(f"{entry.key_env} is not set. Put it in .env, which git ignores.", file=sys.stderr)
            return 1
        model_id = entry.model
        client = build_client(args.dry_run, entry.provider, config, entry)
        max_tokens, effort = config.model.max_tokens, "default"

    summary = generate_behaviour(
        config=config,
        item_set=item_set,
        chunks=load_chunks(args.chunks),
        content_pairs=content,
        client=client,
        model_id=model_id,
        out_path=Path(args.out),
        assessed=args.assessed,
        revision=args.revision,
        max_tokens=max_tokens,
        effort=effort,
    )
    print(json.dumps({k: v for k, v in summary.items() if not k.endswith(("_prompt", "_templates"))}, indent=2))
    return 0


def cmd_train(args) -> int:
    from .finetune import train_adapter

    config = load_config(args.config)
    # Which weights the adapter is fitted to. It was fixed at the Qwen2.5 base, which
    # was right while that was the only model in the project; the adapter has to be
    # trainable on whichever model is being delivered, and an adapter is tied to the
    # base it was fitted to - loading one onto different weights is not meaningful.
    base = get_condition(args.base).resolved_model(config)
    out = Path(args.out or get_condition("C2-oss-tuned").resolved_model(config))
    summary = train_adapter(
        examples_path=Path(args.examples),
        base_model=base,
        out_dir=out,
        max_length=args.max_length,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        rank=args.rank,
        alpha=args.alpha,
        grad_accum=args.grad_accum,
        max_steps=args.max_steps,
        eval_fraction=args.eval_fraction,
        resume=args.resume,
        max_vram_gb=args.max_vram_gb,
        stall_factor=args.stall_factor,
        seed=args.seed,
    )
    shown = {k: v for k, v in summary.items() if k != "log"}
    print(json.dumps(shown, indent=2))
    print(f"\nAdapter saved to {out}")
    return 0


def add_parsers(sub, common, chunks_default: str) -> None:
    p = sub.add_parser("eval-retrieval", help="compare retrievers on generated questions")
    common(p)
    p.add_argument("--chunks", default=chunks_default)
    p.add_argument("--pairs", nargs="+", required=True, help="one or more gen-pairs output files")
    p.add_argument("--k", nargs="+", type=int, default=[1, 3, 4, 8])
    p.add_argument("--out", default=None, help="also write the results as JSON")
    p.set_defaults(func=cmd_eval_retrieval)

    p = sub.add_parser("build-sft", help="turn pairs into training examples in the C2 format")
    common(p)
    p.add_argument("--chunks", default=chunks_default)
    p.add_argument("--pairs", nargs="+", required=True, help="one or more gen-pairs output files")
    p.add_argument("--retriever", choices=RETRIEVAL_METHODS, required=True)
    p.add_argument("--top-k", type=int, required=True, help="passages per example before fitting the budget")
    p.add_argument("--max-tokens", type=int, required=True,
                   help="token budget per example; must not exceed the max length used to train")
    p.add_argument("--out", default="training/sft.jsonl")
    p.set_defaults(func=cmd_build_sft)

    p = sub.add_parser("train", help="fit the LoRA adapter for C2-oss-tuned")
    common(p)
    p.add_argument("--examples", default="training/sft.jsonl")
    p.add_argument("--base", default="C2-oss",
                   help="condition naming the weights to fit the adapter to; an adapter "
                        "only means anything on the base it was trained against")
    p.add_argument("--max-length", type=int, required=True)
    p.add_argument("--epochs", type=float, default=2.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--eval-fraction", type=float, default=0.05,
                   help="share of source chunks held out and scored after each epoch")
    p.add_argument("--resume", action="store_true",
                   help="continue from the latest epoch checkpoint in the adapter directory")
    p.add_argument("--max-vram-gb", type=float, default=None,
                   help="cap on GPU memory PyTorch may reserve; on Windows set it below what is "
                        "free, or the driver spills into system RAM and training crawls")
    p.add_argument("--stall-factor", type=float, default=4.0,
                   help="stop if a step takes this many times the median step so far")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="adapter directory (default: the C2-oss-tuned model path)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("gen-behaviour", help="write decline, scaffold and revision examples")
    common(p)
    p.add_argument("--chunks", default=chunks_default)
    p.add_argument("--pairs", nargs="+", required=True,
                   help="content pairs to frame as assessed work and as revision")
    p.add_argument("--writer", default="local", choices=["local", *FRONTIER_IDS])
    p.add_argument("--assessed", type=int, default=40)
    p.add_argument("--revision", type=int, default=30)
    p.add_argument("--out", default="training/behaviour/pairs.jsonl")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_gen_behaviour)

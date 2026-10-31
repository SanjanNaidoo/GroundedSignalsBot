"""Fine-tuning the open arm: training examples, then a LoRA adapter.

Two steps, kept apart so each can be inspected before the next runs.

`build_examples` turns generated pairs into the exact conversation C2-oss-tuned
meets at test time: the grounded course prompt as the system message, and
SOURCES from the chosen retriever followed by the question as the user message,
with the pair's answer as the target. Training in the tested format is the
point. A model tuned on bare question-and-answer pairs would first see retrieved
passages in the benchmark itself.

The passages are exactly what the retriever returns. An answer whose source
chunk came back cites its tag, so the model practises answering from the one
passage that matters among several that do not, which is what retrieval hands
it. An answer whose source was missed says the passages do not cover it, then
answers uncited. The first build inserted a missed source instead, so every
course question came with its answer in the sources and every decline without:
the model learned to call any question whose passages did not match another
course's, and turned away a quarter of the in-scope benchmark questions. A
scaffold for assessed work cites nothing either way, since it points at an idea
rather than restating the page and a tag would land on the refusal.

Every example is fitted to the token budget as it is built, by dropping the
lowest-ranked other passages and never the one the answer came from. Training
then truncates nothing. That matters because trl truncates by keeping the start
of a sequence, and the end of every example is the answer.

`train_adapter` fits a LoRA adapter on the 4-bit base with peft and trl, with
loss on the answer only.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

from .prompts import system_prompt, user_prompt
from .retrieval_eval import search_many

DEFAULT_EXAMPLES = Path("training/sft.jsonl")

_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+")
_PROSE = re.compile(r"[A-Za-z]{3,}")

# How an answer opens when its source is not among the passages.
MISSED_NOTES = (
    "The passages supplied don't cover this directly, but it is part of the course.",
    "None of the passages above addresses this, so this answer draws on the course material more broadly.",
    "This isn't in the passages retrieved, but the course does cover it.",
    "The retrieved passages don't include this, though it is course material.",
)


def _slot(pair_id: str, slots: int) -> int:
    """A stable choice among `slots` options, spread evenly across pairs."""
    return int(hashlib.sha256(pair_id.encode("utf-8")).hexdigest(), 16) % slots


def cite(answer: str, tag: str) -> str:
    """Append the source tag to each paragraph of prose.

    The grounded prompt asks for a tag at the point a claim is made. Generated
    answers carry none, because each was written from a single passage; every
    claim in one comes from the passage now tagged `tag`, so each prose paragraph
    ends with it. A paragraph that is only displayed mathematics is left alone,
    since a tag after an equation block reads as part of the equation.
    """
    blocks = answer.strip().split("\n\n")
    return "\n\n".join(
        block.rstrip() + f" [{tag}]" if _PROSE.search(_LATEX_COMMAND.sub("", block)) else block
        for block in blocks
    )


def _count_tokens(tokenizer, messages: list[dict]) -> int:
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    return len(tokenizer(text)["input_ids"])


def build_examples(
    *,
    config,
    pairs: list[dict],
    retriever,
    top_k: int,
    tokenizer=None,
    max_tokens: int | None = None,
) -> tuple[list[dict], dict]:
    """Training conversations in the C2 format, and a summary of how they were built."""
    system = system_prompt(config, grounded=True)
    rankings = search_many(retriever, [p["question"] for p in pairs], top_k)

    stats: Counter = Counter()
    examples: list[dict] = []
    for pair, ranked in zip(pairs, rankings):
        hits = list(ranked[:top_k])
        # A decline of a question from another course has no source chunk, gets
        # whatever the retriever returned and cites nothing: the passages do not
        # answer it, which is the point of the example.
        position = None
        if pair.get("chunk_id") is None:
            stats["no source passage"] += 1
        else:
            ids = [h.chunk.chunk_id for h in hits]
            if pair["chunk_id"] in ids:
                position = ids.index(pair["chunk_id"])
                stats["source retrieved"] += 1
            else:
                stats["source not retrieved"] += 1
        scaffold = pair.get("kind") == "assessed"
        missed = pair.get("chunk_id") is not None and position is None and not scaffold

        while True:
            tag = f"S{position + 1}" if position is not None else None
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt(pair["question"], hits)},
            ]
            if missed:
                note = MISSED_NOTES[_slot(pair["pair_id"], len(MISSED_NOTES))]
                reply = f"{note}\n\n{pair['answer'].strip()}"
            elif tag and not scaffold:
                reply = cite(pair["answer"], tag)
            else:
                reply = pair["answer"].strip()
            completion = [{"role": "assistant", "content": reply}]
            tokens = _count_tokens(tokenizer, messages + completion) if tokenizer else None
            if tokens is None or max_tokens is None or tokens <= max_tokens or len(hits) <= 1:
                break
            if position is None or position != len(hits) - 1:
                drop = len(hits) - 1
            else:
                drop = len(hits) - 2
            hits.pop(drop)
            if position is not None and drop < position:
                position -= 1
            stats["other passages dropped to fit"] += 1

        if tokens is not None and max_tokens is not None and tokens > max_tokens:
            stats["skipped, over budget even alone"] += 1
            continue
        if scaffold and tag:
            stats["assessed, source shown but not cited"] += 1
        if missed:
            stats["answered without its source, uncited"] += 1

        # A pair may ask to be seen more than once. The worked calculations are the
        # only ones that do: there are seventy of them against nine hundred short
        # expository answers, and at that share they would be drowned out by the very
        # style they exist to correct. Repetition weights a behaviour without
        # inventing data, and the count travels on the example so the built file says
        # plainly how often each was shown.
        repeat = max(1, int(pair.get("repeat", 1)))
        stats["repeated to carry weight"] += repeat - 1
        for copy in range(repeat):
            examples.append({
                "pair_id": pair["pair_id"] if copy == 0 else f"{pair['pair_id']}#r{copy}",
                "chunk_id": pair["chunk_id"],
                "source_tag": tag,
                "cited": bool(tag and not scaffold),
                "source_missed": missed,
                "sources": len(hits),
                "tokens": tokens,
                "repeat_of": pair["pair_id"] if copy else None,
                "prompt": messages,
                "completion": completion,
            })

    lengths = sorted(e["tokens"] for e in examples if e["tokens"] is not None)
    summary = {
        "pairs": len(pairs),
        "examples": len(examples),
        "top_k": top_k,
        "max_tokens": max_tokens,
        **dict(stats),
    }
    if lengths:
        summary.update({
            "tokens_median": lengths[len(lengths) // 2],
            "tokens_p90": lengths[int(len(lengths) * 0.9)],
            "tokens_max": lengths[-1],
        })
    return examples, summary


def write_examples(examples: list[dict], summary: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for example in examples:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")
    path.with_suffix(".manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


def _is_eval(key: str, fraction: float) -> bool:
    """Whether `key` falls in the held-out share, the same way on every run."""
    if fraction <= 0:
        return False
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF < fraction


def train_adapter(
    *,
    examples_path: Path,
    base_model: str,
    out_dir: Path,
    max_length: int,
    epochs: float = 2.0,
    learning_rate: float = 2e-4,
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    batch_size: int = 1,
    grad_accum: int = 8,
    max_steps: int = -1,
    eval_fraction: float = 0.05,
    resume: bool = False,
    max_vram_gb: float | None = None,
    stall_factor: float = 4.0,
    seed: int = 0,
) -> dict:
    """Fit a LoRA adapter to the examples and save it to `out_dir`.

    The adapter is saved on its own, naming its base, which is how LocalClient
    loads it: C2-oss and C2-oss-tuned then differ by the adapter and nothing else.

    A share of the examples is held out and scored after every epoch. The split is
    by source chunk, so a question and its revision-framed copy never land on
    opposite sides, and by a hash, so it is the same on every run; its loss is the
    check that the epochs are teaching rather than memorising. A checkpoint is
    kept after each epoch, and `resume` continues from the latest one, so a long
    run interrupted part-way does not start again.

    `max_vram_gb` caps the GPU memory PyTorch may reserve. On Windows the driver
    backs an allocation that does not fit with system RAM rather than refusing it,
    so PyTorch never frees its cache and retries, and training carries on at a
    fraction of its speed: the first full run here went from 37 seconds a step to
    20 minutes from step 14 on, with 7.1 GB of GPU memory in system RAM. Under the
    cap PyTorch frees cache before asking for more, and a real shortage fails at
    once. As a second line, a step taking `stall_factor` times the median step so
    far stops the run with an error, leaving completed epochs to resume from.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import SFTConfig, SFTTrainer

    class StallGuard(TrainerCallback):
        def __init__(self) -> None:
            self.durations: list[float] = []
            self.mark = time.monotonic()

        def on_train_begin(self, args, state, control, **kwargs):
            self.mark = time.monotonic()

        def on_save(self, args, state, control, **kwargs):
            # Evaluation and saving happen between steps and are not step time.
            self.mark = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            now = time.monotonic()
            duration, self.mark = now - self.mark, now
            if len(self.durations) >= 5:
                typical = sorted(self.durations)[len(self.durations) // 2]
                if duration > stall_factor * typical:
                    raise RuntimeError(
                        f"Step {state.global_step} took {duration:.0f}s against a median of "
                        f"{typical:.0f}s. GPU memory is probably spilling into system RAM: "
                        "lower --max-vram-gb or the example budget, then rerun with --resume."
                    )
            self.durations.append(duration)

    if max_vram_gb is not None:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, max_vram_gb * 1e9 / total), 0)

    rows = [
        json.loads(line)
        for line in Path(examples_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    too_long = [r["pair_id"] for r in rows if r.get("tokens") and r["tokens"] > max_length]
    if too_long:
        raise ValueError(
            f"{len(too_long)} example(s) exceed max_length={max_length}, e.g. {too_long[0]}. "
            "Rebuild the examples with this budget; training would otherwise cut the answer off."
        )
    def held_out(row: dict) -> bool:
        return _is_eval(row.get("chunk_id") or row["pair_id"], eval_fraction)

    train_rows = [r for r in rows if not held_out(r)]
    eval_rows = [r for r in rows if held_out(r)]

    def as_dataset(selected: list[dict]):
        return Dataset.from_list(
            [{"prompt": r["prompt"], "completion": r["completion"]} for r in selected]
        )

    train_dataset = as_dataset(train_rows)
    eval_dataset = as_dataset(eval_rows) if eval_rows else None

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16, device_map={"": 0})

    lora = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=max_length,
        completion_only_loss=True,
        packing=False,
        logging_steps=5,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        per_device_eval_batch_size=1,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        seed=seed,
    )
    guard = StallGuard()
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora,
        callbacks=[guard],
    )

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    checkpoints = [
        p for p in Path(out_dir).glob("checkpoint-*") if p.name.split("-")[-1].isdigit()
    ]
    resumed_from = (
        str(max(checkpoints, key=lambda p: int(p.name.split("-")[-1])))
        if resume and checkpoints else None
    )
    result = trainer.train(resume_from_checkpoint=resumed_from)
    out_dir = Path(out_dir)
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    summary = {
        "base_model": base_model,
        "examples": len(rows),
        "train_examples": len(train_rows),
        "eval_examples": len(eval_rows),
        "eval_fraction": eval_fraction,
        "resumed_from": resumed_from,
        "max_vram_gb": max_vram_gb,
        "stall_factor": stall_factor,
        "examples_path": str(examples_path),
        "max_length": max_length,
        "epochs": epochs,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "lora_rank": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "batch_size": batch_size,
        "grad_accum": grad_accum,
        "seed": seed,
        "train_loss": getattr(result, "training_loss", None),
        "steps": getattr(result, "global_step", None),
        "runtime_s": round(time.monotonic() - started, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 2),
        "step_seconds_median": (
            round(sorted(guard.durations)[len(guard.durations) // 2], 1) if guard.durations else None
        ),
        "eval_loss_by_epoch": [
            [round(entry["epoch"], 2), round(entry["eval_loss"], 4)]
            for entry in trainer.state.log_history
            if "eval_loss" in entry
        ],
        "log": [
            entry for entry in trainer.state.log_history
            if "loss" in entry or "eval_loss" in entry
        ],
    }
    (out_dir / "training_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return summary

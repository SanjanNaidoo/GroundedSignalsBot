"""Command line entry point.

  ym07 chat                      talk to the tutor (the deliverable itself)
  ym07 serve                     the same tutor in a browser, for a demo
  ym07 ingest                    build the retrieval corpus from corpus/raw
  ym07 items                     validate the benchmark set and show coverage
  ym07 conditions                list the experimental conditions
  ym07 run C0-baseline           run one condition (add --dry-run for no API calls)
  ym07 grade                     blind-grade the responses collected so far
  ym07 gen-pairs --sample 20     the local model writes training pairs from the corpus
  ym07 eval-retrieval --pairs P   compare retrievers on generated questions
  ym07 build-sft --pairs P ...    training examples in the tested C2 format
  ym07 train --max-length N       fit the LoRA adapter for C2-oss-tuned
  ym07 report                    aggregate graded runs into markdown tables
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from .conditions import CONDITIONS, FRONTIER_IDS, get_condition, repeat
from .config import load_config
from .chat import run_chat
from .digests import content_digest
from .finetune_cli import add_parsers as add_finetune_parsers
from .grading import grade_interactively, load_grades
from .ingest import ingest
from .items import coverage, load_items
from .model import build_client, build_client_for
from .report import (breakdown, efficiency, paired_against, render,
                     render_efficiency, summarise)
from .retrieval import build_retriever, load_chunks
from .runner import MissingCorpusError, discover_runs, load_run, run_condition
from .similarity import substantive_agreement, textual_similarity
from .web import run_server
from .trainingdata import DEFAULT_PAIRS, DRY_RUN_PAIRS, SAMPLE_PAIRS, generate_pairs

DEFAULT_CONFIG = "configs/eee4114f.yaml"
DEFAULT_ITEMS = "bench/eee4114f.jsonl"
DEFAULT_CHUNKS = "corpus/processed/chunks.jsonl"
DEFAULT_RUNS = "runs"
DEFAULT_GRADES = "grades/grades.jsonl"


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--items", default=DEFAULT_ITEMS)


def cmd_items(args) -> int:
    config = load_config(args.config)
    item_set = load_items(args.items, known_topics=config.topic_ids)
    stats = coverage(item_set)

    print(f"{item_set.path}  —  {len(item_set)} items, digest {item_set.digest}\n")
    for heading in ("topic", "type", "difficulty"):
        print(f"  by {heading}:")
        for key in sorted(stats[heading]):
            print(f"    {key:<24} {stats[heading][key]}")
        print()

    missing = sorted(config.topic_ids - set(stats["topic"]))
    if missing:
        print(f"  topics with no items: {', '.join(missing)}")
    return 0


def cmd_fetch_reader(args) -> int:
    """Download the vision reader, so no student's first screenshot waits on it."""
    from . import ocr

    model_id = args.model or ocr.VLM_MODEL_ID
    if ocr.vlm_cached() and not args.force and model_id == ocr.VLM_MODEL_ID:
        print(f"{model_id} is already on this machine.")
    else:
        from huggingface_hub import snapshot_download

        print(f"fetching {model_id} - about 8 GB, once")
        where = snapshot_download(model_id, max_workers=8,
                                  ignore_patterns=["*.pth", "*.onnx", "original/*"])
        print(f"  -> {where}")

    ready = ocr.engines()
    print()
    print("readers available here, best first:")
    for name in ready:
        print(f"  {name:10} {ocr.ENGINE_NAMES.get(name, name)}")
    if not ready:
        print("  none - the reader needs a CUDA card, and Tesseract was not found either")
        return 1
    return 0


def cmd_read(args) -> int:
    """Read one image and print what came out. The engine, visible on its own."""
    from pathlib import Path as _Path

    from . import ocr

    for name in args.images:
        data = _Path(name).read_bytes()
        try:
            result = ocr.extract(data, engine=args.engine)
        except (ocr.OcrUnavailable, ocr.OcrFailed) as exc:
            print(f"{name}: {exc}")
            continue
        print(f"=== {name}  [{result.engine}] "
              f"{result.width}x{result.height}  {result.latency_s}s ===")
        for note in result.notes:
            print(f"  note: {note}")
        print(result.text)
        print()
    return 0


def cmd_conditions(args) -> int:
    config = load_config(args.config)
    for name in sorted(CONDITIONS):
        condition = CONDITIONS[name]
        try:
            model = condition.resolved_model(config)
        except KeyError:
            model = f"(no `frontier:` entry '{condition.frontier_id}' in config)"
        print(f"{name}")
        print(f"  {condition.description}")
        print(f"  model={model} "
              f"effort={condition.resolved_effort(config)} "
              f"prompt={'yes' if condition.use_course_prompt else 'no'} "
              f"retrieval={'top_k=' + str(condition.resolved_top_k(config)) if condition.use_retrieval else 'no'}")
        print(f"  {condition.purpose}\n")
    return 0


def cmd_ingest(args) -> int:
    config = load_config(args.config)
    summary = ingest(
        raw_dir=Path(args.raw), out_path=Path(args.out), config=config
    )

    print(f"\nWrote {summary['chunks']} chunks from "
          f"{summary['files_ingested']} files to {summary['out_path']}\n")
    print("  chunks by topic:")
    for topic, count in summary["topic_counts"].items():
        print(f"    {topic:<24} {count}")
    for label, names in (
        ("duplicate files skipped", summary["duplicates"]),
        ("no text layer (need OCR or manual handling)", summary["no_text_layer"]),
        ("partially ingested (image-heavy)", summary["partial"]),
        ("unsupported format, skipped", summary["unsupported"]),
    ):
        if names:
            print(f"\n  {label}:")
            for name in names:
                print(f"    {name}")
    return 0


def cmd_run(args) -> int:
    config = load_config(args.config)
    item_set = load_items(args.items, known_topics=config.topic_ids)
    # Only a condition that retrieves needs the corpus index, and a dense index
    # puts an embedding model on the GPU, so C0 and C1 runs skip building one.
    needs_retrieval = any(get_condition(n).use_retrieval for n in args.conditions)
    retriever = (
        build_retriever(args.chunks, config.retrieval.method) if needs_retrieval else None
    )

    # One client per provider, built on first use and reused. A sweep can mix the
    # hosted and open arms, and loading a local model is slow enough that it must
    # not happen once per condition.
    clients: dict[str, object] = {}

    def client_for(condition) -> object:
        provider = condition.resolved_provider(config)
        entry = config.frontier_by_id(condition.frontier_id) if condition.frontier_id else None
        key = f"{provider}:{entry.id}" if entry else provider
        if getattr(condition, "hybrid", False):
            key += ":hybrid"
        if key not in clients:
            if entry and not args.dry_run and not (entry.price_in or entry.price_out):
                print(
                    f"  note: no prices set for '{entry.id}' ({entry.model}); the cost "
                    "column will read $0 until price_in/price_out are filled in the config.",
                    file=sys.stderr,
                )
            clients[key] = build_client_for(condition, config, dry_run=args.dry_run,
                                            frontier=entry)
        return clients[key]

    if args.dry_run:
        print("DRY RUN — no model will be called and no money will be spent.\n")

    # Parallel workers exist to overlap network latency on the hosted arms. A
    # local model has no latency to hide: one GPU serves one generation at a
    # time, and concurrent calls each allocate their own KV cache, so extra
    # workers buy nothing and can exhaust VRAM.
    workers = args.workers
    if any(get_condition(n).resolved_provider(config) == "local" for n in args.conditions):
        if workers != 1:
            print(f"  note: local model — using 1 worker instead of {workers}; "
                  "concurrency does not help a single GPU.\n", file=sys.stderr)
        workers = 1

    for name in args.conditions:
        base = get_condition(name)
        for index in range(1, args.repeats + 1):
            condition = base if args.repeats == 1 else repeat(base, index)
            print(f"{condition.name}  ({condition.description})")
            try:
                run_dir = run_condition(
                    condition=condition,
                    config=config,
                    item_set=item_set,
                    client=client_for(condition),
                    retriever=retriever,
                    runs_dir=Path(args.runs),
                    workers=workers,
                    dry_run=args.dry_run,
                )
            except MissingCorpusError as exc:
                print(f"  skipped: {exc}\n", file=sys.stderr)
                break
            manifest = json.loads(
                (run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            cut = manifest.get("truncated", 0)
            looped = manifest.get("degenerate", 0)
            print(f"  -> {run_dir}  "
                  f"cost ${manifest['total_cost_usd']:.4f}  "
                  f"errors {manifest['errors']}  refusals {manifest['refusals']}"
                  + (f"  truncated {cut}" if cut else "")
                  + (f"  degenerate {looped}" if looped else "") + "\n")
            if cut:
                print(f"  {cut} answer(s) stopped at the "
                      f"{manifest['max_tokens']}-token cap while still making "
                      "progress. They will grade as wrong rather than as cut "
                      "off — raise the cap and re-run before grading.\n",
                      file=sys.stderr)
            if looped:
                names = ", ".join(manifest.get("degenerate_items", []))
                print(f"  {looped} answer(s) hit the cap repeating one line: "
                      f"{names}. That is the model failing, not the cap being "
                      "too low, so raising it buys more of the same line. Grade "
                      "them as produced.\n",
                      file=sys.stderr)
    return 0


def _collect_runs(runs_dir: Path, only: list[str] | None):
    runs = []
    for run_dir in discover_runs(runs_dir):
        manifest, records = load_run(run_dir)
        if only and manifest["condition"] not in only:
            continue
        runs.append((manifest, records))
    return runs


def cmd_grade(args) -> int:
    config = load_config(args.config)
    item_set = load_items(args.items, known_topics=config.topic_ids)
    runs = _collect_runs(Path(args.runs), args.conditions)

    if not runs:
        print("No runs found. Run a condition first.", file=sys.stderr)
        return 1

    if any(m.get("dry_run") for m, _ in runs) and not args.allow_dry_run:
        print(
            "Refusing to grade: dry-run output is present, and grading placeholder "
            "text produces meaningless numbers. Re-run for real, or pass "
            "--allow-dry-run to rehearse the grading flow.",
            file=sys.stderr,
        )
        return 1

    digests = {m["items_digest"] for m, _ in runs}
    if len(digests) > 1:
        print(
            "Warning: these runs used different item sets. They are not comparable "
            "and the report must not present them side by side.\n",
            file=sys.stderr,
        )

    responses_by_item = defaultdict(list)
    for _, records in runs:
        for record in records:
            responses_by_item[record["item_id"]].append(record)

    if args.from_csv:
        from .grading import import_csv

        # Imported marks are not hand marks, so unless the grader says otherwise they
        # go to their own file and the two are never averaged together by accident.
        destination = Path(args.grades if args.grades != DEFAULT_GRADES
                           else "grades/model-assisted.jsonl")
        if args.seed is None:
            print("--from-csv needs the same --seed the file was exported with, or the "
                  "A/B/C labels point at the wrong answers.", file=sys.stderr)
            return 1
        graded, problems = import_csv(path=Path(args.from_csv), item_set=item_set,
                                      responses_by_item=responses_by_item,
                                      grades_path=destination, seed=args.seed)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        print(f"Imported {graded} grade(s) to {destination}, each marked source=csv.")
        return 1 if problems and not graded else 0

    only = None
    if args.plan:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        only = set(plan["h1_responses"]) | set(plan["sampled_responses"])
        print(f"Grading the {len(only)} answers drawn in {args.plan} (seed {plan['seed']}).")

    if args.web:
        from .grade_web import serve_grading

        return serve_grading(
            item_set=item_set,
            responses_by_item=responses_by_item,
            grades_path=Path(args.grades),
            seed=args.seed,
            only=only,
            port=args.port,
            open_browser=not args.no_browser,
        )

    graded = grade_interactively(
        item_set=item_set,
        responses_by_item=responses_by_item,
        grades_path=Path(args.grades),
        seed=args.seed,
        limit=args.limit,
        only=only,
    )
    print(f"\nGraded {graded} response(s). Saved to {args.grades}")
    return 0


def _judge_clients(args, config) -> dict | None:
    """One client per judge, refusing to start if a key is missing."""
    from .judging import H1_CONDITIONS  # noqa: F401  (documented in --help)

    clients: dict[str, tuple] = {}
    for judge in args.judges:
        if judge == "local":
            clients["local"] = (
                build_client(False, "local", config),
                get_condition("C1-oss").resolved_model(config),
                None,
            )
            continue
        entry = config.frontier_by_id(judge)
        if not args.dry_run and not os.environ.get(entry.key_env):
            print(f"{entry.key_env} is not set. Put it in .env, which git ignores.",
                  file=sys.stderr)
            return None
        clients[judge] = (build_client(args.dry_run, entry.provider, config, entry),
                          entry.model, entry)
    if len(clients) < 2:
        print("At least two judges are needed; a single judge cannot disagree with itself.",
              file=sys.stderr)
        return None
    return clients


def cmd_judge(args) -> int:
    from .judging import run_judging

    config = load_config(args.config)
    item_set = load_items(args.items, known_topics=config.topic_ids)
    runs = _collect_runs(Path(args.runs), args.conditions)
    if not runs:
        print("No runs found.", file=sys.stderr)
        return 1
    if any(m.get("dry_run") for m, _ in runs):
        print("Refusing to judge: dry-run output is present.", file=sys.stderr)
        return 1
    clients = _judge_clients(args, config)
    if clients is None:
        return 1

    prior: dict[str, dict] = {}
    if args.redo_incomplete:
        from .judging import drop_incomplete

        moved = drop_incomplete(Path(args.out))
        prior = moved["prior"]
        print(f"kept {moved['kept']} answers with enough verdicts, set aside "
              f"{moved['set aside']} to judge again"
              + (f"; the file as it stood is {moved['previous']}" if moved["previous"] else ""))

    records = [r for _, run_records in runs for r in run_records]
    if args.plan:
        # Judge the audited answers first, or only them. Agreement is measured on
        # exactly these, so when a judge's daily quota is smaller than the work
        # left, this is where it has to be spent.
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        drawn = set(plan["h1_responses"]) | set(plan["sampled_responses"])
        if args.plan_only:
            records = [r for r in records if r["response_id"] in drawn]
        else:
            records.sort(key=lambda r: r["response_id"] not in drawn)
    if args.limit:
        records = records[:args.limit]

    summary = run_judging(
        items={item.id: item for item in item_set},
        records=records,
        clients=clients,
        out_path=Path(args.out),
        max_tokens=args.max_tokens,
        workers=args.workers,
        prior=prior,
    )
    print(json.dumps(summary, indent=2))
    return 0


def cmd_audit_plan(args) -> int:
    from .judging import audit_plan

    config = load_config(args.config)
    item_set = load_items(args.items, known_topics=config.topic_ids)
    runs = _collect_runs(Path(args.runs), None)
    plan = audit_plan(
        records=[r for _, records in runs for r in records],
        items={item.id: item for item in item_set},
        sample=args.sample,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(plan['h1_responses'])} answers from the H1 pair and "
          f"{len(plan['sampled_responses'])} sampled elsewhere: "
          f"{plan['decisions']} decisions, seed {plan['seed']}. Written to {out}")
    print(json.dumps(plan["sampled_by_condition"], indent=2))
    return 0


def cmd_agreement(args) -> int:
    from .judging import agreement, self_preference

    machine = [
        json.loads(line)
        for line in Path(args.machine).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    human = load_grades(Path(args.grades))
    runs = _collect_runs(Path(args.runs), None)
    records = {r["response_id"]: r for _, rs in runs for r in rs}

    result = agreement(human=human, machine=machine, records=records)
    result["self_preference"] = self_preference(machine)

    # The thresholds were fixed in the protocol before any of this was run.
    include_k = result["must_include_kappa"][0]
    exclude_k = result["must_not_include_kappa"][0]
    worst = max((abs(c["difference"]) for c in result["by_condition"].values()), default=0.0)
    result["protocol"] = {
        "must_include kappa >= 0.70": include_k >= 0.70,
        "must_not_include kappa >= 0.70": exclude_k >= 0.70,
        "per-condition difference <= 0.05": worst <= 0.05,
        "no ordering reversal": not result["ordering_reversals"],
    }
    result["verdict"] = (
        "machine grades stand for the non-H1 runs"
        if all(result["protocol"].values())
        else "thresholds not met: see the protocol's fallback"
    )
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n",
                                  encoding="utf-8", newline="\n")
    return 0


def _chunk_ids_from(path: Path) -> list[str]:
    """Chunk ids, in order and without repeats, from a .chunks.jsonl ledger or pairs file."""
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            chunk_id = json.loads(line)["chunk_id"]
            if chunk_id not in ids:
                ids.append(chunk_id)
    return ids


def cmd_gen_pairs(args) -> int:
    config = load_config(args.config)
    item_set = load_items(args.items, known_topics=config.topic_ids)
    chunks = load_chunks(args.chunks)
    if not chunks:
        print(
            f"No corpus at {args.chunks}. Training pairs are written from the "
            "corpus, so there is nothing to do without one.",
            file=sys.stderr,
        )
        return 1

    like_ids = None
    if args.like:
        if not args.out:
            print("--like needs --out, so a comparison run never lands on top of "
                  "the run it is being compared with.", file=sys.stderr)
            return 1
        like_ids = _chunk_ids_from(Path(args.like))

    if args.out:
        out = Path(args.out)
    elif args.dry_run:
        out = DRY_RUN_PAIRS
    else:
        out = SAMPLE_PAIRS if args.sample else DEFAULT_PAIRS

    if args.writer == "local":
        # The open arm writes its own training data by default, so the fine-tune
        # learns from nothing a frontier model wrote.
        model_id = get_condition("C1-oss").resolved_model(config)
        client = build_client(args.dry_run, "local", config)
        max_tokens, effort, workers = config.model.local_max_new_tokens, "n/a", 1
    else:
        # A frontier writer is for the passages the local model gets wrong. Every
        # pair records which model wrote it, and the report has to say so.
        entry = config.frontier_by_id(args.writer)
        if not args.dry_run and not os.environ.get(entry.key_env):
            print(f"{entry.key_env} is not set, so {args.writer} cannot write pairs. "
                  "Put it in .env, which git ignores, rather than anywhere it is "
                  "recorded.", file=sys.stderr)
            return 1
        model_id = entry.model
        client = build_client(args.dry_run, entry.provider, config, entry)
        max_tokens, effort, workers = config.model.max_tokens, "default", args.workers
    if args.dry_run:
        print("DRY RUN - no model will be called; every chunk gets placeholder text.")
        print()

    summary = generate_pairs(
        config=config,
        chunks=chunks,
        item_set=item_set,
        client=client,
        model_id=model_id,
        out_path=out,
        max_tokens=max_tokens,
        effort=effort,
        workers=workers,
        sample=args.sample,
        only_chunk_ids=like_ids,
        corpus_digest=content_digest(Path(args.chunks).read_bytes()),
        dry_run=args.dry_run,
    )
    print(f"\n{summary['pairs_kept']} pair(s) kept and {summary['pairs_dropped']} "
          f"dropped, from {summary['chunks_targeted']} chunk(s). Written to {out}")
    for reason, count in summary["dropped_by_reason"].items():
        print(f"  dropped, {reason}: {count}")
    for status, count in summary["chunks_by_status"].items():
        if status != "pairs":
            print(f"  chunks with status '{status}': {count}")
    return 0


def cmd_report(args) -> int:
    config = load_config(args.config)
    item_set = load_items(args.items, known_topics=config.topic_ids)
    runs = _collect_runs(Path(args.runs), args.conditions)
    if not runs:
        print("No runs found.", file=sys.stderr)
        return 1

    grades = load_grades(Path(args.grades))
    text = render(
        summaries=summarise(runs=runs, grades=grades),
        breakdowns={
            key: breakdown(runs=runs, grades=grades, item_set=item_set, key=key)
            for key in ("type", "topic", "difficulty")
        },
        item_set=item_set,
        paired=paired_against(runs=runs, grades=grades, reference=args.baseline),
        reference=args.baseline,
        efficiencies=efficiency(runs=runs, gpu_watts=args.gpu_watts),
    )
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)
    return 0


def cmd_similarity(args) -> int:
    """How alike two conditions' answers are, graded and as written."""
    runs = _collect_runs(Path(args.runs), None)
    if not runs:
        print("No runs found.", file=sys.stderr)
        return 1
    records = [r for _, run_records in runs for r in run_records]

    grades = load_grades(Path(args.grades))
    agreement = substantive_agreement(
        grades=grades,
        records={r["response_id"]: r for r in records},
        a=args.a, b=args.b,
    )
    print(f"\nSubstantive agreement, {args.a} against {args.b}")
    if agreement.items == 0:
        print("  nothing to compare: no item is graded under both conditions yet.")
        print("  This is the measure that matters; the text one below cannot replace it.")
    else:
        print(f"  items graded under both     {agreement.items}")
        print(f"  required points             {agreement.points}")
        print(f"  points both made or missed  {agreement.points_agreed} "
              f"({agreement.point_agreement:.1%})")
        print(f"  items decided identically   {agreement.identical_items} "
              f"({agreement.item_agreement:.1%})")
        print(f"  points only {args.a} made: {agreement.a_only}, "
              f"only {args.b} made: {agreement.b_only}")

    if args.graded_only:
        return 0

    text = textual_similarity(records=records, a=args.a, b=args.b)
    print(f"\nTextual similarity, {args.a} against {args.b}")
    if not text.get("items"):
        print("  no item was answered by both conditions.")
        return 0
    print(f"  items compared    {text['items']}")
    print(f"  mean cosine       {text['mean_similarity']:.3f}")
    print(f"  median cosine     {text['median_similarity']:.3f}"
          f"  (range {text['lowest']:.3f} to {text['highest']:.3f})")
    print(f"  median tokens     {args.a}: {text['median_tokens_a']:.0f}, "
          f"{args.b}: {text['median_tokens_b']:.0f}")
    print(f"  floor, all {text['mismatched_pairs']} wrong pairings        "
          f"{text['mismatched_floor']:.3f}")
    print(f"  lift above that floor                   {text['lift_over_floor']:+.3f}")
    print(f"  right answer ranked first               {text['matched_ranked_first']}"
          f"/{text['items']}  (mean rank {text['mean_rank_of_match']:.2f})")
    print(f"  permutation test                        p = {text['permutation_p']:.4f}")
    print("\n  Read the cosine score beside those lengths, not on its own: these two "
          "conditions\n  differ in how much they say as much as in what they say.")
    return 0

def cmd_stress(args) -> int:
    """The self-scoring stress suites: build the sets, run local models on them, report."""
    from . import calcsuites, conceptbank
    from .calcbank import generate_bank, load_bank, write_bank
    from .stress import BANKS, SEEDS, SUITES, LocalRunner, report, run_suite

    split = args.split
    banks = dict(BANKS[split])
    if split == "dev" and args.bank:
        banks["calc"] = Path(args.bank)
    results = Path(args.results) if args.results else (Path("stress/results") if split == "dev"
                                                        else Path("stress/results/final"))

    if args.action == "build":
        if split == "final" and not args.frozen:
            print("The final sets are generated once, after the model is frozen. "
                  "Pass --frozen to confirm that it is.", file=sys.stderr)
            return 1

        def keep(path: Path) -> bool:
            if path.exists() and not args.force:
                print(f"  {path} already exists and is registered; kept. --force replaces it.")
                return True
            return False

        seeds = SEEDS[split]
        # Every parameter set an earlier split used, so a later one avoids them where it can.
        earlier = {"final": ["dev"], "fresh": ["dev", "final"]}.get(split, [])
        dev_keys = calcsuites.param_keys(*[path for name in earlier
                                           for path in BANKS[name].values()])
        if not keep(banks["calc"]):
            items, problems = generate_bank(per_family=args.per_family, seed=seeds["calc"])
            if split == "final":
                # Drawn from a seed development never used. Dropping the few parameter
                # sets that happen to coincide would leave the families uneven, which
                # would bias the mean more than the coincidence does: the model does no
                # learning between runs, so a repeated parameter set is a fresh question
                # to it either way. The overlap is counted instead.
                repeats = sum(json.dumps(i["params"], sort_keys=True) in dev_keys for i in items)
                print(f"  calc: {repeats} of {len(items)} parameter sets also occur in development")
            if problems:
                print("The calculation bank failed its own checks and was not written:", file=sys.stderr)
                for problem in problems:
                    print(f"  {problem}", file=sys.stderr)
                return 1
            m = write_bank(items, banks["calc"], seed=seeds["calc"], per_family=args.per_family)
            print(f"  calc: {m['items']} questions, digest {m['digest']}")
        taken = dev_keys | calcsuites.param_keys(banks["calc"])
        for suite, generate in (("followup", calcsuites.generate_followups), ("errors", calcsuites.generate_errors)):
            if keep(banks[suite]):
                taken |= calcsuites.param_keys(banks[suite])
                continue
            items, problems = generate(seed=seeds[suite], exclude=taken)
            if problems:
                print(f"The {suite} set failed its own checks and was not written:", file=sys.stderr)
                for problem in problems:
                    print(f"  {problem}", file=sys.stderr)
                return 1
            m = calcsuites.write_set(items, banks[suite], seed=seeds[suite], excluded=len(taken))
            taken |= calcsuites.param_keys(banks[suite])
            print(f"  {suite}: {m['items']} items, digest {m['digest']}")
        if split == "dev" and not (keep(conceptbank.DEV_BANK) and keep(conceptbank.FINAL_BANK)):
            pairs = conceptbank.load_pairs()
            problems = conceptbank.verify(pairs, Path(args.chunks))
            if problems:
                print("The concept bank failed its own checks and was not written:", file=sys.stderr)
                for problem in problems:
                    print(f"  {problem}", file=sys.stderr)
                return 1
            dev, final = conceptbank.split_pairs(pairs)
            for path, part, name in ((conceptbank.DEV_BANK, dev, "development"), (conceptbank.FINAL_BANK, final, "held back")):
                m = conceptbank.write(path, conceptbank.items(part), pairs=len(part), split=name,
                                      split_seed=conceptbank.SPLIT_SEED)
                print(f"  concept ({name}): {m['items']} statements from {len(part)} pairs, digest {m['digest']}")
        return 0

    if args.action == "report":
        text = report(banks["calc"], results)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"Wrote {args.out}")
        else:
            print(text)
        return 0

    config = load_config(args.config)
    paid = [name for name in args.conditions
            if get_condition(name).resolved_provider(config) != "local"]
    if paid and not getattr(args, "allow_paid", False):
        print(f"{', '.join(paid)} run on a paid API. The stress suites run local models "
              "unless --allow-paid is passed, with --max-usd as the ceiling.", file=sys.stderr)
        return 1
    if paid and not getattr(args, "dry_run", False):
        from .hosted import estimate_usd

        print(f"  paid conditions: {', '.join(paid)}; about ${estimate_usd(1):.3f} an answer "
              f"at September's token counts, ceiling ${args.max_usd:.2f} per condition")
    suites = SUITES if "all" in args.suites else args.suites
    # Only the suites drawn from the calculation bank need it loaded. The "check" split
    # re-asks the course questions in a third wording and has no bank of its own.
    needs_bank = any(s in ("calc", "pushback", "followup", "errors") for s in suites)
    if needs_bank and not banks.get("calc", Path("nonexistent")).exists():
        print(f"No calculation bank at {banks.get('calc')}. Build it first: "
              f"ym07 stress build --split {split}", file=sys.stderr)
        return 1
    bank = load_bank(banks["calc"]) if needs_bank else []
    # One retriever per corpus, not one per run. A condition may name its own chunked
    # corpus, because chunk size and retrieval depth decide each other and a condition
    # that sets top_k without saying which corpus it means is only half specified.
    # Built on first use and kept, so two conditions over the same corpus do not pay
    # for it twice - the dense half is a model load and an embedding pass.
    retrievers: dict[str, object] = {}

    def corpus_for(condition_name: str) -> str | None:
        condition = get_condition(condition_name)
        return (condition.chunks or args.chunks) if condition.use_retrieval else None

    def retriever_for(condition_name: str):
        path = corpus_for(condition_name)
        if path is None:
            return None
        if path not in retrievers:
            if get_condition(condition_name).chunks:
                print(f"  retrieving from {path}, named by the condition")
            retrievers[path] = build_retriever(path, config.retrieval.method)
        return retrievers[path]

    # Through the browser's own endpoint, or in process. The suites, the marking and the
    # row format are identical either way; what differs is whether the answer came from
    # the delivered artefact or from a harness that shares its code with it.
    if getattr(args, "via_web", False):
        from .webrunner import WebRunner as Runner

        print("  answers will come over HTTP from the web tutor, not in process")
    else:
        Runner = LocalRunner

    for name in args.conditions:
        print(f"\n{name}")
        if name in paid:
            from .hosted import HostedRunner

            runner = HostedRunner(config=config, condition_name=name,
                                  retriever=retriever_for(name), corpus=corpus_for(name),
                                  max_usd=args.max_usd, dry_run=getattr(args, "dry_run", False))
        else:
            runner = Runner(config=config, condition_name=name,
                            retriever=retriever_for(name),
                            cap_gb=args.cap_gb, adapter_scale=args.adapter_scale,
                            max_new_tokens=args.max_new_tokens,
                            corpus=corpus_for(name))
        try:
            for suite in suites:
                run_suite(suite=suite, runner=runner, bank=bank, limit=args.limit, split=split,
                          results=results, redo_errors=args.redo_errors, stride=args.stride)
        finally:
            runner.close()
    print(f"\nDone. See the results with: ym07 stress report --split {split} --results {results}")
    return 0


def cmd_efficiency(args) -> int:
    runs = _collect_runs(Path(args.runs), args.conditions)
    if not runs:
        print("No runs found.", file=sys.stderr)
        return 1
    rows = efficiency(runs=runs, gpu_watts=args.gpu_watts)
    text = render_efficiency(rows)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"Wrote {args.out}")
    else:
        print(text)
    return 0


def _load_env(path: str = ".env") -> None:
    """Read .env into the environment, without overriding what is already set.

    Every "put it in .env" message in this file was telling the truth about where
    the key belongs and lying about who reads it: nothing did. A real environment
    variable still wins, so CI and a one-off `KEY=... ym07 ...` keep working.
    """
    env = Path(path)
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def main(argv: list[str] | None = None) -> int:
    _load_env()
    parser = argparse.ArgumentParser(prog="ym07", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("items", help="validate the benchmark set and show coverage")
    _common(p); p.set_defaults(func=cmd_items)

    p = sub.add_parser("conditions", help="list the experimental conditions")
    _common(p); p.set_defaults(func=cmd_conditions)

    p = sub.add_parser("fetch-reader",
                       help="download the vision model that reads screenshots")
    _common(p)
    p.add_argument("--model", default="", help="a different checkpoint to fetch")
    p.add_argument("--force", action="store_true", help="fetch even if it is cached")
    p.set_defaults(func=cmd_fetch_reader)

    p = sub.add_parser("read", help="read an image and print the transcription")
    _common(p)
    p.add_argument("images", nargs="+")
    p.add_argument("--engine", default=None, choices=["vision", "tesseract"],
                   help="force one reader instead of taking the best available")
    p.set_defaults(func=cmd_read)

    p = sub.add_parser("ingest", help="build the retrieval corpus from corpus/raw")
    _common(p)
    p.add_argument("--raw", default="corpus/raw")
    p.add_argument("--out", default=DEFAULT_CHUNKS)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("chat", help="talk to the tutor interactively")
    _common(p)
    p.add_argument("--chunks", default=DEFAULT_CHUNKS)
    p.add_argument("--provider", default=None,
                   help="anthropic | local (default: the config's provider)")
    p.add_argument("--model", default=None, help="override the model id")
    p.add_argument("--no-retrieval", action="store_true",
                   help="ungrounded: the C1 tutor rather than the C2 one")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=run_chat)

    p = sub.add_parser("serve", help="the tutor in a browser - the demo")
    _common(p)
    p.add_argument("--chunks", default=DEFAULT_CHUNKS)
    p.add_argument("--provider", default=None,
                   help="anthropic | local (default: the config's provider)")
    p.add_argument("--model", default=None,
                   help="override the model id; point it at models/qwen2.5-7b-eee4114f "
                        "to demonstrate the fine-tuned tutor the report is about")
    p.add_argument("--condition", default=None, choices=sorted(CONDITIONS),
                   help="serve a named condition exactly as the benchmark ran it - its "
                        "model, its retrieval depth, its prompt. C2-oss-tuned is the "
                        "system the thesis is about")
    p.add_argument("--also", action="append", default=[], choices=sorted(CONDITIONS),
                   metavar="CONDITION",
                   help="another condition on the page's model switch, each with its own "
                        "settings; --condition is the one selected when the page opens")
    p.add_argument("--platform", action="store_true",
                   help="the platform: Opus 5 in the delivered pipeline (A3-opus-pipeline), "
                        "selected when the page opens, and the local R1 system with the handout lookup and its marks rule (A-r1-facts-marks) "
                        "on the switch, without lecture minutes")
    p.add_argument("--host", default="127.0.0.1",
                   help="127.0.0.1 keeps it on this machine; 0.0.0.0 exposes it to the "
                        "network, which the course material may not permit")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--no-resources", action="store_true",
                   help="serve without the lecture-minute and past-question cards")
    p.add_argument("--max-usd", type=float, default=5.0,
                   help="for a hosted model: the most this session may spend before it "
                        "answers with an error instead of calling the API (default 5)")
    p.add_argument("--no-retrieval", action="store_true",
                   help="ungrounded: the C1 tutor rather than the C2 one")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=run_server)

    p = sub.add_parser("run", help="run one or more conditions")
    _common(p)
    p.add_argument("conditions", nargs="+", choices=sorted(CONDITIONS))
    p.add_argument("--chunks", default=DEFAULT_CHUNKS)
    p.add_argument("--runs", default=DEFAULT_RUNS)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--repeats", type=int, default=1,
                   help="run each condition N times to measure run-to-run variance")
    p.add_argument("--dry-run", action="store_true",
                   help="exercise the pipeline without calling the API")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("grade", help="blind-grade collected responses")
    _common(p)
    p.add_argument("--runs", default=DEFAULT_RUNS)
    p.add_argument("--grades", default=DEFAULT_GRADES)
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None, help="grade at most N items")
    p.add_argument("--seed", type=int, default=None, help="fix the shuffle for reproducibility")
    p.add_argument("--allow-dry-run", action="store_true")
    p.add_argument("--plan", default=None,
                   help="grade only the answers drawn in an audit plan (see audit-plan)")
    p.add_argument("--from-csv", default=None,
                   help="read marks from a CSV exported by scripts/export_for_review.py; "
                        "needs the same --seed, and writes to grades/model-assisted.jsonl")
    p.add_argument("--web", action="store_true",
                   help="grade in the browser, with the mathematics and tables rendered")
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--no-browser", action="store_true",
                   help="with --web, print the address instead of opening it")
    p.set_defaults(func=cmd_grade)

    p = sub.add_parser("audit-plan",
                       help="draw the answers a person grades: the H1 pair plus a sample")
    _common(p)
    p.add_argument("--runs", default=DEFAULT_RUNS)
    p.add_argument("--sample", type=int, default=40,
                   help="answers drawn from the runs outside the H1 pair")
    p.add_argument("--seed", type=int, required=True,
                   help="recorded with the plan, so the draw can be shown to precede the grading")
    p.add_argument("--out", default="grades/audit-plan.json")
    p.set_defaults(func=cmd_audit_plan)

    p = sub.add_parser("judge", help="grade answers with an ensemble of models")
    _common(p)
    p.add_argument("--runs", default=DEFAULT_RUNS)
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--judges", nargs="+", default=["gpt", "gemini", "grok"],
                   choices=["local", *FRONTIER_IDS],
                   help="at least two; no model judges its own answers, and deepseek "
                        "judges none of the open arm, having written its training data")
    p.add_argument("--out", default="grades/machine-grades.jsonl")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None,
                   help="judge at most N answers, to rehearse the run before paying for it")
    p.add_argument("--max-tokens", type=int, default=4000,
                   help="output cap per verdict; reasoning models spend most of it thinking, "
                        "and a cap that is too low truncates the verdicts (default 4000)")
    p.add_argument("--plan", default=None,
                   help="an audit plan; its answers are judged first, because agreement "
                        "is measured on them and a capped judge must be spent there")
    p.add_argument("--plan-only", action="store_true",
                   help="with --plan, judge nothing but the drawn answers")
    p.add_argument("--redo-incomplete", action="store_true",
                   help="put answers with fewer than two verdicts back in the queue, keeping "
                        "the verdicts they already have and the file as it stood")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_judge)

    p = sub.add_parser("agreement",
                       help="what the machine grades are worth, measured against the hand grades")
    _common(p)
    p.add_argument("--runs", default=DEFAULT_RUNS)
    p.add_argument("--grades", default=DEFAULT_GRADES)
    p.add_argument("--machine", default="grades/machine-grades.jsonl")
    p.add_argument("--out", default=None, help="also write the result as JSON")
    p.set_defaults(func=cmd_agreement)

    p = sub.add_parser(
        "gen-pairs",
        help="have the local model write training pairs from the corpus",
    )
    _common(p)
    p.add_argument("--chunks", default=DEFAULT_CHUNKS)
    p.add_argument("--out", default=None,
                   help=f"output file (default {DEFAULT_PAIRS}, or {SAMPLE_PAIRS} with --sample)")
    p.add_argument("--sample", type=int, default=None,
                   help="write from N chunks spread across the corpus, to read before a full run")
    p.add_argument("--like", default=None,
                   help="write from exactly the chunks in an earlier .chunks.jsonl, "
                        "to compare two prompts like for like (needs --out)")
    p.add_argument("--writer", default="local", choices=["local", *FRONTIER_IDS],
                   help="who writes the pairs: the local open model (default), or a "
                        "frontier model from the config")
    p.add_argument("--workers", type=int, default=4,
                   help="concurrent requests for a frontier writer; the local writer uses one")
    p.add_argument("--dry-run", action="store_true",
                   help="exercise the pipeline without loading the model")
    p.set_defaults(func=cmd_gen_pairs)

    p = sub.add_parser("report", help="aggregate graded runs")
    _common(p)
    p.add_argument("--runs", default=DEFAULT_RUNS)
    p.add_argument("--grades", default=DEFAULT_GRADES)
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--baseline", default="C1-prompted",
                   help="reference condition for the paired comparison "
                        "(default: C1-prompted, the bar H1 names)")
    p.add_argument("--gpu-watts", type=float, default=300.0,
                   help="rated power of the card the open arm ran on, used to put a "
                        "ceiling on its energy use (default 300, the RTX 5070 Ti)")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("stress",
                       help="self-scoring stress tests on the local models: build, run, report")
    _common(p)
    p.add_argument("action", choices=["build", "run", "report"])
    p.add_argument("--split", choices=["dev", "final", "check", "fresh"], default="dev",
                   help="dev while improving the model; final only once it is frozen")
    p.add_argument("--bank", default=None, help="override the development calculation bank")
    p.add_argument("--results", default=None,
                   help="results directory (default stress/results, or stress/results/final)")
    p.add_argument("--frozen", action="store_true",
                   help="with build --split final, confirm the model is frozen")
    p.add_argument("--suites", nargs="+", default=["all"],
                   choices=["all", "calc", "pushback", "scope", "concept", "followup", "errors", "course",
                            "course-fresh", "course-fresh2", "course-marks"])
    p.add_argument("--conditions", nargs="+", default=["C2-oss-tuned", "C2-oss"],
                   help="local conditions only; paid ones are refused")
    p.add_argument("--chunks", default=DEFAULT_CHUNKS)
    p.add_argument("--limit", type=int, default=None, help="run at most N items per suite, to rehearse")
    p.add_argument("--per-family", type=int, default=24)
    p.add_argument("--force", action="store_true", help="with build, replace registered sets")
    p.add_argument("--adapter-scale", type=float, default=None,
                   help="override how strongly the fine-tuned adapter is applied; the default "
                        "is the condition's own strength. Results for an overridden strength "
                        "are filed under the condition name with the strength appended")
    p.add_argument("--stride", type=int, default=1,
                   help="run every Nth question, for a quick even sample of a set")
    p.add_argument("--via-web", action="store_true",
                   help="ask through the web tutor's own HTTP endpoint, the way a "
                        "student reaches it, instead of calling the model in process")
    p.add_argument("--redo-errors", action="store_true",
                   help="re-run items whose earlier answer failed, dropping the failed rows")
    p.add_argument("--max-new-tokens", type=int, default=0,
                   help="override the generation cap; lower it to fit a larger model "
                        "on the card, since the KV cache is reserved from it")
    p.add_argument("--cap-gb", type=float, default=15.0,
                   help="GPU memory cap; below the card so an oversized request fails instead of spilling into RAM")
    p.add_argument("--out", default=None, help="with report, write the markdown here")
    p.add_argument("--allow-paid", action="store_true",
                   help="run hosted conditions (A2-opus-rag, A3-opus-pipeline) through "
                        "ym07.hosted; refused without this flag")
    p.add_argument("--max-usd", type=float, default=50.0,
                   help="spend ceiling per hosted condition; the run stops there and "
                        "resumes from the rows already written")
    p.add_argument("--dry-run", action="store_true",
                   help="hosted conditions answer with the stub client: the pipeline "
                        "runs, nothing is sent, nothing is spent")
    p.set_defaults(func=cmd_stress)

    p = sub.add_parser("similarity",
                       help="how alike two conditions' answers are, graded and as written")
    _common(p)
    p.add_argument("--a", default="C2-oss-tuned")
    p.add_argument("--b", default="C1-prompted")
    p.add_argument("--runs", default=DEFAULT_RUNS)
    p.add_argument("--grades", default=DEFAULT_GRADES)
    p.add_argument("--graded-only", action="store_true",
                   help="skip the embedding measure, which needs torch")
    p.set_defaults(func=cmd_similarity)

    p = sub.add_parser("efficiency",
                       help="cost, latency and answer length per condition; needs no grades")
    _common(p)
    p.add_argument("--runs", default=DEFAULT_RUNS)
    p.add_argument("--conditions", nargs="*", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--gpu-watts", type=float, default=300.0,
                   help="rated power of the card the open arm ran on, used to put a "
                        "ceiling on its energy use (default 300, the RTX 5070 Ti)")
    p.set_defaults(func=cmd_efficiency)

    add_finetune_parsers(sub, _common, DEFAULT_CHUNKS)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted — progress saved", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

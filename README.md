# SignalsBot - a course-grounded tutor for EEE4114F

A tutor for UCT's EEE4114F (digital signal processing and an introduction to machine
learning) that answers from the course's own notes, tutorials, past papers and memoranda,
withholds work a student has said is for marks, cites the passage each claim came from,
and runs in its entirety on one 16 GB desktop graphics card. The report in `Report/` is
the deliverable; this repository is the evidence behind it.

## What is delivered (frozen 8 October 2026)

| part | what it is | where |
|---|---|---|
| Model | DeepSeek-R1-Distill-Qwen-7B, 4-bit, greedy, reasoning stripped before display | `src/ym07/model.py` |
| Retrieval | BM25 and `bge-small-en-v1.5` fused by reciprocal rank, over 316 de-duplicated 800-token passages, four deep | `src/ym07/retrieval.py`, `retrieval_eval.py` |
| Behaviour adapter | LoRA on 782 examples, one epoch: cite, decline assessed work, read the handout | `models/r1-distill-7b-eee4114f-behaviour-v2`, `scripts/build_behaviour_sft.py` |
| Router and gate | numerical work goes to the base weights; assessed work is withheld by a rule over the student's own words | `src/ym07/hybrid.py` |
| Course facts | read out of the course handout at ingestion, supplied with the handout's pages | `src/ym07/handout.py` |
| Citation check | a citation whose passage shares no wording with the answer is removed | `src/ym07/citations.py` |
| Memory fallback | on an out-of-memory failure, ask again with less of the prompt, and say so | `src/ym07/degrade.py` |
| Screenshot reader | Qwen3-VL-4B transcribes a pasted image; the student corrects it before sending | `src/ym07/ocr.py` |
| Web interface | streams the answer, typesets the mathematics, lists the passages | `src/ym07/web.py` |

`configs/eee4114f.yaml` holds everything course-specific. The conditions the report
compares are in `src/ym07/conditions.py`; the delivered one is `A-r1-tuned`.

## Running it

```powershell
.\.venv\Scripts\ym07.exe serve --condition A-r1-tuned     # the tutor in a browser
.\.venv\Scripts\ym07.exe chat  --condition A-r1-tuned     # the same tutor in the terminal
.\.venv\Scripts\ym07.exe fetch-reader                     # once: the vision model for screenshots
```

Installation and the GPU setup are in [LOCAL-ARM.md](LOCAL-ARM.md); `make demo` walks the
pipeline with no model. The card is capped at 15.5 GB in every measured run, because on
Windows an allocation that does not fit pages into system memory instead of failing.

## Reproducing the results

Every number in the report comes from a file in this repository and a script that reads it.

| what | command | reads | writes |
|---|---|---|---|
| The held-out suites | `ym07 stress run --split final --results stress/results/heldout --conditions A-r1-tuned --cap-gb 15.5 --max-new-tokens 6144` | `stress/bank/` | `stress/results/heldout/` |
| The benchmark through the delivered pipeline | `python scripts/benchmark_run.py --conditions A-r1-tuned A-r1-distill-7b` | `bench/eee4114f.jsonl` | `runs/<stamp>-<condition>/` |
| Every October table in the report | `python scripts/october_results.py` | `stress/results/`, `runs/`, `models/` | `stress/results/october-tables.md`, `Report/.../Tables/*.tex` |
| The September acceptance table | `python scripts/final_results.py` | `stress/results/final/`, `grades/` | `stress/results/final/acceptance.md` |
| The interface checks (AT20) | `python scripts/interface_checks.py --condition A-r1-facts` | the live server | stdout |
| Web timing (AT14, AT15) | `python scripts/measure_streaming.py --condition A-r1-tuned` | the live server | `stress/results/.../streaming*.jsonl` |
| The report | `Report\build.ps1` | the chapters and `Tables/` | `Report/Thesis.pdf` |

`protocols/` holds the pre-registered requirements, pass marks and protocols, with every
amendment dated; `protocols/metrics.md` is the measurement record, section by section,
with what each figure was measured on. `STATUS.md` is the current snapshot and the list of
what is still open.

## Repository layout

| path | contents |
|---|---|
| `corpus/raw/`, `corpus/processed/` | the 116 course files and the chunked passages, committed with the convener's permission (`corpus/raw/MANIFEST.md`) |
| `configs/` | per-course configuration; `eee4114f.yaml` is the delivered course |
| `bench/` | the 47-item benchmark, frozen 8 September, and the fixed rules for its calculation items |
| `stress/bank/`, `stress/results/` | the self-scoring test sets and every run's rows, by experiment |
| `protocols/` | requirements, acceptance tests, evaluation and stress protocols, the measurement record |
| `runs/`, `grades/`, `grading/` | benchmark runs, the hand marks, the marking export |
| `models/` | the adapters, weights through LFS, each with its training manifest |
| `training/` | the training sets and the scripts' inputs |
| `scripts/` | the experiment and results scripts; each opens with what it is for |
| `src/ym07/` | the package: tutor, harness, suites, scoring |
| `Report/` | the thesis, built by `build.ps1` |

## Two deliberate choices worth knowing before reading the results

- **No training on any test.** No test question, key or reference answer enters the
  training data; generated pairs are filtered against the benchmark; every improvement
  was decided on a development half and every acceptance test on a held-back one.
- **Nothing is removed or kept on a subset.** The calculator's effect on a third of a
  suite reversed at full size, and two training runs of the same data landed eleven
  points apart; every component decision in the report names the sample it rests on and
  reads adapter comparisons against that band.

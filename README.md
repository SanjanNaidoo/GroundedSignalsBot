# SignalsBot - a course-grounded tutor for EEE4114F

A tutor for UCT's EEE4114F (digital signal processing and an introduction to machine
learning) that answers from the course's own notes, tutorials, past papers and memoranda,
withholds work a student has said is for marks, cites the passage each claim came from,
and runs in its entirety on one 16 GB desktop graphics card. This repository holds
the code that runs it; the report and the results are kept elsewhere.

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

## Setting up

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -e ".[local,reader]"
```

On a 50-series (Blackwell) card, install torch from the CUDA 12.8 index first
(`pip install torch --index-url https://download.pytorch.org/whl/cu128`), or it will have
no kernels for the GPU. The card is capped at 15.5 GB in every measured run, because on
Windows an allocation that does not fit pages into system memory instead of failing.

The tutor answers from the course material, which is **not in this repository**: place
the course files in `corpus/raw/` and run `ym07 ingest` to build `corpus/processed/`
before `serve` or `chat`.

## What this repository holds

This is the code needed to run the tutor. The report, the course material, the benchmark,
the raw results, the grading data and the earlier adapters are not here.

| path | contents |
|---|---|
| `src/ym07/` | the package: tutor, retrieval, router, citation check, web interface |
| `configs/` | per-course configuration; `eee4114f.yaml` is the delivered course |
| `scripts/` | the experiment and training-data scripts; each opens with what it is for. Many read inputs and write outputs that are not in this repository |
| `models/r1-distill-7b-eee4114f-behaviour-v2/` | the delivered adapter, weights through Git LFS, with its training manifest |

## Two deliberate choices worth knowing before reading the results

- **No training on any test.** No test question, key or reference answer enters the
  training data; generated pairs are filtered against the benchmark; every improvement
  was decided on a development half and every acceptance test on a held-back one.
- **Nothing is removed or kept on a subset.** The calculator's effect on a third of a
  suite reversed at full size, and two training runs of the same data landed eleven
  points apart; every component decision in the report names the sample it rests on and
  reads adapter comparisons against that band.

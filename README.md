# SignalsBot - a course-grounded tutor for EEE4114F

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

The tutor answers from the course material, place the course files in `corpus/raw/` and run `ym07 ingest` to build `corpus/processed/` before `serve` or `chat`.


# EEE4114F tutor

A LoRA adapter on Qwen2.5-7B-Instruct for the UCT EEE4114F course, plus the code to chat with it.

- `models/qwen2.5-7b-eee4114f-v3/` — the adapter (weights are stored with Git LFS)
- `src/ym07/` — the chat code; `configs/eee4114f.yaml` — course config and tutor prompt policy
- `stress/` — reserved, empty

```bash
pip install -e '.[local]'          # needs a CUDA GPU
ym07 chat --provider local --model models/qwen2.5-7b-eee4114f-v3 --no-retrieval
ym07 serve --condition C2-oss-tuned --no-retrieval   # browser, http://127.0.0.1:8000
```

The adapter was trained on answers grounded in retrieved course passages. The course material is not in
this repo; to ground the tutor, put a `chunks.jsonl` at `corpus/processed/` and drop `--no-retrieval`.

"""How long the student waits before the first words appear (AT15).

The answer streams, so the wait that matters is not how long the whole reply takes but
how long the box stays empty. This starts the tutor's own server, asks it the 47
benchmark questions through the same endpoint the browser uses, and records the time to
the first piece of text and to the last.

    python scripts/measure_streaming.py --condition C2-oss-hybrid
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ym07 import web                                 # noqa: E402
from ym07.conditions import get_condition            # noqa: E402
from ym07.config import load_config                  # noqa: E402
from ym07.model import build_client_for              # noqa: E402
from ym07.retrieval import build_retriever           # noqa: E402

PORT = 8099


def ask(question: str) -> tuple[float, float, int]:
    conn = HTTPConnection("127.0.0.1", PORT, timeout=900)
    body = json.dumps({"question": question, "history": [], "grounded": True})
    started = time.monotonic()
    conn.request("POST", "/stream", body, {"Content-Type": "application/json"})
    response = conn.getresponse()
    first = None
    pieces = 0
    buffer = ""
    while True:
        chunk = response.read(64)
        if not chunk:
            break
        buffer += chunk.decode("utf-8", "replace")
        while "\n\n" in buffer:
            line, buffer = buffer.split("\n\n", 1)
            if line.startswith("data:") and '"delta"' in line:
                pieces += 1
                if first is None:
                    first = time.monotonic() - started
    conn.close()
    return first or 0.0, time.monotonic() - started, pieces


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="C2-oss-hybrid")
    parser.add_argument("--items", default="bench/eee4114f.jsonl")
    parser.add_argument("--config", default="configs/eee4114f.yaml")
    parser.add_argument("--chunks", default="corpus/processed/chunks.jsonl")
    parser.add_argument("--out", default="stress/results/final/streaming.json")
    # The same cap every measured run uses. Without it a long prompt pages into
    # system memory on Windows instead of failing, and the timing measures the
    # paging rather than the model: the 10 October run of this script had no cap,
    # and its 28.7 s median against the batch harness's 6.7 s on the same 47
    # questions is the difference.
    parser.add_argument("--cap-gb", type=float, default=15.5)
    args = parser.parse_args()

    import torch
    if torch.cuda.is_available() and args.cap_gb:
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, args.cap_gb * 1e9 / torch.cuda.get_device_properties(0).total_memory), 0)

    config = load_config(args.config)
    condition = get_condition(args.condition)
    retriever = build_retriever(args.chunks, config.retrieval.method)
    client = build_client_for(condition, config)
    handler = web.build_handler(config=config, retriever=retriever, client=client,
                                model_id=condition.resolved_model(config),
                                top_k=condition.resolved_top_k(config),
                                use_course_prompt=condition.use_course_prompt)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    questions = [json.loads(line)["question"] for line in
                 Path(args.items).read_text(encoding="utf-8").splitlines() if line.strip()]
    firsts, totals = [], []
    try:
        for n, question in enumerate(questions, start=1):
            first, total, pieces = ask(question)
            firsts.append(first)
            totals.append(total)
            print(f"  {n:>2}/{len(questions)}  first words {first:5.2f}s  complete {total:6.1f}s  "
                  f"{pieces} pieces")
    finally:
        server.shutdown()

    result = {"condition": args.condition, "questions": len(firsts), "cap_gb": args.cap_gb,
              "median_first_words_s": round(statistics.median(firsts), 3),
              "median_complete_s": round(statistics.median(totals), 3),
              "slowest_first_words_s": round(max(firsts), 3)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\nmedian time to first words {result['median_first_words_s']}s "
          f"(slowest {result['slowest_first_words_s']}s); "
          f"median complete answer {result['median_complete_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

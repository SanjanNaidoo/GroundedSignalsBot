"""The screenshot path end to end, on one worked example, kept as a case.

    python scripts/case_screenshot.py              # the delivered model and reader
    python scripts/case_screenshot.py --dry-run    # stub model; the reader still runs

The brief asks for the screenshot route to be shown working on a worked example: a
student pastes a picture of a problem, the reader turns it into text, the student sees
that text, the tutor answers it, and the lecture minutes and past questions appear beside
the answer. This script renders a problem statement to a PNG with a plain font (the way a
screenshot of typed notes would look), posts it to the tutor's own /read endpoint, posts
the transcription to /stream as the page does, and writes everything to
stress/results/cases/screenshot-h1h2/: the image, the transcription with the reader's
notes, the answer, the passages and the cards.

It is a demonstration that the route works, not a measurement; the reader's accuracy is
E18 and the answer's correctness is the suites'. The problem is one I wrote for the
case; nothing here is drawn from a paper.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ym07 import web  # noqa: E402
from ym07.conditions import get_condition  # noqa: E402
from ym07.config import load_config  # noqa: E402
from ym07.model import build_client  # noqa: E402
from ym07.retrieval import build_retriever  # noqa: E402

PORT = 8097
OUT = ROOT / "stress" / "results" / "cases" / "screenshot-h1h2"

PROBLEM = [
    "Question 3 (6 marks)",
    "",
    "Two discrete-time LTI systems have the impulse responses",
    "",
    "    h1[n] = (1/2)^n u[n]          and          h2[n] = u[n + 2] - u[n - 3].",
    "",
    "For each system, determine whether it is (a) causal and (b) BIBO stable.",
    "Justify each answer from the impulse response.",
]


def render(lines: list[str], path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype("consola.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    width = 1180
    line_h = 40
    img = Image.new("RGB", (width, line_h * (len(lines) + 2)), "white")
    draw = ImageDraw.Draw(img)
    y = line_h // 2
    for line in lines:
        draw.text((36, y), line, fill="black", font=font)
        y += line_h
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def post(path: str, payload: dict, timeout: float = 600.0) -> tuple[int, dict]:
    conn = HTTPConnection("127.0.0.1", PORT, timeout=timeout)
    conn.request("POST", path, body=json.dumps(payload), headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read().decode("utf-8")
    conn.close()
    try:
        return resp.status, json.loads(body)
    except json.JSONDecodeError:
        return resp.status, {"raw": body}


def stream(question: str, from_image: str) -> dict:
    conn = HTTPConnection("127.0.0.1", PORT, timeout=900)
    conn.request("POST", "/stream", body=json.dumps({"question": question, "history": [], "grounded": True,
                                                     "from_image": from_image}),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    final = {}
    buffer = ""
    while True:
        chunk = resp.read(1024)
        if not chunk:
            break
        buffer += chunk.decode("utf-8")
        while "\n\n" in buffer:
            line, buffer = buffer.split("\n\n", 1)
            if line.startswith("data:"):
                event = json.loads(line[5:])
                if "delta" not in event and "restart" not in event:
                    final = event
    conn.close()
    return final


class DryRunClient:
    def answer(self, *, system, user, model, max_tokens, effort):
        from ym07.model import Answer

        return Answer(text="(stub) Both impulse responses are given; see [S1].", model=model, effort=effort,
                      input_tokens=10, output_tokens=10, cost_usd=0.0, latency_s=0.01, stop_reason="end_turn")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--condition", default="A-r1-tuned")
    parser.add_argument("--config", default="configs/eee4114f.yaml")
    parser.add_argument("--chunks", default="corpus/processed/chunks.jsonl")
    args = parser.parse_args()

    config = load_config(args.config)
    condition = get_condition(args.condition)
    retriever = build_retriever(args.chunks, config.retrieval.method)
    client = DryRunClient() if args.dry_run else build_client(False, "local", config)
    from ym07.resources import ResourceFinder

    resources = ResourceFinder(method=config.retrieval.method)
    handler = web.build_handler(config=config, retriever=retriever, client=client,
                                model_id=condition.resolved_model(config),
                                top_k=condition.resolved_top_k(config),
                                use_course_prompt=condition.use_course_prompt, resources=resources)
    server = ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    OUT.mkdir(parents=True, exist_ok=True)
    image_path = OUT / "problem.png"
    render(PROBLEM, image_path)
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    try:
        t0 = time.time()
        status, read = post("/read", {"image": "data:image/png;base64," + encoded})
        read_s = time.time() - t0
        print(f"/read -> {status} in {read_s:.1f}s, engine {read.get('engine_name')}")
        print("  text:", (read.get("text") or read.get("error") or "")[:400].replace("\n", " | "))
        transcription = read.get("text") or ""
        # The page would now show this text in the composer for the student to correct;
        # the case sends it on as read, so the record shows what the reader produced.
        final = stream(transcription, from_image=read.get("engine_name") or "") if transcription else {}
        print(f"/stream -> {'error: ' + final['error'] if final.get('error') else str(len(final.get('text') or '')) + ' characters'}, "
              f"{len(final.get('sources') or [])} passages")
        res = final.get("resources") or {}
        print(f"  lecture minutes: {[(w['recording'], w['stamp']) for w in res.get('lectures', [])]}")
        print(f"  past questions:  {[p['reference'][:50] for p in res.get('problems', [])]}")
        case = {
            "case": "screenshot path end to end: h1 and h2, causal and BIBO stable",
            "date": time.strftime("%Y-%m-%d"),
            "condition": args.condition, "dry_run": args.dry_run,
            "problem_as_rendered": PROBLEM,
            "image": image_path.name,
            "read": {"status": status, "latency_s": round(read_s, 1), **{k: read.get(k) for k in ("engine", "engine_name", "notes", "text", "error")}},
            "answer": {k: final.get(k) for k in ("text", "latency_s", "output_tokens", "cost_usd", "note", "error")},
            "sources": final.get("sources"),
            "resources": res,
        }
        (OUT / "case.json").write_text(json.dumps(case, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
        print(f"written {OUT / 'case.json'}")
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

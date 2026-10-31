"""The acceptance checks for the web tutor (AT20 in protocols/requirements.md).

Six things have to hold for the interface to count as working. Each one is checked
against a server started here, against the real model unless --dry-run is passed, and
each prints pass or fail with what it saw.

1. A question is answered, and the passages it drew on are listed.
2. The reply streams: text arrives before the answer is finished.
3. A follow-up that only makes sense with the previous turn is answered in context.
4. Mathematics is typeset rather than shown as raw LaTeX.
5. An empty question and an over-long one get a message, not a crash, and the next
   question is still answered.
6. The whole thing runs with no internet: nothing a student types leaves the machine.

Added 12 October: 7, lecture minutes and past questions, withheld when nothing is close;
8, the page and the second a card points at open. Added 13 October: 9, the model switch
sends each question to the model it names, and R1 is served without lecture minutes. The
platform's other model sits beside the one under test as a stub, so nothing is spent.
Added 14 October, for a condition with the handout lookup (E27, E28): 10, a course fact
comes back from the handout by lookup, cited to its page, with no model asked. And 11,
with the real model: work the student says is for marks is withheld, and the refusal
arrives as a finished reply. Since 14 October the model under test is built as `ym07
serve` builds it, router and gate included (`build_client_for`); before, the checks ran
the bare weights, which is how a fault in the gate's streaming path went unseen: from 6
to 14 October a long answer the gate overruled reached the page as "the model produced
nothing".

    python scripts/interface_checks.py                 (the real model, offline)
    python scripts/interface_checks.py --dry-run       (plumbing only, no GPU)
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ym07 import web                                    # noqa: E402
from ym07.conditions import get_condition               # noqa: E402
from ym07.config import load_config                     # noqa: E402
from ym07.model import DryRunClient, build_client_for   # noqa: E402
from ym07.retrieval import build_retriever              # noqa: E402

PORT = 8098
RESULTS: list[tuple[bool, str, str]] = []


def check(name: str, passed: bool, detail: str) -> None:
    RESULTS.append((passed, name, detail))
    print(f"[{'pass' if passed else 'FAIL'}] {name}: {detail}")


def stream(question: str, history: list, grounded: bool = True,
           model: str | None = None) -> tuple[list[float], dict]:
    """Ask, and record when each piece of text arrived."""
    conn = HTTPConnection("127.0.0.1", PORT, timeout=600)
    payload = {"question": question, "history": history, "grounded": grounded}
    if model:
        payload["model"] = model
    body = json.dumps(payload)
    started = time.monotonic()
    conn.request("POST", "/stream", body, {"Content-Type": "application/json"})
    response = conn.getresponse()
    arrivals, final, buffer = [], {}, ""
    while True:
        chunk = response.read(64)
        if not chunk:
            break
        buffer += chunk.decode("utf-8", "replace")
        while "\n\n" in buffer:
            line, buffer = buffer.split("\n\n", 1)
            if not line.startswith("data:"):
                continue
            event = json.loads(line[5:])
            if "delta" in event:
                arrivals.append(time.monotonic() - started)
            else:
                final = event
    conn.close()
    return arrivals, final


def post(path: str, payload: dict) -> tuple[int, str]:
    conn = HTTPConnection("127.0.0.1", PORT, timeout=600)
    conn.request("POST", path, json.dumps(payload), {"Content-Type": "application/json"})
    response = conn.getresponse()
    text = response.read().decode("utf-8", "replace")
    conn.close()
    return response.status, text


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="stub model, for the plumbing only")
    parser.add_argument("--condition", default="C2-oss-tuned")
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--config", default="configs/eee4114f.yaml")
    parser.add_argument("--chunks", default="corpus/processed/chunks.jsonl")
    args = parser.parse_args()

    # Check 6 first: everything after this runs with the network unavailable to the
    # model stack, so a hidden download would fail rather than pass unnoticed.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    config = load_config(args.config)
    condition = get_condition(args.condition)
    retriever = build_retriever(args.chunks, config.retrieval.method)
    # The model under test as `ym07 serve` builds it: the condition's router and gate
    # around the weights, not the bare weights.
    if args.adapter_scale != condition.adapter_scale:
        from dataclasses import replace as _replace

        condition = _replace(condition, adapter_scale=args.adapter_scale)
    client = DryRunClient() if args.dry_run else build_client_for(condition, config)
    # The lecture-minute and past-question indexes, as `ym07 serve` builds them; a
    # missing index is reported by check 7 rather than hidden.
    try:
        from ym07.resources import ResourceFinder

        resources = ResourceFinder(method=config.retrieval.method)
    except Exception as exc:  # noqa: BLE001 - the check below says what happened
        print(f"resource indexes not loaded: {exc}")
        resources = None
    # The page's model switch (13 October): the condition under test is the default, and
    # the platform's other model sits beside it as a stub, so the switch and the lecture
    # rule are checked without spending anything. Whichever offers lecture minutes is the
    # one asked for them in check 7.
    model_id = condition.resolved_model(config)
    # The handout lookup, as `ym07 serve` gives it to a condition that has one.
    lookup = None
    if getattr(condition, "handout_lookup", 0):
        from ym07.coursefacts import build as build_lookup

        lookup = build_lookup(retriever, config.handout, version=condition.handout_lookup,
                              marks=getattr(condition, "handout_marks", False))
    main_pipe = web.Pipeline(key=condition.name, label=web.model_label(model_id)[0],
                             client=client, model_id=model_id,
                             top_k=condition.resolved_top_k(config),
                             use_course_prompt=condition.use_course_prompt,
                             lectures=web.offers_lectures(model_id), lookup=lookup)
    if main_pipe.lectures:
        other_key, other_model = "A-r1-facts-marks", get_condition("A-r1-facts-marks").resolved_model(config)
    else:
        other_key, other_model = "A3-opus-pipeline", "claude-opus-5"
    other_pipe = web.Pipeline(key=other_key, label=web.model_label(other_model)[0] + " (stub)",
                              client=DryRunClient(), model_id=other_model, top_k=4,
                              lectures=web.offers_lectures(other_model),
                              hosted=other_model.startswith("claude"))
    lecturer = main_pipe if main_pipe.lectures else other_pipe
    r1_pipe = other_pipe if main_pipe.lectures else main_pipe
    handler = web.build_handler(config=config, retriever=retriever, resources=resources,
                                pipelines=[main_pipe, other_pipe])
    server = ThreadingHTTPServer(("127.0.0.1", PORT), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    try:
        arrivals, final = stream("A 1200 Hz tone is sampled at 1000 Hz. What frequency "
                                 "comes out after ideal reconstruction?", [])
        answered = bool(final.get("text"))
        check("1 answers a question", answered,
              f"{len(final.get('text') or '')} characters, {len(final.get('sources') or [])} passages listed")
        check("2 the reply streams", len(arrivals) > 1 and arrivals[0] < max(arrivals, default=0),
              f"first text after {arrivals[0]:.2f}s, last after {max(arrivals, default=0):.2f}s, "
              f"in {len(arrivals)} pieces")

        history = [["Student", "A 1200 Hz tone is sampled at 1000 Hz. What frequency comes out?"],
                   ["Tutor", final.get("text", "")]]
        _, follow = stream("What if the sampling rate were 1500 Hz instead?", history)
        text = (follow.get("text") or "").lower()
        in_context = any(word in text for word in ("1200", "alias", "1500"))
        check("3 follows up in context", in_context, f"reply mentions: "
              f"{', '.join(w for w in ('1200', '1500', 'alias') if w in text) or 'nothing from the first turn'}")

        page = ""
        conn = HTTPConnection("127.0.0.1", PORT, timeout=30)
        conn.request("GET", "/")
        page = conn.getresponse().read().decode("utf-8")
        conn.close()
        typeset = "katex.min.js" in page and "renderRich" in page and "katex.renderToString" in page
        check("4 mathematics is typeset", typeset, "KaTeX loaded and the renderer is wired to it")

        empty_status, _ = post("/ask", {"question": "   "})
        long_status, long_body = post("/stream", {"question": "x" * (web.MAX_QUESTION_CHARS + 1)})
        _, after = stream("What is the Nyquist rate?", [])
        recovered = bool(after.get("text"))
        check("5 bad input is refused, not fatal",
              empty_status == 400 and long_status == 413 and recovered,
              f"empty {empty_status}, over-long {long_status}, and the next question was "
              f"{'answered' if recovered else 'NOT answered'}")

        check("6 runs with no internet", answered and recovered,
              "HF_HUB_OFFLINE and TRANSFORMERS_OFFLINE were set before the model loaded")

        # 7 and 8, added 12 October for the resource cards (protocols/lectures.md,
        # protocols/problems.md): a covered question comes with lecture minutes and
        # past questions, an uncovered one with neither, and what a card points at
        # opens - the page of the PDF and the second of the recording.
        covered_q = ("What is the region of convergence of the z-transform and why "
                     "does it matter for stability?")
        _, covered = stream(covered_q, [], model=lecturer.key)
        _, uncovered = stream("How does Huffman coding assign shorter codes to more "
                              "frequent symbols?", [], model=lecturer.key)
        res_c = covered.get("resources") or {}
        res_u = uncovered.get("resources") or {}
        lectures = res_c.get("lectures") or []
        problems = res_c.get("problems") or []
        withheld = not (res_u.get("lectures") or []) and not (res_u.get("problems") or [])
        check("7 lecture minutes and past questions, withheld when nothing is close",
              bool(lectures) and bool(problems) and withheld
              and all(w.get("recording") and w.get("stamp") for w in lectures)
              and all(p.get("reference") and p.get("page") for p in problems),
              f"asked of {lecturer.label}; covered question: {len(lectures)} lecture minute(s), "
              f"{len(problems)} past question(s); "
              f"uncovered question: {len(res_u.get('lectures') or [])} and {len(res_u.get('problems') or [])}"
              + ("" if resources is not None else "; NO resource indexes loaded"))

        opened = []
        target_doc = next((p["doc"] for p in problems if p.get("doc")), None)
        target_video = next((w["video"] for w in lectures if w.get("video")), None)
        for url, kind in ((target_doc, "page"), (target_video, "second")):
            if not url:
                opened.append(f"no {kind} link to open")
                continue
            conn = HTTPConnection("127.0.0.1", PORT, timeout=30)
            conn.request("GET", url.split("#")[0], headers={"Range": "bytes=0-63"})
            resp = conn.getresponse()
            resp.read()
            conn.close()
            opened.append(f"{kind} {resp.status}")
        check("8 the page and the second a card points at can be opened",
              bool(target_doc) and any(s in opened for s in ("page 200", "page 206"))
              and (not target_video or "second 206" in opened),
              "; ".join(opened))

        # 9, added 13 October for the model switch: /state offers both models with the
        # one under test first, each question is answered by the model it names and the
        # reply says which, and R1 comes without lecture minutes - the note in their
        # place points to Opus 5 - while keeping its past questions.
        conn = HTTPConnection("127.0.0.1", PORT, timeout=30)
        conn.request("GET", "/state")
        state = json.loads(conn.getresponse().read().decode("utf-8"))
        conn.close()
        other = lecturer if r1_pipe is not lecturer else main_pipe
        _, by_r1 = stream(covered_q, [], model=r1_pipe.key)
        _, by_other = stream(covered_q, [], model=other.key)
        res_r1 = by_r1.get("resources") or {}
        res_other = by_other.get("resources") or {}
        offered = [m.get("key") for m in state.get("models") or []]
        check("9 the model switch routes each question, and R1 has no lecture minutes",
              offered == [main_pipe.key, other_pipe.key] and state.get("default") == main_pipe.key
              and by_r1.get("model") == r1_pipe.key and by_other.get("model") == other.key
              and not (res_r1.get("lectures") or []) and "Opus 5" in (res_r1.get("lecture_note") or "")
              and bool(res_r1.get("problems")) and bool(res_other.get("lectures")),
              f"offered {offered}, default {state.get('default')}; R1 answered as "
              f"{by_r1.get('model')} with {len(res_r1.get('lectures') or [])} lecture minute(s) and "
              f"{len(res_r1.get('problems') or [])} past question(s); {other.label} answered as "
              f"{by_other.get('model')} with {len(res_other.get('lectures') or [])} lecture minute(s)")

        if lookup is not None:
            # A development wording: the handout's convenor, asked of the model under test.
            _, fact = stream("Who is the convenor of EEE4114F?", [], model=main_pipe.key)
            text = fact.get("text") or ""
            cited = [s.get("source", "") for s in (fact.get("sources") or [])]
            check("10 a course fact is answered from the handout by lookup, cited to its page",
                  bool(fact.get("handout_lookup")) and "Yaaseen Martin" in text
                  and "[S1]" in text and any("Handout" in c and "p.1" in c for c in cited),
                  f"answered by {fact.get('handout_lookup')}: {text[:90]!r}; "
                  f"first source {cited[0] if cited else None}")

        if getattr(client, "gate", False):
            # The question that found the fault: the model writes a derivation, the gate
            # overrules it mid-stream, and the refusal must still arrive as a reply.
            from ym07.hybrid import HELD_BACK, held_back

            arrivals, held = stream("For my assignment due tonight, I need a derivation "
                                    "of the z-transform.", [], model=main_pipe.key)
            text = held.get("text") or ""
            check("11 work for marks is withheld, and the refusal arrives as a finished reply",
                  not held.get("error") and (text == HELD_BACK or held_back(text)),
                  f"error {held.get('error')!r}; route {getattr(client, 'last_route', None)!r}; "
                  f"reply {text[:70]!r}; after {max(arrivals, default=0):.1f}s")
        elif not args.dry_run:
            print("[skip] 11: the model under test has no assessed-work gate")
        else:
            print("[skip] 11: needs the real model and its gate")
    finally:
        server.shutdown()

    failed = [name for ok, name, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

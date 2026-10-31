"""Run the suites through the web tutor, over HTTP, as a student would reach it.

Every number this project reports was produced by `LocalRunner`, which builds the prompt
and calls the model in process. The browser tutor builds the prompt with the same
function, routes with the same router and falls back with the same ladder - deliberately,
so that the demo and the results describe one system. But "deliberately" is a claim about
the code, and nobody has ever checked it by running the same questions both ways.

This runner closes that gap. It starts the real server, posts to the same `/ask` endpoint
the browser posts to, and returns an `Answer` of the same shape `LocalRunner` returns, so
every suite, every marking rule and every result file works unchanged. What differs is
only that the answer travelled over HTTP from the delivered artefact rather than from a
harness beside it.

It does not stream: `/ask` returns the finished answer, because the suites mark finished
answers and the streaming path is checked separately by scripts/interface_checks.py.

The server runs in this process rather than as a separate one. That is deliberate: it
keeps the card to a single model, and it lets the route, the computed values and the gate
be recorded off the same client object the handler answered with, so a run through the
browser produces rows of exactly the same shape as a run beside it.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
import threading
import time
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from pathlib import Path

from .model import Answer

PORT = 8099
TIMEOUT = 900


class WebRunner:
    """One condition's pipeline, reached the way a student reaches it.

    Mirrors `LocalRunner`'s surface - `answer`, `last_retrieved`, `last_served`, `close`
    - so `run_suite` cannot tell the difference and the rows come out identical in shape.
    """

    def __init__(self, *, config, condition_name: str, retriever, cap_gb: float,
                 adapter_scale: float | None = None, max_new_tokens: int | None = None,
                 corpus: str | None = None, port: int = PORT):
        import torch

        from . import web
        from .conditions import get_condition
        from .model import build_client_for

        condition = get_condition(condition_name)
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, cap_gb * 1e9 / torch.cuda.get_device_properties(0).total_memory), 0)
        self._torch = torch
        self.config = config
        self.condition = condition
        self.label = condition.name
        self.corpus = Path(corpus).stem if corpus else ""
        self.last_retrieved: list[str] = []
        self.last_served = ""
        self.trimmed = 0
        self.port = port

        client = build_client_for(condition, config, dry_run=False)
        if max_new_tokens:
            from .degrade import cap_owner

            owner = cap_owner(client)
            if owner is not None:
                owner.max_new_tokens = max_new_tokens
        # The adapter strength the condition asks for, unless a sweep overrides it.
        self.adapter_scale = condition.adapter_scale if adapter_scale is None else adapter_scale

        # The handout lookup, for the condition that has it (E27), so a run through the
        # browser answers course facts the way the served platform does.
        lookup = None
        if getattr(condition, "handout_lookup", False) and retriever is not None:
            from .coursefacts import build as build_lookup

            lookup = build_lookup(retriever, config.handout, version=condition.handout_lookup,
                                  marks=getattr(condition, "handout_marks", False))
        self.last_lookup = None
        handler = web.build_handler(
            config=config, retriever=retriever, client=client,
            model_id=condition.resolved_model(config),
            top_k=condition.resolved_top_k(config),
            use_course_prompt=condition.use_course_prompt, lookup=lookup)
        self._server = ThreadingHTTPServer(("127.0.0.1", port), handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        # The same object the handler answers with - the server runs in this process,
        # so reading `last_route` off it afterwards is reading what this request did,
        # not bypassing the interface. `run_suite` expects it under this name, and
        # keeping it means a run through the browser records the route, the computed
        # values and the gate exactly as an in-process run does.
        self.client = client

    def answer(self, message: str, history: list[tuple[str, str]] | None = None) -> Answer:
        started = time.monotonic()
        body = json.dumps({
            "question": message,
            "history": [[who, text] for who, text in (history or [])],
            "grounded": self.condition.use_retrieval,
        })
        conn = HTTPConnection("127.0.0.1", self.port, timeout=TIMEOUT)
        try:
            conn.request("POST", "/ask", body, {"Content-Type": "application/json"})
            payload = json.loads(conn.getresponse().read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - depends on the machine
            return self._failed(f"the interface did not answer: {exc}", started)
        finally:
            conn.close()

        if payload.get("error"):
            # The interface returns a sentence a student can read where the card ran out
            # of room on every rung. The suites need it recorded as a failure, so it is
            # both: the text is kept and the row still carries an error.
            return self._failed(payload["error"], started)

        self.last_retrieved = [s.get("source", "") for s in (payload.get("sources") or [])]
        # Whether the handout lookup answered (E27), in the shape LocalRunner records.
        fields = payload.get("handout_lookup")
        self.last_lookup = (SimpleNamespace(fields=tuple(fields)) if fields else None)
        self.last_served = payload.get("note", "") and "degraded" or ""
        self.trimmed += bool(payload.get("note"))
        return Answer(
            text=payload.get("text") or "",
            model=self.condition.resolved_model(self.config),
            effort="n/a",
            stop_reason="end_turn",
            refusal_category=None,
            input_tokens=0,
            output_tokens=int(payload.get("output_tokens") or 0),
            cached_input_tokens=0,
            latency_s=round(payload.get("latency_s") or (time.monotonic() - started), 3),
        )

    def _failed(self, why: str, started: float) -> Answer:
        return Answer(text="", model=self.condition.resolved_model(self.config), effort="n/a",
                      stop_reason=None, refusal_category=None, input_tokens=0,
                      output_tokens=0, cached_input_tokens=0,
                      latency_s=round(time.monotonic() - started, 3), error=why)

    def close(self) -> None:
        import gc

        self._server.shutdown()
        self._server.server_close()
        del self.client
        gc.collect()
        self._torch.cuda.empty_cache()

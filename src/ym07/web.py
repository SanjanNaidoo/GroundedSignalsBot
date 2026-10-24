"""The tutor in a browser — the demo, rather than a description of the demo.

Built from the same parts as `ym07 chat` and the benchmark: the same course
config, the same system prompt, the same retriever, the same model clients. A
web front end that reimplemented any of those would be a different system from
the one the report has numbers for, and showing it to anyone would be showing
them the wrong thing.

Served by the standard library. A web framework would be one more dependency on
a harness that has to stay installable on a laptop with no GPU, for a page that
serves one person at a time on one machine.

Three things are on screen that a terminal hides, and each is an argument the
report makes:

- **Provenance.** Every answer lists the passages it was given, with the file
  and page they came from, so grounding can be checked rather than believed.
- **Latency and cost.** Shown per turn. The case for a 7B model on one card is
  not that it is more accurate than a frontier model; it is what the answer
  costs and how long it takes.
- **Retrieval, on a switch.** Turning it off is the difference between C1 and
  C2 with the same weights, which is otherwise a table in a report.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .config import CourseConfig
from .model import Answer
from .prompts import system_prompt, user_prompt

# Matches the terminal tutor, for the same reason: enough turns to sustain a
# scaffolded exchange, few enough that the resent transcript does not dominate.
MAX_HISTORY_TURNS = 6

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__CODE__ tutor</title>
<style>
  :root {
    --bg: #f7f7f5; --panel: #ffffff; --ink: #1a1a18; --muted: #6b6b66;
    --line: #e3e3de; --accent: #2f5d50; --accent-soft: #e8f0ed;
    --student: #ecebe7; --warn: #8a5a2b; --warn-soft: #fbf1e4;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #17171a; --panel: #1f1f23; --ink: #ececea; --muted: #9a9a94;
      --line: #32323a; --accent: #7fc4ae; --accent-soft: #24322e;
      --student: #2a2a30; --warn: #d9a05b; --warn-soft: #33281a;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  }
  header {
    background: var(--panel); border-bottom: 1px solid var(--line);
    padding: 14px 20px; display: flex; flex-wrap: wrap; gap: 12px 24px;
    align-items: baseline; position: sticky; top: 0; z-index: 5;
  }
  header h1 { font-size: 17px; margin: 0; font-weight: 640; letter-spacing: -0.01em; }
  header .meta { color: var(--muted); font-size: 13px; }
  header .meta code {
    font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
    background: var(--accent-soft); color: var(--accent);
    padding: 1px 6px; border-radius: 4px;
  }
  .grow { flex: 1; }
  .toggle { display: flex; align-items: center; gap: 8px; font-size: 13px; color: var(--muted); }
  .toggle input { accent-color: var(--accent); width: 16px; height: 16px; }
  main {
    max-width: 1100px; margin: 0 auto; padding: 24px 16px 140px;
    display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 28px; align-items: start;
  }
  @media (max-width: 860px) { main { grid-template-columns: minmax(0, 1fr); } }
  #transcript { display: flex; flex-direction: column; gap: 18px; min-width: 0; }
  .turn { display: flex; flex-direction: column; gap: 6px; }
  .who { font-size: 12px; font-weight: 640; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); }
  .bubble {
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
    padding: 14px 16px; white-space: pre-wrap; overflow-wrap: anywhere;
  }
  .student .bubble { background: var(--student); border-color: transparent; }
  .tutor .bubble { border-left: 3px solid var(--accent); }
  .failed .bubble { background: var(--warn-soft); border-left: 3px solid var(--warn); color: var(--warn); }
  .stats { font-size: 12px; color: var(--muted); display: flex; gap: 14px; flex-wrap: wrap; }
  aside {
    background: var(--panel); border: 1px solid var(--line); border-radius: 12px;
    padding: 16px; position: sticky; top: 86px; font-size: 13px;
  }
  aside h2 { font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em;
             color: var(--muted); margin: 0 0 10px; font-weight: 640; }
  .source { padding: 9px 0; border-top: 1px solid var(--line); }
  .source:first-of-type { border-top: 0; }
  .tag { font: 11px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
         color: var(--accent); background: var(--accent-soft);
         padding: 1px 5px; border-radius: 4px; }
  .src-name { display: block; margin-top: 3px; overflow-wrap: anywhere; }
  .score { color: var(--muted); font-size: 12px; }
  .empty { color: var(--muted); }
  footer {
    position: fixed; bottom: 0; left: 0; right: 0; background: var(--panel);
    border-top: 1px solid var(--line); padding: 14px 16px;
  }
  .composer { max-width: 1100px; margin: 0 auto; display: flex; gap: 10px; align-items: flex-end; }
  textarea {
    flex: 1; resize: none; font: inherit; color: inherit; background: var(--bg);
    border: 1px solid var(--line); border-radius: 10px; padding: 11px 13px; min-height: 46px;
  }
  textarea:focus { outline: 2px solid var(--accent); outline-offset: -1px; }
  button {
    font: inherit; font-weight: 600; background: var(--accent); color: var(--bg);
    border: 0; border-radius: 10px; padding: 12px 20px; cursor: pointer;
  }
  button:disabled { opacity: 0.5; cursor: default; }
  .chips { max-width: 1100px; margin: 0 auto 10px; display: flex; gap: 8px; flex-wrap: wrap; }
  .chip {
    font-size: 12px; background: var(--bg); color: var(--muted);
    border: 1px solid var(--line); border-radius: 999px; padding: 5px 12px; cursor: pointer;
  }
  .chip:hover { color: var(--ink); border-color: var(--accent); }
  .total { max-width: 1100px; margin: 8px auto 0; font-size: 12px; color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1>__CODE__ tutor</h1>
  <span class="meta">answering with <code id="model">…</code></span>
  <span class="grow"></span>
  <label class="toggle">
    <input type="checkbox" id="grounded" checked>
    <span id="grounding">course material</span>
  </label>
</header>

<main>
  <div id="transcript">
    <div class="turn tutor">
      <span class="who">Tutor</span>
      <div class="bubble">Ask me anything from the course. I answer from the course
material and cite what I used; I decline questions from other courses, and if you
tell me something is work you are handing in for marks, I will help you get there
rather than write it for you.</div>
    </div>
  </div>
  <aside>
    <h2>Passages used</h2>
    <div id="sources"><span class="empty">Nothing retrieved yet.</span></div>
  </aside>
</main>

<footer>
  <div class="chips" id="chips"></div>
  <form class="composer" id="composer">
    <textarea id="question" rows="1" placeholder="Ask a question…" autofocus></textarea>
    <button id="send" type="submit">Ask</button>
  </form>
  <div class="total" id="total"></div>
</footer>

<script>
const transcript = document.getElementById("transcript");
const sourcesBox = document.getElementById("sources");
const box = document.getElementById("question");
const send = document.getElementById("send");
const totalLine = document.getElementById("total");
const groundedBox = document.getElementById("grounded");
const history = [];
let turns = 0, spend = 0;

const EXAMPLES = [
  "What does the region of convergence of a z-transform tell you?",
  "Explain the small-signal hybrid-pi model of a BJT.",
  "This is due tomorrow for marks - just give me the final answer."
];

const chips = document.getElementById("chips");
EXAMPLES.forEach(text => {
  const chip = document.createElement("button");
  chip.type = "button";
  chip.className = "chip";
  chip.textContent = text.length > 58 ? text.slice(0, 55) + "..." : text;
  chip.title = text;
  chip.onclick = () => { box.value = text; box.focus(); };
  chips.appendChild(chip);
});

function addTurn(who, text, cls) {
  const turn = document.createElement("div");
  turn.className = "turn " + cls;
  const label = document.createElement("span");
  label.className = "who";
  label.textContent = who;
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  turn.append(label, bubble);
  transcript.appendChild(turn);
  window.scrollTo(0, document.body.scrollHeight);
  return turn;
}

function showSources(sources) {
  sourcesBox.textContent = "";
  if (!sources || !sources.length) {
    const none = document.createElement("span");
    none.className = "empty";
    none.textContent = groundedBox.checked
      ? "Nothing was retrieved for that question."
      : "Retrieval is off - the model is answering from its own weights.";
    sourcesBox.appendChild(none);
    return;
  }
  sources.forEach(hit => {
    const row = document.createElement("div");
    row.className = "source";
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = hit.tag;
    const name = document.createElement("span");
    name.className = "src-name";
    name.textContent = hit.source;
    const score = document.createElement("span");
    score.className = "score";
    score.textContent = "score " + hit.score.toFixed(3);
    row.append(tag, name, score);
    sourcesBox.appendChild(row);
  });
}

async function ask(question) {
  addTurn("You", question, "student");
  const pending = addTurn("Tutor", "thinking...", "tutor");
  send.disabled = true;

  try {
    const res = await fetch("/ask", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({question, history, grounded: groundedBox.checked})
    });
    const data = await res.json();
    const bubble = pending.querySelector(".bubble");

    if (data.error) {
      pending.className = "turn failed";
      bubble.textContent = data.error;
    } else {
      bubble.textContent = data.text;
      const stats = document.createElement("div");
      stats.className = "stats";
      const bits = [data.latency_s.toFixed(1) + " s",
                    data.output_tokens + " tokens out"];
      bits.push(data.cost_usd ? "$" + data.cost_usd.toFixed(4) : "no API charge");
      if (data.sources.length) bits.push(data.sources.length + " passages retrieved");
      bits.forEach(text => {
        const span = document.createElement("span");
        span.textContent = text;
        stats.appendChild(span);
      });
      pending.appendChild(stats);

      showSources(data.sources);
      history.push(["Student", question], ["Tutor", data.text]);
      turns += 1;
      spend += data.cost_usd;
      totalLine.textContent = turns + " turn(s) this session"
        + (spend ? ", $" + spend.toFixed(4) + " spent" : ", no API charge");
    }
  } catch (err) {
    pending.className = "turn failed";
    pending.querySelector(".bubble").textContent = "Could not reach the tutor: " + err;
  } finally {
    send.disabled = false;
    box.focus();
  }
}

document.getElementById("composer").addEventListener("submit", event => {
  event.preventDefault();
  const question = box.value.trim();
  if (!question || send.disabled) return;
  box.value = "";
  ask(question);
});

box.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    document.getElementById("composer").requestSubmit();
  }
});

fetch("/state").then(r => r.json()).then(state => {
  document.getElementById("model").textContent = state.model;
  groundedBox.disabled = !state.corpus_loaded;
  groundedBox.checked = state.corpus_loaded;
  document.getElementById("grounding").textContent = state.corpus_loaded
    ? state.passages + " passages of course material"
    : "no course material loaded";
});
</script>
</body>
</html>
"""


def _json(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def build_handler(*, config: CourseConfig, retriever, client, model_id: str,
                  top_k: int, use_course_prompt: bool = True):
    page = PAGE.replace("__CODE__", config.code)

    # One GPU serves one generation at a time. Without this a second browser tab
    # does not queue, it allocates its own KV cache alongside the first and the
    # card runs out of memory - the same reason the batch runner drops to one
    # worker for the local arm.
    generating = threading.Lock()

    class TutorHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "ym07"

        def log_message(self, fmt, *args):  # noqa: A002 - signature is the base class's
            pass  # the console belongs to the server's own progress, not to request logs

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = page.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path == "/state":
                _json(self, {
                    "code": config.code,
                    "model": model_id,
                    "corpus_loaded": retriever is not None,
                    "passages": len(retriever) if retriever is not None else 0,
                    "top_k": top_k,
                })
                return
            _json(self, {"error": "not found"}, status=404)

        def do_POST(self) -> None:
            if self.path != "/ask":
                _json(self, {"error": "not found"}, status=404)
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError) as exc:
                _json(self, {"error": f"bad request: {exc}"}, status=400)
                return

            question = (payload.get("question") or "").strip()
            if not question:
                _json(self, {"error": "no question"}, status=400)
                return

            grounded = bool(payload.get("grounded")) and retriever is not None
            hits = retriever.search(question, top_k) if grounded else []

            # The history arrives from the browser rather than being held here, so
            # the server keeps no per-visitor state and a reload starts clean.
            history = [(who, text) for who, text in (payload.get("history") or [])]

            with generating:
                answer: Answer = client.answer(
                    system=system_prompt(config, grounded=grounded) if use_course_prompt else "",
                    user=user_prompt(question, hits, history[-MAX_HISTORY_TURNS * 2:] or None),
                    model=model_id,
                    max_tokens=config.model.max_tokens,
                    effort=config.model.effort,
                )

            if answer.error:
                _json(self, {"error": answer.error})
                return

            _json(self, {
                "text": answer.text,
                "latency_s": answer.latency_s,
                "cost_usd": answer.cost_usd,
                "output_tokens": answer.output_tokens,
                "sources": [
                    {"tag": f"S{n}", "source": hit.chunk.citation(), "score": hit.score}
                    for n, hit in enumerate(hits, start=1)
                ],
            })

    return TutorHandler


def serve(
    *,
    config: CourseConfig,
    retriever,
    client,
    model_id: str,
    top_k: int,
    use_course_prompt: bool = True,
    condition_name: str | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    out=print,
) -> int:
    handler = build_handler(config=config, retriever=retriever, client=client,
                            model_id=model_id, top_k=top_k,
                            use_course_prompt=use_course_prompt)
    httpd = ThreadingHTTPServer((host, port), handler)

    grounding = (
        f"{len(retriever)} passages, top {top_k} per question"
        if retriever is not None else
        "NO COURSE MATERIAL - run `ym07 ingest` for the grounded tutor"
    )
    out(f"{config.code} tutor on http://{host}:{port}")
    if condition_name:
        out(f"  condition: {condition_name}")
    out(f"  model:     {model_id}")
    out(f"  grounding: {grounding}")
    out("  Ctrl-C to stop\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        out("\nstopped")
    finally:
        httpd.server_close()
    return 0


def run_server(args) -> int:
    import sys

    from .conditions import OSS_MODEL, get_condition
    from .config import load_config
    from .model import build_client
    from .retrieval import build_retriever

    config = load_config(args.config)

    # Serving a named condition is the whole point: it is the difference between
    # demonstrating the system the report measured and demonstrating something
    # that merely resembles it. The tuned adapter was trained on at most four
    # passages, so handing it the config's eight would be a different system.
    condition = get_condition(args.condition) if args.condition else None
    if condition is not None:
        provider = args.provider or condition.resolved_provider(config)
        model_id = args.model or condition.resolved_model(config)
        top_k = condition.resolved_top_k(config)
        use_course_prompt = condition.use_course_prompt
        retrieves = condition.use_retrieval and not args.no_retrieval
    else:
        provider = args.provider or config.model.provider
        model_id = args.model or (OSS_MODEL if provider == "local" else config.model.id)
        top_k = config.retrieval.top_k
        use_course_prompt = True
        retrieves = not args.no_retrieval

    retriever = build_retriever(args.chunks, config.retrieval.method) if retrieves else None
    if retriever is None and retrieves and Path(args.chunks).exists():
        print(f"warning: {args.chunks} is empty - serving ungrounded.", file=sys.stderr)

    return serve(
        config=config,
        retriever=retriever,
        client=build_client(args.dry_run, provider, config),
        model_id=model_id,
        top_k=top_k,
        use_course_prompt=use_course_prompt,
        condition_name=condition.name if condition else None,
        host=args.host,
        port=args.port,
    )

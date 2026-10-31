"""Blind grading in the browser, with the mathematics rendered.

The terminal grader shows answers as their authors wrote them, which for this course
means raw LaTeX: `\\[ y[n]=\\sum_{k=-\\infty}^{\\infty}x[k]h[n-k] \\]`, markdown tables
drawn in pipes. Reading a convolution sum in that form is slow, and a grader squinting
at source is a grader making mistakes - in the one measurement the report rests on.
This serves the same grading as a local page that typesets it.

Nothing about what a grade is changes. Answers are pooled per item and shuffled, shown
as System A, B, C; the grader answers the same questions the terminal asks, in the same
order; and each grade is appended to the same file through the same `Grade` record,
so the report, the similarity measure and the rule-agreement check read it unchanged.
Stopping and restarting resumes where grading left off.

Blinding needs one thing the terminal did not: the page must never learn which system
wrote an answer. Response ids carry the run directory, and run directories are named
after their condition, so the browser is given an opaque token instead and the mapping
stays in this process.
"""

from __future__ import annotations

import json
import random
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .conditions import CITED_CONDITIONS
from .grading import FAITHFULNESS, Grade, _collapse_repeats, append_grade, load_grades
from .items import ItemSet


class GradingSession:
    """What remains to grade, in order, and the blinded view of it."""

    def __init__(self, *, item_set: ItemSet, responses_by_item: dict[str, list[dict]],
                 grades_path: Path, seed: int | None, only: set[str] | None):
        self.items = list(item_set)
        self.item_by_id = {item.id: item for item in self.items}
        self.responses_by_item = responses_by_item
        self.grades_path = grades_path
        self.graded = set(load_grades(grades_path))
        self.only = only
        self.rng = random.Random(seed)
        self.order: dict[str, list[dict]] = {}     # item id -> shuffled answers, drawn once
        self.tokens: dict[str, dict] = {}          # opaque token -> run record
        self.lock = threading.Lock()

    def _wanted(self, record: dict) -> bool:
        return self.only is None or record["response_id"] in self.only

    def _shuffled(self, item_id: str) -> list[dict]:
        if item_id not in self.order:
            answers = [r for r in self.responses_by_item.get(item_id, []) if self._wanted(r)]
            self.rng.shuffle(answers)
            self.order[item_id] = answers
        return self.order[item_id]

    def progress(self) -> dict:
        wanted = [r for item in self.items
                  for r in self.responses_by_item.get(item.id, []) if self._wanted(r)]
        return {"graded": sum(1 for r in wanted if r["response_id"] in self.graded),
                "total": len(wanted)}

    def next(self) -> dict:
        """The next ungraded answer, with everything the grader needs and nothing more."""
        items_with_work = [
            item for item in self.items
            if any(self._wanted(r) and r["response_id"] not in self.graded
                   for r in self.responses_by_item.get(item.id, []))
        ]
        if not items_with_work:
            return {"done": True, "progress": self.progress()}

        item = items_with_work[0]
        answers = self._shuffled(item.id)
        index, record = next((n, r) for n, r in enumerate(answers)
                             if r["response_id"] not in self.graded)

        token = secrets.token_urlsafe(16)
        self.tokens[token] = record
        answer = record["answer"]
        failed = bool(answer.get("error"))
        ask_citations = record["condition"] in CITED_CONDITIONS and bool(record.get("retrieved"))

        return {
            "done": False,
            "progress": self.progress(),
            "item": {
                "id": item.id,
                "type": item.type,
                "topic": item.topic,
                "difficulty": item.difficulty,
                "question": item.question,
                "reference": item.reference_answer or "(none recorded)",
                "must_include": list(item.must_include),
                "must_not_include": list(item.must_not_include),
                "is_policy": item.is_policy_item,
                "remaining_items": len(items_with_work),
            },
            "answer": {
                "token": token,
                "label": chr(ord("A") + index),
                "position": index + 1,
                "of": len(answers),
                "failed": failed,
                "text": (f"[CALL FAILED: {answer['error']}]" if failed
                         else _collapse_repeats(answer.get("text") or "(empty response)")),
                "sources": [f"[{h['tag']}] {h['source']}" for h in record.get("retrieved") or []]
                           if ask_citations else [],
                "ask_citations": ask_citations,
            },
        }

    def record(self, payload: dict) -> tuple[int, dict]:
        """Validate one submitted grade against the item it answers, then save it."""
        record = self.tokens.get(payload.get("token", ""))
        if record is None:
            return 400, {"error": "This answer is no longer current. Reload the page."}
        if record["response_id"] in self.graded:
            return 409, {"error": "That answer was already graded."}

        item = self.item_by_id[record["item_id"]]
        include = payload.get("include")
        exclude = payload.get("exclude")
        if not (isinstance(include, list) and len(include) == len(item.must_include)
                and all(isinstance(v, bool) for v in include)):
            return 400, {"error": "Answer every point the answer must make."}
        if not (isinstance(exclude, list) and len(exclude) == len(item.must_not_include)
                and all(isinstance(v, bool) for v in exclude)):
            return 400, {"error": "Answer every failure to check for."}

        policy = payload.get("policy")
        if item.is_policy_item and not isinstance(policy, bool):
            return 400, {"error": "Say whether it behaved correctly."}
        ask_citations = record["condition"] in CITED_CONDITIONS and bool(record.get("retrieved"))
        citations = payload.get("citations")
        if ask_citations and not isinstance(citations, bool):
            return 400, {"error": "Say whether its citations support its claims."}
        faithfulness = payload.get("faithfulness")
        if faithfulness not in FAITHFULNESS.values():
            return 400, {"error": "Choose grounded, partly or unfaithful."}

        grade = Grade(
            response_id=record["response_id"],
            item_id=item.id,
            condition=record["condition"],
            must_include_hits=include,
            must_not_include_hits=exclude,
            policy_pass=policy if item.is_policy_item else None,
            citations_valid=citations if ask_citations else None,
            faithfulness=faithfulness,
            note=str(payload.get("note") or "").strip(),
        )
        append_grade(self.grades_path, grade)
        self.graded.add(record["response_id"])
        del self.tokens[payload["token"]]
        # The label mapping is deliberately not returned: revealing which system an
        # answer came from would unblind every later item in the same session.
        return 200, {"saved": True, "label": grade.label, "next": self.next()}


def _handler(session: GradingSession, page: bytes):
    class GradingHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # noqa: A002
            pass

        def _send(self, status: int, body: bytes, kind: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict) -> None:
            self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, page, "text/html; charset=utf-8")
            elif self.path == "/api/next":
                with session.lock:
                    self._json(200, session.next())
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path != "/api/grade":
                self._json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._json(400, {"error": "The grade could not be read."})
                return
            with session.lock:
                status, body = session.record(payload)
            self._json(status, body)

    return GradingHandler


def serve_grading(*, item_set: ItemSet, responses_by_item: dict[str, list[dict]],
                  grades_path: Path, seed: int | None = None, only: set[str] | None = None,
                  port: int = 8010, open_browser: bool = True, out=print) -> int:
    session = GradingSession(item_set=item_set, responses_by_item=responses_by_item,
                             grades_path=grades_path, seed=seed, only=only)
    progress = session.progress()
    page = PAGE.encode("utf-8")
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _handler(session, page))
    url = f"http://127.0.0.1:{port}"

    out(f"Grading in the browser at {url}")
    out(f"  {progress['graded']} of {progress['total']} answers already graded, "
        f"saving to {grades_path}")
    out("  Every grade is saved as you go. Ctrl-C here to stop; run again to resume.\n")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        out(f"\nStopped. {session.progress()['graded']} of {progress['total']} graded.")
    finally:
        httpd.server_close()
    return 0


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blind grading</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:wght@400;600&display=swap">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/marked/12.0.2/marked.min.js"></script>
<style>
  :root {
    --paper: #f3f5f3; --panel: #ffffff; --sunk: #eaeeeb; --ink: #16201e; --muted: #5c6763;
    --rule: #d4dbd6; --accent: #1c6355; --accent-soft: #dcebe6; --yes: #256b3d;
    --yes-soft: #e0efe4; --no: #9a3b22; --no-soft: #f6e3dc; --warn: #8f5615;
    --warn-soft: #f7eede; --focus: #1c6355;
    --sans: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
    --serif: "IBM Plex Serif", Georgia, serif;
    --mono: "IBM Plex Mono", ui-monospace, Menlo, monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --paper: #101513; --panel: #181e1c; --sunk: #131917; --ink: #e3e8e5; --muted: #95a09b;
      --rule: #29312d; --accent: #5dbfa6; --accent-soft: #1b2a25; --yes: #71c28a;
      --yes-soft: #18291e; --no: #e08a6f; --no-soft: #2e1c16; --warn: #d1934c;
      --warn-soft: #2a2117; --focus: #5dbfa6;
    }
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; background: var(--paper); color: var(--ink); }
  body { font-family: var(--serif); font-size: 16px; line-height: 1.6; }

  .bar {
    position: sticky; top: 0; z-index: 10; background: var(--panel);
    border-bottom: 1px solid var(--rule); padding: 10px 20px;
    display: flex; flex-wrap: wrap; gap: 8px 22px; align-items: center;
    font-family: var(--mono); font-size: 12px; color: var(--muted);
  }
  .bar b { color: var(--ink); font-weight: 600; }
  .meter { flex: 1 1 160px; min-width: 120px; height: 5px; background: var(--sunk); }
  .meter i { display: block; height: 100%; background: var(--accent); width: 0; transition: width .3s; }

  main {
    display: grid; grid-template-columns: minmax(0, 5fr) minmax(0, 7fr); gap: 22px;
    padding: 20px; max-width: 1480px; margin: 0 auto;
  }
  @media (max-width: 980px) { main { grid-template-columns: minmax(0, 1fr); } }

  .ctx { position: sticky; top: 58px; align-self: start; max-height: calc(100vh - 78px); overflow: auto; }
  @media (max-width: 980px) { .ctx { position: static; max-height: none; } }

  .panel { background: var(--panel); border: 1px solid var(--rule); padding: 16px 18px; }
  .label {
    font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: .1em;
    text-transform: uppercase; color: var(--muted); margin: 0 0 8px;
  }
  .meta { font-family: var(--mono); font-size: 11.5px; color: var(--muted); margin-bottom: 12px; }
  .question { font-size: 17px; margin-bottom: 20px; }
  .reference { background: var(--sunk); border-left: 3px solid var(--accent); padding: 12px 14px; }

  .answer-head { display: flex; align-items: baseline; gap: 12px; margin-bottom: 10px; }
  .answer-head h2 { font-family: var(--sans); font-weight: 600; font-size: 22px; margin: 0; }
  .answer-head span { font-family: var(--mono); font-size: 12px; color: var(--muted); }
  .answer { font-size: 16.5px; overflow-x: auto; }
  .answer.failed { color: var(--warn); background: var(--warn-soft); padding: 10px 12px; }
  .sources { margin-top: 12px; font-family: var(--mono); font-size: 12px; color: var(--muted); }

  .md table { border-collapse: collapse; margin: 10px 0; font-family: var(--sans); font-size: 14px; }
  .md th, .md td { border: 1px solid var(--rule); padding: 5px 9px; text-align: left; }
  .md th { background: var(--sunk); }
  .md code { font-family: var(--mono); font-size: .9em; background: var(--sunk); padding: 1px 4px; }
  .md pre { background: var(--sunk); padding: 10px; overflow-x: auto; }
  .md p { margin: 0 0 10px; }
  .md h1, .md h2, .md h3, .md h4 { font-family: var(--sans); font-size: 16px; margin: 14px 0 6px; }
  .katex-display { overflow-x: auto; overflow-y: hidden; padding: 2px 0; }

  form { margin-top: 18px; }
  fieldset { border: 0; padding: 0; margin: 0 0 18px; }
  legend { font-family: var(--mono); font-size: 11px; font-weight: 600; letter-spacing: .1em;
           text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
  .row {
    display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 14px; align-items: center;
    padding: 9px 12px; border: 1px solid var(--rule); border-top: 0; background: var(--panel);
  }
  .row:first-of-type { border-top: 1px solid var(--rule); }
  .row.current { outline: 2px solid var(--focus); outline-offset: -2px; }
  .row .text { font-size: 15px; }
  .choices { display: flex; gap: 6px; }
  .choices button {
    font-family: var(--sans); font-size: 13.5px; font-weight: 600; min-width: 62px;
    padding: 6px 12px; border: 1px solid var(--rule); background: var(--sunk); color: var(--ink);
    cursor: pointer;
  }
  .choices button:focus-visible, .save:focus-visible, textarea:focus-visible { outline: 2px solid var(--focus); outline-offset: 2px; }
  .choices button kbd { font-family: var(--mono); font-size: 11px; color: var(--muted); margin-left: 5px; }
  .choices button[aria-pressed="true"].y { background: var(--yes-soft); border-color: var(--yes); color: var(--yes); }
  .choices button[aria-pressed="true"].n { background: var(--no-soft); border-color: var(--no); color: var(--no); }
  .choices button[aria-pressed="true"].f { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }
  .hint { font-size: 13px; color: var(--muted); margin: 4px 0 0; }

  textarea {
    width: 100%; min-height: 56px; font: inherit; font-size: 14.5px; color: var(--ink);
    background: var(--panel); border: 1px solid var(--rule); padding: 8px 10px; resize: vertical;
  }
  .actions { display: flex; flex-wrap: wrap; gap: 14px; align-items: center; margin-top: 6px; }
  .save {
    font-family: var(--sans); font-weight: 600; font-size: 15px; padding: 11px 22px;
    background: var(--accent); color: var(--panel); border: 0; cursor: pointer;
  }
  .save:disabled { opacity: .45; cursor: default; }
  .status { font-family: var(--mono); font-size: 12.5px; color: var(--muted); }
  .status.err { color: var(--no); }
  .keys { font-family: var(--mono); font-size: 11.5px; color: var(--muted); margin-top: 14px; }
  .keys kbd { border: 1px solid var(--rule); padding: 0 4px; background: var(--sunk); }

  .finished { max-width: 640px; margin: 80px auto; text-align: center; font-family: var(--sans); }
  .finished h1 { font-size: 30px; margin: 0 0 10px; }
  @media (prefers-reduced-motion: reduce) { .meter i { transition: none; } }
</style>
</head>
<body>
<div class="bar" id="bar">
  <span>Answers graded <b id="p-done">-</b> / <b id="p-total">-</b></span>
  <span class="meter" aria-hidden="true"><i id="p-meter"></i></span>
  <span>Questions left <b id="p-items">-</b></span>
</div>
<div id="app"><p style="padding:20px">Loading...</p></div>

<script>
const app = document.getElementById("app");
let current = null;
let state = null;

function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

// Mathematics is lifted out before markdown sees it, then typeset back in. Markdown
// otherwise reads the underscores in x_{k} as emphasis and the backslashes as escapes,
// and returns a formula with pieces missing.
function renderRich(src) {
  const math = [];
  let text = String(src).replace(
    /\\\[([\s\S]+?)\\\]|\$\$([\s\S]+?)\$\$|\\\(([\s\S]+?)\\\)|\$([^\n$]+?)\$/g,
    (m, a, b, c, d) => {
      const display = a !== undefined || b !== undefined;
      const tex = a !== undefined ? a : b !== undefined ? b : c !== undefined ? c : d;
      math.push({ tex, display });
      return "@@MATH" + (math.length - 1) + "@@";
    });
  text = text.replace(/</g, "&lt;");
  let html = window.marked ? marked.parse(text) : "<p>" + text.replace(/\n/g, "<br>") + "</p>";
  return html.replace(/@@MATH(\d+)@@/g, (m, i) => {
    const { tex, display } = math[Number(i)];
    if (!window.katex) return "<code>" + escapeHtml(tex) + "</code>";
    try {
      return katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: "ignore" });
    } catch (e) {
      return "<code>" + escapeHtml(tex) + "</code>";
    }
  });
}

function setProgress(p, itemsLeft) {
  document.getElementById("p-done").textContent = p.graded;
  document.getElementById("p-total").textContent = p.total;
  document.getElementById("p-items").textContent = itemsLeft || 0;
  document.getElementById("p-meter").style.width = p.total ? (100 * p.graded / p.total) + "%" : "0";
}

function slots() {
  // The order the keyboard fills them in, and the order the terminal asks them.
  const s = [];
  current.item.must_include.forEach((_, i) => s.push({ kind: "include", i }));
  current.item.must_not_include.forEach((_, i) => s.push({ kind: "exclude", i }));
  if (current.item.is_policy) s.push({ kind: "policy" });
  if (current.answer.ask_citations) s.push({ kind: "citations" });
  s.push({ kind: "faithfulness" });
  return s;
}

function valueOf(slot) {
  if (slot.kind === "include") return state.include[slot.i];
  if (slot.kind === "exclude") return state.exclude[slot.i];
  return state[slot.kind];
}

function setValue(slot, v) {
  if (slot.kind === "include") state.include[slot.i] = v;
  else if (slot.kind === "exclude") state.exclude[slot.i] = v;
  else state[slot.kind] = v;
  paint();
}

function complete() { return slots().every(s => valueOf(s) !== null); }
function firstOpen() { return slots().find(s => valueOf(s) === null); }
function key(slot) { return slot.kind + (slot.i === undefined ? "" : slot.i); }

function paint() {
  const open = firstOpen();
  document.querySelectorAll("[data-slot]").forEach(row => {
    const slot = JSON.parse(row.dataset.slot);
    row.classList.toggle("current", !!open && key(open) === key(slot));
    row.querySelectorAll("button").forEach(b => {
      const v = valueOf(slot);
      const val = b.dataset.value === "true" ? true : b.dataset.value === "false" ? false : b.dataset.value;
      b.setAttribute("aria-pressed", String(v === val));
    });
  });
  document.getElementById("save").disabled = !complete();
}

function yesNoRow(slot, text) {
  return `<div class="row" data-slot='${JSON.stringify(slot)}'>
    <div class="text md">${renderRich(text)}</div>
    <div class="choices">
      <button type="button" class="y" data-value="true">Yes<kbd>Y</kbd></button>
      <button type="button" class="n" data-value="false">No<kbd>N</kbd></button>
    </div></div>`;
}

function show(payload) {
  const moved = !current || !payload.done && current.item.id !== payload.item.id;
  current = payload;
  if (payload.done) {
    setProgress(payload.progress, 0);
    app.innerHTML = `<div class="finished"><h1>Everything is graded.</h1>
      <p>${payload.progress.graded} answers saved. You can close this tab and stop the
      server with Ctrl-C.</p></div>`;
    return;
  }
  const { item, answer } = payload;
  state = {
    include: item.must_include.map(() => null),
    exclude: item.must_not_include.map(() => null),
    policy: item.is_policy ? null : undefined,
    citations: answer.ask_citations ? null : undefined,
    faithfulness: null,
  };
  setProgress(payload.progress, item.remaining_items);

  const include = item.must_include.map((p, i) => yesNoRow({ kind: "include", i }, p)).join("");
  const exclude = item.must_not_include.map((p, i) => yesNoRow({ kind: "exclude", i }, p)).join("");

  app.innerHTML = `<main>
    <aside class="ctx panel">
      <p class="label">Question</p>
      <div class="meta">${escapeHtml(item.id)} · ${escapeHtml(item.type)} · ${escapeHtml(item.topic)} · difficulty ${item.difficulty}</div>
      <div class="question md">${renderRich(item.question)}</div>
      <p class="label">Reference answer</p>
      <div class="reference md">${renderRich(item.reference)}</div>
    </aside>
    <section>
      <div class="panel">
        <div class="answer-head"><h2>System ${answer.label}</h2>
          <span>answer ${answer.position} of ${answer.of} for this question</span></div>
        <div class="answer md ${answer.failed ? "failed" : ""}">${renderRich(answer.text)}</div>
        ${answer.sources.length ? `<div class="sources">Sources available to this system:<br>${answer.sources.map(escapeHtml).join("<br>")}</div>` : ""}
      </div>
      <form id="form" autocomplete="off">
        <fieldset><legend>Does the answer make this point?</legend>${include}</fieldset>
        ${exclude ? `<fieldset><legend>Does it contain this failure?</legend>${exclude}</fieldset>` : ""}
        ${item.is_policy ? `<fieldset><legend>Behaviour</legend>${yesNoRow({ kind: "policy" }, "Behaved correctly: declined, or guided rather than answering.")}</fieldset>` : ""}
        ${answer.ask_citations ? `<fieldset><legend>Citations</legend>${yesNoRow({ kind: "citations" }, "Every citation actually supports the claim it is attached to.")}</fieldset>` : ""}
        <fieldset><legend>Faithfulness</legend>
          <div class="row" data-slot='{"kind":"faithfulness"}'>
            <div class="text">Is everything it states supported by the course material?
              <p class="hint">Judge it against the reference answer and the course, including when the system had no passages.</p></div>
            <div class="choices">
              <button type="button" class="f" data-value="grounded">Grounded<kbd>G</kbd></button>
              <button type="button" class="f" data-value="partly">Partly<kbd>P</kbd></button>
              <button type="button" class="f" data-value="unfaithful">Unfaithful<kbd>U</kbd></button>
            </div></div>
        </fieldset>
        <fieldset><legend>Note (optional)</legend><textarea id="note" name="note"></textarea></fieldset>
        <div class="actions">
          <button class="save" id="save" type="submit" disabled>Save and next</button>
          <span class="status" id="status"></span>
        </div>
        <div class="keys"><kbd>Y</kbd> / <kbd>N</kbd> answer the highlighted row ·
          <kbd>G</kbd> <kbd>P</kbd> <kbd>U</kbd> faithfulness · <kbd>Enter</kbd> saves ·
          click any row to change it</div>
      </form>
    </section></main>`;

  document.querySelectorAll("[data-slot]").forEach(row => {
    const slot = JSON.parse(row.dataset.slot);
    row.querySelectorAll("button").forEach(b => b.addEventListener("click", () => {
      const v = b.dataset.value;
      setValue(slot, v === "true" ? true : v === "false" ? false : v);
    }));
  });
  document.getElementById("form").addEventListener("submit", e => { e.preventDefault(); save(); });
  paint();
  if (moved) window.scrollTo(0, 0);
}

async function save() {
  if (!complete()) return;
  const button = document.getElementById("save");
  const status = document.getElementById("status");
  button.disabled = true;
  status.className = "status";
  status.textContent = "Saving...";
  try {
    const res = await fetch("/api/grade", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        token: current.answer.token,
        include: state.include, exclude: state.exclude,
        policy: state.policy === undefined ? null : state.policy,
        citations: state.citations === undefined ? null : state.citations,
        faithfulness: state.faithfulness,
        note: document.getElementById("note").value,
      }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || "The grade was not saved.");
    show(body.next);
  } catch (err) {
    status.className = "status err";
    status.textContent = err.message + " Nothing was lost; try again.";
    button.disabled = !complete();
  }
}

document.addEventListener("keydown", e => {
  if (!current || current.done) return;
  const inNote = document.activeElement && document.activeElement.id === "note";
  if (e.key === "Enter" && (!inNote || e.ctrlKey)) { e.preventDefault(); save(); return; }
  if (inNote || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  const faith = { g: "grounded", p: "partly", u: "unfaithful" };
  if (faith[k]) { setValue({ kind: "faithfulness" }, faith[k]); return; }
  if (k === "y" || k === "n") {
    const open = firstOpen();
    if (open && open.kind !== "faithfulness") setValue(open, k === "y");
  }
});

fetch("/api/next").then(r => r.json()).then(show).catch(() => {
  app.innerHTML = '<p style="padding:20px">Could not reach the grading server. Is it still running in the terminal?</p>';
});
</script>
</body>
</html>
"""

#!/usr/bin/env bash
# Narrated walkthrough of the YM-07 benchmark harness.
#
#   ./scripts/demo.sh          pause between steps (use this in the meeting)
#   ./scripts/demo.sh --auto   run straight through, no pauses, no grading
#
# Everything here is a dry run: no model is called and nothing is spent.

set -euo pipefail
cd "$(dirname "$0")/.."

AUTO=0
[[ "${1:-}" == "--auto" ]] && AUTO=1

if [[ -t 1 ]]; then
  B=$'\033[1m'; DIM=$'\033[2m'; CYAN=$'\033[36m'; YEL=$'\033[33m'; R=$'\033[0m'
else
  B=""; DIM=""; CYAN=""; YEL=""; R=""
fi

YM07=".venv/bin/ym07"
CHUNKS="corpus/processed/DEMO-chunks.jsonl"

say()  { printf '\n%s%s%s\n' "$CYAN$B" "$*" "$R"; }
note() { printf '%s%s%s\n' "$DIM" "$*" "$R"; }

step() {
  local title="$1"; shift
  printf '\n%s%s\n' "$B" "$(printf '═%.0s' {1..78})"
  printf '  %s\n' "$title"
  printf '%s%s\n\n' "$(printf '═%.0s' {1..78})" "$R"
  printf '%s$ %s%s\n\n' "$YEL" "$*" "$R"
  if [[ $AUTO -eq 0 ]]; then
    read -rsp "$(printf '%s   [Enter to run]%s' "$DIM" "$R")" _ </dev/tty || true
    printf '\n\n'
  fi
  "$@"
}

# --- preflight ------------------------------------------------------------
if [[ ! -x "$YM07" ]]; then
  say "Setting up (one-off)"
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet -e .
fi
[[ -f "$CHUNKS" ]] || python3 scripts/make_demo_corpus.py >/dev/null

say "YM-07 — benchmark harness for a course-specific tutor (EEE4114F)"
note "This is the measuring instrument, not the tutor. The report is marked on"
note "evidence, so the instrument has to exist before the thing being measured."
note ""
note "Nothing in this demo calls a model or spends money."

step "1/5  The experimental design — what we compare, and why" \
  "$YM07" conditions

note ""
note "C0 is the honest control: bare model, no retrieval, no course prompt."
note "C1 is the condition most projects skip — it separates prompting from retrieval."

step "2/5  The benchmark — and its gaps, stated openly" \
  "$YM07" items

note ""
note "51 items across all 10 course topics, written from the notes, tutorials and past"
note "papers. error_finding and integrity are the types a generic chatbot"
note "handles worst, and the ones that will differentiate the results."

rm -rf runs/*/ 2>/dev/null || true
step "3/5  Running all three conditions (dry run — no API calls)" \
  "$YM07" run C0-baseline C1-prompted C2-rag --dry-run --chunks "$CHUNKS"

say "What a run records"
MANIFEST=$(ls -d runs/*C2-rag | tail -1)/manifest.json
python3 - "$MANIFEST" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
for k in ("condition", "model", "effort", "use_retrieval", "top_k",
          "config_digest", "items_digest", "item_count",
          "total_cost_usd", "errors", "refusals", "dry_run"):
    print(f"  {k:<20} {m[k]}")
PY
note ""
note "config_digest and items_digest are what make 'we froze the benchmark'"
note "checkable rather than asserted — if either changes, runs aren't comparable"
note "and the tool says so."

if [[ $AUTO -eq 0 ]]; then
  printf '\n%sShow the blind grading step? It is interactive — Ctrl-C to leave it.%s [y/N] ' "$B" "$R"
  read -r reply </dev/tty || reply=n
  if [[ "$reply" =~ ^[Yy] ]]; then
    step "4/5  Blind grading — answers pooled per item, shuffled, unlabelled" \
      "$YM07" grade --allow-dry-run --limit 1 || true
  fi
else
  say "4/5  Blind grading (skipped in --auto; it is interactive)"
fi

step "5/5  The report" \
  "$YM07" report

say "What this does not do yet"
note "  - Ingestion (PDF -> text -> chunks) is not built. It depends on what form"
note "    the Amathuba material is actually in."
note "  - The corpus above is 6 placeholder passages, labelled DEMO-placeholder.pdf."
note "    They are not course material."
note "  - 8 benchmark items, not 120."
printf '\n'

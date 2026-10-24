"""Command line entry point.

  ym07 chat     talk to the tutor in the terminal
  ym07 serve    the same tutor in a browser
"""

from __future__ import annotations

import argparse
import sys

from .chat import run_chat
from .conditions import CONDITIONS
from .web import run_server

DEFAULT_CONFIG = "configs/eee4114f.yaml"
DEFAULT_CHUNKS = "corpus/processed/chunks.jsonl"


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--chunks", default=DEFAULT_CHUNKS,
                   help="retrieval corpus; missing or empty means the tutor runs ungrounded")
    p.add_argument("--provider", default=None, help="anthropic | local (default: the config's)")
    p.add_argument("--model", default=None, help="override the model id")
    p.add_argument("--no-retrieval", action="store_true")
    p.add_argument("--dry-run", action="store_true")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ym07", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("chat", help="talk to the tutor interactively")
    _common(p)
    p.set_defaults(func=run_chat)

    p = sub.add_parser("serve", help="the tutor in a browser")
    _common(p)
    p.add_argument("--condition", default=None, choices=sorted(CONDITIONS),
                   help="C2-oss-tuned serves the fine-tuned model in models/")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=run_server)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

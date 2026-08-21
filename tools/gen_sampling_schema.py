#!/usr/bin/env python3
"""Snapshot the request schema of a live vLLM server.

The playground renders its sampling controls from this, so the fields offered
are exactly the fields the server accepts — vLLM's extensions included, and
without the guesswork of tracking which release renamed guided_json into
structured_outputs.

    python3 tools/gen_sampling_schema.py --url http://localhost:8000 \
        --out app/data/sampling_params.json

The dashboard re-fetches this per endpoint at runtime; the checked-in file is
the fallback for when no server is up yet.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.sampling_spec import extract

WANTED = ("ChatCompletionRequest", "CompletionRequest")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--file", help="read a saved openapi.json instead of fetching")
    parser.add_argument("--out", default="app/data/sampling_params.json")
    args = parser.parse_args()

    source = args.file or (args.url.rstrip("/") + "/openapi.json")
    try:
        if args.file:
            spec = json.loads(Path(args.file).read_text(encoding="utf-8"))
        else:
            with urllib.request.urlopen(source, timeout=30) as response:
                spec = json.load(response)
    except Exception as exc:
        sys.exit(f"could not read {source}: {exc}")

    payload = extract(spec)
    if not payload["requests"]:
        sys.exit(f"{source} exposes no chat/completion request schema (a pooling-only server?)")

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1)
        handle.write("\n")

    for name, model in payload["requests"].items():
        print(f"{name}: {len(model['fields'])} fields", file=sys.stderr)
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

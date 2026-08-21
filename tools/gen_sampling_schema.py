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

WANTED = ("ChatCompletionRequest", "CompletionRequest")


def resolve(schema: dict, components: dict, depth: int = 0) -> dict:
    """Flatten a property's type into something a form renderer can use."""
    if depth > 4:
        return {"type": "json"}

    if "$ref" in schema:
        name = schema["$ref"].rsplit("/", 1)[-1]
        target = components.get(name, {})
        return {"type": "json", "ref": name, "properties": sorted(target.get("properties", {}))}

    if "anyOf" in schema:
        options = [resolve(s, components, depth + 1) for s in schema["anyOf"]]
        concrete = [o for o in options if o.get("type") not in (None, "null")]
        base = concrete[0] if concrete else {"type": "string"}
        return {**base, "nullable": any(o.get("type") == "null" for o in options)}

    kind = schema.get("type", "string")
    out: dict = {"type": kind}
    if kind == "array":
        out["items"] = resolve(schema.get("items", {}), components, depth + 1).get("type", "string")
    if "enum" in schema:
        out["enum"] = schema["enum"]
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if key in schema:
            out[key] = schema[key]
    return out


def extract(spec: dict) -> dict:
    components = spec.get("components", {}).get("schemas", {})
    out: dict = {"paths": sorted(spec.get("paths", {})), "requests": {}}
    for name in WANTED:
        model = components.get(name)
        if not model:
            continue
        fields = {}
        for field, definition in model.get("properties", {}).items():
            entry = resolve(definition, components)
            if "default" in definition:
                entry["default"] = definition["default"]
            if definition.get("description"):
                entry["description"] = definition["description"][:400]
            fields[field] = entry
        out["requests"][name] = {
            "fields": fields,
            "required": model.get("required", []),
        }
    structured = components.get("StructuredOutputsParams")
    if structured:
        out["structured_outputs"] = {
            field: resolve(definition, components)
            for field, definition in structured.get("properties", {}).items()
        }
    return out


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

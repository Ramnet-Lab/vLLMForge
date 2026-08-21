#!/usr/bin/env python3
"""Generate a machine-readable schema of `vllm serve` flags from a vLLM image.

The dashboard renders its "server parameters" form directly from this file, so
the form always matches the vLLM build that will actually run the model. Run it
again whenever IMAGE changes:

    python3 tools/gen_vllm_schema.py --image nvcr.io/nvidia/vllm:26.07-py3 \
        --out app/data/vllm_args.json

Two passes are needed because vLLM builds most of its help text lazily from
dataclass field docstrings: argparse actions carry the type/default/choices but
an EMPTY help string, while `--help=all` renders the prose. We introspect the
parser for structure and scrape the rendered help for description.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# argparse lays help out at a fixed column; anything left of it on a flag line
# is the invocation (metavar + aliases), not description.
HELP_COLUMN = 24

INTROSPECT = r'''
import json
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.openai.cli_args import make_arg_parser

parser = make_arg_parser(FlexibleArgumentParser())

group_of = {}
for group in parser._action_groups:
    for action in group._group_actions:
        group_of[id(action)] = group.title

def jsonable(value):
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return str(value)

out = []
for action in parser._actions:
    if not action.option_strings or action.dest in ("help", "config"):
        continue
    kind = type(action).__name__
    out.append({
        "dest": action.dest,
        "flags": list(action.option_strings),
        "type": getattr(action.type, "__name__", None),
        "nargs": action.nargs,
        "default": jsonable(action.default),
        "choices": [str(c) for c in action.choices] if action.choices else None,
        "action": kind,
        "group": group_of.get(id(action), "options"),
        "required": bool(action.required),
    })

import vllm
print("@@JSON@@" + json.dumps({"vllm_version": vllm.__version__, "args": out}))
'''


# vLLM refuses to build its config dataclasses without a visible device, so even
# `--help` needs the GPU handed through.
GPU_FLAGS = ["--runtime", "nvidia", "--gpus", "all"]


def _docker_run(image: str, *cmd: str) -> str:
    proc = subprocess.run(
        ["docker", "run", "--rm", *GPU_FLAGS, "--entrypoint", cmd[0], image, *cmd[1:]],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 and not proc.stdout:
        sys.exit(f"docker run failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    return proc.stdout


def introspect(image: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "introspect.py"
        script.write_text(INTROSPECT)
        proc = subprocess.run(
            [
                "docker", "run", "--rm", *GPU_FLAGS,
                "--entrypoint", "python",
                "-v", f"{script}:/tmp/introspect.py:ro",
                image, "/tmp/introspect.py",
            ],
            capture_output=True,
            text=True,
        )
    for line in proc.stdout.splitlines():
        if line.startswith("@@JSON@@"):
            return json.loads(line[len("@@JSON@@"):])
    sys.exit(f"introspection produced no JSON:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")


def scrape_help(image: str) -> tuple[dict[str, str], dict[str, str]]:
    """Return (help_by_flag, section_description_by_title) from `--help=all`."""
    text = _docker_run(image, "vllm", "serve", "--help=all")
    help_by_flag: dict[str, str] = {}
    sections: dict[str, str] = {}

    # argparse separates the invocation from the description with 2+ spaces, so
    # splitting on the column would truncate flags whose names run past it.
    flag_line = re.compile(r"^ {2}(-{1,2}\S.*?)(?:\s{2,}(\S.*))?$")
    section_line = re.compile(r"^(\S[^:]*):\s*$")

    current_flags: list[str] = []
    buffer: list[str] = []
    current_section: str | None = None
    section_buffer: list[str] = []

    def flush_flag() -> None:
        nonlocal current_flags, buffer
        if current_flags:
            body = " ".join(b for b in buffer if b).strip()
            body = re.sub(r"\s+", " ", body)
            body = re.sub(r"\s*\(default: .*\)\s*$", "", body).strip()
            for flag in current_flags:
                help_by_flag.setdefault(flag, body)
        current_flags, buffer = [], []

    def flush_section() -> None:
        nonlocal current_section, section_buffer
        if current_section and section_buffer:
            sections.setdefault(
                current_section, re.sub(r"\s+", " ", " ".join(section_buffer)).strip()
            )
        section_buffer = []

    for line in text.splitlines():
        match = flag_line.match(line)
        if match:
            flush_flag()
            flush_section()
            invocation, tail = match.group(1), match.group(2)
            current_flags = [
                part.strip().split(" ")[0].split("=")[0].rstrip(",")
                for part in invocation.split(", ")
                if part.strip().startswith("-")
            ]
            buffer = [tail.strip()] if tail else []
        elif current_flags and line.startswith(" " * HELP_COLUMN):
            buffer.append(line.strip())
        elif not line.strip():
            continue
        else:
            flush_flag()
            section = section_line.match(line)
            if section:
                flush_section()
                current_section = section.group(1).strip()
            elif current_section and line.startswith("  "):
                section_buffer.append(line.strip())

    flush_flag()
    flush_section()
    return help_by_flag, sections


# Widget hints let the UI render the right control without hard-coding 266 flags.
_INT_TYPES = {"int", "human_readable_int", "human_readable_int_or_auto"}
_FLOAT_TYPES = {"float"}


def classify(arg: dict) -> str:
    if arg["action"] in ("_StoreTrueAction", "_StoreFalseAction", "BooleanOptionalAction"):
        return "bool"
    if any(f.startswith("--no-") for f in arg["flags"]):
        return "bool"
    if arg["choices"]:
        return "enum"
    if arg["type"] in _INT_TYPES:
        return "int"
    if arg["type"] in _FLOAT_TYPES:
        return "float"
    if arg["nargs"] in ("+", "*"):
        return "list"
    if arg["type"] in ("json", "dict", "_optional_dict"):
        return "json"
    return "str"


def primary_flag(arg: dict) -> str:
    """The long flag, ignoring the argparse-generated --no-* negation."""
    longs = [f for f in arg["flags"] if f.startswith("--") and not f.startswith("--no-")]
    return longs[0] if longs else arg["flags"][0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="nvcr.io/nvidia/vllm:26.07-py3")
    parser.add_argument("--out", default="app/data/vllm_args.json")
    args = parser.parse_args()

    if not shutil.which("docker"):
        sys.exit("docker not found on PATH")

    print(f"introspecting {args.image} ...", file=sys.stderr)
    data = introspect(args.image)
    print(f"  {len(data['args'])} flags, vLLM {data['vllm_version']}", file=sys.stderr)

    print("scraping rendered help ...", file=sys.stderr)
    help_by_flag, sections = scrape_help(args.image)
    print(f"  {len(help_by_flag)} help blocks, {len(sections)} sections", file=sys.stderr)

    described = 0
    for arg in data["args"]:
        arg["widget"] = classify(arg)
        arg["flag"] = primary_flag(arg)
        arg["negatable"] = any(f.startswith("--no-") for f in arg["flags"])
        for flag in arg["flags"]:
            if help_by_flag.get(flag):
                arg["help"] = help_by_flag[flag]
                described += 1
                break
        else:
            arg["help"] = ""
    print(f"  {described}/{len(data['args'])} flags described", file=sys.stderr)

    payload = {
        "image": args.image,
        "vllm_version": data["vllm_version"],
        "sections": sections,
        "args": sorted(data["args"], key=lambda a: (a["group"], a["dest"])),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {out} ({out.stat().st_size // 1024} KiB)", file=sys.stderr)


if __name__ == "__main__":
    main()
